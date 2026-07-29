# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""Opt-in GPU profiler that attributes GPU utilization to training phases.

A background thread polls NVML (falling back to ``nvidia-smi``) at a fixed
interval and records, per visible GPU:

  - ``sm_util``       %    SM / "GPU" utilization (compute busy)
  - ``mem_bw_util``   %    memory-controller utilization == HBM bandwidth busy
  - ``mem_used_gb``        allocated framebuffer
  - ``power_w``            board power draw
  - ``sm_clock_mhz``       SM clock
  - ``pcie_tx_mb_s`` /
    ``pcie_rx_mb_s``       PCIe throughput (CPU<->GPU; offload indicator)

Each sample is tagged with the phase currently on top of a thread-safe phase
stack. The stack is driven by ``verl.trainer.ppo.ray_trainer._timer``, so every
phase (gen / old_log_prob / teacher_forward / ref / update_actor / ...) is
tagged automatically with no per-call-site changes.

Why a node-wide NVML sampler works here: ``_timer`` and the rollout loop run in
the *driver* process, and each phase boundary wraps a blocking Ray call into the
GPU workers (``generate_sequences``, ``compute_log_prob``, ``update_actor`` ...).
NVML reports device-wide utilization regardless of process, so the driver-side
phase tag and the device-wide samples line up in wall-clock time.

Everything is a no-op unless ``GPU_PROFILER=1``: when disabled the sampler is
never started and ``push_phase``/``pop_phase``/``mean_util_between`` return
immediately, so there is zero overhead and zero behavior change in normal runs.

Env vars
--------
  GPU_PROFILER=1               enable the profiler
  GPU_PROFILER_INTERVAL=0.3    sample interval, seconds
  GPU_PROFILER_IDLE_THRESH=30  a sample whose mean-across-GPU sm_util is below
                               this counts as "idle"
  GPU_PROFILER_BOUNDARY=step   completing this phase triggers a per-step report
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import threading
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = [
    "enabled",
    "push_phase",
    "pop_phase",
    "mean_util_between",
    "per_gpu_util_between",
    "now",
    "report_and_reset",
]


# [EXPLAIN] `_env_flag` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _env_flag(name: str, default: str = "0") -> bool:
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# [EXPLAIN] `enabled` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def enabled() -> bool:
    # Cached so the hot push/pop path is a single dict lookup + bool.
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    global _ENABLED
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _ENABLED is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _ENABLED = _env_flag("GPU_PROFILER")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _ENABLED


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ENABLED = None
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_INTERVAL = float(os.environ.get("GPU_PROFILER_INTERVAL", "0.3"))
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_IDLE_THRESH = float(os.environ.get("GPU_PROFILER_IDLE_THRESH", "30"))
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_BOUNDARY = os.environ.get("GPU_PROFILER_BOUNDARY", "step").strip()

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_GB = 1024.0 ** 3


# [EXPLAIN] `now` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def now() -> float:
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return time.monotonic()


# --------------------------------------------------------------------------- #
# Metric backends
# --------------------------------------------------------------------------- #
# [EXPLAIN] `_Backend` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class _Backend:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Abstract per-GPU metric reader. Returns a list (one dict per GPU)."""

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n_gpus = 0

    # [EXPLAIN] `sample` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sample(self):  # pragma: no cover - overridden
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError


# [EXPLAIN] `_NvmlBackend` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class _NvmlBackend(_Backend):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import pynvml  # noqa: F401

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._nvml = pynvml
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._nvml.nvmlInit()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.n_gpus = self._nvml.nvmlDeviceGetCount()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._handles = [self._nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.n_gpus)]

    # [EXPLAIN] `_safe` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _safe(self, fn, *a, default=None):
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return fn(*a)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return default

    # [EXPLAIN] `sample` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sample(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nv = self._nvml
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        out = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for h in self._handles:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rec = {}
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            util = self._safe(nv.nvmlDeviceGetUtilizationRates, h)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rec["sm_util"] = float(util.gpu) if util is not None else None
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rec["mem_bw_util"] = float(util.memory) if util is not None else None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mem = self._safe(nv.nvmlDeviceGetMemoryInfo, h)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rec["mem_used_gb"] = (mem.used / _GB) if mem is not None else None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pw = self._safe(nv.nvmlDeviceGetPowerUsage, h)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rec["power_w"] = (pw / 1000.0) if pw is not None else None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            clk = self._safe(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rec["sm_clock_mhz"] = float(clk) if clk is not None else None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tx = self._safe(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_TX_BYTES)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rx = self._safe(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_RX_BYTES)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rec["pcie_tx_mb_s"] = (tx / 1024.0) if tx is not None else None  # KB/s -> MB/s
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rec["pcie_rx_mb_s"] = (rx / 1024.0) if rx is not None else None
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            out.append(rec)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return out


# [EXPLAIN] `_SmiBackend` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class _SmiBackend(_Backend):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Fallback that shells out to nvidia-smi (no PCIe throughput available)."""

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _QUERY = "utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm"

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import shutil
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import subprocess

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._subprocess = subprocess
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if shutil.which("nvidia-smi") is None:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError("nvidia-smi not found")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.n_gpus = len(self._raw())

    # [EXPLAIN] `_raw` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _raw(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cmd = [
            "nvidia-smi",
            f"--query-gpu={self._QUERY}",
            "--format=csv,noheader,nounits",
        ]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        out = self._subprocess.check_output(cmd, timeout=5).decode()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [ln for ln in out.strip().splitlines() if ln.strip()]

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `_f` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _f(tok):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tok = tok.strip()
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return float(tok)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except ValueError:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None

    # [EXPLAIN] `sample` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sample(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rows = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for line in self._raw():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            parts = line.split(",")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(parts) < 5:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sm, membw, mem_used_mib, power, clk = (self._f(p) for p in parts[:5])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rows.append(
                {
                    "sm_util": sm,
                    "mem_bw_util": membw,
                    "mem_used_gb": (mem_used_mib / 1024.0) if mem_used_mib is not None else None,
                    "power_w": power,
                    "sm_clock_mhz": clk,
                    "pcie_tx_mb_s": None,
                    "pcie_rx_mb_s": None,
                }
            )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rows


# [EXPLAIN] `_make_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _make_backend():
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return _NvmlBackend()
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:  # pragma: no cover - depends on host
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nvml_err = e
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return _SmiBackend()
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:  # pragma: no cover - depends on host
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[gpu-profiler] disabled: no NVML ({nvml_err}) and no nvidia-smi ({e})", flush=True)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_METRICS = ("sm_util", "mem_bw_util", "mem_used_gb", "power_w", "sm_clock_mhz", "pcie_tx_mb_s", "pcie_rx_mb_s")


# [EXPLAIN] `_Sampler` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class _Sampler:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, backend, interval):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._backend = backend
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._interval = interval
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.n_gpus = backend.n_gpus
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._lock = threading.Lock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._phase_stack = []
        # samples since last reset: list of (ts, dt, phase, per_gpu_list)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._samples = []
        # lightweight ring of (ts, mean_sm_util) for mean_util_between()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._util_trace = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._stop = threading.Event()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._step_idx = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._thread = threading.Thread(target=self._run, name="gpu-profiler", daemon=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._thread.start()

    # -- phase stack ------------------------------------------------------- #
    # [EXPLAIN] `push` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def push(self, name):
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self._lock:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._phase_stack.append(name)

    # [EXPLAIN] `pop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def pop(self, name):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        report = False
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self._lock:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._phase_stack:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self._phase_stack.pop()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            report = name == _BOUNDARY
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if report:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.report_and_reset(label=f"step {self._step_idx}")
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self._step_idx += 1

    # [EXPLAIN] `_current_phase` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _current_phase(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._phase_stack[-1] if self._phase_stack else "(idle/other)"

    # -- sampling loop ----------------------------------------------------- #
    # [EXPLAIN] `_run` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _run(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prev = now()
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while not self._stop.wait(self._interval):
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                per_gpu = self._backend.sample()
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            t = now()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dt = t - prev
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prev = t
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with self._lock:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                phase = self._current_phase()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self._samples.append((t, dt, phase, per_gpu))
                # store per-GPU sm so callers can see data-parallel imbalance
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                per_gpu_sm = [g.get("sm_util") for g in per_gpu]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self._util_trace.append((t, per_gpu_sm))

    # [EXPLAIN] `mean_util_between` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def mean_util_between(self, t0, t1):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Mean (across GPUs and time) SM util in [t0, t1]."""
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self._lock:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            window = [vals for (ts, vals) in self._util_trace if t0 <= ts <= t1]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        flat = [v for vals in window for v in vals if v is not None]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not flat:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sum(flat) / len(flat)

    # [EXPLAIN] `per_gpu_util_between` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def per_gpu_util_between(self, t0, t1):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Per-GPU mean SM util in [t0, t1] as a list (one entry per GPU).

        Lets callers see data-parallel load imbalance (e.g. a mixed-task batch
        that lands different tasks on different GPUs)."""
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self._lock:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            window = [vals for (ts, vals) in self._util_trace if t0 <= ts <= t1]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not window:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n = max(len(vals) for vals in window)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        out = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for gi in range(n):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            col = [vals[gi] for vals in window if gi < len(vals) and vals[gi] is not None]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            out.append((sum(col) / len(col)) if col else None)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return out

    # -- reporting --------------------------------------------------------- #
    # [EXPLAIN] `report_and_reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def report_and_reset(self, label=""):
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self._lock:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            samples = self._samples
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._samples = []
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._util_trace = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not samples:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _print_report(samples, self.n_gpus, label, self._interval)

    # [EXPLAIN] `stop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def stop(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._stop.set()


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_sampler = None
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_sampler_failed = False
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_sampler_lock = threading.Lock()


# [EXPLAIN] `_ensure_sampler` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _ensure_sampler():
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    global _sampler, _sampler_failed
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _sampler is not None or _sampler_failed:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return _sampler
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with _sampler_lock:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if _sampler is None and not _sampler_failed:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            backend = _make_backend()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if backend is None or backend.n_gpus == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                _sampler_failed = True  # don't retry every phase
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _sampler = _Sampler(backend, _INTERVAL)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(
                f"[gpu-profiler] started: {backend.n_gpus} GPU(s), "
                f"interval={_INTERVAL}s, idle<{_IDLE_THRESH}% sm",
                flush=True,
            )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _sampler


# --------------------------------------------------------------------------- #
# Public API (cheap no-ops when disabled)
# --------------------------------------------------------------------------- #
# [EXPLAIN] `push_phase` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def push_phase(name):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not enabled():
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    s = _ensure_sampler()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if s is not None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        s.push(name)


# [EXPLAIN] `pop_phase` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def pop_phase(name):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not enabled():
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _sampler is not None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _sampler.pop(name)


# [EXPLAIN] `mean_util_between` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def mean_util_between(t0, t1):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not enabled() or _sampler is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _sampler.mean_util_between(t0, t1)


# [EXPLAIN] `per_gpu_util_between` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def per_gpu_util_between(t0, t1):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not enabled() or _sampler is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _sampler.per_gpu_util_between(t0, t1)


# [EXPLAIN] `report_and_reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def report_and_reset(label=""):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not enabled() or _sampler is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _sampler.report_and_reset(label=label)


# --------------------------------------------------------------------------- #
# Aggregation + pretty-print
# --------------------------------------------------------------------------- #
# [EXPLAIN] `_agg_phase` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _agg_phase(rows, n_gpus):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """rows: list of (dt, per_gpu_list) for a single phase."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wall = sum(dt for dt, _ in rows)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = len(rows)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sums = defaultdict(float)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    counts = defaultdict(int)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    per_gpu_sm = [[] for _ in range(n_gpus)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idle_samples = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _dt, per_gpu in rows:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sm_across = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for gi, g in enumerate(per_gpu):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for m in _METRICS:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                v = g.get(m)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if v is not None:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    sums[m] += v
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    counts[m] += 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if g.get("sm_util") is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                sm_across.append(g["sm_util"])
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if gi < n_gpus:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    per_gpu_sm[gi].append(g["sm_util"])
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if sm_across:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mean_sm = sum(sm_across) / len(sm_across)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mean_sm < _IDLE_THRESH:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                idle_samples += 1

    # [EXPLAIN] `mean` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def mean(m):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (sums[m] / counts[m]) if counts[m] else None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    per_gpu_mean = [
        (sum(v) / len(v)) if v else None for v in per_gpu_sm
    ]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {
        "wall": wall,
        "n": n,
        "sm_util": mean("sm_util"),
        "mem_bw_util": mean("mem_bw_util"),
        "mem_used_gb": mean("mem_used_gb"),
        "power_w": mean("power_w"),
        "sm_clock_mhz": mean("sm_clock_mhz"),
        "pcie_tx_mb_s": mean("pcie_tx_mb_s"),
        "pcie_rx_mb_s": mean("pcie_rx_mb_s"),
        "idle_pct": (100.0 * idle_samples / n) if n else 0.0,
        "per_gpu_sm": per_gpu_mean,
    }


# Phases worth ordering first in the table; anything else appended after.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_PHASE_ORDER = [
    "gen",
    "old_log_prob",
    "teacher_forward",
    "ref",
    "values",
    "adv",
    "update_critic",
    "update_actor",
    "reward",
]


# [EXPLAIN] `_fmt` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _fmt(v, spec):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return format(v, spec) if v is not None else "-"


# [EXPLAIN] `_print_report` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _print_report(samples, n_gpus, label, interval):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    by_phase = defaultdict(list)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _ts, dt, phase, per_gpu in samples:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        by_phase[phase].append((dt, per_gpu))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    order = [p for p in _PHASE_ORDER if p in by_phase]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    order += sorted(p for p in by_phase if p not in _PHASE_ORDER)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_wall = sum(dt for _ts, dt, _p, _g in samples)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n_samples = len(samples)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lines = []
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    lines.append(
        f"[gpu-profiler] {label} | {n_gpus} GPU(s) | samples={n_samples} "
        f"| interval~{interval}s | idle<{_IDLE_THRESH}% sm"
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    header = (
        f"{'phase':<20}{'wall_s':>9}{'sm%':>7}{'memBW%':>8}{'idle%':>7}"
        f"{'memGB':>8}{'powerW':>8}{'smClk':>7}{'pcieTX':>8}{'pcieRX':>8}  per-GPU sm%"
    )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    lines.append(header)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    lines.append("-" * len(header))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    weighted_sm = 0.0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for phase in order:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        a = _agg_phase(by_phase[phase], n_gpus)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_gpu = "/".join(_fmt(v, ".0f") for v in a["per_gpu_sm"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        lines.append(
            f"{phase:<20}{a['wall']:>9.1f}{_fmt(a['sm_util'], '>7.1f')}"
            f"{_fmt(a['mem_bw_util'], '>8.1f')}{a['idle_pct']:>7.1f}"
            f"{_fmt(a['mem_used_gb'], '>8.1f')}{_fmt(a['power_w'], '>8.0f')}"
            f"{_fmt(a['sm_clock_mhz'], '>7.0f')}{_fmt(a['pcie_tx_mb_s'], '>8.0f')}"
            f"{_fmt(a['pcie_rx_mb_s'], '>8.0f')}  {per_gpu}"
        )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if a["sm_util"] is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            weighted_sm += a["sm_util"] * a["wall"]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    avg_sm = (weighted_sm / total_wall) if total_wall else 0.0
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    lines.append("-" * len(header))
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    lines.append(
        f"{'TOTAL/step':<20}{total_wall:>9.1f}{avg_sm:>7.1f}"
        f"{'':>8}{'':>7}{'':>8}{'':>8}{'':>7}{'':>8}{'':>8}  "
        f"(wall-weighted mean SM across phases)"
    )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------- #
# Standalone self-test (no GPU needed): python -m verl.utils.gpu_profiler
# --------------------------------------------------------------------------- #
# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":  # pragma: no cover
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import random

    # [EXPLAIN] `_FakeBackend` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
    class _FakeBackend(_Backend):
        # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __init__(self, n=3):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.n_gpus = n

        # [EXPLAIN] `sample` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def sample(self):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            phase = _sampler._current_phase() if _sampler else "?"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            base = {"gen": 60, "update_actor": 96, "teacher_forward": 91}.get(phase, 95)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return [
                {
                    "sm_util": max(0, min(100, base + random.uniform(-10, 10))),
                    "mem_bw_util": base * 0.6,
                    "mem_used_gb": 45 + random.uniform(-1, 1),
                    "power_w": 250 + random.uniform(-30, 30),
                    "sm_clock_mhz": 1700,
                    "pcie_tx_mb_s": 800 if phase == "update_actor" else 50,
                    "pcie_rx_mb_s": 800 if phase == "update_actor" else 50,
                }
                for _ in range(self.n_gpus)
            ]

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["GPU_PROFILER"] = "1"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _ENABLED = True
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _INTERVAL = 0.02
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _sampler = _Sampler(_FakeBackend(3), _INTERVAL)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("running synthetic profile ...")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    push_phase("step")
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for phase, dur in [("gen", 0.6), ("old_log_prob", 0.2), ("teacher_forward", 0.4), ("ref", 0.2), ("update_actor", 1.0)]:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        push_phase(phase)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        t0 = now()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        time.sleep(dur)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if phase == "gen":
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("  GEN-UTIL window:", round(mean_util_between(t0, now()) or -1, 1))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        pop_phase(phase)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    pop_phase("step")  # boundary -> triggers the per-step report
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _sampler.stop()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("self-test complete")
