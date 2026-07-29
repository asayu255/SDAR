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

import os
import threading
import time
from collections import defaultdict

__all__ = [
    "enabled",
    "push_phase",
    "pop_phase",
    "mean_util_between",
    "per_gpu_util_between",
    "now",
    "report_and_reset",
]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    # Cached so the hot push/pop path is a single dict lookup + bool.
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = _env_flag("GPU_PROFILER")
    return _ENABLED


_ENABLED = None
_INTERVAL = float(os.environ.get("GPU_PROFILER_INTERVAL", "0.3"))
_IDLE_THRESH = float(os.environ.get("GPU_PROFILER_IDLE_THRESH", "30"))
_BOUNDARY = os.environ.get("GPU_PROFILER_BOUNDARY", "step").strip()

_GB = 1024.0 ** 3


def now() -> float:
    return time.monotonic()


# --------------------------------------------------------------------------- #
# Metric backends
# --------------------------------------------------------------------------- #
class _Backend:
    """Abstract per-GPU metric reader. Returns a list (one dict per GPU)."""

    n_gpus = 0

    def sample(self):  # pragma: no cover - overridden
        raise NotImplementedError


class _NvmlBackend(_Backend):
    def __init__(self):
        import pynvml  # noqa: F401

        self._nvml = pynvml
        self._nvml.nvmlInit()
        self.n_gpus = self._nvml.nvmlDeviceGetCount()
        self._handles = [self._nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.n_gpus)]

    def _safe(self, fn, *a, default=None):
        try:
            return fn(*a)
        except Exception:
            return default

    def sample(self):
        nv = self._nvml
        out = []
        for h in self._handles:
            rec = {}
            util = self._safe(nv.nvmlDeviceGetUtilizationRates, h)
            rec["sm_util"] = float(util.gpu) if util is not None else None
            rec["mem_bw_util"] = float(util.memory) if util is not None else None
            mem = self._safe(nv.nvmlDeviceGetMemoryInfo, h)
            rec["mem_used_gb"] = (mem.used / _GB) if mem is not None else None
            pw = self._safe(nv.nvmlDeviceGetPowerUsage, h)
            rec["power_w"] = (pw / 1000.0) if pw is not None else None
            clk = self._safe(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM)
            rec["sm_clock_mhz"] = float(clk) if clk is not None else None
            tx = self._safe(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_TX_BYTES)
            rx = self._safe(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_RX_BYTES)
            rec["pcie_tx_mb_s"] = (tx / 1024.0) if tx is not None else None  # KB/s -> MB/s
            rec["pcie_rx_mb_s"] = (rx / 1024.0) if rx is not None else None
            out.append(rec)
        return out


class _SmiBackend(_Backend):
    """Fallback that shells out to nvidia-smi (no PCIe throughput available)."""

    _QUERY = "utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm"

    def __init__(self):
        import shutil
        import subprocess

        self._subprocess = subprocess
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi not found")
        self.n_gpus = len(self._raw())

    def _raw(self):
        cmd = [
            "nvidia-smi",
            f"--query-gpu={self._QUERY}",
            "--format=csv,noheader,nounits",
        ]
        out = self._subprocess.check_output(cmd, timeout=5).decode()
        return [ln for ln in out.strip().splitlines() if ln.strip()]

    @staticmethod
    def _f(tok):
        tok = tok.strip()
        try:
            return float(tok)
        except ValueError:
            return None

    def sample(self):
        rows = []
        for line in self._raw():
            parts = line.split(",")
            if len(parts) < 5:
                continue
            sm, membw, mem_used_mib, power, clk = (self._f(p) for p in parts[:5])
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
        return rows


def _make_backend():
    try:
        return _NvmlBackend()
    except Exception as e:  # pragma: no cover - depends on host
        nvml_err = e
    try:
        return _SmiBackend()
    except Exception as e:  # pragma: no cover - depends on host
        print(f"[gpu-profiler] disabled: no NVML ({nvml_err}) and no nvidia-smi ({e})", flush=True)
        return None


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
_METRICS = ("sm_util", "mem_bw_util", "mem_used_gb", "power_w", "sm_clock_mhz", "pcie_tx_mb_s", "pcie_rx_mb_s")


# daemon threadがNVML/SMIのdevice-wide counterを一定間隔で読むため、値は現在processだけでなく同GPU上の全workloadを含む。phase stack最上位のlabelを各sampleへ付け、lockはhost側のsample buffer整合性だけを守る。
# `mean_util_between`は指定wall-clock窓、`report_and_reset`はtrainer step境界の集計であり、どちらもCUDA synchronizeを挿入しないため測定自体でGPU pipelineを止めない。
class _Sampler:
    def __init__(self, backend, interval):
        self._backend = backend
        self._interval = interval
        self.n_gpus = backend.n_gpus
        self._lock = threading.Lock()
        self._phase_stack = []
        # samples since last reset: list of (ts, dt, phase, per_gpu_list)
        self._samples = []
        # lightweight ring of (ts, mean_sm_util) for mean_util_between()
        self._util_trace = []
        self._stop = threading.Event()
        self._step_idx = 0
        self._thread = threading.Thread(target=self._run, name="gpu-profiler", daemon=True)
        self._thread.start()

    # -- phase stack ------------------------------------------------------- #
    def push(self, name):
        with self._lock:
            self._phase_stack.append(name)

    def pop(self, name):
        report = False
        with self._lock:
            if self._phase_stack:
                self._phase_stack.pop()
            report = name == _BOUNDARY
        if report:
            self.report_and_reset(label=f"step {self._step_idx}")
            self._step_idx += 1

    def _current_phase(self):
        return self._phase_stack[-1] if self._phase_stack else "(idle/other)"

    # -- sampling loop ----------------------------------------------------- #
    def _run(self):
        prev = now()
        while not self._stop.wait(self._interval):
            try:
                per_gpu = self._backend.sample()
            except Exception:
                continue
            t = now()
            dt = t - prev
            prev = t
            with self._lock:
                phase = self._current_phase()
                self._samples.append((t, dt, phase, per_gpu))
                # store per-GPU sm so callers can see data-parallel imbalance
                per_gpu_sm = [g.get("sm_util") for g in per_gpu]
                self._util_trace.append((t, per_gpu_sm))

    def mean_util_between(self, t0, t1):
        """Mean (across GPUs and time) SM util in [t0, t1]."""
        with self._lock:
            window = [vals for (ts, vals) in self._util_trace if t0 <= ts <= t1]
        flat = [v for vals in window for v in vals if v is not None]
        if not flat:
            return None
        return sum(flat) / len(flat)

    def per_gpu_util_between(self, t0, t1):
        """Per-GPU mean SM util in [t0, t1] as a list (one entry per GPU).

        Lets callers see data-parallel load imbalance (e.g. a mixed-task batch
        that lands different tasks on different GPUs)."""
        with self._lock:
            window = [vals for (ts, vals) in self._util_trace if t0 <= ts <= t1]
        if not window:
            return None
        n = max(len(vals) for vals in window)
        out = []
        for gi in range(n):
            col = [vals[gi] for vals in window if gi < len(vals) and vals[gi] is not None]
            out.append((sum(col) / len(col)) if col else None)
        return out

    # -- reporting --------------------------------------------------------- #
    def report_and_reset(self, label=""):
        with self._lock:
            samples = self._samples
            self._samples = []
            self._util_trace = []
        if not samples:
            return
        _print_report(samples, self.n_gpus, label, self._interval)

    def stop(self):
        self._stop.set()


_sampler = None
_sampler_failed = False
_sampler_lock = threading.Lock()


def _ensure_sampler():
    global _sampler, _sampler_failed
    if _sampler is not None or _sampler_failed:
        return _sampler
    with _sampler_lock:
        if _sampler is None and not _sampler_failed:
            backend = _make_backend()
            if backend is None or backend.n_gpus == 0:
                _sampler_failed = True  # don't retry every phase
                return None
            _sampler = _Sampler(backend, _INTERVAL)
            print(
                f"[gpu-profiler] started: {backend.n_gpus} GPU(s), "
                f"interval={_INTERVAL}s, idle<{_IDLE_THRESH}% sm",
                flush=True,
            )
    return _sampler


# --------------------------------------------------------------------------- #
# Public API (cheap no-ops when disabled)
# --------------------------------------------------------------------------- #
def push_phase(name):
    if not enabled():
        return
    s = _ensure_sampler()
    if s is not None:
        s.push(name)


def pop_phase(name):
    if not enabled():
        return
    if _sampler is not None:
        _sampler.pop(name)


def mean_util_between(t0, t1):
    if not enabled() or _sampler is None:
        return None
    return _sampler.mean_util_between(t0, t1)


def per_gpu_util_between(t0, t1):
    if not enabled() or _sampler is None:
        return None
    return _sampler.per_gpu_util_between(t0, t1)


def report_and_reset(label=""):
    if not enabled() or _sampler is None:
        return
    _sampler.report_and_reset(label=label)


# --------------------------------------------------------------------------- #
# Aggregation + pretty-print
# --------------------------------------------------------------------------- #
def _agg_phase(rows, n_gpus):
    """rows: list of (dt, per_gpu_list) for a single phase."""
    wall = sum(dt for dt, _ in rows)
    n = len(rows)
    sums = defaultdict(float)
    counts = defaultdict(int)
    per_gpu_sm = [[] for _ in range(n_gpus)]
    idle_samples = 0
    for _dt, per_gpu in rows:
        sm_across = []
        for gi, g in enumerate(per_gpu):
            for m in _METRICS:
                v = g.get(m)
                if v is not None:
                    sums[m] += v
                    counts[m] += 1
            if g.get("sm_util") is not None:
                sm_across.append(g["sm_util"])
                if gi < n_gpus:
                    per_gpu_sm[gi].append(g["sm_util"])
        if sm_across:
            mean_sm = sum(sm_across) / len(sm_across)
            if mean_sm < _IDLE_THRESH:
                idle_samples += 1

    def mean(m):
        return (sums[m] / counts[m]) if counts[m] else None

    per_gpu_mean = [
        (sum(v) / len(v)) if v else None for v in per_gpu_sm
    ]
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


def _fmt(v, spec):
    return format(v, spec) if v is not None else "-"


def _print_report(samples, n_gpus, label, interval):
    by_phase = defaultdict(list)
    for _ts, dt, phase, per_gpu in samples:
        by_phase[phase].append((dt, per_gpu))

    order = [p for p in _PHASE_ORDER if p in by_phase]
    order += sorted(p for p in by_phase if p not in _PHASE_ORDER)

    total_wall = sum(dt for _ts, dt, _p, _g in samples)
    n_samples = len(samples)

    lines = []
    lines.append(
        f"[gpu-profiler] {label} | {n_gpus} GPU(s) | samples={n_samples} "
        f"| interval~{interval}s | idle<{_IDLE_THRESH}% sm"
    )
    header = (
        f"{'phase':<20}{'wall_s':>9}{'sm%':>7}{'memBW%':>8}{'idle%':>7}"
        f"{'memGB':>8}{'powerW':>8}{'smClk':>7}{'pcieTX':>8}{'pcieRX':>8}  per-GPU sm%"
    )
    lines.append(header)
    lines.append("-" * len(header))

    weighted_sm = 0.0
    for phase in order:
        a = _agg_phase(by_phase[phase], n_gpus)
        per_gpu = "/".join(_fmt(v, ".0f") for v in a["per_gpu_sm"])
        lines.append(
            f"{phase:<20}{a['wall']:>9.1f}{_fmt(a['sm_util'], '>7.1f')}"
            f"{_fmt(a['mem_bw_util'], '>8.1f')}{a['idle_pct']:>7.1f}"
            f"{_fmt(a['mem_used_gb'], '>8.1f')}{_fmt(a['power_w'], '>8.0f')}"
            f"{_fmt(a['sm_clock_mhz'], '>7.0f')}{_fmt(a['pcie_tx_mb_s'], '>8.0f')}"
            f"{_fmt(a['pcie_rx_mb_s'], '>8.0f')}  {per_gpu}"
        )
        if a["sm_util"] is not None:
            weighted_sm += a["sm_util"] * a["wall"]

    avg_sm = (weighted_sm / total_wall) if total_wall else 0.0
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL/step':<20}{total_wall:>9.1f}{avg_sm:>7.1f}"
        f"{'':>8}{'':>7}{'':>8}{'':>8}{'':>7}{'':>8}{'':>8}  "
        f"(wall-weighted mean SM across phases)"
    )
    print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------- #
# Standalone self-test (no GPU needed): python -m verl.utils.gpu_profiler
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import random

    class _FakeBackend(_Backend):
        def __init__(self, n=3):
            self.n_gpus = n

        def sample(self):
            phase = _sampler._current_phase() if _sampler else "?"
            base = {"gen": 60, "update_actor": 96, "teacher_forward": 91}.get(phase, 95)
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

    os.environ["GPU_PROFILER"] = "1"
    _ENABLED = True
    _INTERVAL = 0.02
    _sampler = _Sampler(_FakeBackend(3), _INTERVAL)
    print("running synthetic profile ...")
    push_phase("step")
    for phase, dur in [("gen", 0.6), ("old_log_prob", 0.2), ("teacher_forward", 0.4), ("ref", 0.2), ("update_actor", 1.0)]:
        push_phase(phase)
        t0 = now()
        time.sleep(dur)
        if phase == "gen":
            print("  GEN-UTIL window:", round(mean_util_between(t0, now()) or -1, 1))
        pop_phase(phase)
    pop_phase("step")  # boundary -> triggers the per-step report
    _sampler.stop()
    print("self-test complete")
