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
  - ``nvlink_mb_s``        NVLink data throughput (TX+RX), NVML only. On a
                           single node this is where FSDP all-gather /
                           reduce-scatter traffic shows up -- PCIe counters do
                           NOT see it, so collective bubbles are invisible
                           without this metric.

Alongside the per-GPU sample the driver process's own CPU usage is recorded
(``cpu_pct``, 100% == one core; requires ``psutil``). It separates the two
causes of a GPU-idle window that look identical on the device: the driver
burning CPU (batch assembly, tokenization -> prefetchable) versus the driver
blocked on I/O (disk / network / env.step -> not prefetchable).

Each sample is tagged with the phase currently on top of a thread-safe phase
stack. The stack is driven by ``verl.trainer.ppo.ray_trainer._timer``, so every
phase (gen / old_log_prob / teacher_forward / ref / update_actor / ...) is
tagged automatically with no per-call-site changes. Samples taken while the
stack is empty are tagged ``(idle/other)`` -- for the fixed-dataset trainers
that bucket is the between-step batch-assembly window, which is exactly the
part a prefetch would remove.

Two reports are produced:

  - a **per-step** table, emitted when the boundary phase (default ``step``)
    pops, covering only that step;
  - a **cumulative** table over every step so far, emitted every
    ``GPU_PROFILER_ROLLUP_EVERY`` steps and at process exit. Periodic phases
    (validation, checkpointing) only appear in the step that fires them, so the
    cumulative ``share%`` column is the only place their true cost over a run
    is visible.

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
  GPU_PROFILER_ROLLUP_EVERY=25 emit the cumulative table every N boundary
                               phases (0 disables the periodic one; the
                               at-exit rollup still fires)
  GPU_PROFILER_SYNC_PHASES=1   synchronize the device at each phase boundary.
                               Kernel launches are async, so without this a
                               phase's wall clock is when its work was issued,
                               not when the GPU finished it, and the boundaries
                               smear. Exact attribution, but it serializes what
                               the run would overlap -- the totals get slower.
  GPU_PROFILER_TRACE=<path>    also append every raw sample to this CSV
                               (ts, clock, phase, per-GPU sm%, per-GPU memBW%,
                               driver cpu%). The tables above are per-step
                               aggregates reset at each boundary, so they say a
                               gap happened and how long it was but not *when*;
                               the trace is what lines a dip seen on an external
                               monitor up against the phase that produced it.
"""

import atexit
import os
import re
import threading
import time
from collections import defaultdict

# Leading alignment flag + field width of a format spec such as ">9.0f".
_WIDTH_RE = re.compile(r"([<>^]?)(\d+)")

__all__ = [
    "enabled",
    "push_phase",
    "pop_phase",
    "mean_util_between",
    "per_gpu_util_between",
    "now",
    "report_and_reset",
    "report_cumulative",
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
_ROLLUP_EVERY = int(os.environ.get("GPU_PROFILER_ROLLUP_EVERY", "25"))
_TRACE_PATH = os.environ.get("GPU_PROFILER_TRACE", "").strip()

_GB = 1024.0 ** 3
_KIB_TO_MB = 1024.0 / (1000.0 * 1000.0)  # NVML NVLink counters are KiB

# Two consecutive samples of the same phase separated by more than this many
# sample intervals are treated as different blocks of that phase, so an idle
# stretch is never reported across a busy phase that ran in between.
_CONTIGUITY_SLACK = 2.0


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


# pynvml's c_nvmlValue_t is a union; the meaningful member is picked by the
# value's declared type (NVML_VALUE_TYPE_*).
_NVML_VALUE_ATTR = {0: "dVal", 1: "uiVal", 2: "ulVal", 3: "ullVal", 4: "sllVal"}


class _NvmlBackend(_Backend):
    def __init__(self):
        import pynvml  # noqa: F401

        self._nvml = pynvml
        self._nvml.nvmlInit()
        self.n_gpus = self._nvml.nvmlDeviceGetCount()
        self._handles = [self._nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.n_gpus)]
        # NVLink data-throughput counters are cumulative KiB, so a rate needs the
        # previous reading. Absent on GPUs / drivers without NVLink -> stays None.
        self._fi_tx = getattr(pynvml, "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX", None)
        self._fi_rx = getattr(pynvml, "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX", None)
        self._nvlink_fields = [f for f in (self._fi_tx, self._fi_rx) if f is not None]
        self._prev_nvlink = [None] * self.n_gpus  # per GPU: (t, tx_kib, rx_kib)

    def _safe(self, fn, *a, default=None):
        try:
            return fn(*a)
        except Exception:
            return default

    def _nvlink_counters(self, h):
        """Cumulative (tx_kib, rx_kib) across all links, or (None, None)."""
        if not self._nvlink_fields:
            return (None, None)
        values = self._safe(self._nvml.nvmlDeviceGetFieldValues, h, list(self._nvlink_fields))
        if not values:
            return (None, None)
        got = {}
        for fv in values:
            try:
                if int(fv.nvmlReturn) != 0:  # NVML_SUCCESS
                    continue
                attr = _NVML_VALUE_ATTR.get(int(fv.valueType))
                if attr is None:
                    continue
                got[int(fv.fieldId)] = float(getattr(fv.value, attr))
            except Exception:
                continue
        return (got.get(self._fi_tx), got.get(self._fi_rx))

    def _nvlink_rate_mb_s(self, gi, h, t):
        """Differentiate the cumulative counters into MB/s (TX+RX)."""
        tx, rx = self._nvlink_counters(h)
        prev = self._prev_nvlink[gi]
        if tx is None or rx is None:
            self._prev_nvlink[gi] = None
            return None
        self._prev_nvlink[gi] = (t, tx, rx)
        if prev is None:
            return None  # first reading: no interval to divide by
        dt = t - prev[0]
        d_tx = tx - prev[1]
        d_rx = rx - prev[2]
        if dt <= 0 or d_tx < 0 or d_rx < 0:  # counter reset / clock anomaly
            return None
        return (d_tx + d_rx) * _KIB_TO_MB / dt

    def sample(self):
        nv = self._nvml
        t = now()
        out = []
        for gi, h in enumerate(self._handles):
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
            rec["nvlink_mb_s"] = self._nvlink_rate_mb_s(gi, h, t)
            out.append(rec)
        return out


class _SmiBackend(_Backend):
    """Fallback that shells out to nvidia-smi (no PCIe/NVLink throughput)."""

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
                    "nvlink_mb_s": None,
                }
            )
        return rows


class _HostSampler:
    """Driver-process CPU usage, sampled alongside the GPU metrics.

    ``cpu_pct`` is this process's CPU time as a percentage of one core, so it
    can exceed 100 on a multi-threaded driver. Its value during a GPU-idle
    phase says whether the driver was *working* (assembling a batch: prefetch
    helps) or *waiting* (I/O: prefetch does not).
    """

    def __init__(self):
        import psutil

        self._proc = psutil.Process()
        self._proc.cpu_percent(None)  # prime: the first call always returns 0.0
        self.n_cpus = psutil.cpu_count() or 1

    def sample(self):
        try:
            return {"cpu_pct": float(self._proc.cpu_percent(None))}
        except Exception:
            return {"cpu_pct": None}


def _make_host_sampler():
    try:
        return _HostSampler()
    except Exception as e:  # pragma: no cover - psutil is optional
        print(f"[gpu-profiler] host CPU sampling off (no psutil: {e})", flush=True)
        return None


def _make_backend():
    try:
        backend = _NvmlBackend()
        print("[gpu-profiler] backend: NVML", flush=True)
        return backend
    except Exception as e:  # pragma: no cover - depends on host
        nvml_err = e
    try:
        backend = _SmiBackend()
        # Say so loudly: the fallback silently drops two whole metrics, and a
        # blank NVLink column otherwise reads as "no collective traffic"
        # rather than "not measured".
        print(
            f"[gpu-profiler] backend: nvidia-smi fallback -- NVLink and PCIe "
            f"throughput are NOT collected (NVML unavailable: {nvml_err}). "
            f"Install nvidia-ml-py to enable them.",
            flush=True,
        )
        return backend
    except Exception as e:  # pragma: no cover - depends on host
        print(f"[gpu-profiler] disabled: no NVML ({nvml_err}) and no nvidia-smi ({e})", flush=True)
        return None


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
_METRICS = (
    "sm_util",
    "mem_bw_util",
    "mem_used_gb",
    "power_w",
    "sm_clock_mhz",
    "pcie_tx_mb_s",
    "pcie_rx_mb_s",
    "nvlink_mb_s",
)


class _Sampler:
    def __init__(self, backend, interval, host=None):
        self._backend = backend
        self._host = host
        self._interval = interval
        self.n_gpus = backend.n_gpus
        self._lock = threading.Lock()
        self._phase_stack = []
        # samples since last reset: list of (ts, dt, phase, per_gpu_list, host)
        self._samples = []
        # lightweight ring of (ts, mean_sm_util) for mean_util_between()
        self._util_trace = []
        self._stop = threading.Event()
        self._step_idx = 0
        # Cumulative per-phase accumulators across every step so far. Aggregated
        # incrementally (sums/counts, not raw samples) so memory stays flat over
        # a long run.
        self._cum = {}
        self._cum_steps = 0
        self._trace = None
        if _TRACE_PATH:
            try:
                self._trace = open(_TRACE_PATH, "w", buffering=1)
                self._trace.write("ts,clock,phase,sm_pct_per_gpu,membw_pct_per_gpu,driver_cpu_pct\n")
                print(f"[gpu-profiler] per-sample trace -> {_TRACE_PATH}", flush=True)
            except OSError as e:
                print(f"[gpu-profiler] could not open GPU_PROFILER_TRACE={_TRACE_PATH}: {e}", flush=True)
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
            if _ROLLUP_EVERY > 0 and self._step_idx % _ROLLUP_EVERY == 0:
                self.report_cumulative()

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
            host = self._host.sample() if self._host is not None else None
            t = now()
            dt = t - prev
            prev = t
            with self._lock:
                phase = self._current_phase()
                self._samples.append((t, dt, phase, per_gpu, host))
                # store per-GPU sm so callers can see data-parallel imbalance
                per_gpu_sm = [g.get("sm_util") for g in per_gpu]
                self._util_trace.append((t, per_gpu_sm))
            self._write_trace(t, phase, per_gpu, host)

    def _write_trace(self, t, phase, per_gpu, host):
        """Append one row to GPU_PROFILER_TRACE, if it is set.

        The tables are per-step aggregates and are reset at every step boundary,
        which is the right shape for "where does the time go" but the wrong one
        for a transient that happens once an hour: they say a gap existed and how
        long it was, not when. This keeps every sample with its wall-clock time
        and the phase it fell in, so a dip seen on an external monitor can be
        lined up with what the trainer was doing at that moment.

        Written outside the sampler lock, and never allowed to kill the sampler:
        a profiler that takes the run down with it is worse than no profiler.
        """
        if self._trace is None:
            return
        try:
            sm = ";".join("" if g.get("sm_util") is None else f"{g['sm_util']:.0f}" for g in per_gpu)
            bw = ";".join("" if g.get("mem_bw_util") is None else f"{g['mem_bw_util']:.0f}" for g in per_gpu)
            cpu = (host or {}).get("cpu_pct")
            self._trace.write(
                f"{t:.3f},{time.strftime('%H:%M:%S', time.localtime())},{phase},"
                f"{sm},{bw},{'' if cpu is None else f'{cpu:.0f}'}\n"
            )
            # Flushed every row: the interesting runs are the ones that end in a
            # crash or a Ctrl-C, and a buffered tail would lose exactly those.
            self._trace.flush()
        except Exception:
            self._trace = None

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
        by_phase = _accumulate(samples, self.n_gpus)
        # Fold this step into the run-level totals before printing it, so a run
        # cut short by Ctrl-C still has every completed step in the rollup.
        with self._lock:
            for phase, acc in by_phase.items():
                dst = self._cum.get(phase)
                if dst is None:
                    self._cum[phase] = dst = _acc_new(self.n_gpus)
                _acc_merge(dst, acc, self.n_gpus)
            self._cum_steps += 1
        _print_report(by_phase, self.n_gpus, label, self._interval)

    def report_cumulative(self, label=""):
        with self._lock:
            snapshot = {p: _acc_copy(a, self.n_gpus) for p, a in self._cum.items()}
            n_steps = self._cum_steps
        if not snapshot or n_steps == 0:
            return
        _print_report(
            snapshot,
            self.n_gpus,
            label or f"CUMULATIVE over {n_steps} step(s)",
            self._interval,
            cumulative_steps=n_steps,
        )

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
            host = _make_host_sampler()
            _sampler = _Sampler(backend, _INTERVAL, host=host)
            rollup = f"every {_ROLLUP_EVERY} step(s)" if _ROLLUP_EVERY > 0 else "at exit only"
            print(
                f"[gpu-profiler] started: {backend.n_gpus} GPU(s), "
                f"interval={_INTERVAL}s, idle<{_IDLE_THRESH}% sm, "
                f"host-cpu={'on' if host is not None else 'off'}, rollup {rollup}",
                flush=True,
            )
            atexit.register(_atexit_rollup)
    return _sampler


def _atexit_rollup():
    """Best-effort final rollup. The periodic one is the reliable path: a run
    killed with Ctrl-C inside a Ray actor may never reach atexit."""
    if _sampler is not None:
        try:
            _sampler.report_cumulative(label="CUMULATIVE (final)")
        except Exception:
            pass


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


def report_cumulative(label=""):
    """Print the run-level table (every phase, every step so far) on demand."""
    if not enabled() or _sampler is None:
        return
    _sampler.report_cumulative(label=label)


# --------------------------------------------------------------------------- #
# Aggregation + pretty-print
# --------------------------------------------------------------------------- #
def _acc_new(n_gpus):
    """A mergeable per-phase accumulator.

    Sums and counts rather than raw samples, so folding every step of a long
    run into one cumulative view costs O(phases), not O(samples).
    """
    return {
        "wall": 0.0,
        "n": 0,
        "sums": defaultdict(float),
        "counts": defaultdict(int),
        "per_gpu_sum": [0.0] * n_gpus,
        "per_gpu_n": [0] * n_gpus,
        "idle_n": 0,
        "idle_wall": 0.0,
        "max_gap": 0.0,
        "cpu_sum": 0.0,
        "cpu_n": 0,
    }


def _acc_copy(acc, n_gpus):
    out = _acc_new(n_gpus)
    _acc_merge(out, acc, n_gpus)
    return out


def _acc_merge(dst, src, n_gpus):
    dst["wall"] += src["wall"]
    dst["n"] += src["n"]
    for k, v in src["sums"].items():
        dst["sums"][k] += v
    for k, v in src["counts"].items():
        dst["counts"][k] += v
    for gi in range(n_gpus):
        dst["per_gpu_sum"][gi] += src["per_gpu_sum"][gi]
        dst["per_gpu_n"][gi] += src["per_gpu_n"][gi]
    dst["idle_n"] += src["idle_n"]
    dst["idle_wall"] += src["idle_wall"]
    # A gap is a property of one contiguous block, so the run-level worst case
    # is the max over blocks -- never their sum.
    dst["max_gap"] = max(dst["max_gap"], src["max_gap"])
    dst["cpu_sum"] += src["cpu_sum"]
    dst["cpu_n"] += src["cpu_n"]


def _accumulate(samples, n_gpus):
    """Fold time-ordered samples into one accumulator per phase.

    Idle *runs* are tracked here rather than at merge time because contiguity
    is only knowable from the timestamps: a phase can be entered several times
    within one step (e.g. ``step`` before and after ``update_actor``), and two
    idle stretches on either side of a busy phase must not be joined.
    """
    by_phase = {}
    prev_ts_by_phase = {}
    run_by_phase = {}
    for ts, dt, phase, per_gpu, host in samples:
        acc = by_phase.get(phase)
        if acc is None:
            by_phase[phase] = acc = _acc_new(n_gpus)
        acc["wall"] += dt
        acc["n"] += 1

        sm_across = []
        for gi, g in enumerate(per_gpu):
            for m in _METRICS:
                v = g.get(m)
                if v is not None:
                    acc["sums"][m] += v
                    acc["counts"][m] += 1
            v = g.get("sm_util")
            if v is not None:
                sm_across.append(v)
                if gi < n_gpus:
                    acc["per_gpu_sum"][gi] += v
                    acc["per_gpu_n"][gi] += 1

        cpu = (host or {}).get("cpu_pct")
        if cpu is not None:
            acc["cpu_sum"] += cpu
            acc["cpu_n"] += 1

        prev_ts = prev_ts_by_phase.get(phase)
        contiguous = prev_ts is not None and (ts - prev_ts) <= _CONTIGUITY_SLACK * _INTERVAL
        prev_ts_by_phase[phase] = ts

        is_idle = bool(sm_across) and (sum(sm_across) / len(sm_across)) < _IDLE_THRESH
        if is_idle:
            acc["idle_n"] += 1
            acc["idle_wall"] += dt
            run = (run_by_phase.get(phase, 0.0) if contiguous else 0.0) + dt
            run_by_phase[phase] = run
            if run > acc["max_gap"]:
                acc["max_gap"] = run
        else:
            run_by_phase[phase] = 0.0
    return by_phase


def _acc_finalize(acc, n_gpus):
    def mean(m):
        c = acc["counts"][m]
        return (acc["sums"][m] / c) if c else None

    per_gpu_mean = [
        (acc["per_gpu_sum"][gi] / acc["per_gpu_n"][gi]) if acc["per_gpu_n"][gi] else None
        for gi in range(n_gpus)
    ]
    n = acc["n"]
    return {
        "wall": acc["wall"],
        "n": n,
        "sm_util": mean("sm_util"),
        "mem_bw_util": mean("mem_bw_util"),
        "mem_used_gb": mean("mem_used_gb"),
        "power_w": mean("power_w"),
        "sm_clock_mhz": mean("sm_clock_mhz"),
        "pcie_tx_mb_s": mean("pcie_tx_mb_s"),
        "pcie_rx_mb_s": mean("pcie_rx_mb_s"),
        "nvlink_mb_s": mean("nvlink_mb_s"),
        "idle_pct": (100.0 * acc["idle_n"] / n) if n else 0.0,
        "idle_wall": acc["idle_wall"],
        "max_gap": acc["max_gap"],
        "cpu_pct": (acc["cpu_sum"] / acc["cpu_n"]) if acc["cpu_n"] else None,
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
    # Worker-side stages inside update_actor (dp_actor._actor_phase). Listed in
    # execution order so the interior of a step reads top to bottom.
    "actor.fwd",
    # Resolving the teacher at the ids the student's forward just chose, out of
    # the hidden states it cached (student_indexed_topk). Between fwd and bwd
    # because that is where it runs: the ids do not exist before the forward, and
    # the KL it feeds is built before the backward.
    "actor.teacher_lookup",
    "actor.bwd",
    "actor.task_metrics",
    "actor.optim",
]


def _fmt(v, spec):
    if v is not None:
        return format(v, spec)
    # Pad the placeholder to the column width taken from the numeric spec.
    # Returning a bare "-" collapses the column, and a run of unavailable
    # metrics then renders as "1736---" with every later column shifted --
    # which is exactly what an nvidia-smi-backed host (no PCIe, no NVLink)
    # produces. Specs without a width (the per-GPU list) stay unpadded.
    m = _WIDTH_RE.match(spec)
    if not m:
        return "-"
    return format("-", f"{m.group(1) or '>'}{m.group(2)}")


def _print_report(by_phase, n_gpus, label, interval, cumulative_steps=None):
    """Render one table. ``by_phase`` maps phase -> accumulator (_acc_new)."""
    order = [p for p in _PHASE_ORDER if p in by_phase]
    order += sorted(p for p in by_phase if p not in _PHASE_ORDER)

    total_wall = sum(a["wall"] for a in by_phase.values())
    n_samples = sum(a["n"] for a in by_phase.values())

    scope = f"steps={cumulative_steps}" if cumulative_steps else "1 step"
    lines = []
    lines.append(
        f"[gpu-profiler] {label} | {n_gpus} GPU(s) | {scope} | samples={n_samples} "
        f"| interval~{interval}s | idle<{_IDLE_THRESH}% sm"
    )
    header = (
        f"{'phase':<20}{'n':>6}{'wall_s':>9}{'share%':>8}{'sm%':>7}{'memBW%':>8}"
        f"{'idle%':>7}{'maxGap':>8}{'cpu%':>8}{'memGB':>8}{'powerW':>8}{'smClk':>7}"
        f"{'nvlink':>9}{'pcieTX':>8}{'pcieRX':>8}  per-GPU sm%"
    )
    lines.append(header)
    lines.append("-" * len(header))

    weighted_sm = 0.0
    for phase in order:
        a = _acc_finalize(by_phase[phase], n_gpus)
        share = (100.0 * a["wall"] / total_wall) if total_wall else 0.0
        per_gpu = "/".join(_fmt(v, ".0f") for v in a["per_gpu_sm"])
        # n is printed so a row backed by one or two samples is not mistaken for
        # a measurement: short phases can be narrower than the sample interval.
        lines.append(
            f"{phase:<20}{a['n']:>6}{a['wall']:>9.1f}{share:>8.1f}"
            f"{_fmt(a['sm_util'], '>7.1f')}{_fmt(a['mem_bw_util'], '>8.1f')}"
            f"{a['idle_pct']:>7.1f}{a['max_gap']:>8.1f}{_fmt(a['cpu_pct'], '>8.0f')}"
            f"{_fmt(a['mem_used_gb'], '>8.1f')}{_fmt(a['power_w'], '>8.0f')}"
            f"{_fmt(a['sm_clock_mhz'], '>7.0f')}{_fmt(a['nvlink_mb_s'], '>9.0f')}"
            f"{_fmt(a['pcie_tx_mb_s'], '>8.0f')}{_fmt(a['pcie_rx_mb_s'], '>8.0f')}  {per_gpu}"
        )
        if a["sm_util"] is not None:
            weighted_sm += a["sm_util"] * a["wall"]

    avg_sm = (weighted_sm / total_wall) if total_wall else 0.0
    total_label = "TOTAL (run)" if cumulative_steps else "TOTAL/step"
    lines.append("-" * len(header))
    lines.append(
        f"{total_label:<20}{n_samples:>6}{total_wall:>9.1f}{100.0:>8.1f}{avg_sm:>7.1f}"
        f"{'':>8}{'':>7}{'':>8}{'':>8}{'':>8}{'':>8}{'':>7}{'':>9}{'':>8}{'':>8}  "
        f"(wall-weighted mean SM across phases)"
    )
    if cumulative_steps:
        lines.append(
            f"{'per step':<20}{'':>6}{total_wall / cumulative_steps:>9.1f}"
            f"  (mean wall per boundary phase, incl. periodic phases amortized)"
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
            # "(idle/other)" stands in for the between-step batch assembly:
            # GPU parked, driver CPU busy -- the signature a prefetch removes.
            base = {
                "gen": 60,
                "update_actor": 96,
                "teacher_forward": 91,
                "save_checkpoint": 3,
                "(idle/other)": 2,
            }.get(phase, 95)
            return [
                {
                    "sm_util": max(0, min(100, base + random.uniform(-4, 4))),
                    "mem_bw_util": base * 0.6,
                    "mem_used_gb": 45 + random.uniform(-1, 1),
                    "power_w": 250 + random.uniform(-30, 30),
                    "sm_clock_mhz": 1700,
                    "pcie_tx_mb_s": 800 if phase == "update_actor" else 50,
                    "pcie_rx_mb_s": 800 if phase == "update_actor" else 50,
                    "nvlink_mb_s": 12000 if phase == "update_actor" else 100,
                }
                for _ in range(self.n_gpus)
            ]

    class _FakeHost:
        """Driver CPU: busy while assembling a batch, idle while the GPU works."""

        n_cpus = 8

        def sample(self):
            phase = _sampler._current_phase() if _sampler else "?"
            return {"cpu_pct": 380.0 if phase == "(idle/other)" else 40.0}

    os.environ["GPU_PROFILER"] = "1"
    _ENABLED = True
    _INTERVAL = 0.02
    _ROLLUP_EVERY = 2
    _sampler = _Sampler(_FakeBackend(3), _INTERVAL, host=_FakeHost())
    print("running synthetic profile ...")
    for step in range(2):
        time.sleep(0.15)  # between-step batch assembly -> "(idle/other)"
        push_phase("step")
        for phase, dur in [
            ("gen", 0.3),
            ("old_log_prob", 0.1),
            ("teacher_forward", 0.2),
            ("ref", 0.1),
            ("update_actor", 0.5),
        ]:
            push_phase(phase)
            t0 = now()
            time.sleep(dur)
            if phase == "gen" and step == 0:
                print("  GEN-UTIL window:", round(mean_util_between(t0, now()) or -1, 1))
            pop_phase(phase)
        if step == 1:  # periodic phase: only visible in the cumulative table
            push_phase("save_checkpoint")
            time.sleep(0.2)
            pop_phase("save_checkpoint")
        pop_phase("step")  # boundary -> per-step report (+ rollup every 2)
    _sampler.stop()
    print("self-test complete")
