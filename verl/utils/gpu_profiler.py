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
  GPU_PROFILER_TRACE=<path>    also write every raw sample to a CSV (ts, clock,
                               pid, phase, and per-GPU sm%, memBW%, power W, SM
                               clock, PCIe RX, NVLink). The tables above are
                               per-step aggregates reset at each boundary, so
                               they say a gap happened and how long it was but
                               not *when*; the trace is what lines a dip seen on
                               an external monitor up against the phase that
                               produced it. The pid is inserted before the
                               extension (``/tmp/t.csv`` -> ``/tmp/t.1234.csv``)
                               because two processes sample -- see
                               ``_trace_path_for_pid``. Analyse with
                               ``scripts/gpu_stall_scan.py``.
"""

import atexit
import contextlib
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict

# Leading alignment flag + field width of a format spec such as ">9.0f".
_WIDTH_RE = re.compile(r"([<>^]?)(\d+)")

__all__ = [
    "enabled",
    "push_phase",
    "pop_phase",
    "mean_util_between",
    "per_gpu_util_between",
    "residency_between",
    "format_residency",
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


# What the driver's threads are doing, counted rather than stacked.
#
# push_phase/pop_phase keep a single list, which is correct for one training
# thread and meaningless for validation: three pipeline slots push and pop
# concurrently and the top of the stack becomes whichever thread moved last.
#
# The question that decides where the remaining idle comes from is not "which
# phase" but "how many slots were in each at once" -- every card empty while
# all three slots sit in envstep is the environment, and the same picture while
# they sit in preproc is the driver's own Python. A counter answers that; a
# stack cannot.
_ACTIVITY_LOCK = threading.Lock()
_ACTIVITY: Dict[str, int] = {}


@contextlib.contextmanager
def activity(name: str):
    """Count this thread as being in ``name`` for as long as the block runs."""
    with _ACTIVITY_LOCK:
        _ACTIVITY[name] = _ACTIVITY.get(name, 0) + 1
    try:
        yield
    finally:
        with _ACTIVITY_LOCK:
            remaining = _ACTIVITY.get(name, 0) - 1
            if remaining > 0:
                _ACTIVITY[name] = remaining
            else:
                _ACTIVITY.pop(name, None)


def activity_snapshot() -> Dict[str, int]:
    with _ACTIVITY_LOCK:
        return dict(_ACTIVITY)


# --------------------------------------------------------------------------- #
# Gauges: how much work EXISTS, beside how much is running
# --------------------------------------------------------------------------- #
# The census and the frames both answer "what are the threads doing". Neither
# answers "was there anything else they could have been doing", and without that
# the two explanations for an idle GPU are indistinguishable:
#
#   ready=0,  retriever_inflight=10, gen_inflight=0  -> RETRIEVER_DEPENDENCY
#   ready=32, retriever_inflight=10, gen_inflight=0  -> SCHEDULER_STARVATION
#
# The observable part is identical -- every card idle with the retriever busy --
# and the prescriptions are opposite: make the retriever faster, or stop
# lock-stepping and submit the work that was already waiting. This arm has one
# reading of exactly that shape and cannot yet say which of the two it is.
#
# Counts, not durations. A gauge says how many of a thing exist right now, and
# the sampler records it beside the utilisation, so an excursion can be read as
# a STATE rather than as a phase name.
_GAUGE_LOCK = threading.Lock()
_GAUGES: Dict[str, int] = {}

# The ones the classifier reads. Named here so the trace schema is stable across
# runs that instrument different amounts, and so a typo at a call site shows up
# as an unknown gauge rather than as a silently missing one.
GAUGE_NAMES = (
    "ready",               # prepared and submittable, but not submitted
    "gen_inflight",        # handed to the engine, not yet returned
    "retriever_inflight",  # waiting on the retrieval service
    "env_inflight",        # inside env.step / a tool
    "future_wait",         # driver threads blocked on a Future
    "slots_free",          # pipeline slots that are idle
    "placeable_ready",     # queued batches a free slot could actually run
)


def gauge_set(name: str, value: int) -> None:
    with _GAUGE_LOCK:
        if value:
            _GAUGES[name] = int(value)
        else:
            _GAUGES.pop(name, None)


def gauge_add(name: str, delta: int) -> None:
    if not delta:
        return
    with _GAUGE_LOCK:
        total = _GAUGES.get(name, 0) + int(delta)
        if total > 0:
            _GAUGES[name] = total
        else:
            _GAUGES.pop(name, None)


@contextlib.contextmanager
def inflight(name: str, n: int = 1):
    """Hold ``n`` on a gauge for the duration of the block.

    Paired in a finally, like activity(), because the interesting blocks are the
    ones that raise: a retrieval that times out must not leave the gauge reading
    "10 in flight" for the rest of the run, which would classify every later
    excursion as RETRIEVER_DEPENDENCY.
    """
    gauge_add(name, n)
    try:
        yield
    finally:
        gauge_add(name, -n)


# Some gauges already have an authoritative owner and should not be mirrored.
# gen_inflight is len(PumpClient._pending), mutated at five places -- submit, the
# three completion paths and _fail_all -- and a gauge that misses one of them
# reads high forever, which classifies every later excursion as GPU-SIDE. So the
# owner registers a reader instead and the sampler pulls it.
#
# The callable must not block: it runs on the sampler thread once per interval,
# and a sampler that stalls stops recording the very excursion it was called
# during. PumpClient.in_flight takes its lock for a dict length and no more.
_GAUGE_SOURCES: Dict[str, Any] = {}

# Stack states, interned. The same handful of states repeat thousands of times
# over a run -- the interesting question is which ones, not how many bytes they
# take -- so the trace carries an integer and a sidecar carries the text once.
# Bounded: a run that somehow produces more distinct states than this writes -1
# rather than growing a dict without limit inside the sampler.
_STACK_TABLE_MAX = int(os.environ.get("GPU_PROFILER_STACK_TABLE_MAX", "20000"))
_STACK_IDS: Dict[tuple, int] = {}
_STACK_BY_ID: Dict[int, tuple] = {}


def stack_state_id(keys):
    """Intern one sample's stack state, dropping threads outside this repo.

    They are dropped from the KEY, not merely from the display: a hundred-odd
    parked infrastructure threads drift in and out constantly, and keeping them
    would make almost every sample a distinct state -- an intern table with no
    repeats, which is a list. What remains is our frames, which is what anyone
    reading an excursion wants, plus a count of the rest.
    """
    ours = tuple(sorted(k for k in keys if not k.endswith(" -")))
    outside = len(keys) - len(ours)
    # KEYED ON OURS ALONE. The parked count drifts by one or two constantly --
    # Ray reaps and spawns pool workers all run -- and folding it into the key
    # made "one of our frames, 137 parked" and "the same frame, 138 parked" two
    # states. That is an intern table with no repeats, which is a list. The
    # count is still reported, from the first sighting, because "alone" and
    # "alongside 138 parked threads" are different situations to read.
    known = _STACK_IDS.get(ours)
    if known is not None:
        return known, (ours, outside)
    if len(_STACK_IDS) >= _STACK_TABLE_MAX:
        return -1, (ours, outside)
    ident = len(_STACK_IDS)
    _STACK_IDS[ours] = ident
    _STACK_BY_ID[ident] = (ours, outside)
    return ident, (ours, outside)


def register_gauge_source(name: str, read) -> None:
    with _GAUGE_LOCK:
        _GAUGE_SOURCES[name] = read


def unregister_gauge_source(name: str) -> None:
    with _GAUGE_LOCK:
        _GAUGE_SOURCES.pop(name, None)


def gauge_snapshot() -> Dict[str, int]:
    with _GAUGE_LOCK:
        snap = dict(_GAUGES)
        sources = dict(_GAUGE_SOURCES)
    for name, read in sources.items():
        try:
            value = int(read())
        except Exception:  # pragma: no cover - a broken reader is not a run-ender
            continue
        if value:
            snap[name] = value
    return snap


def reset_gauges() -> None:
    """For tests. A leaked gauge is a wrong classification, not a wrong number."""
    with _GAUGE_LOCK:
        _GAUGES.clear()
        _GAUGE_SOURCES.clear()


# Where the threads actually are, asked of the interpreter rather than of a tag.
#
# The census answers "which tagged phase", and twice now the answer has been
# "none of them": 1.0 of 3 slots, then 2.1 of 4, in no tagged phase while every
# card was idle. Both times the response was to guess at a region and tag it --
# record/assemble, then the env reset -- and both guesses measured below 0.05 of
# a slot. A residual that survives two prescriptions is not going to be named by
# a third guess.
#
# sys._current_frames() does not guess. It returns the frame every live thread
# is executing right now, so a thread blocked in a socket read says socket.py
# and a thread rebuilding tensors says the line it is on. Two frames are kept
# per thread: the deepest one inside this repository, which says WHICH of our
# code is responsible, and the innermost frame of all, which says what it is
# doing there -- and those differ precisely in the interesting cases, where our
# code is waiting inside somebody else's.
# Measured at 49.8 us per snapshot with 14 live threads, which at the 0.3 s
# sampling cadence is 0.63 s over a 3800 s run -- and it is spent on the
# profiler thread, not on a slot. Off by env var anyway, because an instrument
# that cannot be turned off cannot be ruled out as the cause of what it sees.
_STACKS = os.environ.get("GPU_PROFILER_STACKS", "1").strip().lower() not in ("0", "false", "no", "off")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _frame_key(frame) -> str:
    code = frame.f_code
    return f"{os.path.basename(code.co_filename)}:{frame.f_lineno} {code.co_name}"


# Innermost frames that mean "parked", by (file, function). A thread here is
# waiting and burning no CPU, which is a different finding from a thread
# running Python -- they need opposite fixes and the list must not merge them.
# The set is not exhaustive and cannot be: an unlisted blocking primitive reads
# as RUNNING, which is why the cpu_pct band, not this, decides the verdict.
_PARKED_FRAMES = frozenset({
    ("threading.py", "wait"), ("threading.py", "acquire"),
    ("threading.py", "_wait_for_tstate_lock"), ("threading.py", "join"),
    ("thread.py", "_worker"),                      # idle concurrent.futures worker
    ("queue.py", "get"), ("queue.py", "put"),
    ("_base.py", "wait"), ("_base.py", "result"),  # concurrent.futures
    ("selectors.py", "select"), ("selectors.py", "poll"),
    ("socket.py", "readinto"), ("socket.py", "accept"),
    ("ssl.py", "read"), ("ssl.py", "recv_into"),
    ("connection.py", "_recv"), ("connection.py", "_recv_bytes"),
    ("subprocess.py", "_try_wait"), ("subprocess.py", "wait"),
})


def _is_parked(frame) -> bool:
    code = frame.f_code
    return (os.path.basename(code.co_filename), code.co_name) in _PARKED_FRAMES


def stack_snapshot() -> List[str]:
    """One key per live thread: our deepest frame, and the innermost frame.

    Prefixed "R " when the innermost frame is running and "B " when it is a
    known wait, and "-" instead of a repo frame when the thread is not in this
    repository's code at all. The first reading of this printed 126.58 threads
    on concurrent.futures' idle _worker, above every frame that meant anything:
    a machine running 140 threads has ~135 parked ones, and a top-N list that
    ranks by count is a list of them.
    """
    if not _STACKS:
        return []
    out = []
    me = threading.get_ident()
    for ident, frame in sys._current_frames().items():
        if ident == me:
            continue
        innermost, repo, walk, depth = frame, None, frame, 0
        while walk is not None and depth < 200:
            name = walk.f_code.co_filename
            if "gpu_profiler" in name:
                repo = None
                innermost = None
                break
            if repo is None and name.startswith(_REPO_ROOT):
                repo = walk
            walk = walk.f_back
            depth += 1
        if innermost is None:
            continue  # this thread is inside the profiler; it is not the subject
        state = "B" if _is_parked(innermost) else "R"
        inner_key = _frame_key(innermost)
        if repo is None:
            out.append(f"{state} -")
        elif repo is innermost:
            out.append(f"{state} {inner_key}")
        else:
            out.append(f"{state} {_frame_key(repo)} <- {inner_key}")
    return out


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

# Per-GPU columns of the trace, in file order. sm/memBW came first and stay
# first so traces written before the others existed still parse.
_TRACE_PER_GPU = (
    ("sm_pct_per_gpu", "sm_util", "{:.0f}"),
    ("membw_pct_per_gpu", "mem_bw_util", "{:.0f}"),
    ("power_w_per_gpu", "power_w", "{:.0f}"),
    ("smclk_mhz_per_gpu", "sm_clock_mhz", "{:.0f}"),
    ("pcie_rx_mb_s_per_gpu", "pcie_rx_mb_s", "{:.0f}"),
    ("nvlink_mb_s_per_gpu", "nvlink_mb_s", "{:.0f}"),
)
_TRACE_HEADER = (
    "ts,clock,pid,phase," + ",".join(c for c, _, _ in _TRACE_PER_GPU) + ",driver_cpu_pct,"
    # The gauges, in a fixed order, so a scanner can index them positionally and
    # a run that instruments nothing still writes the columns (as zeros) rather
    # than a narrower row that silently changes the schema.
    + ",".join(GAUGE_NAMES)
    # WHERE, per sample. The gauges say which dependency an excursion was on;
    # these say which of our code was live during it. Without them a dip in the
    # trace carries a reason and a phase name and nothing that points at a line,
    # and the frames only existed as an average over a whole reporting window --
    # which is the wrong shape for "this 1.8 s dip".
    + ",activity,stack_id\n"
)


def _trace_path_for_pid(path: str) -> str:
    """One trace file per process.

    A sampler starts in the driver (``ray_trainer._timer``) and again in rank 0's
    worker (``dp_actor._actor_phase``). They are different processes holding
    different file offsets, so opening one path ``"w"`` from both makes each
    overwrite the other's bytes: the file ends up with roughly ONE sampler's
    worth of rows, stitched together from two streams at whatever offset each
    had reached. Aggregate means survive that -- both samplers read the same
    devices -- but anything that treats consecutive rows as consecutive in time
    does not, and that is most of what a per-sample trace is for.
    """
    root, ext = os.path.splitext(path)
    return f"{root}.{os.getpid()}{ext or '.csv'}"


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
        self._cpu_trace = []
        self._act_trace = []
        self._stack_trace = []
        self._gauge_trace = []
        self._stacks_seen = set()
        self._stop = threading.Event()
        self._step_idx = 0
        # Cumulative per-phase accumulators across every step so far. Aggregated
        # incrementally (sums/counts, not raw samples) so memory stays flat over
        # a long run.
        self._cum = {}
        self._cum_steps = 0
        self._trace = None
        self._stacks_file = None
        if _TRACE_PATH:
            path = _trace_path_for_pid(_TRACE_PATH)
            try:
                self._trace = open(path, "w", buffering=1)
                self._trace.write(_TRACE_HEADER)
                print(f"[gpu-profiler] per-sample trace -> {path}", flush=True)
            except OSError as e:
                print(f"[gpu-profiler] could not open GPU_PROFILER_TRACE={path}: {e}", flush=True)
            try:
                # Beside the trace, named after it, so a scanner given one path
                # can find the other without being told.
                self._stacks_file = open(path + ".stacks", "w", buffering=1)
                self._stacks_file.write("stack_id\tthreads_outside_repo\tframes\n")
            except OSError:
                self._stacks_file = None
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
                # Kept in its own list rather than widened into the tuple above,
                # which two other readers unpack by shape.
                self._cpu_trace.append((t, (host or {}).get("cpu_pct")))
                self._act_trace.append((t, activity_snapshot()))
                gauges = gauge_snapshot()
                self._gauge_trace.append((t, gauges))
                stacks = stack_snapshot()
                stack_id, _key = stack_state_id(stacks)
                # The ID, not the strings. stack_snapshot builds a new string per
                # thread per sample and they are not interned, so at a 0.1 s
                # interval over a 3800 s run keeping them costs about 0.3 GB
                # against about 1 MB for the ids. Sub-second excursions are what
                # 0.1 s sampling is for, so this is what makes that interval
                # affordable rather than a memory bomb an hour in.
                self._stack_trace.append((t, stack_id))
                census = activity_snapshot()
            self._write_trace(t, phase, per_gpu, host, gauges, census, stack_id)

    def _write_trace(self, t, phase, per_gpu, host, gauges=None, census=None, stacks=None):
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
            cols = [
                ";".join("" if g.get(key) is None else fmt.format(g[key]) for g in per_gpu)
                for _, key, fmt in _TRACE_PER_GPU
            ]
            cpu = (host or {}).get("cpu_pct")
            g = gauges or {}
            # Semicolons inside the field, because the file is comma-separated
            # and a census is a mapping. Same convention as the per-GPU columns.
            act = ";".join(f"{k}:{v}" for k, v in sorted((census or {}).items()))
            sid, key = ((stacks, _STACK_BY_ID.get(stacks, ((), 0))) if isinstance(stacks, int)
                        else stack_state_id(stacks or []))
            if self._stacks_file is not None and sid >= 0 and sid not in self._stacks_seen:
                self._stacks_seen.add(sid)
                frames, outside = key
                self._stacks_file.write(f"{sid}\t{outside}\t{' | '.join(frames)}\n")
            self._trace.write(
                f"{t:.3f},{time.strftime('%H:%M:%S', time.localtime())},{os.getpid()},{phase},"
                f"{','.join(cols)},{'' if cpu is None else f'{cpu:.0f}'},"
                f"{','.join(str(g.get(name, 0)) for name in GAUGE_NAMES)},{act},{sid}\n"
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

    def residency_between(self, t0, t1, busy_thresh=None):
        """How MANY GPUs were busy at once, over [t0, t1].

        The two instruments that already exist both miss this, in opposite
        directions, and between them they cost two wrong conclusions:

        * ``[val-pipeline]`` counts a slot as running whenever it is inside a
          batch. A slot blocked in ``env.step`` is running by that measure while
          every GPU is empty -- it reported "NOTHING running 0.1%" for a run in
          which NVML saw 285 s of node-wide idle.
        * ``genGPU%`` measures only the window inside ``generate``, so it cannot
          see a gap that happens between generates at all.

        This one asks the only question that decides whether more work in flight
        would help: at this instant, how many cards had something to do? Zero is
        recoverable by overlapping another batch; three-at-87% is the decode
        step's duty cycle and is not.

        Returns ``None`` if the window holds no samples, else a dict:
        ``n_gpus``, ``samples``, ``wall``, ``counts`` (busy-count -> samples),
        ``wall_by_count``, and ``pct`` (busy-count -> % of samples).
        """
        thresh = _IDLE_THRESH if busy_thresh is None else busy_thresh
        with self._lock:
            window = [(ts, vals) for (ts, vals) in self._util_trace if t0 <= ts <= t1]
        if not window:
            return None
        n_gpus = max(len(vals) for _ts, vals in window)
        counts = {k: 0 for k in range(n_gpus + 1)}
        wall_by_count = {k: 0.0 for k in range(n_gpus + 1)}
        # A gap far larger than the sample interval is the sampler having been
        # stopped, not the GPU having been idle for that long -- charge one
        # interval for it rather than the whole gap.
        cap = _INTERVAL * _CONTIGUITY_SLACK
        previous = None
        for ts, vals in window:
            busy = sum(1 for v in vals if v is not None and v >= thresh)
            dt = _INTERVAL if previous is None else min(ts - previous, cap)
            counts[busy] += 1
            wall_by_count[busy] += dt
            previous = ts
        total = sum(counts.values())
        # Per-GPU means over the same window. Whether the partly-busy time is one
        # rank always waiting or all three taking turns decides the prescription:
        # a lopsided column is a rank finishing its chunk of a collective call
        # early and idling until the slowest one returns, which more batches in
        # flight cannot fix because the worker group runs one call at a time.
        per_gpu = []
        for gi in range(n_gpus):
            col = [vals[gi] for _ts, vals in window if gi < len(vals) and vals[gi] is not None]
            per_gpu.append((sum(col) / len(col)) if col else None)
        # The driver's own CPU, split the same way. This is what separates the
        # two reasons every card can be empty at once, which look identical on
        # the GPU and need opposite fixes: Python holding the GIL (cpu near or
        # above one core) against the whole process waiting on something off the
        # box (cpu at idle, which is what a retriever round trip looks like).
        empty_ts = {ts for ts, vals in window if not any(v is not None and v >= thresh for v in vals)}
        busy_ts = {ts for ts, vals in window if sum(1 for v in vals if v is not None and v >= thresh) == n_gpus}
        with self._lock:
            cpu_at = {ts: pct for ts, pct in self._cpu_trace if pct is not None}
            act_at = dict(self._act_trace)
            stack_at = dict(self._stack_trace)
            gauge_at = dict(self._gauge_trace)
        def _mean(stamps):
            vals = [cpu_at[ts] for ts in stamps if ts in cpu_at]
            return (sum(vals) / len(vals)) if vals else None

        def _busy_util():
            vals = [
                v
                for _ts, per in window
                if sum(1 for g in per if g is not None and g >= thresh) == n_gpus
                for v in per
                if v is not None
            ]
            return (sum(vals) / len(vals)) if vals else None

        def _bands(stamps):
            """Share of these samples with the driver blocked / between / running."""
            vals = [cpu_at[ts] for ts in stamps if ts in cpu_at]
            if not vals:
                return None
            n = len(vals)
            return {
                "blocked": 100.0 * sum(1 for v in vals if v < 20) / n,
                "between": 100.0 * sum(1 for v in vals if 20 <= v < 60) / n,
                "running": 100.0 * sum(1 for v in vals if v >= 60) / n,
            }

        def _census(stamps, table):
            """Mean number of threads in each activity, over these samples."""
            seen = [table[ts] for ts in stamps if ts in table]
            if not seen:
                return None
            names = {k for snap in seen for k in snap}
            return {k: sum(snap.get(k, 0) for snap in seen) / len(seen) for k in names}

        def _gauges(stamps):
            """Mean of each gauge over these samples.

            The mean, not the max: one excursion with 40 ready episodes and
            ninety with none is not "ready was 40", and the classifier below has
            to distinguish a standing backlog from a single blip.
            """
            seen = [gauge_at[ts] for ts in stamps if ts in gauge_at]
            if not seen:
                return None
            return {
                name: sum(snap.get(name, 0) for snap in seen) / len(seen)
                for name in GAUGE_NAMES
            }

        def _stacks(stamps, top=60):
            """The frames the threads were actually on, over these samples.

            Mean threads per sample on each frame, so it reads in the same unit
            as the census beside it -- "1.2 of 3 slots were here" -- and the two
            can be compared directly instead of one being a share and the other
            a count.
            """
            seen = [stack_at[ts] for ts in stamps if ts in stack_at]
            if not seen:
                return None
            counted = Counter()
            for entry in seen:
                # An interned id from the sampler, or a literal list of keys from
                # a test double. Both, because the doubles are far more readable
                # written out.
                if isinstance(entry, int):
                    ours, outside = _STACK_BY_ID.get(entry, ((), 0))
                    counted.update(ours)
                    if outside:
                        # Weighted, not expanded: expanding is a 140-element list
                        # per sample, the allocation the ids exist to avoid.
                        counted["B -"] += outside
                else:
                    counted.update(entry)
            # Deep, and truncated by the FORMATTER after it has split running
            # from parked. Truncating here ranks by raw count, and the raw count
            # is led by a hundred-odd parked infrastructure threads -- which is
            # how a six-entry list came back holding nothing but them.
            return [(key, n / len(seen)) for key, n in counted.most_common(top)]
        wall = sum(wall_by_count.values()) or 1.0
        return {
            "n_gpus": n_gpus,
            "samples": total,
            "wall": wall,
            "counts": counts,
            "wall_by_count": wall_by_count,
            # WALL, not sample count. These print beside wall_by_count and were
            # a share of samples, which is the same thing only if every sample
            # covers the same interval -- and the one bucket where it does not
            # is the one this whole exercise is about. While the GPUs are EMPTY
            # the driver is running Python holding the GIL, the sampler thread
            # is starved, and its interval stretches: measured at 3541s / 312s,
            # EMPTY was 8.8% of the seconds and printed as 6.8% of the samples.
            # Counting samples therefore UNDER-reports exactly the idle it is
            # there to find, by about a fifth of it, and does so silently
            # because the seconds sitting next to it looked consistent.
            "pct": {k: 100.0 * v / wall for k, v in wall_by_count.items()},
            "per_gpu": per_gpu,
            "cpu_when_empty": _mean(empty_ts),
            "cpu_when_busy": _mean(busy_ts),
            # The mean names neither mode of a two-mode distribution. A run
            # whose EMPTY samples are half at 5% (blocked on a future) and half
            # at 100% (working) reads 52%, which the verdict then calls
            # UNRESOLVED -- true of the mean and true of nothing that happened.
            "empty_cpu_bands": _bands(empty_ts),
            # And the census restricted to the samples where the driver WAS
            # working, which is the only place "where" is a meaningful question.
            "activity_when_empty_running": _census(
                {ts for ts in empty_ts if cpu_at.get(ts, 0.0) >= 60}, act_at
            ),
            # Mean util across the cards over the samples where ALL of them had
            # work. This is the duty cycle -- the engine's own per-step host
            # processing showing between kernel launches -- and it is the number
            # an engine setting is aimed at. Node util cannot stand in for it:
            # node util moves when EMPTY or PARTIAL move, and multi-step
            # scheduling or async scheduling touch neither.
            "util_when_busy": _busy_util(),
            "activity_when_empty": _census(empty_ts, act_at),
            # The same question asked of the interpreter instead of the tags,
            # for the part of EMPTY that no tag has ever claimed.
            "gauges_when_empty": _gauges(empty_ts),
            "gauges_when_busy": _gauges(busy_ts),
            "stacks_when_empty": _stacks(empty_ts),
            "stacks_when_busy": _stacks(busy_ts),
            "activity_when_busy": _census(busy_ts, act_at),
            # The most threads ever counted at once, which is the slot count in
            # practice. Without it "envstep 1.1" has no denominator and reads as
            # a majority when it is a third.
            "slots_seen": max((sum(snap.values()) for snap in act_at.values()), default=0),
        }

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
            self._cpu_trace = []
            self._act_trace = []
            self._stack_trace = []
            self._gauge_trace = []
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


def ensure_started():
    """Start the sampler now, for a caller that only reads util windows.

    ``push_phase`` is the only other thing that starts it, and the validation
    path never calls it -- so on an evaluation the sampler never existed and
    ``mean_util_between`` returned None for every window. The turn table printed
    its genGPU% and perGPU% columns as "-" on every evaluation ever run,
    GPU_PROFILER=1 included, which reads as "the GPU was not measured" when it
    actually means "nothing ever asked it to be".

    Returns whether a sampler is running. Cheap to call repeatedly: a pointer
    check once the sampler exists, and it does not retry a failed backend.
    """
    if not enabled():
        return False
    return _ensure_sampler() is not None


def mean_util_between(t0, t1):
    if not enabled() or _sampler is None:
        return None
    return _sampler.mean_util_between(t0, t1)


def per_gpu_util_between(t0, t1):
    if not enabled() or _sampler is None:
        return None
    return _sampler.per_gpu_util_between(t0, t1)


def residency_between(t0, t1, busy_thresh=None):
    if not enabled() or _sampler is None:
        return None
    return _sampler.residency_between(t0, t1, busy_thresh=busy_thresh)


def classify_empty(gauges) -> Optional[str]:
    """Why the cards were empty, from what work existed at the time.

    Rule-based on purpose, and it keeps an UNINSTRUMENTED answer. The failure
    mode of a classifier like this is that it always names something: every gap
    gets attributed to whichever gauge happens to be instrumented, and the
    profiler then argues confidently for a fix aimed at the wrong thing. Twice
    in this arm a phase was named and then measured at under 0.05 of a slot.
    So an excursion that matches no rule says so.

    The distinction the whole thing exists for is the first two:

      ready == 0 and the retriever busy   -> nothing COULD be submitted
      ready >  0 and the retriever busy   -> something could and was not

    Same observable, opposite prescriptions.
    """
    if not gauges:
        return None
    ready = gauges.get("ready", 0)
    gen = gauges.get("gen_inflight", 0)
    retr = gauges.get("retriever_inflight", 0)
    env = gauges.get("env_inflight", 0)
    wait = gauges.get("future_wait", 0)

    if ready >= 1 and gen < 0.5:
        return ("SCHEDULER STARVATION -- work was ready and the engine had none of it. "
                "The fix is on the submitting side, not the retriever's")
    if gen >= 0.5:
        return ("GPU-SIDE -- requests were with the engine while the cards read idle: "
                "either the engine's own host work, or a sample window artefact")
    if retr >= 1 and ready < 1:
        return ("RETRIEVER DEPENDENCY -- nothing was submittable and the retrieval "
                "service was the thing being waited on")
    if env >= 1 and ready < 1:
        return ("ENVIRONMENT DEPENDENCY -- nothing was submittable and env.step was "
                "the thing being waited on")
    if wait >= 1:
        return "DRIVER WAITING ON A FUTURE with no downstream work recorded"
    return None


def format_residency(res) -> str:
    """One line: how much of the window had 0, 1, ... n cards with work."""
    if not res:
        return ""
    n = res["n_gpus"]
    parts = []
    for k in range(n, -1, -1):
        parts.append(f"{k}gpu {res['pct'][k]:.1f}% ({res['wall_by_count'][k]:.0f}s)")
    empty = res["pct"][0]
    partial = sum(res["pct"][k] for k in range(1, n))
    busy_util = res.get("util_when_busy")
    duty = f", reading {busy_util:.1f}%" if busy_util is not None else ""
    per_gpu = res.get("per_gpu") or []
    spread = ""
    known = [v for v in per_gpu if v is not None]
    if len(known) > 1:
        cols = " ".join("--" if v is None else f"{v:.0f}" for v in per_gpu)
        spread = f" | per-gpu {cols} (spread {max(known) - min(known):.0f} pt)"
    # WHY every card was empty. Two readings, and the verdict is the one they
    # agree on -- an earlier version took the top activity alone and named the
    # retriever on a run whose driver was burning 82% of a core, which is the
    # opposite conclusion and the opposite fix.
    #
    #   CPU says WHETHER the driver was working. A process blocked on a socket
    #   burns no CPU; that part is unambiguous at these extremes.
    #   The census says WHERE, but only when one phase actually dominates the
    #   slots -- "envstep 1.1" out of three slots is a third of one, not a
    #   verdict.
    why = ""
    empty_cpu, busy_cpu = res.get("cpu_when_empty"), res.get("cpu_when_busy")
    census = res.get("activity_when_empty") or {}
    slots = res.get("slots_seen") or 0
    ranked = sorted(census.items(), key=lambda kv: -kv[1])
    accounted = sum(census.values())
    top_name, top_n = (ranked[0] if ranked else (None, 0.0))
    # Against the slot count, not against the accounted total: a phase can be
    # most of what was tagged while most of the slots were somewhere untagged.
    dominates = slots > 0 and top_n >= 0.5 * slots

    driver_running = empty_cpu is not None and empty_cpu >= 60
    driver_blocked = empty_cpu is not None and empty_cpu < 20

    # What work EXISTED while the cards were empty, and what that implies.
    # Printed before the census, because "nothing was submittable" and "32 things
    # were submittable" want different questions asked of the census after them.
    gauges = res.get("gauges_when_empty") or {}
    if any(v >= 0.05 for v in gauges.values()):
        shown = ", ".join(f"{k} {v:.1f}" for k, v in gauges.items() if v >= 0.05)
        why += f"\n[gpu-residency]    while EMPTY the work outstanding was: {shown}"
        verdict = classify_empty(gauges)
        why += (
            f"\n[gpu-residency]    -> {verdict}" if verdict else
            "\n[gpu-residency]    -> UNINSTRUMENTED: no rule matches these gauges, which is"
            " an answer -- the gap is somewhere nothing counts"
        )

    untagged = max(0.0, slots - accounted) if slots else 0.0
    if ranked:
        shown = ", ".join(f"{k} {v:.1f}" for k, v in ranked if v >= 0.05) or "nothing recorded"
        line = f"\n[gpu-residency] while EMPTY the slots were in: {shown}"
        if slots:
            line += f" (of {slots:.0f} slots; {untagged:.1f} in no tagged phase)"
        why += line

    # The frames, whenever the tags leave a third of a thread or more
    # unaccounted for.
    #
    # THESE TWO LINES ARE NOT IN THE SAME UNIT and saying so is the whole of
    # this comment. The census counts threads that entered a tagged phase --
    # four of them on this run. The frames count every live thread in the
    # process, and a machine running the pump, Ray, a retriever pool and three
    # slots has well over a hundred, nearly all parked. The first reading put
    # "126.58 (no repo frame) <- thread.py:81 _worker" at the top, which is
    # concurrent.futures' idle workers waiting for work that is not coming, and
    # pushed the one thread burning 65% of a core off the end of the list.
    #
    # So: threads outside this repository collapse to a count, because nothing
    # here can act on them; threads inside it are listed and split by whether
    # they are running Python or parked, because those need opposite fixes.
    stacks = res.get("stacks_when_empty") or []
    if stacks and untagged >= 0.33:
        outside = sum(mean for key, mean in stacks if key.endswith(" -"))
        ours = [(key[2:], key[0], mean) for key, mean in stacks if not key.endswith(" -")]
        running = [(k, m) for k, state, m in ours if state == "R" and m >= 0.05]
        parked = [(k, m) for k, state, m in ours if state == "B" and m >= 0.05]
        why += (
            f"\n[gpu-residency]    {untagged:.1f} of the tagged slots are in NO tagged phase. "
            f"Mean live threads per EMPTY sample, from the interpreter "
            f"({outside:.0f} outside this repo, parked or otherwise):"
        )
        for label, rows in (("RUNNING Python", running), ("PARKED, burning nothing", parked)):
            if rows:
                why += f"\n[gpu-residency]      -- {label} --"
                for key, mean in rows[:8]:
                    why += f"\n[gpu-residency]      {mean:5.2f}  {key}"
                if len(rows) > 8:
                    why += f"\n[gpu-residency]      ... and {len(rows) - 8} more below {rows[8][1]:.2f}"
        if not running:
            why += (
                "\n[gpu-residency]      -- nothing of ours was RUNNING; the CPU above is being "
                "burned outside this repo, or by a wait this list does not know is one --"
            )

    if driver_blocked:
        why += ("\n[gpu-residency] -> EMPTY is a WAIT OFF THE BOX: the driver burned "
                f"{empty_cpu:.0f}% of one core, and a process waiting on a socket burns none")
        if dominates and top_name != "envstep":
            why += f" (though {top_name} was the phase -- these disagree, trust neither yet)"
    elif driver_running:
        named = {
            "preproc": "tokenising",
            "decode": "detokenising",
            "glue": "padding and DataProto union between the phases",
            "dataload": "the val loader on the calling thread",
            "prepare": "batch preparation on the calling thread",
            "scoring": "reward accumulation on the calling thread",
            "envstep": "envstep -- the driver's work around the call, not the retriever",
            "record": "per-turn bookkeeping (to_list_of_dict expands the whole batch every turn)",
            "assemble": "gather_rollout_data, padding every turn of every trajectory",
        }.get(top_name, top_name)
        where = f", mostly in {named}" if dominates else ", and no single phase dominates"
        why += (f"\n[gpu-residency] -> EMPTY is the DRIVER RUNNING PYTHON: {empty_cpu:.0f}% of one "
                f"core while EMPTY against {busy_cpu:.0f}% while busy{where}")

    elif empty_cpu is not None:
        bands = res.get("empty_cpu_bands") or {}
        if bands and max(bands.get("blocked", 0), bands.get("running", 0)) >= 25:
            # Two modes, not one middling state. Say how EMPTY splits between
            # them and what the working half was doing -- the mean was hiding
            # both answers behind a number that described neither.
            running_census = res.get("activity_when_empty_running") or {}
            top = max(running_census.items(), key=lambda kv: -(-kv[1]), default=(None, 0.0))
            where = f", mostly in {top[0]}" if top[0] and top[1] >= 0.5 * max(slots, 1) else ""
            why += (
                f"\n[gpu-residency] -> EMPTY SPLITS: driver BLOCKED (cpu<20%) for "
                f"{bands.get('blocked', 0):.0f}% of it, RUNNING (cpu>=60%) for "
                f"{bands.get('running', 0):.0f}%, between for {bands.get('between', 0):.0f}%"
                f" (mean {empty_cpu:.0f}% of one core against {busy_cpu:.0f}% while busy)."
            )
            if running_census:
                shown = ", ".join(
                    f"{k} {v:.1f}" for k, v in sorted(running_census.items(), key=lambda kv: -kv[1])
                    if v >= 0.05
                )
                why += f"\n[gpu-residency]    while RUNNING-and-empty the slots were in: {shown}{where}"
            why += (
                "\n[gpu-residency]    BLOCKED is work the GPU does not have yet -- the tail of a "
                "turn, or a slot waiting on another. RUNNING is driver Python. They need "
                "different fixes and the mean names neither."
            )
        else:
            why += (f"\n[gpu-residency] -> EMPTY is UNRESOLVED: driver CPU {empty_cpu:.0f}% of one core "
                    f"(vs {busy_cpu:.0f}% while busy) is neither blocked nor clearly running")
    elif dominates:
        why += f"\n[gpu-residency] -> EMPTY is mostly {top_name}, but with no CPU sample to corroborate it"
    return (
        f"[gpu-residency] {res['wall']:.0f}s sampled: " + ", ".join(parts) + spread +
        f"\n[gpu-residency] EMPTY {empty:.1f}% (more batches in flight fill this), "
        f"PARTIAL {partial:.1f}% (a rank idle inside a collective call -- only the "
        f"pump fills this). All {n} cards had work {res['pct'][n]:.1f}% of the time"
        + duty + " -- THAT is the engine's duty cycle, and the only number an "
        "engine setting (multi-step, async scheduling) can move." + why
    )


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
    #
    # actor.h2d is the first thing a step does and the one a Ray actor cannot
    # overlap: calls run one at a time, so moving the batch onto the device
    # happens strictly after the previous step returned, whatever the driver is
    # doing at that moment. Whatever is left OUTSIDE it -- still tagged
    # (idle/other) in a worker trace -- is Ray deserialising the argument before
    # the method body runs, which no phase inside the body can reach.
    "actor.h2d",
    "actor.fwd",
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
