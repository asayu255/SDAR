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
"""Two idle GPUs that look identical and need opposite fixes.

    ready=0,  retriever_inflight=10, gen_inflight=0   nothing COULD be submitted
    ready=32, retriever_inflight=10, gen_inflight=0   something could and was not

Every device-side reading of those two is the same: three cards at zero with the
retriever busy. The census and the frame sampler cannot separate them either --
both answer "what are the threads doing", and in both cases the answer is "ten
of them are in search.py waiting". What separates them is whether any work
existed that nobody had submitted, and nothing in this profiler counted that.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from verl.utils import gpu_profiler as gp  # noqa: E402

SCANNER = Path(__file__).resolve().parents[2] / "scripts" / "gpu_stall_scan.py"


@pytest.fixture(autouse=True)
def clean_gauges():
    gp.reset_gauges()
    yield
    gp.reset_gauges()


# --------------------------------------------------------------------------- #
# the gauge itself
# --------------------------------------------------------------------------- #
def test_a_gauge_is_released_even_when_the_block_raises():
    """A retrieval that times out must not leave the gauge reading 10 forever.

    It would not look like a bug. It would look like every later excursion in
    the run being caused by the retriever.
    """
    with pytest.raises(RuntimeError):
        with gp.inflight("retriever_inflight", 10):
            assert gp.gauge_snapshot()["retriever_inflight"] == 10
            raise RuntimeError("the retriever timed out")
    assert "retriever_inflight" not in gp.gauge_snapshot()


def test_gauges_add_across_threads_and_come_back_to_zero():
    started, release = threading.Event(), threading.Event()

    def hold():
        with gp.inflight("env_inflight", 126):
            started.set()
            release.wait(5)

    threads = [threading.Thread(target=hold) for _ in range(3)]
    for t in threads:
        t.start()
    started.wait(5)
    time.sleep(0.05)
    assert gp.gauge_snapshot()["env_inflight"] == 3 * 126
    release.set()
    for t in threads:
        t.join()
    assert gp.gauge_snapshot() == {}


def test_a_pulled_gauge_comes_from_its_owner():
    """gen_inflight is len(PumpClient._pending), mutated in five places.

    Mirroring it means one missed site reads high forever, and a gen_inflight
    that never falls classifies every excursion as GPU_SIDE -- pointing the next
    fix at the engine when the engine was not involved.
    """
    pending = {"n": 7}
    gp.register_gauge_source("gen_inflight", lambda: pending["n"])
    assert gp.gauge_snapshot()["gen_inflight"] == 7
    pending["n"] = 0
    assert "gen_inflight" not in gp.gauge_snapshot()
    gp.unregister_gauge_source("gen_inflight")


def test_a_broken_gauge_source_does_not_take_the_sampler_down():
    gp.register_gauge_source("gen_inflight", lambda: 1 / 0)
    gp.gauge_add("ready", 3)
    assert gp.gauge_snapshot() == {"ready": 3}


# --------------------------------------------------------------------------- #
# the classification
# --------------------------------------------------------------------------- #
def test_the_two_cases_that_look_identical_on_the_device_are_separated():
    starving = gp.classify_empty({"ready": 32, "retriever_inflight": 10, "gen_inflight": 0})
    waiting = gp.classify_empty({"ready": 0, "retriever_inflight": 10, "gen_inflight": 0})
    assert "SCHEDULER STARVATION" in starving
    assert "RETRIEVER DEPENDENCY" in waiting


def test_an_unmatched_state_says_so_rather_than_naming_the_nearest_gauge():
    """A classifier with no UNINSTRUMENTED branch always finds a culprit.

    This arm has named a cause and then measured it under 0.05 of a slot twice
    -- record/assemble, then the env reset. Both times the honest answer was
    "nothing instrumented covers this".
    """
    assert gp.classify_empty({}) is None
    assert gp.classify_empty({"ready": 0, "gen_inflight": 0}) is None


def test_ready_beats_the_retriever_because_the_fix_is_different():
    """Both gauges are up; the prescription follows ready, not the retriever."""
    verdict = gp.classify_empty({"ready": 32, "retriever_inflight": 99, "env_inflight": 99})
    assert "SCHEDULER STARVATION" in verdict


# --------------------------------------------------------------------------- #
# it reaches the residency report, and the trace, and the scanner
# --------------------------------------------------------------------------- #
def _sampler(trace, gauges):
    import inspect
    import re

    # Every self._X_trace the real sampler creates, so a buffer added later is
    # present and empty here rather than an AttributeError in forty tests.
    buffers = re.findall(r"self\.(_\w+_trace)\s*=\s*\[\]", inspect.getsource(gp._Sampler.__init__))

    class F:
        residency_between = gp._Sampler.residency_between

        def __init__(self):
            self._lock = threading.Lock()
            for name in buffers:
                setattr(self, name, [])
            self._util_trace = trace
            self._cpu_trace = [(ts, 8.0) for ts, _ in trace]
            self._act_trace = [(ts, {"gen": 1}) for ts, _ in trace]
            self._stack_trace = [(ts, ["B -"]) for ts, _ in trace]
            self._gauge_trace = gauges

    return F()


def test_the_residency_report_names_the_reason():
    ts = [100.0 + i * 0.3 for i in range(4)]
    trace = list(zip(ts, [[0, 0, 0]] * 3 + [[90, 90, 90]]))
    out = gp.format_residency(
        _sampler(trace, [(t, {"ready": 32, "retriever_inflight": 10}) for t in ts]).residency_between(0, 1e9)
    )
    assert "while EMPTY the work outstanding was" in out
    assert "SCHEDULER STARVATION" in out


def test_the_gauges_are_averaged_over_the_empty_samples_only():
    ts = [100.0 + i * 0.3 for i in range(4)]
    trace = list(zip(ts, [[0, 0, 0], [0, 0, 0], [90, 90, 90], [90, 90, 90]]))
    gauges = [(ts[0], {"ready": 4}), (ts[1], {"ready": 0}),
              (ts[2], {"ready": 99}), (ts[3], {"ready": 99})]
    res = _sampler(trace, gauges).residency_between(0, 1e9)
    assert res["gauges_when_empty"]["ready"] == pytest.approx(2.0)   # (4+0)/2
    assert res["gauges_when_busy"]["ready"] == pytest.approx(99.0)


def test_the_trace_schema_carries_every_gauge():
    header = gp._TRACE_HEADER.strip().split(",")
    for name in gp.GAUGE_NAMES:
        assert name in header, header
    # Contiguous and in order, so a scanner may index positionally -- but NOT
    # pinned to the end: activity and stack_id were appended after them, and a
    # test that treats the last column as fixed fails on every later extension
    # while saying nothing about it.
    first = header.index(gp.GAUGE_NAMES[0])
    assert header[first:first + len(gp.GAUGE_NAMES)] == list(gp.GAUGE_NAMES)
    assert header[-2:] == ["activity", "stack_id"]


def _trace(path, blocks):
    """blocks: list of (sm, cpu, gauges, n_samples)."""
    cols = ",".join(gp.GAUGE_NAMES)
    lines = ["ts,clock,pid,phase,sm_pct_per_gpu,membw_pct_per_gpu,power_w_per_gpu,"
             "smclk_mhz_per_gpu,pcie_rx_mb_s_per_gpu,nvlink_mb_s_per_gpu,driver_cpu_pct," + cols]
    t = 0.0
    for sm, cpu, gauges, n in blocks:
        for _ in range(n):
            lines.append(
                f"{t:.3f},00:00:00,1,gen,{';'.join(str(v) for v in sm)},;;,;;,;;,;;,;;,{cpu},"
                + ",".join(str(gauges.get(name, 0)) for name in gp.GAUGE_NAMES)
            )
            t += 0.1
    path.write_text("\n".join(lines) + "\n")
    return path


def _scan(path):
    done = subprocess.run([sys.executable, str(SCANNER), str(path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_the_scanner_separates_the_same_two_cases_from_a_trace(tmp_path):
    busy = ([98, 97, 98], 20, {"gen_inflight": 200}, 30)
    path = _trace(tmp_path / "t.csv", [
        busy, ([0, 1, 0], 6, {"retriever_inflight": 10}, 12),
        busy, ([0, 1, 0], 6, {"retriever_inflight": 10, "ready": 32}, 12),
        busy, ([0, 1, 0], 6, {"retriever_inflight": 10}, 12),
        busy,
    ])
    out = _scan(path)
    assert "why the cards were idle, by lost GPU-seconds" in out
    assert "SCHEDULER_STARVATION" in out
    assert "RETRIEVER_DEPENDENCY" in out
    # Ranked by lost GPU-seconds: two retriever dips against one starvation dip.
    starve = out.index("SCHEDULER_STARVATION")
    retr = out.index("RETRIEVER_DEPENDENCY")
    assert retr < starve, out


def test_a_trace_written_before_the_gauges_existed_still_reads(tmp_path):
    """And every excursion in it is UNINSTRUMENTED, which is the truth.

    There are ten of these logs on the box already. A scanner that refused them,
    or that quietly called their gaps RETRIEVER_DEPENDENCY, would be worse than
    one that says it does not know.
    """
    path = tmp_path / "old.csv"
    lines = ["ts,clock,pid,phase,sm_pct_per_gpu,membw_pct_per_gpu,power_w_per_gpu,"
             "smclk_mhz_per_gpu,pcie_rx_mb_s_per_gpu,nvlink_mb_s_per_gpu,driver_cpu_pct"]
    t = 0.0
    for sm, cpu, n in (([98, 97, 98], 20, 30), ([0, 1, 0], 6, 12), ([98, 97, 98], 20, 30)):
        for _ in range(n):
            lines.append(f"{t:.3f},00:00:00,1,gen,{';'.join(str(v) for v in sm)},;;,;;,;;,;;,;;,{cpu}")
            t += 0.1
    path.write_text("\n".join(lines) + "\n")
    out = _scan(path)
    assert "UNINSTRUMENTED" in out
    assert "RETRIEVER_DEPENDENCY" not in out


# --------------------------------------------------------------------------- #
# and the call site that makes `ready` mean something
# --------------------------------------------------------------------------- #
def test_ready_rises_when_a_batch_is_prepared_and_every_slot_is_busy():
    """This is the gauge the whole distinction rests on.

    If it never rises, SCHEDULER_STARVATION is unreachable and the classifier
    will always answer RETRIEVER_DEPENDENCY -- confidently, and without any way
    to notice.
    """
    from verl.utils.val_pipeline import Slot, run_pipelined

    seen = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            seen.append(gp.gauge_snapshot())
            time.sleep(0.02)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    slots = [Slot(f"s{i}", envs=None, collector=None) for i in range(2)]
    list(run_pipelined(range(4), lambda x: x, lambda _p: None,
                       lambda _p, _s: time.sleep(0.15), slots))
    stop.set()
    watcher.join()

    assert any(s.get("ready", 0) >= 1 for s in seen), seen
    assert any(s.get("future_wait", 0) >= 1 for s in seen), seen
    assert gp.gauge_snapshot() == {}, "a gauge leaked past the end of the pipeline"


# --------------------------------------------------------------------------- #
# and which LINE, per excursion, not averaged over the run
# --------------------------------------------------------------------------- #
def test_stack_states_are_interned_and_ignore_threads_outside_the_repo():
    """The intern table has to have repeats or it is a list.

    A hundred-odd parked infrastructure threads drift in and out constantly, so
    keying on them would make almost every sample a distinct state. They are
    dropped from the key and counted instead.
    """
    a, _ = gp.stack_state_id(["R rollout_loop.py:1 f", "B -", "B -"])
    b, _ = gp.stack_state_id(["R rollout_loop.py:1 f", "B -", "B -", "B -", "B -"])
    c, _ = gp.stack_state_id(["R rollout_loop.py:9 g", "B -"])
    assert a == b, "the parked count must not split one state into two"
    assert a != c
    # ...but the count is kept, because "one of ours and 138 parked" and "one of
    # ours alone" are different situations.
    _, key_a = gp.stack_state_id(["R rollout_loop.py:1 f", "B -", "B -"])
    _, key_b = gp.stack_state_id(["R rollout_loop.py:1 f"])
    assert key_a[1] == 2 and key_b[1] == 0


class _StubBackend:
    """Three cards that are always busy. No GPU needed."""

    n_gpus = 3

    def sample(self):
        return [{"sm_util": 90.0} for _ in range(self.n_gpus)]


class _StubHost:
    def sample(self):
        return {"cpu_pct": 40.0}


def test_the_sampler_writes_a_sidecar_the_scanner_can_join(tmp_path, monkeypatch):
    """End to end: the ids in the trace resolve against the file beside it."""
    import importlib

    for key in ("GPU_PROFILER", "GPU_PROFILER_INTERVAL", "GPU_PROFILER_TRACE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GPU_PROFILER", "1")
    monkeypatch.setenv("GPU_PROFILER_INTERVAL", "0.02")
    monkeypatch.setenv("GPU_PROFILER_TRACE", str(tmp_path / "trace.csv"))
    # The ORIGINAL module object is put back in the finally, not a fresh import
    # of the same name. Every other test in this file holds `gp`, and callers
    # like val_pipeline._gauge resolve through sys.modules at call time -- so
    # leaving a different object there sends their gauges into a second _GAUGES
    # dict that `gp` cannot see. That is how the ready-gauge test started
    # failing only when run after this one.
    original = sys.modules.pop("verl.utils.gpu_profiler", None)
    mod = importlib.import_module("verl.utils.gpu_profiler")
    try:
        # Driven with a stub backend, so this runs on a box with no NVML -- the
        # same way test_gpu_profiler_trace does it.
        sampler = mod._Sampler(_StubBackend(), 0.02, host=_StubHost())
        with mod.activity("gen"), mod.inflight("ready", 4):
            time.sleep(0.2)
        sampler._stop.set()

        written = Path(mod._trace_path_for_pid(str(tmp_path / "trace.csv")))
        rows = [r.split(",") for r in written.read_text().strip().splitlines()]
        header, body = rows[0], rows[1:]
        assert body, written.read_text()
        col = {name: i for i, name in enumerate(header)}
        assert any(r[col["ready"]] == "4" for r in body)
        assert any("gen:1" in r[col["activity"]] for r in body)

        sidecar = Path(str(written) + ".stacks")
        assert sidecar.exists(), "no stacks file beside the trace"
        table = {int(l.split("\t")[0]) for l in sidecar.read_text().splitlines()[1:]}
        ids = {int(r[col["stack_id"]]) for r in body if r[col["stack_id"]] not in ("", "-1")}
        assert ids and ids <= table, (ids, table)
    finally:
        if original is not None:
            sys.modules["verl.utils.gpu_profiler"] = original
            # And the package attribute, which importlib also rebound; see the
            # note in test_gpu_profiler_trace.py.
            import verl.utils

            verl.utils.gpu_profiler = original
        else:  # pragma: no cover - only if it was never imported
            sys.modules.pop("verl.utils.gpu_profiler", None)


def test_a_trace_whose_sidecar_was_left_behind_still_scans(tmp_path):
    """Copying a CSV off a box without its sidecar must lose the frame line only."""
    busy = ([98, 97, 98], 20, {"gen_inflight": 200}, 30)
    path = _trace(tmp_path / "t.csv", [busy, ([0, 1, 0], 6, {"retriever_inflight": 10}, 12), busy])
    out = _scan(path)      # _trace writes no sidecar
    assert "RETRIEVER_DEPENDENCY" in out
    assert "costliest excursions" in out
