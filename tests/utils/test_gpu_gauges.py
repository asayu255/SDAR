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
EVAL_SH = Path(__file__).resolve().parents[2] / "examples" / "sft_trainer" / "eval_checkpoints.sh"


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
        # placeable_ready as well as slots_free: once slots became task-typed,
        # "a slot was free" stopped being enough to mean the dispatcher missed
        # something -- the free slot also has to be one a ready batch fits.
        busy, ([0, 1, 0], 6, {"retriever_inflight": 10, "ready": 32,
                              "slots_free": 1, "placeable_ready": 1}, 12),
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


# --------------------------------------------------------------------------- #
# NVML smooths, so the deepest sample is not the causing sample
# --------------------------------------------------------------------------- #
def _lagged_trace(path, *, stall_samples=3, window=3, lag=2):
    """A retrieval stall whose SM minimum lands AFTER the retrieval finished.

    utilization.gpu is a moving-window average. The true sequence is

        retriever busy, queue empty      SM falls
        retriever done, generate resumes SM keeps falling for a window, then rises

    so the sample where SM bottoms out has retriever_inflight=0 and
    gen_inflight>0 -- and a classifier reading that one sample calls a retrieval
    stall GPU_SIDE. Every synthetic trace in this file until now flipped the
    gauges and the SM on the same row, which is the one shape that cannot catch
    this.
    """
    names = ",".join(gp.GAUGE_NAMES)
    lines = ["ts,clock,pid,phase,sm_pct_per_gpu,membw_pct_per_gpu,power_w_per_gpu,"
             "smclk_mhz_per_gpu,pcie_rx_mb_s_per_gpu,nvlink_mb_s_per_gpu,driver_cpu_pct,"
             + names + ",activity,stack_id"]
    true_sm, gauges = [], []

    def add(sm_true, g):
        true_sm.append(sm_true)
        gauges.append(g)

    for _ in range(40):
        add(100, {"gen_inflight": 200})
    for _ in range(stall_samples):          # queue empty, retriever running
        add(0, {"retriever_inflight": 8})
    for _ in range(40):                     # recovered
        add(100, {"gen_inflight": 200})

    t = 0.0
    for i, g in enumerate(gauges):
        # TWO effects, and only the second one moves the minimum past the event.
        # A trailing mean alone bottoms out on the LAST stall sample, where the
        # gauges still say retriever -- so a mean-only fixture cannot express
        # this bug, which is how the first attempt at this test passed against
        # the broken scanner. NVML also recomputes on its own cadence, so a poll
        # returns a value already up to one update period old; that staleness is
        # what carries the minimum into the recovery, where the gauges have
        # flipped to gen_inflight.
        end = i - lag
        lo = max(0, end - window + 1)
        smoothed = sum(true_sm[lo:end + 1]) / (end - lo + 1) if end >= 0 else 100
        cpu = 6 if g.get("retriever_inflight") else 20
        lines.append(
            f"{t:.3f},00:00:00,1,gen,{';'.join([f'{smoothed:.0f}'] * 3)},;;,;;,;;,;;,;;,{cpu},"
            + ",".join(str(g.get(n, 0)) for n in gp.GAUGE_NAMES) + ",gen:1,0"
        )
        t += 0.1
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_short_retrieval_stall_is_not_called_gpu_side(tmp_path):
    """The regression the previous synthetic traces could not express.

    At the SM minimum the retriever has finished and the engine is busy again,
    so a single-sample classifier reads gen_inflight>0 and answers GPU_SIDE --
    pointing the next fix at the engine for a stall the engine did not cause.
    """
    out = _scan(_lagged_trace(tmp_path / "lag.csv"))
    assert "RETRIEVER_DEPENDENCY" in out, out
    assert "GPU_SIDE" not in out.split("why the cards were idle")[1], out


def test_the_deepest_sample_really_does_say_gpu_side(tmp_path):
    """Guard on the fixture: if it stops lagging, the test above proves nothing.

    A regression test for smoothing has to actually contain smoothing, and the
    only way to be sure is to check that the naive reading still gets it wrong.
    """
    import csv as _csv

    path = _lagged_trace(tmp_path / "lag.csv")
    with open(path) as f:
        rows = list(_csv.DictReader(f))
    deepest = min(rows, key=lambda r: float(r["sm_pct_per_gpu"].split(";")[0]))
    assert int(deepest["retriever_inflight"]) == 0, "the fixture is not lagged"
    assert int(deepest["gen_inflight"]) > 0, "the fixture is not lagged"


def test_a_retrieval_alongside_a_busy_engine_is_not_the_reason(tmp_path):
    """gen_inflight is required to be absent for every dependency answer.

    Twenty retrievals running while a hundred requests are with the engine is
    not why a card went idle, however busy the retriever looks.
    """
    from importlib.machinery import SourceFileLoader

    scanner = SourceFileLoader(
        "stall_scan", str(Path(__file__).resolve().parents[2] / "scripts" / "gpu_stall_scan.py")
    ).load_module()
    assert scanner.why_one({"retriever_inflight": 20, "gen_inflight": 100}) == "GPU_SIDE"
    assert scanner.why_one({"retriever_inflight": 20, "gen_inflight": 0}) == "RETRIEVER_DEPENDENCY"
    assert scanner.why_one({}) is None


def test_the_scanner_finds_the_trace_the_profiler_actually_wrote(tmp_path):
    """GPU_PROFILER_TRACE=X.csv writes X.<pid>.csv, one file per process.

    So the name in the run command is never the name on disk, and the scanner
    answered "no such trace file" about a trace sitting right beside the path it
    was given -- after an hour-long run, which is the worst moment to discover a
    naming convention.
    """
    busy = ([98, 97, 98], 20, {"gen_inflight": 200}, 30)
    real = _trace(tmp_path / "t504.31415.csv",
                  [busy, ([0, 1, 0], 6, {"retriever_inflight": 10}, 12), busy])
    assert real.exists()

    out = _scan(tmp_path / "t504.csv")          # the name a person would type
    assert "t504.31415.csv" in out
    assert "RETRIEVER_DEPENDENCY" in out


def test_a_genuinely_missing_trace_says_where_to_look(tmp_path):
    done = subprocess.run(
        [sys.executable, str(SCANNER), str(tmp_path / "nothing.csv")],
        capture_output=True, text=True,
    )
    assert done.returncode != 0
    assert "X.<pid>.csv" in done.stderr
    assert "ls " in done.stderr


def test_one_request_in_flight_is_not_the_engine_having_work(tmp_path):
    """The defect the first real run exposed.

    Excursions read `ready=1 gen_inflight=1` against a batch of 504, and the
    classifier answered GPU_SIDE for fifteen samples out of sixteen -- "requests
    were with the engine". One request on three A6000s is a turn's last row
    draining, not the engine having work, and the difference decides whether the
    next fix goes to the engine or to the scheduler.
    """
    busy = ([98, 97, 98], 20, {"gen_inflight": 500}, 40)
    tail = ([0, 1, 0], 15, {"gen_inflight": 2, "ready": 1, "future_wait": 1}, 15)
    out = _scan(_trace(tmp_path / "tail.csv", [busy, tail, busy, tail, busy]))
    # A TAIL_ reason, whichever slot state this fixture happens to describe --
    # what this test is about is that it is not GPU_SIDE.
    assert "TAIL_NO_FREE_SLOT_PINNED" in out
    assert "GPU_SIDE" not in out.split("why the cards were idle")[1].split("per phase")[0]
    # And it says what it calibrated to, so the reader can disagree with it.
    assert "calibrated from this run" in out


def test_the_threshold_comes_from_the_run_not_from_a_constant():
    """The same scanner reads 126-row batches and 504-row batches."""
    from importlib.machinery import SourceFileLoader

    scanner = SourceFileLoader(
        "stall_scan2", str(Path(__file__).resolve().parents[2] / "scripts" / "gpu_stall_scan.py")
    ).load_module()
    busy = [98, 97, 98]
    wide = [{"sm": busy, "gauges": {"gen_inflight": 500}} for _ in range(9)]
    narrow = [{"sm": busy, "gauges": {"gen_inflight": 40}} for _ in range(9)]
    assert scanner.gen_full_threshold(wide) == 125.0
    assert scanner.gen_full_threshold(narrow) == 10.0
    # A trace with no gen gauge at all falls back rather than dividing by zero.
    assert scanner.gen_full_threshold([{"sm": busy, "gauges": {}}]) == 4


def test_a_tail_with_nothing_waiting_behind_it_is_named_differently(tmp_path):
    """TAIL_DRAIN costs nothing actionable; a tail with a queue behind it does."""
    from importlib.machinery import SourceFileLoader

    scanner = SourceFileLoader(
        "stall_scan3", str(Path(__file__).resolve().parents[2] / "scripts" / "gpu_stall_scan.py")
    ).load_module()
    assert scanner.why_one({"gen_inflight": 2, "ready": 1}, gen_full=100) == "TAIL_BLOCKS_READY_UNKNOWN"
    assert scanner.why_one({"gen_inflight": 2}, gen_full=100) == "TAIL_DRAIN"
    assert scanner.why_one({"gen_inflight": 400}, gen_full=100) == "GPU_SIDE"


def _scanner():
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader(
        "stall_scan_cal", str(Path(__file__).resolve().parents[2] / "scripts" / "gpu_stall_scan.py")
    ).load_module()


def test_the_tails_do_not_calibrate_the_threshold_that_finds_them():
    """A bias that hides the finding, and grows with it.

    gen_full was the median over every sample with gen_inflight > 0. A run full
    of turn tails has thousands of samples reading gen=1..4; those drag the
    median down, the threshold falls, and the tails stop classifying as tails.
    The more tail a run has, the less of it this finds.

    The existing tests could not see it: they fed nine identical samples, so
    tail and busy were never mixed.
    """
    scanner = _scanner()
    busy = [{"sm": [98, 97, 98], "gauges": {"gen_inflight": 500}} for _ in range(20)]
    tails = [{"sm": [0, 1, 0], "gauges": {"gen_inflight": n}} for n in (1, 2, 3, 4) for _ in range(30)]

    # Over ALL non-zero samples the median is a tail value; over the busy ones
    # it is a working load.
    assert scanner.gen_full_threshold(busy + tails) == 125.0
    naive = sorted(r["gauges"]["gen_inflight"] for r in busy + tails)
    assert naive[len(naive) // 2] <= 4, "the fixture is not mixed enough to show the bias"


def test_the_primary_reason_is_what_the_excursion_was_mostly_made_of():
    """Fixed rank let one sample outvote fifteen.

    "gpu:15 scheduler:1" came back reading SCHEDULER_STARVATION -- right by
    luck on that run, and wrong the next time. Dwell decides; the causal
    lead-in is reported separately rather than as a verdict that overwrites
    what the excursion consisted of.
    """
    scanner = _scanner()
    rows = []
    t = 0.0
    # One sample of a fully-starved queue, then fourteen of a draining tail.
    rows.append({"ts": t, "sm": [0, 0, 0], "cpu": 10, "gauges": {"ready": 1, "slots_free": 1}})
    for _ in range(14):
        t += 0.1
        rows.append({"ts": t, "sm": [0, 0, 0], "cpu": 10, "gauges": {"gen_inflight": 2, "ready": 1}})
    primary, lead_in, dwell = scanner.why(rows, 1, len(rows) - 1, pre_roll=0.4, gen_full=100)
    assert primary == "TAIL_BLOCKS_READY_UNKNOWN", (primary, dwell)
    assert lead_in == "SLOT_FREE_UNKNOWN", (lead_in, dwell)
    assert dwell["TAIL_BLOCKS_READY_UNKNOWN"] == 14 and dwell["SLOT_FREE_UNKNOWN"] == 1


def test_the_lead_in_is_dropped_when_it_is_the_primary():
    scanner = _scanner()
    rows = [{"ts": i * 0.1, "sm": [0, 0, 0], "cpu": 10, "gauges": {"ready": 1, "slots_free": 1}}
            for i in range(10)]
    primary, lead_in, _ = scanner.why(rows, 1, 9, pre_roll=0.4, gen_full=100)
    assert primary == "SLOT_FREE_UNKNOWN" and lead_in is None


def test_the_kv_guard_counts_depth_not_just_width():
    """504x3 and 378x4 are the same envelope; 504x4 is a third more.

    The 468 tokens per row of width is an observed peak measured at depth 3, so
    scaling by width alone passes 504x4 while it asks for 33% more than the
    number the check was calibrated on.
    """
    import re
    text = EVAL_SH.read_text()
    assert "VAL_PIPELINE_DEPTH" in text, "the guard still ignores depth"
    line = next((l for l in text.splitlines() if "_NEEDED=" in l), "")
    assert "_DEPTH" in line and " d " in line, line

    def needed(w, d):
        return w * 468 * d / 3

    budget = (0.85 * 48 - 10.9) * 8920
    assert needed(504, 3) == needed(378, 4)
    assert needed(504, 3) <= 0.92 * budget
    assert needed(504, 4) > 0.92 * budget


def test_starvation_needs_a_free_slot_not_merely_a_queue():
    """`ready` changed meaning underneath the classifier.

    It used to be "one batch that cannot be placed"; once the pipeline grew a
    lookahead queue it means "the queue is a queue", true nearly all the time.
    Reading starvation off it alone took SCHEDULER_STARVATION from 49.9 to
    109.0 GPU-s in the same commit that added the queue -- a change in the
    instrument, reported as a change in the run.
    """
    scanner = _scanner()
    queued_busy = {"ready": 8, "slots_free": 0}
    queued_free = {"ready": 8, "slots_free": 2}
    # A free slot, but no placeable_ready column to say whether the queue fit
    # it: known free, unknown usable.
    assert scanner.why_one(queued_free, gen_full=100) == "SLOT_FREE_UNKNOWN"
    assert scanner.why_one({**queued_free, "placeable_ready": 3},
                           gen_full=100) == "SCHEDULER_STARVATION"
    assert scanner.why_one(queued_busy, gen_full=100) == "SLOTS_BUSY_NOT_GENERATING"
    # A dependency still outranks the catch-all when it is present.
    assert scanner.why_one({"ready": 8, "slots_free": 0, "env_inflight": 378},
                           gen_full=100) == "ENV_DEPENDENCY"
    # And a trace written before slots_free existed says so, rather than either
    # inventing starvation from a missing gauge or -- as it did until a real
    # trace was read this way -- reporting "every slot was occupied", which is
    # a claim about four slots from a column that is not in the file.
    assert scanner.why_one({"ready": 1}, gen_full=100) == "READY_BUT_IDLE_UNKNOWN"


def test_slots_free_is_published_by_the_pipeline():
    """The gauge has to actually move, or the branch above is unreachable."""
    import threading

    from verl.utils.val_pipeline import Slot, run_pipelined

    gp.reset_gauges()
    seen = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            seen.append(gp.gauge_snapshot())
            time.sleep(0.01)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    slots = [Slot(f"s{i}", envs=None, collector=None) for i in range(3)]
    list(run_pipelined(range(8), lambda x: x, lambda _p: None,
                       lambda _p, _s: time.sleep(0.05), slots))
    stop.set()
    watcher.join()
    assert any("slots_free" in s for s in seen), seen
    assert any(s.get("slots_free", 0) == 0 for s in seen), "never fully occupied"
    assert gp.gauge_snapshot().get("slots_free", 0) in (0, 3)


def test_a_gauge_column_the_scanner_has_never_heard_of_is_still_read():
    """The gauge list was duplicated and the copy went stale.

    scripts/gpu_stall_scan.py kept its own tuple of gauge names, imported from
    verl when importable and hard-coded when not. `slots_free` was added to the
    profiler, written into every row of the trace -- and dropped by the scanner,
    which classified every excursion as though the gauge did not exist. The
    columns are the file's own header now; anything not fixed and not a per-GPU
    series is a gauge.
    """
    import csv as _csv
    import tempfile

    scanner = _scanner()
    tmp = Path(tempfile.mkdtemp()) / "t.csv"
    tmp.write_text(
        "ts,clock,pid,phase,sm_pct_per_gpu,membw_pct_per_gpu,power_w_per_gpu,"
        "smclk_mhz_per_gpu,pcie_rx_mb_s_per_gpu,nvlink_mb_s_per_gpu,driver_cpu_pct,"
        "ready,gen_inflight,slots_free,a_gauge_invented_tomorrow,activity,stack_id\n"
        "0.000,00:00:00,1,gen,0;1;0,;;,;;,;;,;;,;;,6,32,0,1,7,gen:1,0\n"
    )
    rows, _skipped = scanner.read_trace(str(tmp))
    # Zeros included: a declared column that read zero is a reading, and telling
    # it apart from a column the trace never had is what lets the classifier
    # know which questions this trace can answer.
    assert rows[0]["gauges"] == {"ready": 32, "gen_inflight": 0, "slots_free": 1,
                                 "a_gauge_invented_tomorrow": 7}
    # and the per-GPU series are not mistaken for gauges
    assert not any(k.endswith("_per_gpu") for k in rows[0]["gauges"])
    assert _csv  # (the import is what read_trace uses)


def test_a_free_slot_that_fits_no_queued_batch_is_not_a_dispatcher_miss():
    """Slots are task-typed; "free" and "usable" are different questions.

    A free alfworld slot cannot take a queued webshop batch. Both states read
    ready>0, slots_free>0, gen_inflight=0 -- and they want opposite fixes:
    dispatch sooner, versus change what the queue is holding. Reported as one
    reason, the second sends the next change at a dispatcher doing its job.
    """
    scanner = _scanner()
    fits = {"ready": 4, "slots_free": 1, "placeable_ready": 2, "gen_inflight": 0}
    fits_not = {"ready": 4, "slots_free": 1, "placeable_ready": 0, "gen_inflight": 0}
    assert scanner.why_one(fits, cpu=10, gen_full=100) == "SCHEDULER_STARVATION"
    assert scanner.why_one(fits_not, cpu=10, gen_full=100) == "SLOT_COMPATIBILITY_BLOCK"


def test_a_trace_from_before_the_placeable_gauge_says_so():
    """The absent column must not turn every starved sample into a block.

    A missing gauge reads as zero everywhere, so a classifier that treats zero
    as "nothing fits" relabels every SCHEDULER_STARVATION in every older trace
    -- inventing a finding out of an instrument that was never installed. Nor
    may it keep the confident older name: the sample is genuinely undecided
    between the two, and the table has to show that.
    """
    scanner = _scanner()
    old = {"ready": 4, "slots_free": 1, "gen_inflight": 0}
    assert scanner.why_one(old, cpu=10, gen_full=100) == "SLOT_FREE_UNKNOWN"


def test_a_placeable_column_reading_zero_is_not_an_absent_column(tmp_path):
    """End to end, through the CSV, which is where the two could be confused.

    The reader used to drop zero-valued gauges to keep one printed line short.
    That collapsed "the gauge said nothing fits" into "there is no such gauge",
    which is exactly the pair this classifier has to separate -- and it would
    have done so silently, answering SCHEDULER_STARVATION for every block.
    """
    scanner = _scanner()
    tmp = tmp_path / "t.csv"
    tmp.write_text(
        "ts,clock,pid,phase,sm_pct_per_gpu,membw_pct_per_gpu,power_w_per_gpu,"
        "smclk_mhz_per_gpu,pcie_rx_mb_s_per_gpu,nvlink_mb_s_per_gpu,driver_cpu_pct,"
        "ready,gen_inflight,slots_free,placeable_ready,activity,stack_id\n"
        "0.000,00:00:00,1,gen,0;0;0,;;,;;,;;,;;,;;,6,4,0,1,0,gen:0,0\n"
    )
    rows, _skipped = scanner.read_trace(str(tmp))
    assert "placeable_ready" in rows[0]["gauges"]
    assert scanner.why_one(rows[0]["gauges"], cpu=10, gen_full=100) == "SLOT_COMPATIBILITY_BLOCK"


def test_the_reason_table_does_not_run_its_columns_together(tmp_path):
    """The column width was a literal, sized to the longest reason of the day.

    Adding SLOT_COMPATIBILITY_BLOCK, at 24 characters, printed it flush against
    the events count -- a table that reads as one token. The width comes from
    the names now, so the next reason added cannot do it again.
    """
    busy = ([98, 97, 98], 20, {"gen_inflight": 200}, 30)
    path = _trace(tmp_path / "t.csv", [
        busy,
        ([0, 1, 0], 6, {"ready": 4, "slots_free": 1, "placeable_ready": 0}, 14),
        busy,
    ])
    out = _scan(path)
    line = next(ln for ln in out.splitlines() if "SLOT_COMPATIBILITY_BLOCK" in ln
                and ln.strip().startswith("SLOT_COMPATIBILITY_BLOCK"))
    rest = line.strip()[len("SLOT_COMPATIBILITY_BLOCK"):]
    assert rest.startswith(" "), line
    # and the dwell abbreviation separates it from the other SLOT reason
    assert scanner_abbr("SLOT_COMPATIBILITY_BLOCK") != scanner_abbr("SLOTS_BUSY_NOT_GENERATING")


def scanner_abbr(reason):
    return _scanner()._abbr(reason)


def test_the_report_names_the_gauges_the_trace_carries(tmp_path):
    """Without it the table is not readable, and the failure is silent.

    Several reasons fall back to a coarser name when their gauge is missing, so
    "SCHEDULER_STARVATION 0" means either "no slot was ever free while work
    waited" or "this trace has no slots_free column" -- indistinguishable in the
    table, and one of them is a finding while the other is an absent instrument.
    """
    busy = ([98, 97, 98], 20, {"gen_inflight": 200}, 20)
    out = _scan(_trace(tmp_path / "t.csv", [busy, ([0, 1, 0], 6, {"ready": 4}, 10), busy]))
    line = next(ln for ln in out.splitlines() if ln.startswith("gauges in this trace:"))
    assert "slots_free" in line and "placeable_ready" in line

    # ...and a trace from before them says so rather than listing nothing.
    old = tmp_path / "old.csv"
    old.write_text(
        "ts,clock,pid,phase,sm_pct_per_gpu,membw_pct_per_gpu,power_w_per_gpu,"
        "smclk_mhz_per_gpu,pcie_rx_mb_s_per_gpu,nvlink_mb_s_per_gpu,driver_cpu_pct\n"
        + "".join(f"{i * 0.1:.3f},00:00:00,1,gen,98;97;98,;;,;;,;;,;;,;;,20\n" for i in range(20))
    )
    assert "NONE" in next(ln for ln in _scan(old).splitlines()
                          if ln.startswith("gauges in this trace:"))


def test_the_report_survives_being_piped_into_head(tmp_path):
    """It is long, and reading the top of it is the obvious thing to do.

    Unhandled, the BrokenPipeError prints a traceback over the output it just
    produced -- and this arm has already lost a day to a SIGPIPE that turned a
    present line into an absent one.
    """
    busy = ([98, 97, 98], 20, {"gen_inflight": 200}, 20)
    path = _trace(tmp_path / "t.csv", [busy, ([0, 1, 0], 6, {"ready": 4}, 10), busy])
    done = subprocess.run(f"{sys.executable} {SCANNER} {path} 2>&1 | head -4",
                          shell=True, capture_output=True, text=True)
    assert "BrokenPipeError" not in done.stdout, done.stdout
    assert "Traceback" not in done.stdout, done.stdout


def test_a_draining_tail_says_whether_a_slot_was_actually_open():
    """The branch never read a slot; its report line claimed one was waited for.

    "the engine was draining a turn's last few rows while a whole batch waited
    FOR A SLOT" was printed for 60% of one run's idle time by a rule that only
    tested gen_inflight and ready. Every slot busy on its own tail, and a free
    slot the queued batch cannot run, are opposite findings -- the first wants
    more slots, the second cannot be helped by any number of them.
    """
    scanner = _scanner()
    tail = {"gen_inflight": 3, "ready": 2}
    why = lambda g: scanner.why_one({**tail, **g}, gen_full=100)

    # No free slot. What to do then depends on WHAT was queued, which is a
    # second gauge: work only one slot in the topology can run is not helped by
    # adding copies of the slots that already cannot run it.
    assert why({"slots_free": 0, "ready_scalable": 1, "ready_pinned": 0}) == "TAIL_NO_FREE_SLOT_SCALABLE"
    assert why({"slots_free": 0, "ready_scalable": 0, "ready_pinned": 2}) == "TAIL_NO_FREE_SLOT_PINNED"

    # A slot was free. Either the queue could not use it, or it could and the
    # driver had not got back to dispatch yet.
    assert why({"slots_free": 1, "placeable_ready": 0}) == "TAIL_SLOT_FREE_UNUSABLE"
    assert why({"slots_free": 1, "placeable_ready": 2}) == "TAIL_SLOT_FREE_PLACEABLE"

    # No queue behind the tail: nothing was blocked, whatever the slots say.
    assert scanner.why_one({"gen_inflight": 3, "ready": 0, "slots_free": 0}, gen_full=100) == "TAIL_DRAIN"


def test_no_free_slot_does_not_claim_the_busy_slots_were_generating():
    """`slots_free == 0` says there was no free slot. Nothing more.

    The occupied slots may be in env.step, in prepare(), or waiting on a
    retrieval. An earlier name for this leaf, TAIL_ALL_SLOTS_BUSY, read as
    "every slot was draining a tail" -- a claim about four slots derived from a
    gauge that counts one number.
    """
    scanner = _scanner()
    for name in scanner._REASON_RANK:
        assert "ALL_SLOTS_BUSY" not in name, name
    text = SCANNER.read_text()
    start = text.index("meaning = {")
    end = text.index("}", text.index("UNINSTRUMENTED", start))
    for leaf in ("TAIL_NO_FREE_SLOT_SCALABLE", "TAIL_NO_FREE_SLOT_PINNED", "TAIL_NO_FREE_SLOT_UNKNOWN"):
        line = text[text.index(f'"{leaf}":', start):]
        line = line[:line.index('",\n') + 1] if '",\n' in line else line[:300]
        assert "no slot was free" in line or "no slot was free" in line.lower(), (leaf, line[:200])


def test_a_tail_in_a_trace_without_slots_free_stays_undecided():
    """The coarse name has to survive, for the same reason as before: a missing
    column reads as zero, and zero here would mean "every slot was busy" -- a
    definite finding manufactured out of an instrument that was not installed."""
    scanner = _scanner()
    assert scanner.why_one({"gen_inflight": 3, "ready": 2}, gen_full=100) == "TAIL_BLOCKS_READY_UNKNOWN"
    # And one gauge short of the bottom says so too, rather than borrowing the
    # neighbouring leaf's answer.
    assert scanner.why_one({"gen_inflight": 3, "ready": 2, "slots_free": 1},
                           gen_full=100) == "TAIL_SLOT_FREE_UNKNOWN"
    assert scanner.why_one({"gen_inflight": 3, "ready": 2, "slots_free": 0},
                           gen_full=100) == "TAIL_NO_FREE_SLOT_UNKNOWN"


def test_no_leaf_claims_a_slot_fact_from_a_trace_with_no_slot_gauge():
    """The rule the tail branch got, applied to the branch beside it.

    Without `slots_free` the engine-empty path fell through to
    SLOTS_BUSY_NOT_GENERATING -- "every slot was occupied and none was feeding
    the engine" -- a claim about four slots read off a column that was not in
    the file. The first real trace read this way had 22.2 GPU-s reported that
    way by a run whose profiler never wrote the gauge. Three findings
    (starvation, shape mismatch, genuinely all busy) were in that one bucket.
    """
    scanner = _scanner()
    empty = {"gen_inflight": 0, "ready": 4}
    assert scanner.why_one(empty, cpu=10, gen_full=100) == "READY_BUT_IDLE_UNKNOWN"
    # With the gauge, free == 0 IS established, and the name may say so.
    assert scanner.why_one({**empty, "slots_free": 0}, cpu=10,
                           gen_full=100) == "SLOTS_BUSY_NOT_GENERATING"
    # A slot free but no placeable_ready column: free is known, fit is not.
    assert scanner.why_one({**empty, "slots_free": 2}, cpu=10,
                           gen_full=100) == "SLOT_FREE_UNKNOWN"
    # A dependency reads its own gauge -- but with work QUEUED it still cannot
    # be called the cause without knowing whether a slot was free. This line
    # asserted RETRIEVER_DEPENDENCY here, which is the error itself written
    # down as a test. See
    # test_a_dependency_is_only_a_cause_when_nothing_else_could_have_run.
    assert scanner.why_one({**empty, "retriever_inflight": 3}, cpu=10,
                           gen_full=100) == "READY_BUT_IDLE_UNKNOWN"


def test_a_trace_matched_by_name_is_dated_in_the_report(tmp_path):
    """The pid fallback matches on NAME, so a run whose trace never appeared
    silently gets the previous run's file under the same base name.

    That happened: a comparison table was built from a trace last written
    twelve minutes BEFORE the run it was supposed to describe had started, and
    nothing in the report said so. The file's write time is the one fact that
    exposes it.
    """
    import os
    import time as _time

    busy = ([98, 97, 98], 20, {"gen_inflight": 200}, 20)
    real = _trace(tmp_path / "t.4242.csv", [busy, ([0, 1, 0], 6, {"ready": 4}, 10), busy])
    old_mtime = _time.time() - 3 * 3600
    os.utime(real, (old_mtime, old_mtime))

    # asked for the name WITHOUT the pid, which is what the profiler never writes
    out = _scan(tmp_path / "t.csv")
    assert "CHECK THE TIMES" in out, out
    assert "3.0 h ago" in out, out
    # ...and on the analysed file itself, not only in the fallback notice
    body = out[out.index("=== t.4242.csv ==="):]
    assert "last written" in body.splitlines()[1], body[:200]


def test_the_report_totals_what_it_could_not_attribute(tmp_path):
    """Summing it off the table by eye is how a structural zero got read as a
    measurement for two turns. On the first real trace read this way, two
    thirds of the loss was in leaves no gauge could resolve."""
    names = ("ready", "gen_inflight", "retriever_inflight", "env_inflight", "future_wait")
    head = ("ts,clock,pid,phase,sm_pct_per_gpu,membw_pct_per_gpu,power_w_per_gpu,"
            "smclk_mhz_per_gpu,pcie_rx_mb_s_per_gpu,nvlink_mb_s_per_gpu,driver_cpu_pct,"
            + ",".join(names))
    lines, t = [head], 0.0
    for sm, gauges, n in ((([98, 97, 98]), {"gen_inflight": 200}, 30),
                          (([0, 1, 0]), {"gen_inflight": 3, "ready": 2}, 20),
                          (([98, 97, 98]), {"gen_inflight": 200}, 30)):
        for _ in range(n):
            lines.append(f"{t:.3f},00:00:00,1,gen,{';'.join(str(v) for v in sm)},;;,;;,;;,;;,;;,20,"
                         + ",".join(str(gauges.get(k, 0)) for k in names))
            t += 0.1
    path = tmp_path / "old.csv"
    path.write_text("\n".join(lines) + "\n")
    out = _scan(path)
    line = next(ln for ln in out.splitlines() if "UNATTRIBUTED" in ln)
    assert "100.0%" in line, line
    assert "slots_free" in out


def test_a_dependency_is_only_a_cause_when_nothing_else_could_have_run():
    """"The retriever was running" is an observation, not an explanation.

    With work queued and no slot gauge, a free slot may have been sitting there
    with a placeable batch while some other slot happened to be in a retrieval.
    The dependency branch ran before the slot check and reported 64.2 GPU-s of
    one real trace as RETRIEVER_DEPENDENCY on exactly that reasoning -- the
    same "name a cause the instrument did not observe" this classifier has now
    been wrong about four times.

    With ready == 0 the dependency stands on its own: nothing was submittable,
    so no slot state could have changed the outcome.
    """
    scanner = _scanner()
    why = lambda g: scanner.why_one({"gen_inflight": 0, **g}, cpu=10, gen_full=100)

    assert why({"ready": 4, "retriever_inflight": 3}) == "READY_BUT_IDLE_UNKNOWN"
    assert why({"ready": 4, "env_inflight": 3}) == "READY_BUT_IDLE_UNKNOWN"
    # Nothing queued: the dependency IS the answer, gauge or no gauge.
    assert why({"ready": 0, "retriever_inflight": 3}) == "RETRIEVER_DEPENDENCY"
    assert why({"ready": 0, "env_inflight": 3}) == "ENV_DEPENDENCY"
    # With the gauge, a full slot set makes the dependency an explanation again.
    assert why({"ready": 4, "slots_free": 0, "retriever_inflight": 3}) == "RETRIEVER_DEPENDENCY"


def test_every_unknown_leaf_is_named_unknown():
    """A reader must be able to total the unattributed share off the names.

    Two thirds of the first trace's loss landed in leaves that could not be
    resolved for want of a gauge. That is a fact about the instrument, and it
    has to be legible in the table rather than hidden inside confident names.
    """
    scanner = _scanner()
    bare = {"gen_inflight": 0, "ready": 4}
    tail = {"gen_inflight": 3, "ready": 4}
    for gauges in (bare, tail, {**tail, "slots_free": 1}, {**tail, "slots_free": 0},
                   {**bare, "slots_free": 1}):
        reason = scanner.why_one(gauges, cpu=10, gen_full=100)
        resolved = all(k in gauges for k in ("slots_free", "placeable_ready", "ready_scalable"))
        assert resolved or reason.endswith("UNKNOWN"), (gauges, reason)


def test_the_new_reason_did_not_reshuffle_the_existing_ranks():
    """Inserting a name into the rank tuple must not reorder what was there.

    The rank breaks dwell ties, so moving two existing reasons past each other
    would change verdicts on traces that have nothing to do with this change.
    """
    scanner = _scanner()
    before = ("SCHEDULER_STARVATION", "SLOTS_BUSY_NOT_GENERATING",
              "RETRIEVER_DEPENDENCY", "ENV_DEPENDENCY", "FUTURE_RAY_WAIT",
              "DRIVER_CPU", "TAIL_DRAIN", "GPU_SIDE")
    kept = [r for r in scanner._REASON_RANK if r in before]
    assert tuple(kept) == before
    assert "SLOT_COMPATIBILITY_BLOCK" in scanner._REASON_RANK
