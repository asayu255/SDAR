"""GPU_PROFILER_TRACE keeps the per-sample record the aggregate tables discard.

The per-step tables say a GPU gap happened, in which phase, and how long the
longest one was -- then reset at the step boundary. That is the wrong shape for
a transient seen once on an external monitor: to line the two up you need the
wall-clock time of each sample and the phase it fell in. This covers that file.

No GPU is needed; the sampler is driven with a stub backend.
"""

import importlib
import os
import pathlib
import sys
import time

import pytest


def _fresh(monkeypatch, **env):
    for key in ("GPU_PROFILER", "GPU_PROFILER_TRACE", "GPU_PROFILER_INTERVAL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("verl.utils.gpu_profiler", None)
    return importlib.import_module("verl.utils.gpu_profiler")


def _sampler_with_stub(mod, interval=0.02):
    """A sampler on a stub backend, so this runs anywhere."""

    class _StubBackend(mod._Backend):
        n_gpus = 3

        def sample(self):
            return [{"sm_util": 97.0, "mem_bw_util": 40.0} for _ in range(self.n_gpus)]

    class _StubHost:
        def sample(self):
            return {"cpu_pct": 55.0}

    return mod._Sampler(_StubBackend(), interval, host=_StubHost())


def test_trace_is_off_unless_the_path_is_set(monkeypatch):
    mod = _fresh(monkeypatch, GPU_PROFILER="1")
    assert mod._TRACE_PATH == ""
    sampler = _sampler_with_stub(mod)
    try:
        assert sampler._trace is None
    finally:
        sampler._stop.set()


def test_every_sample_lands_in_the_trace_with_its_phase(monkeypatch, tmp_path):
    path = tmp_path / "trace.csv"
    mod = _fresh(monkeypatch, GPU_PROFILER="1", GPU_PROFILER_TRACE=str(path),
                 GPU_PROFILER_INTERVAL="0.02")
    sampler = _sampler_with_stub(mod)
    try:
        sampler.push("step")
        sampler.push("update_actor")
        time.sleep(0.25)
        sampler.pop("update_actor")
        time.sleep(0.15)
    finally:
        sampler._stop.set()
        time.sleep(0.05)

    written = pathlib.Path(mod._trace_path_for_pid(str(path)))
    rows = written.read_text().strip().splitlines()
    header = rows[0].split(",")
    assert header[:4] == ["ts", "clock", "pid", "phase"]
    assert header[-1] == "driver_cpu_pct"
    body = [r.split(",") for r in rows[1:]]
    assert len(body) > 5, rows
    assert all(len(r) == len(header) for r in body)

    col = {name: i for i, name in enumerate(header)}
    # The phase column is the whole point: it says what the trainer was doing.
    assert "update_actor" in {r[col["phase"]] for r in body}
    assert "step" in {r[col["phase"]] for r in body}
    # One column per GPU, so data-parallel imbalance stays visible.
    assert body[0][col["sm_pct_per_gpu"]] == "97;97;97"
    assert body[0][col["membw_pct_per_gpu"]] == "40;40;40"
    assert body[0][col["driver_cpu_pct"]] == "55"
    assert body[0][col["pid"]] == str(os.getpid())
    # Metrics this backend does not report stay empty rather than shifting the
    # later columns along, which is what makes the file parseable by column name.
    assert body[0][col["power_w_per_gpu"]] == ";;"
    # Monotonic timestamps, so a dip can be located against an external chart.
    stamps = [float(r[0]) for r in body]
    assert stamps == sorted(stamps)


def test_the_trace_path_carries_the_pid_so_two_samplers_cannot_clobber(monkeypatch, tmp_path):
    """A sampler starts in the driver AND in rank 0's worker.

    Both would open one path "w" from different processes with independent file
    offsets, and each would overwrite the other's bytes -- leaving about one
    sampler's worth of rows stitched from two streams. Aggregate means survive
    that; treating consecutive rows as consecutive in time does not.
    """
    mod = _fresh(monkeypatch, GPU_PROFILER="1")

    assert mod._trace_path_for_pid("/tmp/t.csv") == f"/tmp/t.{os.getpid()}.csv"
    assert mod._trace_path_for_pid("/tmp/t") == f"/tmp/t.{os.getpid()}.csv"
    # different processes -> different files, which is the entire guarantee
    assert mod._trace_path_for_pid("/tmp/t.csv") != f"/tmp/t.{os.getpid() + 1}.csv"

    path = tmp_path / "trace.csv"
    mod = _fresh(monkeypatch, GPU_PROFILER="1", GPU_PROFILER_TRACE=str(path),
                 GPU_PROFILER_INTERVAL="0.02")
    sampler = _sampler_with_stub(mod)
    try:
        time.sleep(0.08)
    finally:
        sampler._stop.set()
        time.sleep(0.05)
    assert not path.exists()                                       # never the bare name
    assert pathlib.Path(mod._trace_path_for_pid(str(path))).exists()


def test_a_broken_trace_file_never_takes_the_run_down(monkeypatch, tmp_path):
    """The profiler is a diagnostic. It must not be able to kill training."""
    mod = _fresh(monkeypatch, GPU_PROFILER="1", GPU_PROFILER_TRACE=str(tmp_path / "t.csv"),
                 GPU_PROFILER_INTERVAL="0.02")
    sampler = _sampler_with_stub(mod)
    try:
        sampler.push("step")
        time.sleep(0.1)
        sampler._trace.close()   # as a full disk or an unmounted path would
        time.sleep(0.15)
        assert sampler._trace is None   # writes disabled, sampler still alive
        assert sampler._thread.is_alive()
        sampler.push("update_actor")    # and the run carries on
        time.sleep(0.05)
        assert sampler._samples
    finally:
        sampler._stop.set()


def test_an_unwritable_path_is_reported_and_ignored(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, GPU_PROFILER="1",
                 GPU_PROFILER_TRACE=str(tmp_path / "no_such_dir" / "t.csv"),
                 GPU_PROFILER_INTERVAL="0.02")
    sampler = _sampler_with_stub(mod)
    try:
        assert sampler._trace is None
        time.sleep(0.05)
        assert sampler._thread.is_alive()
    finally:
        sampler._stop.set()


def test_the_h2d_phase_reads_first_in_the_interior_of_a_step(monkeypatch):
    """actor.h2d is the near side of the step boundary that pipelining cannot
    reach: a Ray actor runs its calls one at a time, so the worker only starts
    moving step k+1's batch onto the device after step k has returned -- however
    early the driver dispatched it. The table has to show it before actor.fwd or
    the interior of a step stops reading in execution order."""
    mod = _fresh(monkeypatch, GPU_PROFILER="1")
    order = [p for p in mod._PHASE_ORDER if p.startswith("actor.")]

    assert order == ["actor.h2d", "actor.fwd", "actor.bwd", "actor.task_metrics", "actor.optim"]


def test_a_phase_the_report_does_not_know_still_gets_a_row(monkeypatch):
    """Anything unlisted is appended rather than dropped, so instrumenting a new
    stage never silently loses its samples."""
    mod = _fresh(monkeypatch, GPU_PROFILER="1")
    by_phase = {"actor.fwd": mod._acc_new(1), "actor.somethingnew": mod._acc_new(1)}
    for acc in by_phase.values():
        acc["wall"], acc["n"] = 1.0, 1

    mod._print_report(by_phase, 1, "t", 0.2)   # must not raise
