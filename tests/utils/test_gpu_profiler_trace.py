"""GPU_PROFILER_TRACE keeps the per-sample record the aggregate tables discard.

The per-step tables say a GPU gap happened, in which phase, and how long the
longest one was -- then reset at the step boundary. That is the wrong shape for
a transient seen once on an external monitor: to line the two up you need the
wall-clock time of each sample and the phase it fell in. This covers that file.

No GPU is needed; the sampler is driven with a stub backend.
"""

import importlib
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

    written = mod._trace_path_for_pid(str(path))
    rows = open(written).read().strip().splitlines()
    assert rows[0] == (
        "ts,clock,pid,phase,sm_pct_per_gpu,membw_pct_per_gpu,power_w_per_gpu,"
        "smclk_mhz_per_gpu,pcie_rx_mb_s_per_gpu,nvlink_mb_s_per_gpu,driver_cpu_pct"
    )
    body = [r.split(",") for r in rows[1:]]
    assert len(body) > 5, rows

    # The phase column is the whole point: it says what the trainer was doing.
    assert "update_actor" in {r[3] for r in body}
    assert "step" in {r[3] for r in body}
    # One column per GPU, so data-parallel imbalance stays visible.
    assert body[0][4] == "97;97;97"
    assert body[0][5] == "40;40;40"
    # A metric the stub backend does not report comes out empty rather than
    # absent, so the column count is the same on every row whatever the driver
    # can answer.
    assert body[0][6] == ";;"
    assert body[0][-1] == "55"
    assert len({len(r) for r in body}) == 1, "ragged rows -- the columns shifted"
    # Monotonic timestamps, so a dip can be located against an external chart.
    stamps = [float(r[0]) for r in body]
    assert stamps == sorted(stamps)


def test_each_process_gets_its_own_trace_file(monkeypatch, tmp_path):
    """Two samplers run: one in the driver (ray_trainer._timer) and one in
    rank 0's worker (dp_actor._actor_phase). They are separate processes with
    separate file offsets, so opening one path "w" from both makes each
    overwrite the other's bytes -- the file ends up with about ONE sampler's
    worth of rows stitched from two streams at whatever offset each had reached.

    Aggregate means survive that (both read the same devices). Anything that
    treats consecutive rows as consecutive in time does not, which is most of
    what a per-sample trace is for -- and all of what gpu_stall_scan.py does.
    """
    import os

    mod = _fresh(monkeypatch, GPU_PROFILER="1", GPU_PROFILER_TRACE=str(tmp_path / "trace.csv"))
    got = mod._trace_path_for_pid(str(tmp_path / "trace.csv"))
    assert got.endswith(f".{os.getpid()}.csv")
    assert got != str(tmp_path / "trace.csv")
    # A path with no extension still gets one, rather than becoming "trace.1234"
    assert mod._trace_path_for_pid("/tmp/trace").endswith(f"/tmp/trace.{os.getpid()}.csv")


def test_the_pid_column_says_which_sampler_wrote_the_row(monkeypatch, tmp_path):
    """Belt and braces with the filename: if the files are ever concatenated,
    the rows still say which stream they came from."""
    import os

    path = tmp_path / "trace.csv"
    mod = _fresh(monkeypatch, GPU_PROFILER="1", GPU_PROFILER_TRACE=str(path),
                 GPU_PROFILER_INTERVAL="0.02")
    sampler = _sampler_with_stub(mod)
    try:
        sampler.push("step")
        time.sleep(0.12)
    finally:
        sampler._stop.set()
        time.sleep(0.05)

    rows = open(mod._trace_path_for_pid(str(path))).read().strip().splitlines()[1:]
    assert rows
    assert {r.split(",")[2] for r in rows} == {str(os.getpid())}


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
