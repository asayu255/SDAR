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

    rows = path.read_text().strip().splitlines()
    assert rows[0] == "ts,clock,phase,sm_pct_per_gpu,membw_pct_per_gpu,driver_cpu_pct"
    body = [r.split(",") for r in rows[1:]]
    assert len(body) > 5, rows

    # The phase column is the whole point: it says what the trainer was doing.
    assert "update_actor" in {r[2] for r in body}
    assert "step" in {r[2] for r in body}
    # One column per GPU, so data-parallel imbalance stays visible.
    assert body[0][3] == "97;97;97"
    assert body[0][4] == "40;40;40"
    assert body[0][5] == "55"
    # Monotonic timestamps, so a dip can be located against an external chart.
    stamps = [float(r[0]) for r in body]
    assert stamps == sorted(stamps)


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
