"""Unit tests for the phase-attributed GPU profiler's aggregation.

Covered (CPU-only; no GPU, no NVML, no Ray):
* per-phase sample count and wall time (the ``n`` column: a row backed by one
  sample must not read like a measurement);
* the longest-contiguous-idle-gap metric, including the case that motivates it
  — two idle stretches on either side of a busy phase must NOT be reported as
  one long gap;
* accumulator merging (the run-level rollup): walls add, gaps take the max;
* host CPU% and NVLink means, and graceful degradation when a backend does not
  supply them.

If the verl stack is not importable, the module is skipped.
"""

import pytest

try:
    from verl.utils import gpu_profiler as gp
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


N_GPUS = 2
BUSY = 95.0
IDLE = 1.0  # below the default GPU_PROFILER_IDLE_THRESH of 30


def _gpus(sm, nvlink=None):
    return [
        {
            "sm_util": sm,
            "mem_bw_util": sm * 0.6,
            "mem_used_gb": 40.0,
            "power_w": 250.0,
            "sm_clock_mhz": 1700.0,
            "pcie_tx_mb_s": 50.0,
            "pcie_rx_mb_s": 50.0,
            "nvlink_mb_s": nvlink,
        }
        for _ in range(N_GPUS)
    ]


def _samples(spec, dt=0.1, cpu=None):
    """spec: list of (phase, sm). Timestamps advance by ``dt`` (contiguous)."""
    out = []
    ts = 0.0
    for phase, sm in spec:
        ts += dt
        host = None if cpu is None else {"cpu_pct": cpu(phase)}
        out.append((ts, dt, phase, _gpus(sm), host))
    return out


def _finalize(spec, **kw):
    by_phase = gp._accumulate(_samples(spec, **kw), N_GPUS)
    return {p: gp._acc_finalize(a, N_GPUS) for p, a in by_phase.items()}


def test_per_phase_sample_count_and_wall():
    res = _finalize([("update_actor", BUSY)] * 3 + [("save_checkpoint", IDLE)])

    assert res["update_actor"]["n"] == 3
    assert res["update_actor"]["wall"] == pytest.approx(0.3)
    # A single-sample row is exactly what the n column exists to expose.
    assert res["save_checkpoint"]["n"] == 1
    assert res["save_checkpoint"]["wall"] == pytest.approx(0.1)


def test_sm_util_is_averaged_across_gpus_and_samples():
    res = _finalize([("update_actor", 90.0), ("update_actor", 100.0)])
    assert res["update_actor"]["sm_util"] == pytest.approx(95.0)
    assert res["update_actor"]["per_gpu_sm"] == [pytest.approx(95.0)] * N_GPUS


def test_idle_gap_is_the_longest_contiguous_run():
    # idle, idle, busy, idle  ->  longest run is the leading pair (0.2s)
    res = _finalize(
        [
            ("(idle/other)", IDLE),
            ("(idle/other)", IDLE),
            ("(idle/other)", BUSY),
            ("(idle/other)", IDLE),
        ]
    )
    a = res["(idle/other)"]
    assert a["max_gap"] == pytest.approx(0.2)
    assert a["idle_wall"] == pytest.approx(0.3)  # total idle, not contiguous
    assert a["idle_pct"] == pytest.approx(75.0)


def test_idle_gap_not_joined_across_an_intervening_busy_phase():
    """The reason gaps are tracked at accumulate time, not merge time.

    ``step`` is entered twice within one training step (before and after
    ``update_actor``). Its two idle stretches are far apart in wall-clock time
    and must stay separate runs, or the report would invent a gap twice as long
    as any that actually occurred.
    """
    spec = [("step", IDLE), ("step", IDLE)]
    spec += [("update_actor", BUSY)] * 8
    spec += [("step", IDLE), ("step", IDLE)]

    res = _finalize(spec)
    # Two runs of 2 samples each, separated by 0.8s of update_actor.
    assert res["step"]["max_gap"] == pytest.approx(0.2)
    assert res["step"]["idle_wall"] == pytest.approx(0.4)
    assert res["update_actor"]["max_gap"] == pytest.approx(0.0)


def test_merge_adds_wall_and_takes_max_of_gap():
    step_a = gp._accumulate(_samples([("(idle/other)", IDLE)] * 2), N_GPUS)
    step_b = gp._accumulate(_samples([("(idle/other)", IDLE)] * 5), N_GPUS)

    cum = gp._acc_new(N_GPUS)
    gp._acc_merge(cum, step_a["(idle/other)"], N_GPUS)
    gp._acc_merge(cum, step_b["(idle/other)"], N_GPUS)
    out = gp._acc_finalize(cum, N_GPUS)

    assert out["n"] == 7
    assert out["wall"] == pytest.approx(0.7)  # walls add
    assert out["max_gap"] == pytest.approx(0.5)  # gaps do NOT add
    assert out["sm_util"] == pytest.approx(IDLE)


def test_copy_does_not_alias_the_source():
    src = gp._accumulate(_samples([("update_actor", BUSY)] * 2), N_GPUS)["update_actor"]
    snapshot = gp._acc_copy(src, N_GPUS)
    gp._acc_merge(src, src, N_GPUS)  # mutate the original afterwards

    assert gp._acc_finalize(snapshot, N_GPUS)["n"] == 2
    assert gp._acc_finalize(src, N_GPUS)["n"] == 4


def test_host_cpu_percent_is_averaged_per_phase():
    # The signature that tells a prefetchable stall (driver busy) from an I/O
    # wait (driver idle) while the GPU sits at the same low utilization.
    res = _finalize(
        [("(idle/other)", IDLE), ("(idle/other)", IDLE), ("update_actor", BUSY)],
        cpu=lambda phase: 400.0 if phase == "(idle/other)" else 20.0,
    )
    assert res["(idle/other)"]["cpu_pct"] == pytest.approx(400.0)
    assert res["update_actor"]["cpu_pct"] == pytest.approx(20.0)


def test_missing_host_sampler_leaves_cpu_pct_none():
    res = _finalize([("update_actor", BUSY)])
    assert res["update_actor"]["cpu_pct"] is None


def test_nvlink_mean_and_absence():
    with_link = gp._accumulate(
        [(0.1, 0.1, "update_actor", _gpus(BUSY, nvlink=1000.0), None)], N_GPUS
    )
    assert gp._acc_finalize(with_link["update_actor"], N_GPUS)["nvlink_mb_s"] == pytest.approx(1000.0)

    # nvidia-smi fallback / no-NVLink hardware reports None and must not crash.
    without = gp._accumulate([(0.1, 0.1, "gen", _gpus(BUSY, nvlink=None), None)], N_GPUS)
    assert gp._acc_finalize(without["gen"], N_GPUS)["nvlink_mb_s"] is None


def test_report_renders_both_scopes(capsys):
    by_phase = gp._accumulate(
        _samples(
            [("update_actor", BUSY)] * 3 + [("(idle/other)", IDLE)],
            cpu=lambda phase: 300.0 if phase == "(idle/other)" else 10.0,
        ),
        N_GPUS,
    )

    gp._print_report(by_phase, N_GPUS, "step 0", 0.1)
    per_step = capsys.readouterr().out
    assert "TOTAL/step" in per_step
    assert "maxGap" in per_step and "cpu%" in per_step and "nvlink" in per_step

    gp._print_report(by_phase, N_GPUS, "CUMULATIVE", 0.1, cumulative_steps=4)
    cumulative = capsys.readouterr().out
    assert "TOTAL (run)" in cumulative
    assert "steps=4" in cumulative
    assert "per step" in cumulative


def test_missing_metric_placeholder_keeps_the_column_width():
    # An nvidia-smi-backed host reports no PCIe and no NVLink. Three unpadded
    # placeholders in a row render as "1736---" and shift every later column.
    assert gp._fmt(None, ">9.0f") == "        -"
    assert gp._fmt(None, ">8.1f") == "       -"
    assert gp._fmt(12.0, ">8.1f") == "    12.0"
    # The per-GPU list is joined with "/" and must stay unpadded.
    assert gp._fmt(None, ".0f") == "-"


def test_row_stays_aligned_when_a_backend_supplies_no_link_metrics(capsys):
    no_link = [
        {
            "sm_util": BUSY,
            "mem_bw_util": 41.0,
            "mem_used_gb": 44.0,
            "power_w": 232.0,
            "sm_clock_mhz": 1740.0,
            "pcie_tx_mb_s": None,
            "pcie_rx_mb_s": None,
            "nvlink_mb_s": None,
        }
        for _ in range(N_GPUS)
    ]
    by_phase = gp._accumulate([(0.1, 0.1, "update_actor", no_link, None)], N_GPUS)
    gp._print_report(by_phase, N_GPUS, "step 0", 0.1)
    out = capsys.readouterr().out.splitlines()

    header = next(ln for ln in out if ln.lstrip().startswith("phase"))
    row = next(ln for ln in out if ln.startswith("update_actor"))
    assert "---" not in row
    # The per-GPU column is the only ragged one, so compare up to where it starts.
    assert len(row[: header.index("per-GPU")]) == header.index("per-GPU")


def test_share_column_sums_to_one_hundred(capsys):
    by_phase = gp._accumulate(
        _samples([("update_actor", BUSY)] * 3 + [("save_checkpoint", IDLE)]), N_GPUS
    )
    gp._print_report(by_phase, N_GPUS, "step 0", 0.1)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]

    shares = []
    for ln in lines:
        if ln.startswith(("update_actor", "save_checkpoint")):
            shares.append(float(ln.split()[3]))
    assert sum(shares) == pytest.approx(100.0, abs=0.2)
