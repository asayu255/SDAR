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
"""The stall scan, checked against traces whose true answer is known.

Two claims in the script are the ones worth testing, because both are places
the previous read of this data went wrong.

The estimator: ``utilization.gpu`` is a trailing moving average, so a stall is
never reported at its true depth for its true duration -- the window smears it.
Counting time below a threshold therefore under-reports, and by an amount that
depends on the window width, which makes a 0.2 s trace and wandb's 15 s samples
disagree for no physical reason. Integrating the deficit recovers the stall
whatever the window, and that is what ``test_the_deficit_integral_*`` pin down.

The per-card scan: one rank stalling while the others spin in a collective is
the signature that matters, and a node mean hides it. ``test_a_solo_stall_*``
covers that a 1-of-3 event is found and labelled.
"""

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "gpu_stall_scan.py"
_spec = importlib.util.spec_from_file_location("gpu_stall_scan", _SCRIPT)
scan = importlib.util.module_from_spec(_spec)
sys.modules["gpu_stall_scan"] = scan
_spec.loader.exec_module(scan)

HEADER = (
    "ts,clock,pid,phase,sm_pct_per_gpu,membw_pct_per_gpu,power_w_per_gpu,"
    "smclk_mhz_per_gpu,pcie_rx_mb_s_per_gpu,nvlink_mb_s_per_gpu,driver_cpu_pct\n"
)


def write_trace(path, samples, dt=0.2, phase="actor.fwd", ngpu=3):
    """``samples`` is a list of per-GPU sm lists (or a scalar for all cards)."""
    with open(path, "w") as f:
        f.write(HEADER)
        for i, sm in enumerate(samples):
            row = [sm] * ngpu if not isinstance(sm, (list, tuple)) else list(sm)
            ph = phase(i) if callable(phase) else phase
            f.write(
                f"{1000.0 + i * dt:.3f},01:00:00,4242,{ph},"
                f"{';'.join(f'{v:.0f}' for v in row)},"
                f"{';'.join('50' for _ in row)},"
                f"{';'.join('250' for _ in row)},"
                f"{';'.join('1700' for _ in row)},"
                f"{';'.join('100' for _ in row)},"
                f"{';'.join('0' for _ in row)},100\n"
            )
    return path


def analyse(path, **kw):
    rows, skipped = scan.read_trace(str(path))
    dts, n_gaps, gap_time = scan.sample_dts(rows)
    return rows, dts, skipped, n_gaps, gap_time


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
def _idle_seconds(rows, dts, gi):
    return sum((100.0 - r["sm"][gi]) / 100.0 * dts[k] for k, r in enumerate(rows))


def readings_for_stall(n_stall, window, pad=25):
    """What NVML reports for a stall of ``n_stall`` samples.

    ``utilization.gpu`` is the busy fraction of a trailing window, so this is
    literally a moving average of width ``window`` over a signal that is idle
    for ``n_stall`` consecutive samples. Note what it does when the window is
    wider than the stall: the reading never reaches 0, and with a window twice
    the stall it never even reaches 50.
    """
    busy = [1.0] * pad + [0.0] * n_stall + [1.0] * pad
    out = []
    for i in range(len(busy)):
        w = busy[max(0, i - window + 1): i + 1]
        out.append(100.0 * sum(w) / len(w))
    return out


def test_the_deficit_integral_recovers_a_stall_the_threshold_count_misses(tmp_path):
    """A 0.4 s stall seen through a 1 s trailing window, sampled at 0.2 s.

    The window is wider than the stall, so the deepest reading is 60% and NOT
    ONE SAMPLE falls below 50. Counting time under a threshold reports that
    nothing happened. The stall still happened.
    """
    samples = readings_for_stall(n_stall=2, window=5)
    p = write_trace(tmp_path / "t.csv", samples)
    rows, dts, *_ = analyse(p)

    below_50 = sum(dts[k] for k, r in enumerate(rows) if r["sm"][0] < 50)
    integral = _idle_seconds(rows, dts, 0)

    assert min(r["sm"][0] for r in rows) == pytest.approx(60.0)
    assert below_50 == 0.0                              # what the old read reported
    assert integral == pytest.approx(0.4, abs=1e-6)     # what was actually lost


@pytest.mark.parametrize("window_samples", [1, 2, 5, 10])
def test_the_deficit_integral_is_independent_of_the_averaging_window(tmp_path, window_samples):
    """Same 1.0 s stall, four different NVML window widths.

    The estimator has to give the same answer for all four, or a 0.2 s trace
    can never be compared with wandb's 15 s system metrics -- which is exactly
    the comparison that was in dispute.
    """
    samples = readings_for_stall(n_stall=5, window=window_samples)
    p = write_trace(tmp_path / f"t{window_samples}.csv", samples, dt=0.2)
    rows, dts, *_ = analyse(p)

    assert _idle_seconds(rows, dts, 0) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("window_samples", [1, 2, 5, 10])
def test_an_excursion_reports_the_stall_not_the_smear(tmp_path, window_samples):
    """And the same must hold for the per-excursion number, which is what the
    listing prints: one event, costing the 1.0 s that was lost, however wide
    the window that smeared it."""
    samples = readings_for_stall(n_stall=5, window=window_samples)
    p = write_trace(tmp_path / f"e{window_samples}.csv", samples, dt=0.2)
    rows, dts, *_ = analyse(p)

    found = scan.excursions(rows, dts, 0, floor=90.0, baseline=100.0)
    assert len(found) == 1
    assert found[0]["lost_s"] == pytest.approx(1.0, abs=1e-6)


def test_a_perfectly_busy_trace_reports_no_loss_and_no_excursions(tmp_path):
    p = write_trace(tmp_path / "t.csv", [100] * 50)
    rows, dts, *_ = analyse(p)

    assert _idle_seconds(rows, dts, 0) == pytest.approx(0.0)
    assert scan.excursions(rows, dts, 0, floor=90.0, baseline=100.0) == []


def test_ambient_dust_is_counted_but_is_not_an_excursion(tmp_path):
    """98-99% everywhere and no event anywhere.

    This is the case that decides what to do next: there is real lost time, but
    no excursion to go and look at, so the answer is fewer/larger kernels rather
    than a hunt for a straggler.
    """
    samples = [99 if i % 2 else 98 for i in range(200)]
    p = write_trace(tmp_path / "t.csv", samples)
    rows, dts, *_ = analyse(p)

    lost = _idle_seconds(rows, dts, 0)
    assert lost == pytest.approx((100 * 2 + 100 * 1) / 100.0 * 0.2, abs=1e-6)   # 0.6 s, real
    assert scan.excursions(rows, dts, 0, floor=90.0, baseline=median_of(rows, 0)) == []


def median_of(rows, gi):
    return scan.median([r["sm"][gi] for r in rows])


# --------------------------------------------------------------------------- #
# The per-card scan
# --------------------------------------------------------------------------- #
def test_a_solo_stall_is_found_and_named(tmp_path):
    """One card down, the other two at 100 -- a straggler, and the node mean
    would have shown a shallow 67% dip that looks like a global slowdown."""
    samples = [[100, 100, 100]] * 20 + [[0, 100, 100]] * 5 + [[100, 100, 100]] * 20
    p = write_trace(tmp_path / "t.csv", samples)
    rows, dts, *_ = analyse(p)

    found = scan.excursions(rows, dts, 0, floor=90.0, baseline=100.0)
    assert len(found) == 1
    assert found[0]["lost_s"] == pytest.approx(1.0, abs=1e-6)
    assert scan.classify(rows, found[0], floor=90.0) == "solo"
    # and nothing is attributed to the cards that were fine
    assert scan.excursions(rows, dts, 1, floor=90.0, baseline=100.0) == []


def test_a_global_gap_is_named_differently(tmp_path):
    """Every card down together is a driver-side gap, not a straggler."""
    samples = [[100, 100, 100]] * 20 + [[0, 0, 0]] * 5 + [[100, 100, 100]] * 20
    p = write_trace(tmp_path / "t.csv", samples, phase="(idle/other)")
    rows, dts, *_ = analyse(p)

    found = scan.excursions(rows, dts, 0, floor=90.0, baseline=100.0)
    assert scan.classify(rows, found[0], floor=90.0) == "all"


def test_the_excursion_window_grows_past_the_detection_threshold(tmp_path):
    """The shoulders of a dip carry real deficit.

    Detection fires at 90 but the event starts when the card leaves its
    baseline, so a window cut at the threshold would drop the ramp -- which for
    a trailing-average instrument is most of the event.
    """
    shoulders_and_pit = [95, 92, 60, 20, 60, 92, 95]
    samples = [100] * 20 + shoulders_and_pit + [100] * 20
    p = write_trace(tmp_path / "t.csv", samples)
    rows, dts, *_ = analyse(p)

    found = scan.excursions(rows, dts, 0, floor=90.0, baseline=100.0)
    assert len(found) == 1
    # 7 samples wide, not the 3 that are under 90
    assert found[0]["i1"] - found[0]["i0"] + 1 == 7
    assert found[0]["lost_s"] == pytest.approx(sum(100 - v for v in shoulders_and_pit) / 100 * 0.2, abs=1e-6)


def test_two_separate_dips_are_two_excursions(tmp_path):
    samples = [100] * 10 + [0] * 3 + [100] * 30 + [0] * 3 + [100] * 10
    p = write_trace(tmp_path / "t.csv", samples)
    rows, dts, *_ = analyse(p)

    found = scan.excursions(rows, dts, 0, floor=90.0, baseline=100.0)
    assert len(found) == 2
    assert all(e["lost_s"] == pytest.approx(0.6, abs=1e-6) for e in found)


def test_an_excursion_keeps_the_phase_it_happened_in(tmp_path):
    samples = [100] * 20 + [0] * 5 + [100] * 20
    p = write_trace(tmp_path / "t.csv", samples, phase=lambda i: "actor.bwd" if 20 <= i < 25 else "actor.fwd")
    rows, dts, *_ = analyse(p)

    found = scan.excursions(rows, dts, 0, floor=90.0, baseline=100.0)
    assert found[0]["phases"] == ["actor.bwd"]
    assert rows[found[0]["argmin"]]["phase"] == "actor.bwd"


# --------------------------------------------------------------------------- #
# Reading real files
# --------------------------------------------------------------------------- #
def test_a_sampling_gap_is_excluded_rather_than_charged_as_idle(tmp_path):
    """A gap means the sampler did not run. It is not evidence either way, so
    it must not land in the denominator -- a clobbered trace was mostly gaps."""
    with open(tmp_path / "t.csv", "w") as f:
        f.write(HEADER)
        for i, ts in enumerate([0.0, 0.2, 0.4, 60.0, 60.2, 60.4]):
            f.write(f"{ts:.3f},01:00:00,1,actor.fwd,100;100;100,50;50;50,250;250;250,"
                    f"1700;1700;1700,100;100;100,0;0;0,100\n")
    rows, dts, skipped, n_gaps, gap_time = analyse(tmp_path / "t.csv")

    assert n_gaps == 1
    assert gap_time == pytest.approx(59.6, abs=1e-6)
    assert sum(dts) == pytest.approx(0.2 * 5, abs=1e-6)   # the gap contributes nothing


def test_a_torn_final_line_is_dropped_not_fatal(tmp_path):
    """Runs that end in Ctrl-C are the ones worth looking at."""
    p = write_trace(tmp_path / "t.csv", [100] * 5)
    with open(p, "a") as f:
        f.write("1001.2,01:00:00,4242,actor.f")
    rows, _, skipped, _, _ = analyse(p)

    assert len(rows) == 5
    assert skipped == 1


def test_an_old_trace_without_the_power_columns_still_parses(tmp_path):
    """Traces written before those columns existed must not become unreadable."""
    with open(tmp_path / "old.csv", "w") as f:
        f.write("ts,clock,phase,sm_pct_per_gpu,membw_pct_per_gpu,driver_cpu_pct\n")
        f.write("1000.0,01:00:00,actor.fwd,99;99;99,50;50;50,100\n")
        f.write("1000.2,01:00:00,actor.fwd,98;99;99,50;50;50,100\n")
    rows, *_ = analyse(tmp_path / "old.csv")

    assert len(rows) == 2
    assert rows[0]["sm"] == [99.0, 99.0, 99.0]
    assert rows[0]["power"] == [None]      # absent, and absent is not zero
    assert rows[0]["cpu"] == 100.0


def test_the_report_runs_end_to_end(tmp_path, capsys):
    samples = [[100, 100, 100]] * 30 + [[0, 100, 100]] * 5 + [[99, 99, 98]] * 30
    p = write_trace(tmp_path / "t.csv", samples)

    assert scan.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "no-kernel" in out
    assert "solo" in out
    assert "ambient, everywhere else" in out
    assert "actor.fwd" in out


def test_globs_are_expanded_so_both_samplers_can_be_passed_at_once(tmp_path, capsys):
    write_trace(tmp_path / "trace.111.csv", [100] * 20)
    write_trace(tmp_path / "trace.222.csv", [100] * 20)

    assert scan.main([str(tmp_path / "trace.*.csv")]) == 0
    out = capsys.readouterr().out
    assert "scanning 2 trace file(s)" in out
    assert "trace.111.csv" in out and "trace.222.csv" in out


def test_a_missing_file_is_an_error_not_an_empty_report(tmp_path):
    with pytest.raises(SystemExit):
        scan.main([str(tmp_path / "nope.csv")])
