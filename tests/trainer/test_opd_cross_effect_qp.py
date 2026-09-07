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

"""The calibration script, on the three failures review found in it.

This is the code that produces the coefficient vector three GPU runs are then
labelled by, so the tests are about what it does with BAD input, not about
reproducing its happy path.
"""

import importlib.util
import math
import os

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "opd_cross_effect_qp",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "opd_cross_effect_qp.py"),
)
qp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qp)


def _synthetic(n=8, seed=0, kill_rl_row=None, kill_batches=()):
    """Per-batch dot products, RL norms and OPD norms with a known ranking.

    Column scores are built so that search > alfworld > webshop, the ranking the
    step-300 payload gives, and ``kill_rl_row`` zeroes one task's RL gradient on
    ``kill_batches`` -- the degenerate-group case that used to yield b = NaN.
    """
    rng = np.random.default_rng(seed)
    base = np.array([[-0.02, 0.03, -0.10],
                     [-0.01, 0.02, -0.08],
                     [0.01, 0.02, -0.06]])
    nr = np.abs(rng.normal(1.0, 0.05, size=(n, 3))) + 0.5
    nd = np.tile(np.array([0.4563, 0.8859, 0.4084]), (n, 1)) * (1 + rng.normal(0, 0.02, (n, 3)))
    C = (base[None] + rng.normal(0, 0.004, (n, 3, 3))) * nr[:, :, None] * nd[:, None, :]
    if kill_rl_row is not None:
        for b in kill_batches:
            nr[b, kill_rl_row] = 0.0
            C[b, kill_rl_row, :] = 0.0
    D = np.tile(np.diag([0.4563, 0.8859, 0.4084]) ** 2, (n, 1, 1))
    return C, nr, nd, D


# ---------------------------------------------------------------------------
# 1. a row with no RL signal


def test_a_zero_rl_row_used_to_give_nan_and_now_drops_only_that_cell():
    C, nr, nd, _ = _synthetic(kill_rl_row=2, kill_batches=(0, 1))
    # the old code path, for the record: straight division through a zero norm
    with np.errstate(invalid="ignore", divide="ignore"):
        naive = (C / (nr[:, :, None] * nd[:, None, :])).mean(0)
    assert np.isnan(naive).any(), "the fixture must reproduce the failure it is about"

    cos, ok = qp.cosine_cells(C, nr, nd)
    mean, counts = qp.masked_mean(cos, ok)
    assert np.isfinite(mean).all()
    assert (counts[2] == 6).all(), "webshop's row loses exactly the two dead batches"
    assert (counts[0] == 8).all() and (counts[1] == 8).all()

    b, c, q, _ = qp.redistribute(cos, ok, nd, [0, 1, 2])
    assert np.isfinite(b).all()
    assert float((q * b).sum()) == pytest.approx(1.0, abs=1e-9)


def test_it_refuses_when_a_row_is_dead_in_most_batches():
    C, nr, nd, _ = _synthetic(kill_rl_row=0, kill_batches=(0, 1, 2, 3, 4, 5))
    cos, ok = qp.cosine_cells(C, nr, nd)
    with pytest.raises(ValueError, match="too few usable batches"):
        qp.redistribute(cos, ok, nd, [0, 1, 2])
    # and the message names the cell, so the reader knows which row died
    with pytest.raises(ValueError, match=r"C\[RL alfworld, OPD "):
        qp.redistribute(cos, ok, nd, [0, 1, 2])
    # lowering the bar deliberately is allowed; silently passing is not
    b, _, q, _ = qp.redistribute(cos, ok, nd, [0, 1, 2], min_fraction=0.2)
    assert np.isfinite(b).all()


def test_deliberately_dropping_the_diagonal_is_not_treated_as_missing_data():
    """diag=False zeroes three cells by construction; the coverage check must
    not read that as a dead measurement, or the honest variant of the rule
    becomes the one that cannot run."""
    C, nr, nd, _ = _synthetic()
    cos, ok = qp.cosine_cells(C, nr, nd)
    b_on, _, _, _ = qp.redistribute(cos, ok, nd, [0, 1, 2], diag=True)
    b_off, _, _, _ = qp.redistribute(cos, ok, nd, [0, 1, 2], diag=False)
    assert np.isfinite(b_off).all()
    assert not np.allclose(b_on, b_off), "the diagonal does move b -- see the design doc section 5"


# ---------------------------------------------------------------------------
# 2. the ranking test is about the sample, not about one remembered answer


def test_the_measured_ranking_is_read_off_the_data_not_hardcoded():
    C, nr, nd, _ = _synthetic()
    cos, ok = qp.cosine_cells(C, nr, nd)
    b, _, _, _ = qp.redistribute(cos, ok, nd, [0, 1, 2])
    assert qp._order(b) == ("search", "alfworld", "webshop")

    # A different vector must report a different order, which a hardcoded
    # P(search > alfworld > webshop) could not: alfworld middle, search last.
    assert qp._order(np.array([1.076, 0.5, 1.191])) == ("webshop", "alfworld", "search")
    assert qp._order(np.array([1.0, 1.0, 1.0]))[0] == "alfworld"  # ties: stable, by index


def test_reproducibility_and_agreement_are_two_numbers(tmp_path):
    """A sample whose own ranking is stable but DISAGREES with the reference
    must score high on one and low on the other. Reporting only the second, as
    the script used to, would call a perfectly stable measurement unstable."""
    C, nr, nd, D = _synthetic()
    # make webshop the strongest column, so the measured order is not the
    # reference order
    C[:, :, 2] *= -1.0
    cos, ok = qp.cosine_cells(C, nr, nd)
    b, _, _, _ = qp.redistribute(cos, ok, nd, [0, 1, 2])
    measured = qp._order(b)
    assert measured != qp.REFERENCE_ORDER

    rng = np.random.default_rng(0)
    orders = []
    for _ in range(300):
        idx = rng.integers(0, len(C), len(C))
        bb = qp.redistribute(cos[idx], ok[idx], nd[idx], [0, 1, 2], check=False)[0]
        orders.append(qp._order(bb))
    repro = np.mean([o == measured for o in orders])
    agree = np.mean([o == qp.REFERENCE_ORDER for o in orders])
    assert repro > 0.8
    assert agree < 0.2


# ---------------------------------------------------------------------------
# 3. the composite norm has one definition per name


def test_the_two_composite_conventions_are_reported_separately():
    C, nr, nd, D = _synthetic()
    b = np.array([1.076431, 1.191101, 0.5])
    lin, rms, mean = qp.budgets(b, D, nd)
    per_batch = np.array([math.sqrt(b @ D[n] @ b) for n in range(len(D))])
    assert rms == pytest.approx(float(np.sqrt((per_batch ** 2).mean())))
    assert mean == pytest.approx(float(per_batch.mean()))
    assert rms >= mean  # RMS >= mean, always; two names, two numbers


def test_the_uniform_arm_factor_comes_from_the_rms():
    C, nr, nd, D = _synthetic()
    b = np.array([1.076431, 1.191101, 0.5])
    scale = qp.uniform_match(b, D, nd)
    _, rms_b, _ = qp.budgets(b, D, nd)
    _, rms_1, _ = qp.budgets(np.ones(3), D, nd)
    assert scale == pytest.approx(rms_b / rms_1)
    # a uniform vector at that scale reproduces the redistributed composite norm
    _, rms_u, _ = qp.budgets(np.full(3, scale), D, nd)
    assert rms_u == pytest.approx(rms_b, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. the rule's invariant, on synthetic data as well as on the payload


def test_the_budget_invariant_holds_for_any_input_the_rule_accepts():
    for seed in range(5):
        C, nr, nd, _ = _synthetic(seed=seed)
        cos, ok = qp.cosine_cells(C, nr, nd)
        b, c, q, _ = qp.redistribute(cos, ok, nd, [0, 1, 2])
        inside_box = (0.5 < b) & (b < 1.5)
        assert float((q * b).sum()) == pytest.approx(1.0, abs=1e-9) or not inside_box.all()
        # and the proxy it is built to improve never gets worse
        assert float((q * c * b).sum()) >= float((q * c).sum()) - 1e-12


def test_the_payload_flag_parser_does_not_eat_a_flag_value(monkeypatch):
    monkeypatch.setattr(qp.sys, "argv",
                        ["x", "--redistribute", "--json", "out.json", "payload.json"])
    assert qp._payload_arg() == "payload.json"
    monkeypatch.setattr(qp.sys, "argv", ["x", "--redistribute", "--min-batches", "0.25"])
    assert qp._payload_arg().endswith("terms_n8_fixed.json")


# ---------------------------------------------------------------------------
# 5. the go condition, and what the script hands over when it fails


def _payload(path, cos_target, nd_rows, nr_rows, seed=0, jitter=0.0):
    """A terms payload in the shape load() reads, from a target cosine matrix.

    Written out rather than reduced to the arrays, because the failure this
    guards -- a non-trivial b surviving a failed admission test -- lives in
    redistribution_report, which takes a FILE.
    """
    import json

    rng = np.random.default_rng(seed)
    tasks = qp.TASKS
    batches = []
    for n in range(len(nd_rows)):
        nd, nr = np.asarray(nd_rows[n], float), np.asarray(nr_rows[n], float)
        cos = np.asarray(cos_target, float) + rng.normal(0, jitter, (3, 3))
        w = {}
        for i, t in enumerate(tasks):
            w[f"sq:{t}:rl"] = float(nr[i] ** 2)
            w[f"sq:{t}:opd"] = float(nd[i] ** 2)
        for i, ti in enumerate(tasks):
            for j, tj in enumerate(tasks):
                w[f"dot:{ti}:rl:{tj}:opd"] = float(cos[i, j] * nr[i] * nd[j])
                if i < j:
                    w[f"dot:{ti}:rl:{tj}:rl"] = 0.0
                    w[f"dot:{ti}:opd:{tj}:opd"] = 0.0
        batches.append(w)
    path.write_text(json.dumps({
        "checkpoint": "synthetic", "pg_loss_coef": 1.0, "teacher_kl_loss_coef": 0.01,
        "batch_moments": batches,
        "advantages": [{t: {"live_groups": 5, "prompt_groups": 5} for t in tasks}
                       for _ in batches],
    }))
    return path


def test_a_failed_go_condition_returns_the_uniform_vector_not_the_candidate(tmp_path, capsys):
    """The bug: the script printed "kappa = 0" and returned a non-trivial b.

    Columns are near-tied and the per-batch noise is large, so the ranking the
    sample gives does not survive its own bootstrap.
    """
    p = _payload(tmp_path / "wobbly.json",
                 cos_target=[[-0.02, -0.02, -0.021]] * 3,
                 nd_rows=[[0.46, 0.89, 0.41]] * 8,
                 nr_rows=[[1.0, 1.0, 1.0]] * 8,
                 seed=1, jitter=0.05)
    out = tmp_path / "out.json"
    approved = qp.redistribution_report(str(p), json_out=str(out), n_boot=300)
    text = capsys.readouterr().out
    assert "GO CONDITION NOT MET" in text

    import json
    saved = json.loads(out.read_text())
    assert saved["rank_reproducibility"] < 0.90
    assert saved["approved"] is False
    assert saved["approved_b"] == [1.0, 1.0, 1.0]
    assert saved["candidate_b"] != saved["approved_b"], (
        "the candidate must still be reported -- it is the thing under inspection"
    )
    assert np.allclose(approved, 1.0), "the RETURN value is what a caller would use"


def test_a_passing_go_condition_returns_the_candidate(tmp_path):
    p = _payload(tmp_path / "clean.json",
                 cos_target=[[-0.02, 0.03, -0.25]] * 3,
                 nd_rows=[[0.46, 0.89, 0.41]] * 8,
                 nr_rows=[[1.0, 1.0, 1.0]] * 8,
                 seed=2, jitter=0.002)
    out = tmp_path / "out.json"
    approved = qp.redistribution_report(str(p), json_out=str(out), n_boot=300)
    import json
    saved = json.loads(out.read_text())
    assert saved["approved"] is True
    assert saved["approved_b"] == saved["candidate_b"]
    assert not np.allclose(approved, 1.0)


def test_the_admission_threshold_is_the_designs_and_is_named_once():
    assert qp.RANK_REPRO_MIN == 0.90


# ---------------------------------------------------------------------------
# 6. an OPD norm of exactly zero is a measurement, not a missing value


def test_a_zero_opd_norm_counts_in_the_mean_that_builds_q():
    """The reviewer's example, exactly: norms (1,1,1) and (0,1,1).

    Dropping the zero reports the norm of the batches where the teacher DID
    push, which is a bigger number than the one the budget is made of, and q
    comes out uniform instead of skewed.
    """
    nd = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    nd_mean, counts = qp.masked_mean(nd, np.isfinite(nd))
    assert nd_mean.tolist() == [0.5, 1.0, 1.0]
    q = nd_mean / nd_mean.sum()
    assert q.tolist() == pytest.approx([0.2, 0.4, 0.4])

    dropped, _ = qp.masked_mean(nd, np.isfinite(nd) & (nd > 0))
    assert dropped.tolist() == [1.0, 1.0, 1.0]
    q_dropped = dropped / dropped.sum()
    assert q_dropped.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_the_budgets_helper_uses_the_zero_inclusive_mean():
    nd = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    D = np.tile(np.eye(3), (2, 1, 1))
    lin, _, _ = qp.budgets(np.ones(3), D, nd)
    assert lin == pytest.approx(0.5 + 1.0 + 1.0)
