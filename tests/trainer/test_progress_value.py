"""Unit tests for the progress_value_gae estimator (verl/trainer/ppo/progress_value.py).

What has to hold, and why each part is tested:

- The TD/GAE arithmetic, against the closed form on hand-made trajectories for
  (gamma, lam) in {(1, 1), (1, .9), (.95, 1), (.95, .9)}, and the two identities
  that hold for ANY table: the deltas telescope (sum_t gamma^t delta_t =
  gamma^T R - V_0), and lam = 1 is the Monte Carlo return minus the value.
- The batch arrives reordered by _balance_batch with adjust_batch's padding
  copies in it: every row's result must not depend on the row order, a copy must
  get its original's advantage, and copies must stay out of every statistic and
  out of the table update. A turn missing from a trajectory must fail loudly.
- The table: frozen within a step (the batch is scored before it is added),
  discounted counts, the cell -> (task, type, rem) -> task shrinkage by hand,
  WebShop goals the environment cannot pay (V = 0, no update), the empty-table
  fallback, the JSON round trip and the configuration fingerprint.
- The format term: tied vs all, the control's GRPO z-score in a tied group, and
  exactly zero -- no float32 phantom -- in a uniform group.

CPU and numpy only: the module is loaded from its file, so neither torch nor the
verl package is imported.
"""

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "verl", "trainer", "ppo", "progress_value.py")
_SPEC = importlib.util.spec_from_file_location("progress_value_under_test", _PATH)
pv = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pv
_SPEC.loader.exec_module(pv)

GAMMA_LAM = [(1.0, 1.0), (1.0, 0.9), (0.95, 1.0), (0.95, 0.9)]
CAPS = {"alfworld": 50, "webshop": 15, "search": 4}
GAMEFILE = "json_2.1.1/train/pick_and_place_simple-Book-None-Desk-310/trial_T2019_1/game.tw-pddl"
GAMEFILE_TWO = "json_2.1.1/train/pick_two_obj_and_place-Book-None-Desk-311/trial_T2019_2/game.tw-pddl"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def trajectory(uid, tuid, T, reward, task="alfworld", *, k=None, k_after=None, stag=None, valid=None,
               cap=None, gamefile=GAMEFILE, capped=None, buynow=None):
    return dict(uid=uid, tuid=tuid, T=T, reward=reward, task=task, k=k, k_after=k_after, stag=stag,
                valid=valid, cap=cap, gamefile=gamefile, capped=capped, buynow=buynow)


def batch(specs):
    """Per-row columns, trajectory by trajectory, turns ascending (the rollout's order)."""
    cols = {c: [] for c in ("traj_uid", "uid", "pv_t", "pv_cap", "task_name", "episode_rewards",
                            "is_action_valid", "pv_k_before", "pv_k_after", "pv_stag_before", "gamefile",
                            "goal_capped", "pv_buynow_b")}
    for s in specs:
        T = s["T"]
        k = s["k"] if s["k"] is not None else [0] * T
        k_after = s["k_after"] if s["k_after"] is not None else list(k[1:]) + [k[-1]]
        stag = s["stag"] if s["stag"] is not None else list(range(T))
        valid = s["valid"] if s["valid"] is not None else [True] * T
        cap = s["cap"] if s["cap"] is not None else CAPS[s["task"]]
        capped = s["capped"] if s["capped"] is not None else (0.0 if s["task"] == "webshop" else float("nan"))
        # WebShop's buy-now score: 0 (nothing open) unless given; NaN on the other tasks, as recorded.
        buynow = (s["buynow"] if s["buynow"] is not None
                  else [0.0 if s["task"] == "webshop" else float("nan")] * T)
        for t in range(T):
            cols["traj_uid"].append(s["tuid"])
            cols["uid"].append(s["uid"])
            cols["pv_t"].append(float(t))
            cols["pv_cap"].append(float(cap))
            cols["task_name"].append(s["task"])
            cols["episode_rewards"].append(float(s["reward"]))
            cols["is_action_valid"].append(bool(valid[t]))
            cols["pv_k_before"].append(float(k[t]))
            cols["pv_k_after"].append(float(k_after[t]))
            cols["pv_stag_before"].append(float(stag[t]))
            cols["gamefile"].append(s["gamefile"] if s["task"] == "alfworld" else "")
            cols["goal_capped"].append(float(capped))
            cols["pv_buynow_b"].append(float(buynow[t]))
    out = {}
    for c, v in cols.items():
        if c in ("traj_uid", "uid", "task_name", "gamefile"):
            out[c] = np.array(v, dtype=object)
        elif c == "is_action_valid":
            out[c] = np.array(v, dtype=bool)
        else:
            out[c] = np.array(v, dtype=np.float64)
    return out


def permute(cols, order):
    return {c: np.asarray(v)[order] for c, v in cols.items()}


def with_padding(cols, copies, seed=None):
    """adjust_batch(copy): the copies appended at the end, flagged; then optionally shuffled (balance)."""
    n = len(cols["traj_uid"])
    idx = np.concatenate([np.arange(n), np.asarray(copies, dtype=np.int64)])
    out = permute(cols, idx)
    out["is_padding_row"] = np.concatenate([np.zeros(n, dtype=bool), np.ones(len(copies), dtype=bool)])
    if seed is not None:
        out = permute(out, np.random.default_rng(seed).permutation(len(idx)))
    return out


def by_key(cols, arr, real_only=True):
    """{(traj_uid, pv_t): value} over the real rows."""
    pad = cols.get("is_padding_row", np.zeros(len(arr), dtype=bool))
    return {(str(cols["traj_uid"][i]), int(cols["pv_t"][i])): arr[i]
            for i in range(len(arr)) if not (real_only and pad[i])}


def config(**kw):
    kw.setdefault("gamma", 1.0)
    kw.setdefault("lam", 1.0)
    return pv.ProgressValueConfig(**kw)


class StubTable(pv.ProgressValueTable):
    """A table that answers V(z) = values[k]: every row's k picks its value directly."""

    def __init__(self, cfg, values):
        super().__init__(cfg)
        self._values = values

    def has_mass(self, task):
        return True

    def value(self, cell, parent):
        return self._values[cell[1]]


def closed_form(values, reward, gamma, lam):
    """A_t = sum_l (gamma lam)^l (gamma V_{t+l+1} - V_{t+l}), V_T = R, written out as the double sum."""
    v = list(values) + [reward]
    T = len(values)
    return [sum((gamma * lam) ** l * (gamma * v[t + l + 1] - v[t + l]) for l in range(T - t)) for t in range(T)]


def mixed_batch():
    """Two ALFWorld groups, one WebShop group, one Search group: mixed, tied, different lengths, invalid turns."""
    return batch([
        trajectory("g1", "a1", 4, 10.0, k=[0, 1, 1, 2], stag=[0, 0, 1, 0], valid=[1, 0, 1, 1]),
        trajectory("g1", "a2", 6, 0.0, k=[0, 0, 1, 1, 1, 1], stag=[0, 1, 0, 1, 2, 3], valid=[1, 1, 0, 0, 1, 1]),
        trajectory("g1", "a3", 3, 10.0, k=[0, 1, 2], stag=[0, 0, 0], gamefile=GAMEFILE),
        trajectory("g2", "b1", 5, 0.0, k=[0, 0, 0, 1, 1], valid=[0, 1, 1, 1, 1], gamefile=GAMEFILE_TWO),
        trajectory("g2", "b2", 5, 0.0, k=[0, 0, 0, 0, 0], valid=[1, 1, 1, 1, 0], gamefile=GAMEFILE_TWO),
        trajectory("w1", "c1", 3, 10.0, "webshop", k=[0, 1, 2], stag=[0, 0, 0], valid=[1, 1, 1]),
        trajectory("w1", "c2", 5, 0.0, "webshop", k=[0, 1, 1, 1, 1], stag=[0, 0, 1, 2, 3], valid=[1, 0, 1, 1, 1]),
        trajectory("s1", "d1", 2, 1.0, "search", k=[0, 1], valid=[1, 1]),
        trajectory("s1", "d2", 4, 0.0, "search", k=[0, 0, 0, 0], valid=[1, 0, 1, 0]),
    ])


def trained_table(cfg, steps=3):
    """A table with mass in every task: the mixed batch scored and committed a few times."""
    table = pv.ProgressValueTable(cfg)
    for _ in range(steps):
        res = pv.compute_progress_value_advantage(mixed_batch(), table)
        table.stage(res.records)
        table.commit()
    return table


# --------------------------------------------------------------------------- #
# TD / GAE arithmetic
# --------------------------------------------------------------------------- #

HAND = {  # V = (0.2, 0.5, 0.7, 0.4), R = 1, worked by hand
    (1.0, 1.0): [0.8, 0.5, 0.3, 0.6],
    (1.0, 0.9): [0.6744, 0.416, 0.24, 0.6],
    (0.95, 1.0): [0.61450625, 0.357375, 0.2025, 0.55],
    (0.95, 0.9): [0.52591150625, 0.29346375, 0.15025, 0.55],
}


@pytest.mark.parametrize("gamma,lam", GAMMA_LAM)
def test_gae_matches_the_closed_form(gamma, lam):
    values = [0.2, 0.5, 0.7, 0.4]
    delta, adv = pv.gae(values, 1.0, gamma, lam)
    np.testing.assert_allclose(adv, HAND[(gamma, lam)], rtol=0, atol=1e-12)
    np.testing.assert_allclose(delta, [gamma * 0.5 - 0.2, gamma * 0.7 - 0.5, gamma * 0.4 - 0.7, gamma - 0.4],
                               atol=1e-12)
    for reward in (0.0, 1.0):
        _, adv = pv.gae(values, reward, gamma, lam)
        np.testing.assert_allclose(adv, closed_form(values, reward, gamma, lam), atol=1e-12)


@pytest.mark.parametrize("gamma,lam", GAMMA_LAM)
def test_the_estimator_puts_the_closed_form_on_every_row(gamma, lam):
    # Rows carry k = a distinct index per turn; the stub table answers V by k, so V is known per row.
    specs = [trajectory("g", "t1", 4, 10.0, k=[0, 1, 2, 3], valid=[1, 1, 1, 1]),
             trajectory("g", "t2", 3, 0.0, k=[4, 5, 6], valid=[1, 1, 1])]
    values = {0: 0.2, 1: 0.5, 2: 0.7, 3: 0.4, 4: 0.9, 5: 0.1, 6: 0.3}
    cfg = config(gamma=gamma, lam=lam, features={"alfworld": ("k",)})
    cols = batch(specs)
    res = pv.compute_progress_value_advantage(cols, StubTable(cfg, values))
    want = closed_form([0.2, 0.5, 0.7, 0.4], 1.0, gamma, lam) + closed_form([0.9, 0.1, 0.3], 0.0, gamma, lam)
    np.testing.assert_allclose(res.a_rl, 2.0 * np.array(want), atol=1e-12)
    np.testing.assert_allclose(res.value, [values[k] for k in range(7)], atol=0)
    assert not res.a_fmt.any()                  # every turn valid: the format term is exactly zero
    np.testing.assert_array_equal(res.advantage, res.a_rl)
    np.testing.assert_allclose(res.returns, res.a_rl / 2.0 + res.value, atol=0)


@pytest.mark.parametrize("gamma", [1.0, 0.95, 0.8])
def test_the_deltas_telescope(gamma):
    rng = np.random.default_rng(0)
    for _ in range(200):
        T = int(rng.integers(1, 30))
        values, reward = rng.random(T), float(rng.integers(0, 2))
        delta, _ = pv.gae(values, reward, gamma, float(rng.random()))
        assert math.isclose(float(np.sum(gamma ** np.arange(T) * delta)), gamma ** T * reward - values[0],
                            abs_tol=1e-12)


@pytest.mark.parametrize("gamma", [1.0, 0.95])
def test_lam_one_is_the_return_minus_the_value_for_any_table(gamma):
    # An arbitrary table: random targets thrown into the cells the batch visits, over several updates.
    cfg = config(gamma=gamma, lam=1.0, n0=3.0, retention=0.7)
    table = pv.ProgressValueTable(cfg)
    rng = np.random.default_rng(1)
    cols = mixed_batch()
    records = pv.compute_progress_value_advantage(cols, table).records
    for _ in range(4):
        table.update([(c, p, float(rng.random())) for c, p, _ in records if rng.random() < 0.7])
    res = pv.compute_progress_value_advantage(cols, table)
    assert not res.fallback.any()
    T = {}
    for tu in cols["traj_uid"]:
        T[tu] = T.get(tu, 0) + 1
    R = (cols["episode_rewards"] > 0).astype(float)
    want = np.array([gamma ** (T[tu] - t) for tu, t in zip(cols["traj_uid"], cols["pv_t"])]) * R - res.value
    np.testing.assert_allclose(res.a_rl / cfg.adv_scale, want, atol=1e-12)
    # ... and the deltas of each trajectory telescope onto gamma^T R - V_0 through the estimator too.
    for tu in T:
        rows = np.flatnonzero(cols["traj_uid"] == tu)
        rows = rows[np.argsort(cols["pv_t"][rows])]
        s = float(np.sum(gamma ** np.arange(len(rows)) * res.delta[rows]))
        assert math.isclose(s, gamma ** len(rows) * R[rows[0]] - res.value[rows[0]], abs_tol=1e-12)


def test_prefix_discount_weights_the_gae_term_by_gamma_t():
    cfg0 = config(gamma=0.9, lam=0.8)
    cfg1 = config(gamma=0.9, lam=0.8, prefix_discount=True)
    cols = mixed_batch()
    r0 = pv.compute_progress_value_advantage(cols, trained_table(cfg0))
    r1 = pv.compute_progress_value_advantage(cols, trained_table(cfg1))
    np.testing.assert_allclose(r1.a_rl, r0.a_rl * 0.9 ** cols["pv_t"], atol=1e-12)
    np.testing.assert_array_equal(r1.value, r0.value)


def test_eta_mixes_in_the_episode_term_with_a_trajectory_weighted_baseline():
    gamma, eta = 0.9, 0.5
    specs = [trajectory("g", "a", 2, 10.0), trajectory("g", "b", 5, 0.0), trajectory("g", "c", 1, 10.0)]
    cols = batch(specs)
    cfg1 = config(gamma=gamma, lam=0.7)
    cfgm = config(gamma=gamma, lam=0.7, eta=eta)
    r1 = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(cfg1))
    rm = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(cfgm))
    U = {"a": gamma ** 2, "b": 0.0, "c": gamma ** 1}
    # one weight per trajectory: b's five turns count once, not five times
    base = {"a": (U["b"] + U["c"]) / 2, "b": (U["a"] + U["c"]) / 2, "c": (U["a"] + U["b"]) / 2}
    ep = np.array([U[tu] - base[tu] for tu in cols["traj_uid"]])
    np.testing.assert_allclose(rm.a_rl, 2.0 * ((1 - eta) * ep + eta * r1.a_rl / 2.0), atol=1e-12)
    # eta = 1 (the default) is the single GAE, bit for bit
    r_default = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config(gamma=gamma, lam=0.7,
                                                                                         eta=1.0)))
    np.testing.assert_array_equal(r_default.a_rl, r1.a_rl)


def test_adv_scale_is_a_fixed_multiplier():
    cols = mixed_batch()
    r2 = pv.compute_progress_value_advantage(cols, trained_table(config(gamma=0.95, lam=0.9)))
    r5 = pv.compute_progress_value_advantage(cols, trained_table(config(gamma=0.95, lam=0.9, adv_scale=5.0)))
    np.testing.assert_allclose(r5.a_rl, r2.a_rl * 2.5, atol=1e-12)
    np.testing.assert_allclose(r5.returns, r2.returns, atol=1e-12)


# --------------------------------------------------------------------------- #
# Row order, padding copies, loud failures
# --------------------------------------------------------------------------- #

FIELDS = ("advantage", "a_rl", "a_fmt", "value", "returns", "delta", "fallback")


@pytest.mark.parametrize("gamma,lam", GAMMA_LAM)
def test_every_row_is_the_same_whatever_the_row_order(gamma, lam):
    cfg = config(gamma=gamma, lam=lam, eta=0.6, prefix_discount=True, format_scope="all")
    cols = mixed_batch()
    ref = pv.compute_progress_value_advantage(cols, trained_table(cfg))
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(cols["traj_uid"]))
        shuf = permute(cols, order)
        res = pv.compute_progress_value_advantage(shuf, trained_table(cfg))
        for f in FIELDS:
            assert by_key(shuf, getattr(res, f)) == by_key(cols, getattr(ref, f)), f
        assert res.records == ref.records
        assert res.metrics == ref.metrics


def test_the_empty_table_fallback_is_row_order_free_too():
    cols = mixed_batch()
    cfg = config(gamma=0.95, lam=0.9)
    ref = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(cfg))
    shuf = permute(cols, np.random.default_rng(7).permutation(len(cols["traj_uid"])))
    res = pv.compute_progress_value_advantage(shuf, pv.ProgressValueTable(cfg))
    for f in FIELDS:
        assert by_key(shuf, getattr(res, f)) == by_key(cols, getattr(ref, f)), f


def test_padding_copies_get_their_originals_values_and_count_nowhere():
    cfg = config(gamma=0.95, lam=0.9, format_scope="all")
    cols = mixed_batch()
    ref = pv.compute_progress_value_advantage(cols, trained_table(cfg))
    # copies of an invalid turn (row 1) and of a success's turns: counted, they would move the format
    # term's mean/std, the LOO baselines and the table update
    copies = [1, 1, 0, 5, 20, 22]
    for seed in (None, 3, 4):
        padded = with_padding(cols, copies, seed)
        res = pv.compute_progress_value_advantage(padded, trained_table(cfg))
        for f in FIELDS:
            assert by_key(padded, getattr(res, f)) == by_key(cols, getattr(ref, f)), f
            got = getattr(res, f)
            want = by_key(cols, getattr(ref, f))
            for i in np.flatnonzero(padded["is_padding_row"]):
                assert got[i] == want[(str(padded["traj_uid"][i]), int(padded["pv_t"][i]))], f
        assert res.records == ref.records
        assert res.metrics == ref.metrics


def test_a_gap_in_pv_t_raises():
    cols = batch([trajectory("g", "t", 3, 0.0)])
    cols["pv_t"] = np.array([0.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="missing or repeated"):
        pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config()))


def test_a_repeated_turn_raises():
    cols = batch([trajectory("g", "t", 3, 0.0)])
    cols["pv_t"] = np.array([0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="missing or repeated"):
        pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config()))


def test_a_trajectory_not_starting_at_zero_raises():
    cols = batch([trajectory("g", "t", 2, 0.0)])
    cols["pv_t"] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="missing or repeated"):
        pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config()))


@pytest.mark.parametrize("bad", [float("nan"), 0.5, -1.0])
def test_a_non_turn_pv_t_raises(bad):
    cols = batch([trajectory("g", "t", 2, 0.0)])
    cols["pv_t"] = np.array([0.0, bad])
    with pytest.raises(ValueError, match="not a turn index"):
        pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config()))


def test_a_padding_copy_without_an_original_raises():
    cols = with_padding(batch([trajectory("g", "t", 3, 0.0)]), [2])
    cols["pv_t"][-1] = 7.0
    with pytest.raises(ValueError, match="no real row"):
        pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config()))


def test_a_trajectory_longer_than_its_cap_raises():
    cols = batch([trajectory("g", "t", 5, 0.0, "search")])
    with pytest.raises(ValueError, match="above its cap"):
        pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config()))


@pytest.mark.parametrize("column,match", [("is_action_valid", "is_action_valid"),
                                          ("episode_rewards", "episode_rewards"), ("pv_cap", "pv_cap")])
def test_a_non_finite_input_on_a_real_row_raises(column, match):
    # a mixed group, outside the tied format scope: the check must not depend on the scope
    cols = batch([trajectory("g", "a", 2, 10.0), trajectory("g", "b", 2, 0.0)])
    cols[column] = cols[column].astype(object)
    cols[column][1] = float("nan")
    with pytest.raises(ValueError, match=match):
        pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config(format_scope="tied")))


def test_missing_columns_raise():
    cols = mixed_batch()
    for c in ("goal_capped", "gamefile", "pv_k_before", "pv_stag_before", "is_action_valid"):
        broken = {k: v for k, v in cols.items() if k != c}
        with pytest.raises(KeyError, match=c):
            pv.compute_progress_value_advantage(broken, pv.ProgressValueTable(config()))
    # Search alone reads neither the gamefile, goal_capped nor stag (its default features are k, rem)
    search = batch([trajectory("s", "d", 2, 1.0, "search")])
    for c in ("goal_capped", "gamefile", "pv_stag_before", "pv_k_after"):
        search.pop(c)
    pv.compute_progress_value_advantage(search, pv.ProgressValueTable(config()))
    assert pv.required_columns(config(), ["search"]) == [
        "traj_uid", "uid", "pv_t", "pv_cap", "task_name", "episode_rewards", "is_action_valid", "pv_k_before"]


def test_an_unknown_task_raises():
    cols = batch([trajectory("g", "t", 2, 0.0)])
    cols["task_name"] = np.array(["sokoban", "sokoban"], dtype=object)
    with pytest.raises(ValueError, match="sokoban"):
        pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config()))


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def test_the_table_is_frozen_within_a_step_and_updated_after_scoring():
    cfg = config(gamma=0.95, lam=0.9)
    table = pv.ProgressValueTable(cfg)
    cols = mixed_batch()
    first = pv.compute_progress_value_advantage(cols, table)
    # The batch's own outcomes are not in the table that scores it: step 1 runs on the fallback.
    assert first.fallback.all()
    empty = table.state_dict()
    again = pv.compute_progress_value_advantage(cols, table)
    for f in FIELDS:
        np.testing.assert_array_equal(getattr(again, f), getattr(first, f))
    table.stage(first.records)
    assert table.state_dict() == empty          # staged is not committed
    assert table.commit() is True
    assert table.pending is None
    after = pv.compute_progress_value_advantage(cols, table)
    assert not after.fallback.any()
    assert not np.array_equal(after.value, first.value)
    # A commit with nothing staged changes nothing -- no decay without a batch.
    state = table.state_dict()
    assert table.commit() is False
    assert table.state_dict() == state


def test_discard_drops_the_staged_batch_and_leaves_the_table_as_it_was():
    # A grad_probe batch: scored, staged, then dropped -- no decay, no add, the table unchanged.
    cfg = config()
    table = trained_table(cfg)
    state, updates = table.state_dict(), table.updates
    table.stage(pv.compute_progress_value_advantage(mixed_batch(), table).records)
    assert table.discard() is True
    assert table.pending is None and table.updates == updates
    assert table.state_dict() == state
    assert table.discard() is False                     # nothing staged
    assert table.commit() is False                      # and nothing left to commit


def test_a_second_stage_replaces_the_first():
    cfg = config()
    a, b = pv.ProgressValueTable(cfg), pv.ProgressValueTable(cfg)
    recs1 = pv.compute_progress_value_advantage(mixed_batch(), a).records
    recs2 = recs1[:5]
    a.stage(recs1)
    a.stage(recs2)
    a.commit()
    b.update(recs2)
    assert a.state_dict() == b.state_dict()


def test_counts_are_discounted_before_the_new_batch_is_added():
    cfg = config(retention=0.5)
    table = pv.ProgressValueTable(cfg)
    c1, p1 = ("search", 0, 0), ("search", "search", 0)
    c2, p2 = ("search", 1, 0), ("search", "search", 0)
    table.update([(c1, p1, 1.0), (c1, p1, 0.0), (c2, p2, 1.0)])
    assert table.cells[c1] == [1.0, 2.0] and table.cells[c2] == [1.0, 1.0]
    assert table.parents[p1] == [2.0, 3.0] and table.roots["search"] == [2.0, 3.0]
    table.update([(c2, p2, 0.0), (c2, p2, 1.0)])
    assert table.cells[c1] == [0.5, 1.0]                  # decayed, nothing added
    assert table.cells[c2] == [0.5 + 1.0, 0.5 + 2.0]      # decayed, then added
    assert table.parents[p1] == [1.0 + 1.0, 1.5 + 2.0]
    assert table.roots["search"] == [2.0, 3.5]
    assert table.updates == 2


def test_nodes_below_the_floor_are_dropped_and_the_task_falls_back():
    cfg = config(retention=0.01)
    table = pv.ProgressValueTable(cfg)
    cols = batch([trajectory("s", "d1", 2, 1.0, "search"), trajectory("s", "d2", 2, 0.0, "search")])
    table.update(pv.compute_progress_value_advantage(cols, table).records)
    assert table.has_mass("search")
    for _ in range(3):                                   # n = 4 * 0.01^3 = 4e-6: kept
        table.update([])
    assert table.has_mass("search")
    table.update([])                                     # 4e-8: dropped, every level
    assert not table.cells and not table.parents and not table.roots
    res = pv.compute_progress_value_advantage(cols, table)
    assert res.fallback.all()


def test_shrinkage_by_hand():
    cfg = config(n0=2.0, retention=1.0, features={"alfworld": ("type", "k", "rem")})
    table = pv.ProgressValueTable(cfg)
    A = ("alfworld", "pick_and_place_simple", 1, 0)
    B = ("alfworld", "pick_and_place_simple", 2, 0)
    C = ("alfworld", "pick_two_obj_and_place", 1, 3)
    P1 = ("alfworld", "pick_and_place_simple", 0)
    P2 = ("alfworld", "pick_two_obj_and_place", 3)
    table.update([(A, P1, 1.0), (A, P1, 0.0), (B, P1, 1.0), (C, P2, 1.0)])
    v_root = 3 / 4
    v_p1 = (2 + 2 * v_root) / (3 + 2)                    # 0.7
    v_p2 = (1 + 2 * v_root) / (1 + 2)                    # 5/6
    assert math.isclose(table.value(A, P1), (1 + 2 * v_p1) / (2 + 2), abs_tol=1e-15)   # 0.6
    assert math.isclose(table.value(B, P1), (1 + 2 * v_p1) / (1 + 2), abs_tol=1e-15)   # 0.8
    assert math.isclose(table.value(C, P2), (1 + 2 * v_p2) / (1 + 2), abs_tol=1e-15)   # 8/9
    assert math.isclose(table.value(A, P1), 0.6, abs_tol=1e-15)
    assert math.isclose(table.value(C, P2), 8 / 9, abs_tol=1e-15)
    # an unseen cell is its parent's value; an unseen parent is the root's
    assert math.isclose(table.value(("alfworld", "pick_and_place_simple", 0, 0), P1), 0.7, abs_tol=1e-15)
    unseen_parent = ("alfworld", "look_at_obj_in_light", 5)
    assert math.isclose(table.value(("alfworld", "look_at_obj_in_light", 0, 5), unseen_parent), 0.75,
                        abs_tol=1e-15)
    assert table.task_stats("alfworld") == (3, 4.0)
    assert not table.has_mass("webshop")


def test_rows_land_in_the_cells_their_columns_name():
    cfg = config()
    cols = batch([trajectory("g", "t", 12, 0.0, k=[0] * 5 + [1] * 7, stag=[0, 1, 2, 3, 4, 0, 1, 2, 3, 6, 10, 25],
                             gamefile=GAMEFILE_TWO)])
    recs = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(cfg)).records
    stag_b = [0, 1, 1, 2, 2, 0, 1, 1, 2, 3, 4, 5]
    rem_b = [0] * 10 + [1, 1]                            # rem = 50 - t > 40 for t <= 9; 40 and 39 are bucket 1
    for t, (cell, parent, y) in enumerate(recs):
        assert cell == ("alfworld", "pick_two_obj_and_place", 0 if t < 5 else 1, stag_b[t], rem_b[t])
        assert parent == ("alfworld", "pick_two_obj_and_place", rem_b[t])
        assert y == 0.0


def test_buckets_are_the_offline_checks():
    # evaluate_features.py's rb/sb, verbatim
    def rb(x, H):
        return 0 if x > 0.8 * H else 1 if x > 0.6 * H else 2 if x > 0.4 * H else 3 if x > 0.2 * H else \
            4 if x > 0.1 * H else 5

    def sb(s):
        return 0 if s == 0 else 1 if s < 3 else 2 if s < 6 else 3 if s < 10 else 4 if s < 20 else 5

    cfg = config()
    for H in (50, 15, 4):
        for t in range(H):
            assert cfg.rem_bucket(H - t, H) == rb(H - t, H), (H, t)
    for s in range(80):
        assert cfg.stag_bucket(float(s)) == sb(s), s
    assert cfg.stag_bucket(float("nan")) is None


def test_alfworld_type_from_the_gamefile_path():
    assert pv.alfworld_type(GAMEFILE) == "pick_and_place_simple"
    assert pv.alfworld_type("/home/x/data/alfworld/" + GAMEFILE_TWO) == "pick_two_obj_and_place"
    assert pv.alfworld_type("json_2.1.1/valid_seen/look_at_obj_in_light-Bowl-None-DeskLamp-301/t/game.tw-pddl") \
        == "look_at_obj_in_light"
    assert pv.alfworld_type("") == "unknown"
    assert pv.alfworld_type(None) == "unknown"


# --------------------------------------------------------------------------- #
# WebShop goals the environment cannot pay; the empty-table fallback
# --------------------------------------------------------------------------- #

def test_goal_capped_trajectories_have_zero_value_and_no_update():
    cfg = config(gamma=0.95, lam=0.9, format_scope="tied")
    specs = [trajectory("wc", "x1", 3, 0.0, "webshop", k=[0, 1, 2], capped=1.0, valid=[1, 0, 1]),
             trajectory("wc", "x2", 4, 0.0, "webshop", k=[0, 1, 1, 1], capped=1.0, valid=[1, 1, 1, 1]),
             trajectory("wo", "y1", 2, 10.0, "webshop", k=[0, 3], capped=0.0),
             trajectory("wo", "y2", 3, 0.0, "webshop", k=[0, 1, 1], capped=0.0)]
    cols = batch(specs)
    for table in (pv.ProgressValueTable(cfg), trained_table(cfg)):
        assert table.has_mass("webshop") == (table.updates > 0)
        res = pv.compute_progress_value_advantage(cols, table)
        capped = cols["traj_uid"] == "x1"
        capped |= cols["traj_uid"] == "x2"
        assert (res.value[capped] == 0.0).all()
        assert (res.delta[capped] == 0.0).all() and (res.a_rl[capped] == 0.0).all()
        assert not res.fallback[capped].any()
        # no record from a capped trajectory: 2 + 3 rows of the payable goal only
        assert len(res.records) == 5
        # the format term still applies to them (a tied group with one invalid turn)
        assert res.a_fmt[capped].any()
        assert res.metrics["progress_value/webshop/capped_share"] == 0.5


VALUE_SIDE_METRICS = ("fallback_share", "missing_feature_share", "mean_v", "calibration", "mean_abs_a_rl",
                      "a_rl_success", "a_rl_failure", "failure_pos_share", "delta_progress", "delta_stagnant")


def test_capped_rows_stay_out_of_the_value_side_metrics():
    """A capped goal's V, delta and A_rl are 0 by rule, not by the table: counted in, they pull every
    value-side mean towards 0 by the capped share and put fallback_share below 1 on an empty table.
    So the value-side metrics of a batch with a capped group are exactly those of the batch without it."""
    cfg = config(gamma=0.95, lam=0.9)
    payable = [trajectory("wo", "y1", 2, 10.0, "webshop", k=[0, 3], capped=0.0),
               trajectory("wo", "y2", 4, 0.0, "webshop", k=[0, 1, 1, 2], capped=0.0, valid=[1, 0, 1, 1]),
               trajectory("wp", "z1", 3, 0.0, "webshop", k=[0, 1, 2], capped=0.0),
               trajectory("wp", "z2", 3, 10.0, "webshop", k=[0, 1, 3], capped=0.0)]
    capped = [trajectory("wc", "x1", 5, 0.0, "webshop", k=[0, 1, 2, 2, 2], capped=1.0, valid=[1, 0, 1, 1, 1]),
              trajectory("wc", "x2", 6, 0.0, "webshop", k=[0, 1, 1, 1, 1, 1], capped=1.0)]
    p = "progress_value/webshop/"
    for table in (pv.ProgressValueTable(cfg), trained_table(cfg)):
        without = pv.compute_progress_value_advantage(batch(payable), table).metrics
        cols = batch(payable + capped)
        res = pv.compute_progress_value_advantage(cols, table)
        m = res.metrics
        for key in VALUE_SIDE_METRICS:
            assert m[p + key] == without[p + key], key
        # The counts and the format term still see every real row (the format term acts on capped rows).
        assert m[p + "rows"] == len(cols["traj_uid"]) and m[p + "trajectories"] == 6.0
        assert m[p + "capped_share"] == 2 / 6 and without[p + "capped_share"] == 0.0
        assert m[p + "mean_abs_a_fmt"] != without[p + "mean_abs_a_fmt"]
        # mean_v and calibration are over one population: their difference is the batch's mean target.
        scored = ~np.isin(cols["traj_uid"], ["x1", "x2"])
        y = [0.95 ** (T - t) * R for T, R in ((2, 1.0), (4, 0.0), (3, 0.0), (3, 1.0)) for t in range(T)]
        assert math.isclose(m[p + "mean_v"] - m[p + "calibration"], np.mean(y), rel_tol=1e-12)
        assert math.isclose(m[p + "mean_v"], float(np.mean(res.value[scored])), rel_tol=1e-12)
        # An empty table: every scored row is on the fallback, whatever share of the batch is capped.
        assert m[p + "fallback_share"] == (0.0 if table.has_mass("webshop") else 1.0)


def test_the_empty_table_falls_back_to_the_leave_one_out_group_mean():
    cfg = config(gamma=1.0, lam=1.0)
    specs = [trajectory("g", "a", 2, 10.0), trajectory("g", "b", 3, 0.0), trajectory("g", "c", 1, 0.0),
             trajectory("g", "d", 2, 10.0), trajectory("solo", "e", 2, 10.0)]
    cols = batch(specs)
    res = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(cfg))
    want = {"a": 1 / 3, "b": 2 / 3, "c": 2 / 3, "d": 1 / 3, "e": 0.0}
    for i, tu in enumerate(cols["traj_uid"]):
        assert math.isclose(res.value[i], want[tu], abs_tol=1e-15), tu
    assert res.fallback.all()
    assert res.metrics["progress_value/alfworld/fallback_share"] == 1.0
    assert res.metrics["progress_value/alfworld/table_mass"] == 0.0
    # lam = 1: the return minus that baseline
    R = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "e": 1.0}
    np.testing.assert_allclose(res.a_rl, [2.0 * (R[tu] - want[tu]) for tu in cols["traj_uid"]], atol=1e-15)
    table = pv.ProgressValueTable(cfg)
    table.stage(res.records)
    table.commit()
    res2 = pv.compute_progress_value_advantage(cols, table)
    assert res2.metrics["progress_value/alfworld/fallback_share"] == 0.0
    assert res2.metrics["progress_value/alfworld/table_mass"] == 10.0


def test_the_fallback_is_per_task():
    cfg = config()
    table = pv.ProgressValueTable(cfg)
    only_search = batch([trajectory("s", "d1", 2, 1.0, "search"), trajectory("s", "d2", 3, 0.0, "search")])
    table.update(pv.compute_progress_value_advantage(only_search, table).records)
    res = pv.compute_progress_value_advantage(mixed_batch(), table)
    cols = mixed_batch()
    is_search = cols["task_name"] == "search"
    assert not res.fallback[is_search].any() and res.fallback[~is_search].all()


# --------------------------------------------------------------------------- #
# The format term
# --------------------------------------------------------------------------- #

def _format_batch():
    return batch([
        # tied group (all fail), x = [0, -1, 0 | 0]: mean -1/4, std (ddof 1) 1/2
        trajectory("tied", "t1", 3, 0.0, valid=[1, 0, 1]),
        trajectory("tied", "t2", 1, 0.0, valid=[1]),
        # mixed group, x = [0, -1 | -1, 0, 0, 0]
        trajectory("mixed", "m1", 2, 10.0, valid=[1, 0]),
        trajectory("mixed", "m2", 4, 0.0, valid=[0, 1, 1, 1]),
        # uniform groups: every turn invalid / every turn valid
        trajectory("bad", "u1", 3, 0.0, valid=[0, 0, 0]),
        trajectory("bad", "u2", 2, 0.0, valid=[0, 0]),
        trajectory("good", "v1", 2, 10.0, valid=[1, 1]),
        trajectory("good", "v2", 3, 10.0, valid=[1, 1, 1]),
    ])


def test_the_format_term_tied_scope():
    cols = _format_batch()
    res = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config(format_scope="tied")))
    tu = cols["traj_uid"]
    np.testing.assert_allclose(res.a_fmt[np.isin(tu, ["t1", "t2"])], [0.5, -1.5, 0.5, 0.5], atol=1e-15)
    assert (res.a_fmt[np.isin(tu, ["m1", "m2"])] == 0.0).all()           # mixed: outside the scope
    # no phantom: a uniform group's term is exactly zero, not float32 round-off
    assert (res.a_fmt[np.isin(tu, ["u1", "u2", "v1", "v2"])] == 0.0).all()
    np.testing.assert_allclose(res.advantage, res.a_rl + res.a_fmt, atol=0)


def test_the_format_term_all_scope_and_its_coefficient():
    cols = _format_batch()
    res = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config(format_scope="all",
                                                                                  format_coef=0.5)))
    tu = cols["traj_uid"]
    x = np.array([0, -1, -1, 0, 0, 0], dtype=float)
    z = (x - x.mean()) / x.std(ddof=1)
    np.testing.assert_allclose(res.a_fmt[np.isin(tu, ["m1", "m2"])], z, atol=1e-15)
    np.testing.assert_allclose(res.a_fmt[np.isin(tu, ["t1", "t2"])], [0.5, -1.5, 0.5, 0.5], atol=1e-15)
    assert (res.a_fmt[np.isin(tu, ["u1", "u2", "v1", "v2"])] == 0.0).all()
    np.testing.assert_allclose(res.advantage, res.a_rl + 0.5 * res.a_fmt, atol=0)


def test_the_format_term_is_the_controls_grpo_z_in_a_tied_group():
    # The control's row score in an all-fail group: 0 - coef * invalid (ALFWorld/WebShop 0.1, Search 0.01);
    # GRPO's z over the group's turn rows (torch.std, ddof 1, + 1e-6).
    cols = _format_batch()
    res = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config()))
    rows = np.isin(cols["traj_uid"], ["t1", "t2"])
    for coef in (0.1, 0.01):
        s = -coef * (~cols["is_action_valid"][rows]).astype(np.float64)
        z = (s - s.mean()) / (s.std(ddof=1) + 1e-6)
        np.testing.assert_allclose(res.a_fmt[rows], z, rtol=3e-4)   # GRPO's 1e-6 is 2e-4 of 0.01 * 0.5


def test_a_one_row_group_gets_no_format_term():
    cols = batch([trajectory("one", "o", 1, 0.0, valid=[0])])
    res = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(config()))
    assert res.a_fmt[0] == 0.0


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def test_state_dict_round_trips_through_json():
    cfg = config(gamma=0.95, lam=0.9, n0=4.0, retention=0.8)
    table = trained_table(cfg)
    state = json.loads(json.dumps(table.state_dict()))
    fresh = pv.ProgressValueTable(config(gamma=0.95, lam=0.9, n0=4.0, retention=0.8))
    fresh.load_state_dict(state)
    assert fresh.state_dict() == table.state_dict()
    assert fresh.updates == 3
    cols = mixed_batch()
    a = pv.compute_progress_value_advantage(cols, table)
    b = pv.compute_progress_value_advantage(cols, fresh)
    for f in FIELDS:
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    # and both keep learning identically
    table.update(a.records)
    fresh.update(b.records)
    assert fresh.state_dict() == table.state_dict()


@pytest.mark.parametrize("change", [dict(n0=5.0), dict(retention=0.5), dict(gamma=0.9),
                                    dict(rem_buckets=(0.9, 0.5)), dict(stag_buckets=(1, 5)),
                                    dict(features={"search": ("rem",)})])
def test_a_table_saved_under_another_configuration_is_refused(change):
    base = dict(gamma=0.95, lam=0.9, n0=4.0, retention=0.8)
    state = json.loads(json.dumps(trained_table(config(**base)).state_dict()))
    other = pv.ProgressValueTable(config(**{**base, **change}))
    with pytest.raises(ValueError, match="another configuration"):
        other.load_state_dict(state)


def test_what_the_table_does_not_depend_on_is_not_in_the_fingerprint():
    base = dict(gamma=0.95, lam=0.9)
    state = json.loads(json.dumps(trained_table(config(**base)).state_dict()))
    other = pv.ProgressValueTable(config(gamma=0.95, lam=0.5, eta=0.3, adv_scale=1.0, prefix_discount=True,
                                         format_scope="all", format_coef=0.0))
    other.load_state_dict(state)
    assert other.updates == 3


def test_loading_nothing_leaves_the_table_empty_and_a_bad_version_raises():
    table = pv.ProgressValueTable(config())
    table.load_state_dict(None)
    table.load_state_dict({})
    assert not table.roots
    with pytest.raises(ValueError, match="version"):
        table.load_state_dict({"version": 99, "fingerprint": config().fingerprint()})


# --------------------------------------------------------------------------- #
# Configuration and metrics
# --------------------------------------------------------------------------- #

def test_config_from_the_yaml_node():
    cfg = pv.ProgressValueConfig.from_config(
        {"n0": 4, "retention": None, "features": {"webshop": ["k", "ongoal", "optnow", "rem"]},
         "format_scope": "all"}, gamma=0.95, lam=0.9)
    assert cfg.n0 == 4.0 and cfg.retention == 0.9 and cfg.format_scope == "all"
    assert cfg.features["webshop"] == ("k", "ongoal", "optnow", "rem")
    assert cfg.features["alfworld"] == ("type", "k", "stag", "rem")      # untouched tasks keep the default
    assert (cfg.gamma, cfg.lam, cfg.eta, cfg.adv_scale, cfg.format_coef) == (0.95, 0.9, 1.0, 2.0, 1.0)
    assert pv.required_columns(cfg, ["webshop"])[-4:] == ["pv_k_before", "pv_ongoal_b", "pv_optnow_b",
                                                          "goal_capped"]
    assert pv.ProgressValueConfig.from_config(None, gamma=1.0, lam=1.0).features == pv.DEFAULT_FEATURES


@pytest.mark.parametrize("bad", [dict(format_scope="mixed"), dict(eta=1.5), dict(n0=0.0), dict(retention=0.0),
                                 dict(gamma=0.0), dict(features={"alfworld": ("k", "colour")}),
                                 dict(rem_buckets=(0.2, 0.4)), dict(stag_buckets=(3, 1))])
def test_bad_config_raises(bad):
    with pytest.raises(AssertionError):
        config(**bad)


def test_current_state_features_are_read_from_their_columns():
    cfg = config(features={"webshop": ("k", "ongoal", "optnow", "rem")})
    cols = batch([trajectory("w", "c", 3, 0.0, "webshop", k=[0, 1, 2])])
    cols["pv_ongoal_b"] = np.array([0.0, 1.0, 1.0])
    cols["pv_optnow_b"] = np.array([0.0, 0.0, 1.0])
    recs = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(cfg)).records
    assert [r[0] for r in recs] == [("webshop", 0, 0, 0, 0), ("webshop", 1, 1, 0, 0), ("webshop", 2, 1, 1, 0)]


def test_buynow_is_keyed_in_quarters():
    """WebShop's default state includes the buy-now score, binned to 0..4 (the offline check's "bn");
    a missing score is a cell of its own."""
    assert pv.DEFAULT_FEATURES["webshop"] == ("k", "stag", "rem", "buynow")
    cfg = config()
    cols = batch([trajectory("w", "c", 5, 0.0, "webshop", k=[0, 1, 2, 2, 2], stag=[0, 0, 0, 1, 2],
                             buynow=[0.0, 0.1, 0.4, 0.88, float("nan")])])
    recs = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(cfg)).records
    assert [r[0][-1] for r in recs] == [0, 0, 2, 4, None]
    assert [r[0][:3] for r in recs][:3] == [("webshop", 0, 0), ("webshop", 1, 0), ("webshop", 2, 0)]


def test_metrics_by_hand():
    cfg = config(gamma=1.0, lam=1.0)
    specs = [trajectory("g", "a", 3, 10.0, k=[0, 1, 1], k_after=[1, 1, 2], valid=[1, 0, 1]),
             trajectory("g", "b", 2, 0.0, k=[0, 0], k_after=[0, 0], valid=[1, 1])]
    cols = batch(specs)
    m = pv.compute_progress_value_advantage(cols, pv.ProgressValueTable(cfg)).metrics
    p = "progress_value/alfworld/"
    # fallback: V = 0 on a's rows (b failed), 1 on b's (a won)
    assert m[p + "rows"] == 5.0 and m[p + "trajectories"] == 2.0
    assert m[p + "fallback_share"] == 1.0
    assert math.isclose(m[p + "mean_v"], 2 / 5)
    assert math.isclose(m[p + "calibration"], 2 / 5 - 3 / 5)
    assert math.isclose(m[p + "a_rl_success"], 2.0) and math.isclose(m[p + "a_rl_failure"], -2.0)
    assert m[p + "failure_pos_share"] == 0.0
    assert math.isclose(m[p + "mean_abs_a_rl"], 2.0)
    assert m[p + "mean_abs_a_fmt"] == 0.0            # a mixed group, tied scope
    # delta: a = (0, 0, 1) at V = 0 -> progress turns t=0 (0->1) and t=2 (1->2); b: V = 1 -> (0, -1)
    assert math.isclose(m[p + "delta_progress"], 0.5) and math.isclose(m[p + "delta_stagnant"], -1 / 3)
    assert m[p + "table_cells"] == 0.0 and m[p + "table_mass"] == 0.0
    assert m[p + "missing_feature_share"] == 0.0
    assert "progress_value/alfworld/capped_share" not in m
