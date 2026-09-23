"""The saturated-group gate (sat_gate) and the shadow columns (revisits, committed,
done_walkset).

WHAT IT PROTECTS.
  * sat_gate off: bit for bit what sat did before, Search included when listed.
  * sat_gate on: a task's saturated groups fire only while EMA(saturated share) >
    EMA(stuck share); a stuck-heavy step gives verdict "gate_closed" and adds
    nothing, a run of saturated-heavy steps opens it and the term fires; the
    condition and q are reported every step; the EMAs survive a state round-trip.
  * Search is held by the CONDITION, not by name: listed and stuck-heavy, it is
    gated; listed with the gate off, it fires.
  * Shadows: RevisitCounter counts exact repeats; AlfworldWalkSet.raw_done is the
    done count with no won => K; the Search manager writes revisits and committed;
    the loop carries the columns (NaN elsewhere); the controller records them per
    trajectory and reports win/fail revisits, fail_uncommitted and, in fired
    saturated groups, how often revisits order the winners like turns do.
No model and no GPU.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "verl")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from agent_system.environments import progress as P  # noqa: E402
from verl.trainer.ppo import progress_rank as pr  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def build(groups, resp=4):
    """Rows from ``[(uid, task, [(traj, turns, reward, k, K, adv, revisits, committed), ...]), ...]``."""
    cols = {k: [] for k in ("uids", "tuids", "tasks", "rew", "ks", "Ks", "adv", "rv", "cm")}
    for uid, task, trajs in groups:
        for traj, turns, r, k, K, a, rv, cm in trajs:
            for _ in range(turns):
                for key, v in zip(cols, (uid, traj, task, r, k, K, a, rv, cm)):
                    cols[key].append(v)
    n = len(cols["uids"])
    mask = torch.zeros(n, resp)
    mask[:, :3] = 1.0
    obj = lambda v: np.array(v, dtype=object)  # noqa: E731
    return dict(advantages=torch.tensor(cols["adv"], dtype=torch.float32).unsqueeze(-1) * mask, mask=mask,
                uids=obj(cols["uids"]), tuids=obj(cols["tuids"]), task_names=obj(cols["tasks"]),
                episode_rewards=obj(cols["rew"]), k_rows=obj(cols["ks"]), total_rows=obj(cols["Ks"]),
                real_rows=np.ones(n, dtype=bool), stat_rows=np.ones(n, dtype=bool),
                revisit_rows=np.array([float("nan") if v is None else float(v) for v in cols["rv"]]),
                committed_rows=np.array([float("nan") if v is None else float(v) for v in cols["cm"]]))


def rows_of(b, traj):
    return np.where(b["tuids"] == traj)[0]


# ALFWorld: a live group (the scales), a saturated group 4/8/12 turns whose revisits
# 0/2/5 order like turns, stuck groups to weigh the gate down.
LIVE = ("live", "alfworld", [("L1", 3, 10.0, 2, 2, 1.0, 0, None), ("L2", 3, 0.0, 1, 2, -1.0, 4, None)])
SAT = ("sat", "alfworld", [("S1", 4, 10.0, 2, 2, 0.0, 0, None), ("S2", 8, 10.0, 2, 2, 0.0, 2, None),
                           ("S3", 12, 10.0, 2, 2, 0.0, 5, None)])
STUCK = lambda i: (f"stuck{i}", "alfworld", [(f"F{i}a", 5, 0.0, 1, 2, 0.0, 3, None), (f"F{i}b", 5, 0.0, 0, 2, 0.0, 6, None)])  # noqa: E731
SAT2 = lambda i: (f"sat{i}", "alfworld", [(f"G{i}a", 4, 10.0, 2, 2, 0.0, 0, None), (f"G{i}b", 9, 10.0, 2, 2, 0.0, 1, None)])  # noqa: E731
# Search: a live group and a saturated group 1 vs 3 turns (spread 2 >= min 1), committed flags on.
SLIVE = ("slive", "search", [("Q1", 2, 1.0, 1, 1, 1.0, 0, True), ("Q2", 3, 0.0, 0, 1, -1.0, 1, True)])
SSAT = ("ssat", "search", [("W1", 1, 1.0, 1, 1, 0.0, 0, True), ("W2", 3, 1.0, 1, 1, 0.0, 1, True)])
SSTUCK = lambda i: (f"sstuck{i}", "search", [(f"X{i}a", 4, 0.0, 1, 1, 0.0, 2, False), (f"X{i}b", 4, 0.0, 0, 1, 0.0, 1, True)])  # noqa: E731


def run(ctl, groups):
    b = build(groups)
    new, m = ctl.apply(**b)
    return b, new, m


def sat_delta(b, new, trajs):
    return float((new - b["advantages"])[np.concatenate([rows_of(b, t) for t in trajs])].abs().sum())


print("1. gate off: what sat did before, Search included when listed")
ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_tasks=["alfworld", "webshop", "search"])
b, new, m = run(ctl, [LIVE, SAT, STUCK(1), STUCK(2), STUCK(3), SLIVE, SSAT, SSTUCK(1), SSTUCK(2)])
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["sat"]["sat_verdict"] == "fired" and sat_delta(b, new, ["S1", "S2", "S3"]) > 0,
      "ALFWorld's saturated group fires on a stuck-heavy step")
check(recs["ssat"]["sat_verdict"] == "fired" and sat_delta(b, new, ["W1", "W2"]) > 0,
      "...and so does Search's, by name")
check(m["progress_rank/alfworld/sat_gate_condition"] == 0.0 and m["progress_rank/alfworld/sat_gated_out"] == 0.0,
      f"the condition is reported (closed, q={m['progress_rank/alfworld/sat_gate_q']:.2f}) but not enforced")

print("2. gate on: a stuck-heavy step is held, saturated-heavy steps open it")
ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_tasks=["alfworld", "webshop", "search"], sat_gate=True)
b, new, m = run(ctl, [LIVE, SAT, STUCK(1), STUCK(2), STUCK(3), SLIVE, SSAT, SSTUCK(1), SSTUCK(2)])
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["sat"]["sat_verdict"] == "gate_closed" and sat_delta(b, new, ["S1", "S2", "S3"]) == 0.0,
      "ALFWorld: verdict gate_closed, nothing added")
check(recs["ssat"]["sat_verdict"] == "gate_closed" and sat_delta(b, new, ["W1", "W2"]) == 0.0,
      "Search: held by the condition")
check(m["progress_rank/alfworld/sat_gated_out"] == 1.0 and m["progress_rank/alfworld/sat_gate_stuck_ema"] > m["progress_rank/alfworld/sat_gate_sat_ema"],
      "metrics: gated out, stuck EMA above saturated EMA")
check(m.get("progress_rank/alfworld/sat_gate_closed") == 1.0, "the verdict is counted")
opened_at = None
for step in range(2, 40):
    b, new, m = run(ctl, [LIVE, SAT, SAT2(step), SAT2(step + 100), SAT2(step + 200), SLIVE, SSAT, SSTUCK(1), SSTUCK(2)])
    if m["progress_rank/alfworld/sat_gate_condition"] == 1.0:
        opened_at = step
        break
recs = {r["uid"]: r for r in ctl.last_group_records}
check(opened_at is not None and recs["sat"]["sat_verdict"] == "fired" and sat_delta(b, new, ["S1", "S2", "S3"]) > 0,
      f"ALFWorld opens after {opened_at} saturated-heavy steps and fires")
check(recs["ssat"]["sat_verdict"] == "gate_closed", "Search, still stuck-heavy, stays held on the same step")
check(0.0 <= m["progress_rank/alfworld/sat_gate_q"] < 0.5, f"q < 1/2 when open ({m['progress_rank/alfworld/sat_gate_q']:.2f})")

print("3. the EMAs survive a state round-trip")
state = ctl.state_dict()
ctl2 = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_tasks=["alfworld", "webshop", "search"], sat_gate=True)
ctl2.load_state_dict(state)
check(state["version"] == 4 and ctl2.gate_stuck_ema == ctl.gate_stuck_ema and ctl2.gate_sat_ema == ctl.gate_sat_ema,
      "version 4 carries both gate EMAs")
ctl3 = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_gate=True)
ctl3.load_state_dict({"version": 3, "ema": {}, "success_ema": {}})
check(all(v is None for v in ctl3.gate_stuck_ema.values()), "a version-3 state loads with the gate EMAs unset")

print("4. the shadows in the controller")
b, new, m = run(ctl, [LIVE, SAT, SAT2(1), SAT2(2), SAT2(3), SLIVE, SSAT, SSTUCK(1), SSTUCK(2)])
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["sat"]["revisits"] == [0.0, 2.0, 5.0] and recs["sstuck1"]["committed"] == [False, True],
      "records carry revisits and committed per trajectory")
check(m["progress_rank/alfworld/sat_revisit_agreement"] == 1.0, "revisits order the 4/8/12-turn winners like turns: agreement 1")
check(abs(m["traj/search/fail_uncommitted"] - 0.4) < 1e-9, "Search failures that never answered: 2 of 5")
check("traj/alfworld/fail_uncommitted" not in m, "ALFWorld has no terminal action: nothing reported")
check(abs(m["traj/alfworld/win_revisits"] - np.mean([0, 0, 2, 5, 0, 1, 0, 1, 0, 1])) < 1e-9, "win_revisits is the winners' mean")
REV = ("rev", "alfworld", [("R1", 4, 10.0, 2, 2, 0.0, 5, None), ("R2", 12, 10.0, 2, 2, 0.0, 0, None)])
b, new, m = run(ctl, [LIVE, REV, SAT2(1), SAT2(2), SAT2(3)])
check(m["progress_rank/alfworld/sat_revisit_agreement"] < 1.0, "a group where the fast winner revisited more lowers the agreement")

print("4b. what the gate held back, and where the base gradient sits")
ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_tasks=["alfworld", "webshop", "search"], sat_gate=True)
b, new, m = run(ctl, [LIVE, SAT, STUCK(1), STUCK(2), STUCK(3), SLIVE, SSAT, SSTUCK(1), SSTUCK(2)])
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["sat"]["sat_verdict"] == "gate_closed" and recs["sat"]["sat_verdict_ungated"] == "fired",
      "a held group keeps what it would have been (sat_verdict_ungated = fired)")
check(m.get("progress_rank/alfworld/sat_fired_ungated") == 1.0 and m.get("progress_rank/alfworld/sat_gate_kept_share") == 0.0,
      "one group would have fired, the gate let none through: kept share 0")
check("progress_rank/alfworld/stuck_base_mass_share" in m and "progress_rank/alfworld/saturated_base_mass_share" in m,
      "base |A| mass shares by group kind are reported")
check(all("_base_mass" not in r for r in ctl.last_group_records), "the working column does not leak into the records")
# a batch whose stuck groups carry base |A| (the format penalty's effect) and a live group
FMT_STUCK = ("fmt", "alfworld", [("P1", 5, 0.0, 1, 2, 2.0, 0, None), ("P2", 5, 0.0, 0, 2, -2.0, 0, None)])
ctl = pr.ProgressRankController(rho=0.0)
b, new, m = run(ctl, [LIVE, FMT_STUCK])
# live: 3 rows x |1| + 3 rows x |1| = 6 rows x 3 tokens; stuck: 10 rows x |2| x 3 tokens
check(abs(m["progress_rank/alfworld/stuck_base_mass_share"] - (60.0 / (60.0 + 18.0))) < 1e-9,
      f"stuck share of the base mass is what it carries: {m['progress_rank/alfworld/stuck_base_mass_share']:.3f}")

print("4c. the scale modes: the order inside a group never moves, the weight between groups does")
def with_doc(b, doc):
    b = dict(b); b["doc_len_rows"] = np.array([doc.get(t, np.nan) for t in b["tuids"]], dtype=float); return b
SHORT = ("short", "alfworld", [("A1", 4, 10.0, 2, 2, 0.0, 0, None), ("A2", 8, 10.0, 2, 2, 0.0, 0, None)])
LONG = ("long", "alfworld", [("B1", 4, 10.0, 2, 2, 0.0, 0, None), ("B2", 8, 10.0, 2, 2, 0.0, 0, None)])
doc = {"A1": 4, "A2": 4, "B1": 8, "B2": 8}
out = {}
for mode in ("task_constant", "document", "group_mean"):
    ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_turn_scale_mode=mode, cap_kappa=100.0)
    b = with_doc(build([LIVE, SHORT, LONG]), doc)
    new, m = ctl.apply(**b)
    d = (new - b["advantages"])[:, 0].numpy()
    out[mode] = {t: float(d[rows_of(b, t)][0]) for t in ("A1", "A2", "B1", "B2")}
    recs = {r["uid"]: r for r in ctl.last_group_records}
    check(out[mode]["A1"] > 0 > out[mode]["A2"] and out[mode]["B1"] > 0 > out[mode]["B2"],
          f"{mode}: the faster winner goes up in both groups")
    if mode == "document":
        check(recs["short"]["sat_scale"] == 4.0 and recs["long"]["sat_scale"] == 8.0, "document: each group's own length")
check(abs(out["task_constant"]["A1"] - out["task_constant"]["B1"]) < 1e-9, "task_constant: equal turn gaps weigh the same")
check(abs(out["document"]["A1"] / out["document"]["B1"] - 2.0) < 1e-6, "document: the 4-line game weighs twice the 8-line one")
check(abs(out["group_mean"]["A1"] - out["group_mean"]["B1"]) < 1e-9, "group_mean: equal mean turns, equal weight")
try:
    pr.ProgressRankController(rho=0.0, sat_turn_scale_mode="turns")
    check(False, "an unknown scale mode is refused")
except AssertionError:
    check(True, "an unknown scale mode is refused")
b = build([LIVE, SHORT]); b["gamefile_rows"] = np.array(["/g/short.tw-pddl"] * len(b["tuids"]), dtype=object)
ctl = pr.ProgressRankController(rho=0.0)
ctl.apply(**b)
check({r["uid"]: r["gamefile"] for r in ctl.last_group_records}["short"] == "/g/short.tw-pddl", "records carry the group's game file")

print("4d. the placebo: same groups, same zero-sum, same mass -- the ranking's content destroyed")
# Three saturated groups with three distinct winners each, so a permutation can move something.
P3 = lambda i, a, b, c: (f"p{i}", "alfworld", [(f"P{i}a", a, 10.0, 2, 2, 0.0, 0, None),  # noqa: E731
                                              (f"P{i}b", b, 10.0, 2, 2, 0.0, 0, None),
                                              (f"P{i}c", c, 10.0, 2, 2, 0.0, 0, None)])
BATCH = [LIVE, P3(1, 4, 8, 14), P3(2, 5, 9, 12), P3(3, 3, 7, 16), P3(4, 6, 10, 13)]
true_ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, cap_kappa=100.0)
b, new_true, m_true = run(true_ctl, BATCH)
fake_ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, cap_kappa=100.0, sat_placebo="shuffle")
b2, new_fake, m_fake = run(fake_ctl, BATCH)
rt = {r["uid"]: r for r in true_ctl.last_group_records}
rf = {r["uid"]: r for r in fake_ctl.last_group_records}
pg = [f"p{i}" for i in range(1, 5)]
check(all(rt[u]["sat_verdict"] == rf[u]["sat_verdict"] == "fired" for u in pg), "the same groups fire")
zs, l1 = [], []
for u in pg:
    w = np.asarray(rf[u]["turns"], dtype=float)
    s_fake, s_true = np.asarray(rf[u]["sat_score"]), np.asarray(rf[u]["sat_score_true"])
    zs.append(abs(float((w * s_fake).sum())))
    l1.append(abs(float((w * np.abs(s_fake)).sum() - (w * np.abs(s_true)).sum())))
check(max(zs) < 1e-12, f"turn-weighted zero-sum kept in every group (max |sum w s| {max(zs):.1e})")
check(max(l1) < 1e-12, f"turn-weighted L1 mass kept in every group (max diff {max(l1):.1e})")
check(all(np.allclose(rf[u]["sat_score_true"], rt[u]["sat_score"]) for u in pg),
      "the records keep the true ranking beside the applied one")
check(any(not np.allclose(rf[u]["sat_score"], rf[u]["sat_score_true"]) for u in pg),
      "and the applied scores differ from it in at least one group")
mass_true = float((new_true - b["advantages"]).abs().sum())
mass_fake = float((new_fake - b2["advantages"]).abs().sum())
check(abs(mass_true - mass_fake) / mass_true < 1e-6,
      f"the injected |dA| mass matches the true arm's ({mass_fake:.6f} vs {mass_true:.6f})")
check(abs(m_true["progress_rank/alfworld/sat_score_turn_corr"] + 1.0) < 1e-9,
      "true ranking: applied score vs turns correlates at -1")
check(m_fake["progress_rank/alfworld/sat_score_turn_corr"] > -0.99,
      f"placebo: the correlation is broken ({m_fake['progress_rank/alfworld/sat_score_turn_corr']:+.2f})")
again = pr.ProgressRankController(rho=0.0, sat_rho=0.2, cap_kappa=100.0, sat_placebo="shuffle")
b3, new_again, _ = run(again, BATCH)
check(torch.equal(new_again, new_fake), "deterministic: the same groups draw the same permutation (a resume redraws it)")
check(all(r["sat_placebo"] == "none" for r in true_ctl.last_group_records)
      and all(r["sat_placebo"] == "shuffle" for r in fake_ctl.last_group_records), "records name the arm")
try:
    pr.ProgressRankController(rho=0.0, sat_placebo="random")
    check(False, "an unknown placebo mode is refused")
except AssertionError:
    check(True, "an unknown placebo mode is refused")
yaml_cfg = OmegaConf.load(os.path.join(REPO, "verl/trainer/config/ppo_trainer.yaml"))
check(yaml_cfg.algorithm.progress_rank.sat_placebo == "none", "the config default is none: every existing arm is unchanged")

print("4e. the dose-matched placebo: the APPLIED token mass equals the true ranking's, cap binding")
# Real runs bind the cap (kappa 0.5) and rows differ in length, which is where "shuffle" drifted.
def build_tok(groups, tok):
    """build() with a per-trajectory number of response tokens per row (1..resp)."""
    b = build(groups)
    m = torch.zeros_like(b["mask"])
    for i, t in enumerate(b["tuids"]):
        m[i, :tok.get(str(t), 3)] = 1.0
    b["advantages"] = b["advantages"][:, :1] * m
    b["mask"] = m
    return b
TOK = {"P1a": 1, "P1b": 4, "P1c": 2, "P2a": 4, "P2b": 1, "P2c": 3, "P3a": 2, "P3b": 4, "P3c": 1, "P4a": 3, "P4b": 2, "P4c": 4}
# A live group with a small success push S, so kappa * S / max|s| binds under a strong sat_rho.
LIVE_S = ("live", "alfworld", [("L1", 3, 10.0, 2, 2, 0.1, 0, None), ("L2", 3, 0.0, 1, 2, -0.1, 4, None)])
BATCH_CAP = [LIVE_S, P3(1, 4, 8, 14), P3(2, 5, 9, 12), P3(3, 3, 7, 16), P3(4, 6, 10, 13)]
def token_mass(ctl):
    bb = build_tok(BATCH_CAP, TOK)
    new, mm = ctl.apply(**bb)
    rows = np.array([str(x).startswith("P") for x in bb["tuids"]])
    d = (new - bb["advantages"]).abs() * bb["mask"]
    return float(d[torch.from_numpy(rows)].sum()), mm, ctl
mt, m_t, c_t = token_mass(pr.ProgressRankController(rho=0.0, sat_rho=2.0, cap_kappa=0.5))
ms, m_s, _ = token_mass(pr.ProgressRankController(rho=0.0, sat_rho=2.0, cap_kappa=0.5, sat_placebo="shuffle"))
md, m_d, c_d = token_mass(pr.ProgressRankController(rho=0.0, sat_rho=2.0, cap_kappa=0.5, sat_placebo="shuffle_dose"))
check(m_t.get("progress_rank/alfworld/sat_capped", 0.0) == 1.0, "the cap binds in this batch (as in the real runs)")
check(abs(ms - mt) / mt > 0.01, f"the old shuffle placebo drifts off the true dose ({ms / mt:.3f} of it)")
check(abs(md - mt) / mt < 1e-6, f"shuffle_dose applies the true dose ({md:.6f} vs {mt:.6f})")
check(abs(m_d["progress_rank/alfworld/sat_placebo_mass_ratio"] - 1.0) < 1e-9, "and says so in its metric (mass ratio 1)")
check("progress_rank/alfworld/sat_placebo_peak_over_cap" in m_d, "its peak push is reported, not capped")
check(m_d["progress_rank/alfworld/sat_score_turn_corr"] > -0.99, "the content is still destroyed (correlation broken)")
rd = {r["uid"]: r for r in c_d.last_group_records}
check(all(rd[u]["sat_placebo"] == "shuffle_dose" for u in pg), "records name the dose-matched arm")
_, _, c_d2 = token_mass(pr.ProgressRankController(rho=0.0, sat_rho=2.0, cap_kappa=0.5, sat_placebo="shuffle_dose"))
check(all(np.allclose(rd[u]["sat_score"], {r["uid"]: r for r in c_d2.last_group_records}[u]["sat_score"]) for u in pg),
      "deterministic across instances (a resume redraws the same permutation)")

print("4f. the sign placebo: every magnitude of the true arm, only the direction randomised")
P8 = [P3(i, 3 + i % 3, 8 + i % 4, 13 + i % 5) for i in range(1, 9)]
TOK8 = {f"P{i}{x}": 1 + (i + j) % 4 for i in range(1, 9) for j, x in enumerate("abc")}
def run_tok(ctl, groups, tok):
    bb = build_tok(groups, tok)
    new, mm = ctl.apply(**bb)
    return bb, new, mm, ctl
B8 = [LIVE_S] + P8
bt, nt, mt8, ct = run_tok(pr.ProgressRankController(rho=0.0, sat_rho=2.0, cap_kappa=0.5), B8, TOK8)
bs, ns, ms8, cs = run_tok(pr.ProgressRankController(rho=0.0, sat_rho=2.0, cap_kappa=0.5, sat_placebo="sign"), B8, TOK8)
check(mt8.get("progress_rank/alfworld/sat_capped", 0.0) == 1.0, "the cap binds here too")
dt = ((nt - bt["advantages"]) * bt["mask"]); ds = ((ns - bs["advantages"]) * bs["mask"])
check(torch.allclose(dt.abs(), ds.abs(), atol=1e-7), "every row's |dA| is exactly the true arm's (mass, peak and cap identical)")
check(abs(ms8["progress_rank/alfworld/sat_c"] - mt8["progress_rank/alfworld/sat_c"]) < 1e-12, "the same coefficient")
rs8 = {r["uid"]: r for r in cs.last_group_records}
signs = [rs8[f"p{i}"].get("sat_placebo_sign") for i in range(1, 9)]
check(1.0 in signs and -1.0 in signs, f"some groups keep the direction, some flip it ({signs})")
check(all(np.allclose(np.asarray(rs8[f"p{i}"]["sat_score"]), signs[i - 1] * np.asarray(rs8[f"p{i}"]["sat_score_true"])) for i in range(1, 9)),
      "each group's applied scores are its true scores times its coin")
check(abs(ms8["progress_rank/alfworld/sat_score_turn_corr"]) < 1.0, "the average score-turns correlation is no longer -1")
check(abs(ms8["progress_rank/alfworld/sat_placebo_flip_share"] - signs.count(-1.0) / 8) < 1e-12,
      f"the flip share is reported ({ms8['progress_rank/alfworld/sat_placebo_flip_share']:.3f})")
check(all(r.get("sat_placebo_sign") is None for r in ct.last_group_records)
      and "progress_rank/alfworld/sat_placebo_flip_share" not in mt8, "the true arm carries no coin")
_, ns2, _, _ = run_tok(pr.ProgressRankController(rho=0.0, sat_rho=2.0, cap_kappa=0.5, sat_placebo="sign"), B8, TOK8)
check(torch.equal(ns2, ns), "deterministic (a resume redraws the same coins)")

print("5. the counters")
rc = P.RevisitCounter()
for a in ("go to cabinet 1", "Go To  Cabinet 1", "go to cabinet 2", "", "go to cabinet 2"):
    rc.step(a)
check(rc.revisits == 2, "exact repeats (case / spaces folded, instance numbers kept): 2")
w = P.AlfworldWalkSet(["go to desk 1", "take bowl 1 from desk 1", "go to dresser 1", "use desklamp 1"])
w.step("go to desk 1", "You arrive at desk 1.")
w.step("take bowl 1 from desk 1", "You pick up the bowl 1.", won=True)
check(w.k == w.total == 4 and w.raw_done == 2, "won => k = K, but raw_done stays at the 2 lines done")

print("6. the Search manager writes revisits and committed; the loop carries them")
from agent_system.environments.env_manager import SearchEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.search.projection import search_projection  # noqa: E402


class _Envs:
    group_n = 2
    is_train = True

    def reset(self, kwargs=None):
        return ["q"] * 2, [{} for _ in range(2)]

    def step(self, actions):
        return ["<information> Paris. </information>"] * 2, [0.0, 0.0], [False, False], [{"won": 0.0} for _ in actions]


cfg = OmegaConf.create({"env": {"history_length": 0, "rollout": {"n": 2}},
                        "algorithm": {"progress_rank": {"enable": True}, "oci_sat": {"enable": False},
                                      "oci_slots": {"enable": False}, "oci_rank": {"enable": False}}})
mgr = SearchEnvironmentManager(_Envs(), search_projection, cfg)
mgr.reset([{"question": "q", "ground_truth": {"target": ["Paris"]}}] * 2)
_, _, _, i1 = mgr.step(["<search> capital of france </search>", "<search> a </search>"])
_, _, _, i2 = mgr.step(["<search> capital of france </search>", "<answer> Paris </answer>"])
check([i["revisits"] for i in i2] == [1, 0] and [i["committed"] for i in i2] == [False, True],
      "row 0 repeated its query (revisits 1), row 1 answered (committed)")
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector  # noqa: E402
from verl import DataProto  # noqa: E402

c = TrajectoryCollector.__new__(TrajectoryCollector)
c._progress_rank_on = True
c._queue_row_for_prefetch = lambda *a: None
batch = DataProto.from_single_dict({"input_ids": torch.zeros(2, 3, dtype=torch.long),
                                    "traj_uid": np.array(["a", "b"], dtype=object)})
tbl, tinf = [[], []], [[], []]
c._record_turn(batch=batch, active_idx=np.array([0, 1]), active_masks=np.array([True, True]),
               infos=[dict(i2[1]), {"progress_done_walkset": 3, "revisits": 7}], traj_uid=np.array(["a", "b"], dtype=object),
               total_batch_list=tbl, total_infos=tinf, batch_size=2)
r0, r1 = tbl[0][0], tbl[1][0]
check(r0["committed"] == 1.0 and r0["revisits"] == 0.0 and np.isnan(r0["progress_done_walkset"]),
      "a Search row: committed 1, revisits 0, no walkthrough count")
check(np.isnan(r1["committed"]) and r1["revisits"] == 7.0 and r1["progress_done_walkset"] == 3.0,
      "an ALFWorld-like row: no committed flag, revisits and the walkthrough count")

print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
