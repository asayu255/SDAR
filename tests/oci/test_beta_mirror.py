"""scale_mode=beta_mirror: the per-task Beta from the tied-group shares and the strength it sets.

WHAT IT PROTECTS.
  * The fit: (a, b) -> (q_f, q_s) -> (a, b) round-trips; the residual is recomputed at the parameters
    returned; pairs no finite Beta can produce ((1, .01), (.5, .5)) are reported as not converged.
  * The joint smoothing turns every raw pair -- all-fail (1, 0) included -- into a solvable one, and the
    gate's condition on the smoothed shares equals a > b.
  * The strengths: m_f(a, b) = m_s(b, a), both in [0, 1], and equal to a Monte Carlo estimate of the
    posterior E[2 sqrt(p(1-p))].
  * apply(): per TASK normalisation -- the mean |push| over the trajectories of the fired groups is m
    (a zero score inside a fired group counts, padding copies do not), bigger spreads push harder, each
    trajectory's rows get c * score; Search's saturated groups fire without the gate; a failed fit adds
    nothing and says so; the budget_cap counterfactual and the diagnostics are reported; the state
    round-trips and a resumed controller equals a continuous one; what would silently mix is refused.
  * The default (budget_cap) is untouched: no beta metrics or record fields appear.
No model and no GPU.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "verl")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from verl.trainer.ppo import progress_rank as pr  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def build(groups, resp=4, pad=()):
    """Rows from ``[(uid, task, [(traj, turns, reward, k, K, adv, revisits, committed), ...]), ...]``;
    ``pad`` names trajectories whose first row is appended again as a padding copy (real_rows False)."""
    cols = {k: [] for k in ("uids", "tuids", "tasks", "rew", "ks", "Ks", "adv", "rv", "cm", "real")}
    for uid, task, trajs in groups:
        for traj, turns, r, k, K, a, rv, cm in trajs:
            for _ in range(turns):
                for key, v in zip(cols, (uid, traj, task, r, k, K, a, rv, cm, True)):
                    cols[key].append(v)
    for uid, task, trajs in groups:
        for traj, turns, r, k, K, a, rv, cm in trajs:
            if traj in pad:
                for key, v in zip(cols, (uid, traj, task, r, k, K, a, rv, cm, False)):
                    cols[key].append(v)
    n = len(cols["uids"])
    mask = torch.zeros(n, resp)
    mask[:, :3] = 1.0
    obj = lambda v: np.array(v, dtype=object)  # noqa: E731
    real = np.array(cols["real"], dtype=bool)
    return dict(advantages=torch.tensor(cols["adv"], dtype=torch.float32).unsqueeze(-1) * mask, mask=mask,
                uids=obj(cols["uids"]), tuids=obj(cols["tuids"]), task_names=obj(cols["tasks"]),
                episode_rewards=obj(cols["rew"]), k_rows=obj(cols["ks"]), total_rows=obj(cols["Ks"]),
                real_rows=real, stat_rows=real.copy(),
                revisit_rows=np.array([float("nan") if v is None else float(v) for v in cols["rv"]]),
                committed_rows=np.array([float("nan") if v is None else float(v) for v in cols["cm"]]))


def beta_ctl(**kw):
    args = dict(rho=0.1, sat_rho=0.1, sat_tasks=["alfworld", "webshop", "search"], scale_mode="beta_mirror")
    args.update(kw)
    return pr.ProgressRankController(**args)


G = 8

print("1. the fit round-trips and the residual is honest")
worst = 0.0
for a in (0.05, 0.13, 0.3, 1.0, 3.0):
    for b in (0.05, 0.09, 0.48, 1.0, 3.0):
        lf, ls = pr.beta_tied_log_shares(a, b, G)
        qf, qs = math.exp(lf), math.exp(ls)
        fa, fb, res, conv = pr.fit_beta_ends(qf, qs, group_size=G)
        worst = max(worst, abs(math.log(fa / a)), abs(math.log(fb / b)))
        check(conv and res < 1e-8, f"Beta({a}, {b}) -> q=({qf:.4f}, {qs:.4f}) -> ({fa:.4f}, {fb:.4f}), residual {res:.1e}") if (a, b) in ((0.13, 0.75), (0.3, 0.09), (1.0, 1.0)) else None
check(worst < 1e-5, f"every grid point recovers (a, b) (worst |log ratio| {worst:.1e})")
lf, ls = pr.beta_tied_log_shares(0.13, 0.75, G)
_, _, res_cold, conv_cold = pr.fit_beta_ends(math.exp(lf), math.exp(ls), group_size=G, init=(50.0, 0.001))
check(conv_cold and res_cold < 1e-8, "a far-off warm start still converges (fallback starts)")
for pair in ((1.0, 0.01), (0.5, 0.5), (0.0, 0.3), (0.6, 0.45)):
    _, _, res, conv = pr.fit_beta_ends(*pair, group_size=G)
    check(not conv, f"{pair} has no finite Beta: reported as not converged (residual {res})")

print("2. the joint smoothing makes every raw pair solvable, and the gate equals a > b on it")
for raw in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.0, 0.0), (0.74, 0.01), (0.002, 0.008)):
    qf, qs = pr.regularize_tied_shares(*raw, n_eff=75.0, pseudo_count=1.0)
    a, b, res, conv = pr.fit_beta_ends(qf, qs, group_size=G)
    check(0.0 < qf and 0.0 < qs and qf + qs < 1.0 and conv,
          f"raw {raw} -> ({qf:.4f}, {qs:.4f}) solvable (a={a:.3f}, b={b:.3f}, residual {res:.1e})")
rng = np.random.default_rng(0)
agree = total = 0
for _ in range(400):
    x = rng.dirichlet([0.5, 0.5, 0.5])
    qf, qs = pr.regularize_tied_shares(x[0], x[1], n_eff=float(rng.integers(5, 200)), pseudo_count=1.0)
    a, b, res, conv = pr.fit_beta_ends(qf, qs, group_size=G)
    if conv and abs(qs - qf) > 1e-9:
        total += 1
        agree += int((qs > qf) == (a > b))
check(total > 350 and agree == total, f"q_s > q_f <=> a > b on the smoothed shares ({agree}/{total})")

print("3. the strengths: mirror symmetry, range, and the posterior expectation")
sym = max(abs(pr.beta_mirror_strengths(a, b, G)[0] - pr.beta_mirror_strengths(b, a, G)[1])
          for a in (0.05, 0.3, 1.0, 4.0) for b in (0.07, 0.5, 2.0))
check(sym < 1e-12, f"m_f(a, b) = m_s(b, a) (max diff {sym:.1e})")
mf, ms = pr.beta_mirror_strengths(0.7, 0.7, G)
check(abs(mf - ms) < 1e-12 and 0.0 <= mf <= 1.0, f"a = b gives equal strengths ({mf:.4f}) inside [0, 1]")
mc_rng = np.random.default_rng(1)
for a, b in ((0.13, 0.75), (0.37, 0.09), (1.5, 2.0)):
    mf, ms = pr.beta_mirror_strengths(a, b, G)
    pf = mc_rng.beta(a, b + G, size=2_000_000)
    ps = mc_rng.beta(a + G, b, size=2_000_000)
    ef, es = float(np.mean(2 * np.sqrt(pf * (1 - pf)))), float(np.mean(2 * np.sqrt(ps * (1 - ps))))
    check(abs(mf - ef) < 3e-3 and abs(ms - es) < 3e-3,
          f"Beta({a}, {b}): m_f {mf:.4f} vs MC {ef:.4f}, m_s {ms:.4f} vs MC {es:.4f}")
pmf = pr.beta_binomial_pmf(0.3, 0.2, G)
check(abs(sum(pmf) - 1.0) < 1e-12, "the Beta-binomial pmf sums to 1")

print("4. apply(): per-task normalisation, zero scores counted, padding not, bigger spreads push harder")
LIVE = ("live", "alfworld", [("L1", 3, 10.0, 2, 2, 1.0, 0, None), ("L2", 3, 0.0, 1, 2, -1.0, 4, None)])
ST1 = ("st1", "alfworld", [("A1", 5, 0.0, 0, 2, 0.0, 3, None), ("A2", 5, 0.0, 1, 2, 0.0, 6, None)])
ST2 = ("st2", "alfworld", [("B1", 5, 0.0, 0, 2, 0.0, 1, None), ("B2", 5, 0.0, 1, 2, 0.0, 2, None),
                           ("B3", 5, 0.0, 2, 2, 0.0, 0, None)])
SA1 = ("sa1", "alfworld", [("C1", 4, 10.0, 2, 2, 0.0, 0, None), ("C2", 6, 10.0, 2, 2, 0.0, 1, None)])
SA2 = ("sa2", "alfworld", [("D1", 4, 10.0, 2, 2, 0.0, 0, None), ("D2", 20, 10.0, 2, 2, 0.0, 9, None)])
BATCH = [LIVE, ST1, ST2, SA1, SA2]
ctl = beta_ctl()
b = build(BATCH)
new, m = ctl.apply(**b)
P = "progress_rank/alfworld"
recs = {r["uid"]: r for r in ctl.last_group_records}
check(m[f"{P}/beta/fit_failed"] == 0.0 and m[f"{P}/beta/residual"] < 1e-8, "the task's Beta was fitted")
sa = [s for u in ("st1", "st2") for s in recs[u]["score"]]
check(recs["st2"]["score"][1] == 0.0 and recs["st1"]["verdict"] == recs["st2"]["verdict"] == "fired",
      "both stuck groups fire; the middle rollout of st2 scores exactly 0")
d_a = float(np.mean(np.abs(sa)))
check(abs(m[f"{P}/beta/D_a"] - d_a) < 1e-12 and abs(d_a - 0.3) < 1e-12, f"D_a counts the zero score (0.3 over 5 trajectories)")
check(abs(m[f"{P}/beta/c_a"] * d_a - m[f"{P}/beta/m_f"]) < 1e-12, "c_a * mean|s| = m_f: the fired trajectories' mean |push| is m_f")
ss = {t: s for u in ("sa1", "sa2") for t, s in zip(recs[u]["trajs"], recs[u]["sat_score"])}
d_s = float(np.mean(np.abs(list(ss.values()))))
check(abs(m[f"{P}/beta/c_sat"] * d_s - m[f"{P}/beta/m_s"]) < 1e-12, "c_sat * mean|s| = m_s on the saturated side")
dA = (new - b["advantages"])
per_traj_ok = True
for u, key, c in (("st1", "score", m[f"{P}/beta/c_a"]), ("st2", "score", m[f"{P}/beta/c_a"]),
                  ("sa1", "sat_score", m[f"{P}/beta/c_sat"]), ("sa2", "sat_score", m[f"{P}/beta/c_sat"])):
    for t, s in zip(recs[u]["trajs"], recs[u][key]):
        idx = np.where(b["tuids"] == t)[0]
        per_traj_ok &= bool(torch.allclose(dA[idx][:, :3], torch.full((len(idx), 3), c * s), atol=1e-6))
check(per_traj_ok, "every row of a trajectory gets exactly c * its score")
g1 = np.mean([abs(ss[t]) for t in ("C1", "C2")]) * m[f"{P}/beta/c_sat"]
g2 = np.mean([abs(ss[t]) for t in ("D1", "D2")]) * m[f"{P}/beta/c_sat"]
check(g2 > 3 * g1, f"the group with the bigger spread is pushed harder (mean |push| {g2:.3f} vs {g1:.3f})")
b_pad = build(BATCH, pad=("A1", "C1"))
ctl_pad = beta_ctl()
_, m_pad = ctl_pad.apply(**b_pad)
check(abs(m_pad[f"{P}/beta/c_a"] - m[f"{P}/beta/c_a"]) < 1e-12 and abs(m_pad[f"{P}/beta/c_sat"] - m[f"{P}/beta/c_sat"]) < 1e-12,
      "padding copies change neither D nor c")
check(all(f"{P}/cf/{k}" in m for k in ("c", "sat_c", "sat_gate_open", "injected_mean_abs_adv", "sat_injected_mean_abs_adv")),
      "the budget_cap counterfactual is reported")
check(all(r.get("scale_mode") == "beta_mirror" and "beta_a" in r and "cf_c" in r for r in recs.values()),
      "records carry the Beta, the side's strength and the counterfactual")
check(recs["st1"]["beta_m"] == m[f"{P}/beta/m_f"] and recs["sa1"]["beta_m"] == m[f"{P}/beta/m_s"],
      "a record's beta_m is the strength of its group's side")

print("5. diagnostics: net push by unit, ranking vs uniform part, the premises")
SAT8 = ("s8", "alfworld", [(f"E{i}", 2 if i < 4 else 4, 10.0, 2, 2, 0.0, 0 if i < 4 else 2, None) for i in range(8)])
ctl5 = beta_ctl()
b5 = build([LIVE, SAT8, ST1])
_, m5 = ctl5.apply(**b5)
check(abs(m5[f"{P}/beta/sat_net_traj"] - 1.0 / 3.0) < 1e-12,
      f"turn-weighted centring leaves +1/3 per trajectory on T=[2]*4+[4]*4 ({m5[f'{P}/beta/sat_net_traj']:.4f})")
check(abs(m5[f"{P}/beta/sat_net_turn"]) < 1e-12, "and zero per turn (the turn-weighted sum is 0)")
check(m5[f"{P}/beta/sat_bonus_signed"] > 0.0 and m5[f"{P}/beta/sat_rank_mass"] > 0.0, "sat's uniform part is positive, its ranking part non-zero")
check(m5[f"{P}/beta/hist_obs_k8"] == 1.0 and m5[f"{P}/beta/groups_not_G"] == 2.0,
      "the histogram counts the complete group of 8 (k=8); the two short groups are counted apart")
check(abs(sum(m5[f"{P}/beta/hist_pred_k{k}"] for k in range(G + 1)) - 1.0) < 1e-9 and f"{P}/beta/hist_tv" in m5,
      "the predicted histogram and its distance are reported")
check(m5[f"{P}/beta/sat_win_revisits_spread"] == 2.0 and m5[f"{P}/beta/stuck_top_k_over_K"] == 0.5,
      "the premises: winners' revisit spread, the best failure's k/K")

print("6. Search: saturated groups fire without the gate; searches and answers are reported")
SLIVE = ("slive", "search", [("Q1", 2, 1.0, 1, 1, 1.0, 0, True), ("Q2", 3, 0.0, 0, 1, -1.0, 1, True)])
SSAT = ("ssat", "search", [("W1", 1, 1.0, 1, 1, 0.0, 0, True), ("W2", 3, 1.0, 1, 1, 0.0, 1, True)])
SFLAT = ("sflat", "search", [("V1", 2, 1.0, 1, 1, 0.0, 0, True), ("V2", 2, 1.0, 1, 1, 0.0, 0, True)])
SSTUCK = lambda i: (f"sstuck{i}", "search", [(f"X{i}a", 4, 0.0, 1, 1, 0.0, 2, False), (f"X{i}b", 4, 0.0, 0, 1, 0.0, 1, True)])  # noqa: E731
ctl6 = beta_ctl()
b6 = build([SLIVE, SSAT, SFLAT, SSTUCK(1), SSTUCK(2), SSTUCK(3)])
new6, m6 = ctl6.apply(**b6)
r6 = {r["uid"]: r for r in ctl6.last_group_records}
dd = lambda trajs: float((new6 - b6["advantages"])[np.concatenate([np.where(b6["tuids"] == t)[0] for t in trajs])].abs().sum())  # noqa: E731
check(r6["ssat"]["sat_verdict"] == "fired" and dd(["W1", "W2"]) > 0.0,
      "a Search saturated group with a turn spread fires on a stuck-heavy step (no gate)")
check(r6["sflat"]["sat_verdict"] != "fired" and dd(["V1", "V2"]) == 0.0, "equal turns: nothing fires")
S = "progress_rank/search/beta"
check(m6[f"{S}/search_fail_no_answer_share"] > 0.0 and f"{S}/search_win_two_plus_share" in m6,
      "Search's no-answer share and 2+-search share are reported, with their counts")

print("7. a failed fit adds nothing and says so")
real_fit = pr.fit_beta_ends
pr.fit_beta_ends = lambda *a, **k: (float("nan"), float("nan"), float("inf"), False)
try:
    ctl7 = beta_ctl()
    b7 = build(BATCH)
    new7, m7 = ctl7.apply(**b7)
finally:
    pr.fit_beta_ends = real_fit
check(m7[f"{P}/beta/fit_failed"] == 1.0 and m7[f"{P}/beta/c_a"] == 0.0 and m7[f"{P}/beta/c_sat"] == 0.0,
      "fit_failed = 1 and both coefficients 0")
check(torch.equal(new7, b7["advantages"]), "the advantages are returned unchanged")

print("8. state: round-trip, version 4 still loads, resumed == continuous")
steps = [BATCH, [LIVE, ST1, SA2, SA1], [LIVE, ST2, SA1, ST1, SA2]]
cont = beta_ctl()
outs = [cont.apply(**build(s)) for s in steps]
half = beta_ctl()
for s in steps[:2]:
    half.apply(**build(s))
state = half.state_dict()
resumed = beta_ctl()
resumed.load_state_dict(state)
new_r, m_r = resumed.apply(**build(steps[2]))
check(state["version"] == 5 and state["beta_ab"]["alfworld"] is not None, "version 5 carries the fitted (a, b)")
check(torch.equal(new_r, outs[2][0]) and m_r[f"{P}/beta/a"] == outs[2][1][f"{P}/beta/a"],
      "a controller resumed from step 2's state reproduces step 3 exactly")
old = beta_ctl()
old.load_state_dict({"version": 4, "ema": {}, "success_ema": {}, "gate_stuck_ema": {}, "gate_sat_ema": {}})
check(all(v is None for v in old.beta_ab.values()), "a version-4 state loads (the Beta starts cold)")

print("9. refusals, and the default untouched")
for kw, what in ((dict(sat_gate=True), "the gate"), (dict(sat_placebo="sign"), "a sat placebo"),
                 (dict(mixed_rho=0.1), "the mixed term"), (dict(rho=0.0), "rho = 0 (side off by accident)"),
                 (dict(scale_mode="cap_only"), "an unknown scale_mode")):
    try:
        beta_ctl(**kw)
        check(False, f"beta_mirror with {what} is refused")
    except AssertionError:
        check(True, f"beta_mirror with {what} is refused")
dflt = pr.ProgressRankController(rho=0.1, sat_rho=0.1, sat_tasks=["alfworld", "webshop", "search"])
_, m_d = dflt.apply(**build(BATCH))
check(not any("/beta/" in k or "/cf/" in k for k in m_d) and not any("beta_a" in r for r in dflt.last_group_records),
      "budget_cap (the default) reports no beta metrics and no beta record fields")
cfg = OmegaConf.load(os.path.join(REPO, "verl/trainer/config/ppo_trainer.yaml"))
check(cfg.algorithm.progress_rank.scale_mode == "budget_cap" and cfg.algorithm.progress_rank.beta_group_size == 8,
      "the config default is budget_cap (every existing arm unchanged), group size 8")

print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
