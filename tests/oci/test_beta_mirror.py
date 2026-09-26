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
  * A group without exactly beta_group_size real rollouts is outside the model: it is not in the
    shares, the fit, the histogram or either update, and a step with one gives the complete groups
    exactly what the step without it gives them (budget_cap still uses it, as before).
  * Search's searches are the queries the environment sent (the Search manager counts them from the
    real SearchEnv.step's tool calls, the loop carries them, the controller reports them): a rollout
    that hit the turn cap spent its last turn without one.
  * The launcher composes and passes its own lock; a group size other than the rollout count is refused.
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


def build(groups, resp=4, pad=(), searches=True):
    """Rows from ``[(uid, task, [(traj, turns, reward, k, K, adv, revisits, committed[, searches]), ...]), ...]``;
    ``pad`` names trajectories whose first row is appended again as a padding copy (real_rows False).
    ``searches`` False leaves the searches column out entirely."""
    cols = {k: [] for k in ("uids", "tuids", "tasks", "rew", "ks", "Ks", "adv", "rv", "cm", "ns", "real")}
    for uid, task, trajs in groups:
        for tr in trajs:
            traj, turns, r, k, K, a, rv, cm = tr[:8]
            ns = tr[8] if len(tr) > 8 else None
            for _ in range(turns):
                for key, v in zip(cols, (uid, traj, task, r, k, K, a, rv, cm, ns, True)):
                    cols[key].append(v)
    for uid, task, trajs in groups:
        for tr in trajs:
            traj, turns, r, k, K, a, rv, cm = tr[:8]
            ns = tr[8] if len(tr) > 8 else None
            if traj in pad:
                for key, v in zip(cols, (uid, traj, task, r, k, K, a, rv, cm, ns, False)):
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
                committed_rows=np.array([float("nan") if v is None else float(v) for v in cols["cm"]]),
                search_count_rows=(np.array([float("nan") if v is None else float(v) for v in cols["ns"]])
                                   if searches else None))


def grow(group, n):
    """The group with each rollout copied until it holds ``n`` (the first copy keeps the id). Every
    rollout is copied equally often, so the group's means, its verdict and each score are unchanged;
    what changes is only how many trajectories the group contributes to a task-level mean."""
    uid, task, trajs = group
    assert n % len(trajs) == 0, (uid, len(trajs), n)
    out = []
    for tr in trajs:
        for j in range(n // len(trajs)):
            out.append((tr[0] if j == 0 else f"{tr[0]}~{j}",) + tuple(tr[1:]))
    return (uid, task, out)


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
# Every group grown to 6 rollouts (2 x 3, 3 x 2) and the Beta told so: complete groups only.
BATCH6 = [grow(g, 6) for g in BATCH]
ctl = beta_ctl(beta_group_size=6)
b = build(BATCH6)
new, m = ctl.apply(**b)
P = "progress_rank/alfworld"
recs = {r["uid"]: r for r in ctl.last_group_records}
check(m[f"{P}/beta/fit_failed"] == 0.0 and m[f"{P}/beta/residual"] < 1e-8, "the task's Beta was fitted")
sa = [s for u in ("st1", "st2") for s in recs[u]["score"]]
check(dict(zip(recs["st2"]["trajs"], recs["st2"]["score"]))["B2"] == 0.0
      and recs["st1"]["verdict"] == recs["st2"]["verdict"] == "fired",
      "both stuck groups fire; the middle rollout of st2 scores exactly 0")
d_a = float(np.mean(np.abs(sa)))
check(abs(m[f"{P}/beta/D_a"] - d_a) < 1e-12 and abs(d_a - 3.5 / 12) < 1e-12,
      "D_a counts the zero scores (3.5 over 12 trajectories; 0.35 if the two zeros were dropped)")
check(m[f"{P}/beta/groups_not_G"] == 0.0 and all(r["in_model"] and r["n_real_rollouts"] == 6 for r in recs.values()),
      "every group holds 6 real rollouts: all inside the model")
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
b_pad = build(BATCH6, pad=("A1", "C1"))
ctl_pad = beta_ctl(beta_group_size=6)
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
b5 = build([grow(LIVE, 8), SAT8, grow(ST1, 8)])
_, m5 = ctl5.apply(**b5)
check(abs(m5[f"{P}/beta/sat_net_traj"] - 1.0 / 3.0) < 1e-12,
      f"turn-weighted centring leaves +1/3 per trajectory on T=[2]*4+[4]*4 ({m5[f'{P}/beta/sat_net_traj']:.4f})")
check(abs(m5[f"{P}/beta/sat_net_turn"]) < 1e-12, "and zero per turn (the turn-weighted sum is 0)")
check(m5[f"{P}/beta/sat_bonus_signed"] > 0.0 and m5[f"{P}/beta/sat_rank_mass"] > 0.0, "sat's uniform part is positive, its ranking part non-zero")
check(m5[f"{P}/beta/hist_obs_k8"] == 1.0 and m5[f"{P}/beta/hist_obs_k4"] == 1.0 and m5[f"{P}/beta/hist_obs_k0"] == 1.0
      and m5[f"{P}/beta/groups_not_G"] == 0.0,
      "the histogram counts each complete group of 8 by its successes (k = 8, 4, 0)")
check(abs(sum(m5[f"{P}/beta/hist_pred_k{k}"] for k in range(G + 1)) - 1.0) < 1e-9 and f"{P}/beta/hist_tv" in m5,
      "the predicted histogram and its distance are reported")
check(m5[f"{P}/beta/sat_win_revisits_spread"] == 2.0 and m5[f"{P}/beta/stuck_top_k_over_K"] == 0.5,
      "the premises: winners' revisit spread, the best failure's k/K")

print("6. Search: saturated groups fire without the gate; searches and answers are reported")
# The last field is the searches the environment sent: an answer ends the episode on the turn it is
# sent, and a rollout that reaches the 4-turn cap spends its fourth turn without a search (X*a: 4
# turns, never answered, 3 searches -- turns minus answered would say 4).
SLIVE = ("slive", "search", [("Q1", 2, 1.0, 1, 1, 1.0, 0, True, 1), ("Q2", 3, 0.0, 0, 1, -1.0, 1, True, 2)])
SSAT = ("ssat", "search", [("W1", 1, 1.0, 1, 1, 0.0, 0, True, 0), ("W2", 3, 1.0, 1, 1, 0.0, 1, True, 2)])
SFLAT = ("sflat", "search", [("V1", 2, 1.0, 1, 1, 0.0, 0, True, 1), ("V2", 2, 1.0, 1, 1, 0.0, 0, True, 1)])
SSTUCK = lambda i: (f"sstuck{i}", "search", [(f"X{i}a", 4, 0.0, 1, 1, 0.0, 2, False, 3), (f"X{i}b", 4, 0.0, 0, 1, 0.0, 1, True, 3)])  # noqa: E731
SEARCH_BATCH = [SLIVE, SSAT, SFLAT, SSTUCK(1), SSTUCK(2), SSTUCK(3)]
ctl6 = beta_ctl(beta_group_size=2)
b6 = build(SEARCH_BATCH)
new6, m6 = ctl6.apply(**b6)
r6 = {r["uid"]: r for r in ctl6.last_group_records}
dd = lambda trajs: float((new6 - b6["advantages"])[np.concatenate([np.where(b6["tuids"] == t)[0] for t in trajs])].abs().sum())  # noqa: E731
check(r6["ssat"]["sat_verdict"] == "fired" and dd(["W1", "W2"]) > 0.0,
      "a Search saturated group with a turn spread fires on a stuck-heavy step (no gate)")
check(r6["sflat"]["sat_verdict"] != "fired" and dd(["V1", "V2"]) == 0.0, "equal turns: nothing fires")
S = "progress_rank/search/beta"
check(m6[f"{S}/search_fail_no_answer_share"] > 0.0 and f"{S}/search_win_two_plus_share" in m6,
      "Search's no-answer share and 2+-search share are reported, with their counts")
check(abs(m6[f"{S}/search_fail_searches_mean"] - 20.0 / 7.0) < 1e-12,
      f"failures' searches are the env's count, 20/7 (turns minus answered would give 23/7): "
      f"{m6[f'{S}/search_fail_searches_mean']:.4f}")
check(m6[f"{S}/search_win_searches_mean"] == 1.0 and abs(m6[f"{S}/search_fail_two_plus_share"] - 1.0) < 1e-12,
      "winners 1.0 search on average; every failure searched at least twice")
_, m6n = beta_ctl(beta_group_size=2).apply(**build(SEARCH_BATCH, searches=False))
check(f"{S}/search_fail_searches_mean" not in m6n and f"{S}/search_fail_no_answer_share" in m6n,
      "without the searches column no search count is reported (no turn-count stand-in)")

print("7. a failed fit adds nothing and says so")
real_fit = pr.fit_beta_ends
pr.fit_beta_ends = lambda *a, **k: (float("nan"), float("nan"), float("inf"), False)
try:
    ctl7 = beta_ctl(beta_group_size=6)
    b7 = build(BATCH6)
    new7, m7 = ctl7.apply(**b7)
finally:
    pr.fit_beta_ends = real_fit
check(m7[f"{P}/beta/fit_failed"] == 1.0 and m7[f"{P}/beta/c_a"] == 0.0 and m7[f"{P}/beta/c_sat"] == 0.0,
      "fit_failed = 1 and both coefficients 0")
check(torch.equal(new7, b7["advantages"]), "the advantages are returned unchanged")

print("8. state: round-trip, version 4 still loads, resumed == continuous")
steps = [[grow(g, 6) for g in st] for st in (BATCH, [LIVE, ST1, SA2, SA1], [LIVE, ST2, SA1, ST1, SA2])]
cont = beta_ctl(beta_group_size=6)
outs = [cont.apply(**build(s)) for s in steps]
half = beta_ctl(beta_group_size=6)
for s in steps[:2]:
    half.apply(**build(s))
state = half.state_dict()
resumed = beta_ctl(beta_group_size=6)
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

print("10. a group without exactly beta_group_size real rollouts is outside the model")
LIVE8, ST8 = grow(LIVE, 8), grow(ST1, 8)
ST7 = ("st7", "alfworld", [(f"H{i}", 5, 0.0, i % 2, 2, 0.0, 0, None) for i in range(7)])
SAT7 = ("s7", "alfworld", [(f"J{i}", 2 if i < 3 else 6, 10.0, 2, 2, 0.0, 0, None) for i in range(7)])
FULL, SHORT = [LIVE8, SAT8, ST8], [ST7, SAT7]
ctl_w, ctl_c = beta_ctl(), beta_ctl()
b_w, b_c = build(FULL + SHORT), build(FULL)
new_w, m_w = ctl_w.apply(**b_w)
new_c, m_c = ctl_c.apply(**b_c)
rw = {r["uid"]: r for r in ctl_w.last_group_records}
check(m_w[f"{P}/beta/groups_not_G"] == 2.0 and m_w[f"{P}/beta/groups_not_G_fired_a"] == 1.0
      and m_w[f"{P}/beta/groups_not_G_fired_sat"] == 1.0,
      "the two groups of 7 are counted apart, and each would have fired on its side")
check(rw["st7"]["verdict"] == pr.GROUP_NOT_G and rw["st7"]["verdict_if_complete"] == "fired"
      and rw["s7"]["sat_verdict"] == pr.GROUP_NOT_G and rw["s7"]["sat_verdict_if_complete"] == "fired"
      and not rw["st7"]["in_model"] and rw["s7"]["n_real_rollouts"] == 7 and rw["s8"]["in_model"],
      "their records say so: group_not_G, the verdict it would have had, 7 real rollouts")
rows_of = lambda bb, grps: np.concatenate([np.where(bb["tuids"] == tr[0])[0] for g in grps for tr in g[2]])  # noqa: E731
i7 = rows_of(b_w, SHORT)
check(torch.equal(new_w[i7], b_w["advantages"][i7]), "their rows get nothing added")
check(m_w[f"{P}/beta/n_fired_traj_a"] == 8.0 and m_w[f"{P}/beta/n_fired_traj_sat"] == 8.0,
      "only the complete groups' trajectories enter D (8 per side)")
same = all(m_w[f"{P}/beta/{k}"] == m_c[f"{P}/beta/{k}"]
           for k in ("q_f_raw", "q_s_raw", "n_eff", "a", "b", "m_f", "m_s", "c_a", "c_sat"))
check(same and ctl_w.beta_n_groups["alfworld"] == ctl_c.beta_n_groups["alfworld"] == 3.0
      and ctl_w.gate_stuck_ema["alfworld"] == ctl_c.gate_stuck_ema["alfworld"]
      and ctl_w.gate_sat_ema["alfworld"] == ctl_c.gate_sat_ema["alfworld"],
      "shares, fit, strengths and coefficients equal those of the same step without the short groups")
check(torch.equal(new_w[rows_of(b_w, FULL)], new_c[rows_of(b_c, FULL)]),
      "and the complete groups' advantages are identical to that step's")
dflt10 = pr.ProgressRankController(rho=0.1, sat_rho=0.1, sat_tasks=["alfworld", "webshop", "search"])
new_d, _ = dflt10.apply(**b_w)
check(float((new_d[i7] - b_w["advantages"][i7]).abs().sum()) > 0.0,
      "budget_cap (the default) still ranks the groups of 7, as it always did")

print("11. Search's searches come from the environment's own tool calls")
from agent_system.environments.env_manager import SearchEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv  # noqa: E402
from agent_system.environments.env_package.search.projection import search_projection  # noqa: E402
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv  # noqa: E402


class _RealSearchEnvs:
    """The real SearchEnv.step / _is_done behind the manager; only the retriever call is a stub,
    which records every query it is handed."""
    group_n = 1
    is_train = True

    def __init__(self, n, max_turns):
        self.max_turns, self.calls, self.envs = max_turns, [[] for _ in range(n)], []
        for i in range(n):
            e = SearchEnv.__new__(SearchEnv)
            e._execute_tool = (lambda group, name, query, _i=i:
                               self.calls[_i].append(query) or "\n<information>Paris is the capital.</information>\n")
            self.envs.append(e)

    def reset(self, kwargs=None):
        for e, kw in zip(self.envs, kwargs):
            e.reset({"ground_truth": kw["ground_truth"], "max_turns": self.max_turns})
        return [kw["question"] for kw in kwargs], [{} for _ in kwargs]

    def step(self, actions):
        out = [SearchMultiProcessEnv._sync_step(None, e, a) for e, a in zip(self.envs, actions)]
        return tuple(list(x) for x in zip(*out))


cfg11 = OmegaConf.create({"env": {"history_length": 0, "rollout": {"n": 1}},
                          "algorithm": {"progress_rank": {"enable": True}, "oci_sat": {"enable": False},
                                        "oci_slots": {"enable": False}, "oci_rank": {"enable": False}}})
envs11 = _RealSearchEnvs(3, max_turns=4)
mgr = SearchEnvironmentManager(envs11, search_projection, cfg11)
mgr.reset([{"question": "q", "ground_truth": {"target": ["Paris"]}}] * 3)
TURNS = [["<search> capital of france </search>", "<search> a </search>", "no block at all"],
         ["<search> france capital city </search>", "<answer> Paris </answer>", "<search> b </search>"],
         ["<search> paris </search>", "<search> after the end </search>", "<answer> Paris </answer>"],
         ["<search> one more </search>", "<search> after the end </search>", "<search> after the end </search>"]]
counts, dones_seen, last = [], [], None
for acts in TURNS:
    _, _, dones, last = mgr.step(acts)
    counts.append([int(i["searches"]) for i in last])
    dones_seen.append([bool(d) for d in dones])
per_row = [list(col) for col in zip(*counts)]
check(per_row[0] == [1, 2, 3, 3] and dones_seen[3][0],
      f"four <search> turns under a 4-turn cap: 3 searches, the capped turn sends none ({per_row[0]}; "
      "turns minus answered would say 4)")
check(per_row[1] == [1, 1, 1, 1], f"search then answer: 1, and nothing after the episode ended ({per_row[1]})")
check(per_row[2] == [0, 1, 1, 1], f"a turn with no <search> block sends no query ({per_row[2]})")
real_q = [sum(1 for q in c if q and q[0] is not None) for c in envs11.calls]
check(real_q == [row[-1] for row in per_row], f"each count equals the queries the retriever stub received ({real_q})")
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector  # noqa: E402
from verl import DataProto  # noqa: E402

col = TrajectoryCollector.__new__(TrajectoryCollector)
col._progress_rank_on = True
col._queue_row_for_prefetch = lambda *a: None
b11 = DataProto.from_single_dict({"input_ids": torch.zeros(2, 3, dtype=torch.long),
                                  "traj_uid": np.array(["a", "b"], dtype=object)})
tbl, tinf = [[], []], [[], []]
col._record_turn(batch=b11, active_idx=np.array([0, 1]), active_masks=np.array([True, True]),
                 infos=[dict(last[0]), {"revisits": 7}], traj_uid=np.array(["a", "b"], dtype=object),
                 total_batch_list=tbl, total_infos=tinf, batch_size=2)
check(tbl[0][0]["searches"] == 3.0 and np.isnan(tbl[1][0]["searches"]),
      "the loop carries the count on a Search row and NaN on another task's row")

print("12. the launcher passes its own lock; a group size other than the rollout count is refused")
os.environ.setdefault("RUN_TAG_SUFFIX", "_progress_rank_rho0p1_betamirror_arrive")
os.environ["RHO"] = "0.1"
from hydra import compose, initialize_config_dir  # noqa: E402

from tests.trainer.test_run_script_overrides_compose import _overrides  # noqa: E402
from verl.trainer.main_opd_grpo import inject_opd_grpo_config  # noqa: E402
from verl.utils.expected_config import check_expected_config  # noqa: E402

WRAP = "examples/opd_grpo_trainer/run_multitask_progress_rank_beta_mirror_qwen3.sh"


def composed(extra=()):
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "verl/trainer/config")):
        return compose(config_name="ppo_trainer", overrides=list(_overrides(WRAP)) + list(extra))


cfg12 = composed()
inject_opd_grpo_config(cfg12)
LOCK = os.path.join(REPO, cfg12.trainer.expected_config)
check(cfg12.trainer.expected_config.endswith("expected_multitask_progress_rank_beta_mirror_config.yaml")
      and check_expected_config(cfg12, LOCK) == [], "the wrapper composes and matches its own lock, no waiver")
check(cfg12.algorithm.progress_rank.scale_mode == "beta_mirror" and not cfg12.algorithm.opd.retire.enable
      and list(cfg12.algorithm.progress_rank.sat_tasks) == ["alfworld", "webshop", "search"]
      and cfg12.algorithm.progress_rank.alfworld_k == "milestone_arrive",
      "beta_mirror, the teacher never retired, sat on all three tasks, milestone_arrive")
for extra, what in ((["algorithm.progress_rank.scale_mode=budget_cap"], "the budget_cap rule"),
                    (["++algorithm.opd.retire.enable=True"], "retirement switched back on"),
                    (["algorithm.progress_rank.sat_gate=True"], "the gate"),
                    (["algorithm.progress_rank.alfworld_k=milestone"], "the milestone count"),
                    (["algorithm.progress_rank.beta_pseudo_count=2.0"], "another pseudo-count")):
    c12 = composed(extra)
    inject_opd_grpo_config(c12)
    check(len(check_expected_config(c12, LOCK)) == 1, f"the lock catches {what}")
try:
    inject_opd_grpo_config(composed(["algorithm.progress_rank.beta_group_size=7"]))
    check(False, "beta_group_size 7 against env.rollout.n 8 is refused at launch")
except AssertionError:
    check(True, "beta_group_size 7 against env.rollout.n 8 is refused at launch")

print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
