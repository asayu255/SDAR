#!/usr/bin/env python3
"""Rebuild the 3x3 RL-OPD cross-effect matrix from the N=8 term probe and solve
the coefficient QP of the OPD-coefficient proposal.

    C_ij = mean over batches of <r_i, d_j>      (r = RL gradient of task i,
                                                 d = OPD gradient of task j,
                                                 coefficients included, P = I)
    b* = argmin ||b - 1||^2  s.t.  C_i b >= 0 (i in S),  0 <= b <= 1

Everything is read from ``~/grad_probe/terms_n8_fixed.json`` (the only valid
gradient measurement with an RL term, cross_teacher_theory.md section 4.14). The
QP is three-dimensional, so it is solved on a grid and cross-checked with SLSQP.

    python3 scripts/opd_cross_effect_qp.py [payload.json]            # QP of the original proposal
    python3 scripts/opd_cross_effect_qp.py --redistribute [payload]  # redistribution rule (design doc)
    python3 scripts/opd_cross_effect_qp.py --redistribute --json out.json [payload]

Three things this script used to get wrong, all found by review before any run:

* **A row with no RL signal produced b = NaN.** ``C_ij / (||r_i|| ||d_j||)`` is
  0/0 when a batch gave task i no policy gradient at all -- which is exactly the
  case the design's own appendix says to drop, and the case a GRPO mixture with
  degenerate groups will eventually produce. Now every cell carries its own
  count of usable batches, the unusable ones are dropped from that cell's mean
  rather than poisoning it, and the run REFUSES rather than reporting a b built
  on too little (``--min-batches``, default half).
* **The bootstrap asked whether search > alfworld > webshop reproduced**, which
  is that one measurement's answer written into the test. It now bootstraps the
  ranking the sample itself gives (``rank_reproducibility``) and reports
  agreement with a reference ranking SEPARATELY (``agreement_with_reference``),
  because "this measurement is stable" and "this measurement agrees with the
  step-300 one" are two claims and the go/no-go rule in the design doc needs
  them apart.
* **The composite norm was a plain mean over batches** while the design doc
  quotes an RMS. Both are now reported under their own names, and the uniform
  arm's matching factor is derived from the RMS one (1.110833).
"""
import json
import math
import sys

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TASKS = ["alfworld", "search", "webshop"]


def _dot(w, a, b):
    if a == b:
        return w[f"sq:{a[0]}:{a[1]}"]
    return w.get(f"dot:{a[0]}:{a[1]}:{b[0]}:{b[1]}", w.get(f"dot:{b[0]}:{b[1]}:{a[0]}:{a[1]}"))


def load(path):
    d = json.load(open(path))
    B = d["batch_moments"]
    mat = lambda ta, tb: np.array([[[_dot(w, (ti, ta), (tj, tb)) for tj in TASKS] for ti in TASKS] for w in B])
    C, R, D = mat("rl", "opd"), mat("rl", "rl"), mat("opd", "opd")
    nr = np.sqrt(np.array([[w[f"sq:{t}:rl"] for t in TASKS] for w in B]))
    nd = np.sqrt(np.array([[w[f"sq:{t}:opd"] for t in TASKS] for w in B]))
    return d, C, R, D, nr, nd


_GRID = None


def solve_qp(Cmat, S, grid=201):
    """Grid solution of min ||b-1||^2 s.t. Cmat[i] . b >= 0 for i in S, b in [0,1]^3."""
    global _GRID
    if _GRID is None or len(_GRID) != grid ** 3:
        g = np.linspace(0, 1, grid)
        _GRID = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    bb = _GRID
    feas = np.ones(len(bb), bool)
    for i in S:
        feas &= bb @ Cmat[i] >= -1e-15
    obj = ((bb - 1) ** 2).sum(1)
    obj[~feas] = np.inf
    k = obj.argmin()
    return bb[k], obj[k]


def solve_qp_slsqp(Cmat, S):
    from scipy.optimize import minimize

    cons = [{"type": "ineq", "fun": (lambda b, i=i: Cmat[i] @ b)} for i in S]
    res = minimize(lambda b: ((b - 1) ** 2).sum(), x0=np.full(3, 0.5), bounds=[(0, 1)] * 3,
                   constraints=cons, method="SLSQP")
    return res.x, res.fun


def main(path):
    d, C, R, D, nr, nd = load(path)
    NB = len(C)
    print(f"payload {path}\n  checkpoint {d['checkpoint']}  batches {NB}  "
          f"pg_loss_coef {d['pg_loss_coef']}  teacher_kl_loss_coef {d['teacher_kl_loss_coef']}")
    Cm, Cse = C.mean(0), C.std(0, ddof=1) / math.sqrt(NB)
    t = Cm / Cse
    print("\n=== C_ij = mean_b <r_i, d_j>   rows: RL task i, cols: OPD task j  (alfworld, search, webshop) ===")
    print(Cm)
    print("t = mean / SE:\n", t)
    try:
        from scipy import stats
        p = 2 * stats.t.sf(np.abs(t), NB - 1)
        print(f"two-sided p (df={NB - 1}):\n", p, "\nBonferroni x9:\n", np.minimum(1, 9 * p))
    except Exception:
        pass
    print(f"batches with C_ij < 0 (of {NB}):\n", (C < 0).sum(0))
    print("mean cosine:\n", masked_mean(*cosine_cells(C, nr, nd))[0])
    print("\n||r_i|| / ||d_j|| (same task) per batch:\n", nr / nd)
    print("live GRPO groups per batch:")
    for n, a in enumerate(d["advantages"], 1):
        print(f"  b{n}: " + "  ".join(f"{k} {v['live_groups']}/{v['prompt_groups']}" for k, v in a.items()))

    tot = C.sum((1, 2))
    print("\n<sum_i r_i, sum_j d_j> per batch:", tot, "-> negative in", int((tot < 0).sum()), "of", NB)
    print("row sums C_i . 1 (mean):", Cm.sum(1))
    print("OPD first-order term at b=1 relative to RL self term R_ii:", -Cm.sum(1) / np.diag(R.mean(0)))

    print("\n=== QP: min ||b-1||^2  s.t. C_i b >= 0 (i in S), 0<=b<=1 ===")
    for name, S in [("S = all 3", [0, 1, 2]), ("S = {alfworld, webshop}", [0, 2]), ("S = {alfworld}", [0]),
                    ("S = {webshop}", [2]), ("S = {search}", [1])]:
        bg, og = solve_qp(Cm, S)
        try:
            bx, ox = solve_qp_slsqp(Cm, S)
            extra = f"   SLSQP b* = {bx}"
        except Exception:
            extra = ""
        print(f"{name:<26} b* = {bg}  ||b-1||^2 = {og:.3f}{extra}")
    Ccos = masked_mean(*cosine_cells(C, nr, nd))[0]
    print("\nsame QP on the mean-cosine matrix:  S=all", solve_qp(Ccos, [0, 1, 2])[0],
          "  S={alf,web}", solve_qp(Ccos, [0, 2])[0])
    Ct = np.where(np.abs(t) > 2, Cm, 0.0)
    print("same QP keeping only |t|>2 entries: S=all", solve_qp(Ct, [0, 1, 2])[0])
    print("same QP on the median matrix:       S=all", solve_qp(np.median(C, 0), [0, 1, 2])[0],
          "  S={alf,web}", solve_qp(np.median(C, 0), [0, 2])[0])

    print("\n=== stability ===")
    for n in range(NB):
        print(f"single batch {n + 1}: b* = {solve_qp(C[n], [0, 1, 2], grid=101)[0]}")
    for n in range(NB):
        idx = [k for k in range(NB) if k != n]
        print(f"drop batch {n + 1}:   b* = {solve_qp(C[idx].mean(0), [0, 1, 2], grid=101)[0]}")
    rng = np.random.default_rng(0)
    for name, S in [("S=all", [0, 1, 2]), ("S={alf,web}", [0, 2])]:
        bs = np.array([solve_qp(C[rng.integers(0, NB, NB)].mean(0), S, grid=51)[0] for _ in range(400)])
        print(f"bootstrap {name}: mean b* = {bs.mean(0)}  P(b_j = 0) = {(bs == 0).mean(0)}  "
              f"P(all zero) = {(bs.sum(1) == 0).mean():.2f}")

    print("\n=== relaxations ===")
    print("single summed constraint (1^T C) b >= 0:", solve_qp(Cm.sum(0, keepdims=True), [0])[0])
    Cn = Cm / np.abs(Cm).max()
    g = np.linspace(0, 1, 101)
    bb = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    for mu in [1, 3, 10, 30]:
        obj = ((bb - 1) ** 2).sum(1) + mu * np.maximum(0, -(bb @ Cn.T)).sum(1)
        print(f"hinge penalty mu={mu:>3}: b* = {bb[obj.argmin()]}")


def _payload_arg():
    """The positional payload, skipping the VALUES of the flags that take one."""
    args, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a in ("--json", "--min-batches"):
            skip = True
            continue
        if not a.startswith("--"):
            args.append(a)
    return args[0] if args else "/home/ohara/grad_probe/terms_n8_fixed.json"


# ---------------------------------------------------------------------------
# Redistribution rule (design doc section 1). Added after the review found that
# a coefficient budget sum_j b_j = 3 does not preserve the OPD gradient budget:
# ||d_j|| differ 2.2x across tasks, so the rule now fixes sum_j w_j b_j = 1 with
# w_j proportional to ||d_j||. Both budgets are reported for every b.
# ---------------------------------------------------------------------------
MIN_BATCH_FRACTION = 0.5
# The design's admission test: below this the ranking the coefficients encode is
# not reproducible on its own sample, and the arm to run is the control.
RANK_REPRO_MIN = 0.90
# The ranking the step-300 payload gave. Used ONLY as a reference to report
# agreement against -- never as the thing the bootstrap tests, which is the
# distinction the old hardcoded P(search>alf>web) collapsed.
REFERENCE_ORDER = ("search", "alfworld", "webshop")


def cosine_cells(C, nr, nd):
    """Per-batch cosines and a per-cell mask of which batches can carry one.

    A batch where task i produced no policy gradient at all gives 0/0 here, and
    the design's appendix says to drop it from that ROW -- not to let it turn
    the whole matrix into NaN, which is what dividing straight through did.
    """
    denom = nr[:, :, None] * nd[:, None, :]
    ok = np.isfinite(C) & np.isfinite(denom) & (denom > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(ok, C / np.where(ok, denom, 1.0), 0.0)
    return cos, ok & np.isfinite(cos)


def masked_mean(values, ok, axis=0):
    n = ok.sum(axis)
    total = np.where(ok, values, 0.0).sum(axis)
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan), n


def check_coverage(counts, n_batches, required, labels, min_fraction=MIN_BATCH_FRACTION):
    """Refuse rather than report a coefficient built on too few batches.

    ``required`` says which cells the rule actually reads: a cell excluded on
    purpose (the diagonal under ``diag=False``) has zero usable batches BY
    CONSTRUCTION, and treating that as missing data would make the honest
    variant of the rule the one that cannot run.
    """
    need = max(1, int(math.ceil(min_fraction * n_batches)))
    short = [idx for idx in np.argwhere(required) if counts[tuple(idx)] < need]
    if short:
        detail = ", ".join(
            f"{labels(tuple(int(i) for i in idx))}: {int(counts[tuple(idx)])}/{n_batches}"
            for idx in short
        )
        raise ValueError(
            f"too few usable batches (need {need} of {n_batches}): {detail}. "
            f"A coefficient vector built on this would be a measurement of the "
            f"missingness, not of the interference -- re-measure or lower "
            f"--min-batches deliberately."
        )
    return need


def redistribute(cos, ok, nd, S, *, budget="grad", bmin=0.5, bmax=1.5, diag=True,
                 min_fraction=MIN_BATCH_FRACTION, check=True):
    """b_j = clip(1 + kappa (c_j - cbar), bmin, bmax) with c_j = sum_{i in S} Ctilde_ij.

    ``cos``/``ok``: (N, 3, 3) per-batch cosines and their usability mask.
    ``nd``: (N, 3) per-batch OPD gradient norms. ``budget="coef"`` centres on the
    plain mean (sum b = 3); ``budget="grad"`` centres on the ||d||-weighted mean,
    which is the design's choice and holds sum_j q_j b_j = 1. kappa is the
    largest value keeping every b inside the box, so the most extreme column
    sits on the box edge BY CONSTRUCTION -- b_webshop = 0.5 is that, not a
    precisely measured halving.
    """
    use = ok.copy()
    if not diag:
        for k in range(use.shape[1]):
            use[:, k, k] = False
    M, counts = masked_mean(cos, use)
    # Exactly zero is a measured magnitude, not a missing value: dropping it
    # from the mean reports the norm of the batches where the teacher DID push,
    # which is a larger number than the one the budget is made of. With norms
    # (1,1,1) and (0,1,1) the mean is (0.5,1,1) and q is (0.2,0.4,0.4); treating
    # the zero as missing gives (1,1,1) and q = (1/3,1/3,1/3).
    #
    # A zero makes the COSINE undefined, and that is handled per cell in
    # cosine_cells -- two different things that were sharing one filter.
    nd_ok = np.isfinite(nd)
    nd_mean, nd_counts = masked_mean(nd, nd_ok)
    if check:
        required = np.zeros(use.shape[1:], bool)
        required[np.array(S), :] = True
        if not diag:
            np.fill_diagonal(required, False)
        check_coverage(counts, len(cos), required, min_fraction=min_fraction,
                       labels=lambda ij: f"C[RL {TASKS[ij[0]]}, OPD {TASKS[ij[1]]}]")
        check_coverage(nd_counts, len(cos), np.ones(len(nd_counts), bool),
                       min_fraction=min_fraction, labels=lambda ij: f"||d|| of {TASKS[ij[0]]}")
    M = np.where(np.isfinite(M), M, 0.0)
    c = M[S].sum(0)
    q = nd_mean / nd_mean.sum() if budget == "grad" else np.full(len(nd_mean), 1 / len(nd_mean))
    dev = c - (q * c).sum()
    if np.all(dev == 0):
        return np.ones(len(c)), c, q, counts
    kap = min((1 - bmin) / max(-dev.min(), 1e-12), (bmax - 1) / max(dev.max(), 1e-12))
    return np.clip(1 + kap * dev, bmin, bmax), c, q, counts


def budgets(b, D, nd):
    """The two OPD 'amounts', with the composite one under BOTH conventions.

    linear      sum_j b_j mean_n ||d_j||           -- what the rule preserves
    composite   ||sum_j b_j d_j||, per batch, then RMS and plain mean over
                batches. The design doc quotes the RMS (control 1.0700 ->
                1.1885); the mean is 1.0501 -> 1.1644. Reporting one under the
                other's name is how the uniform arm's factor came out as two
                different numbers in two places.
    """
    nd_ok = np.isfinite(nd)
    nd_mean, _ = masked_mean(nd, nd_ok)
    lin = float((b * nd_mean).sum())
    per_batch = np.array([math.sqrt(max(b @ D[n] @ b, 0.0)) for n in range(len(D))])
    return lin, float(np.sqrt((per_batch ** 2).mean())), float(per_batch.mean())


def uniform_match(b, D, nd):
    """The scalar that puts a uniform arm on the redistributed COMPOSITE norm (RMS)."""
    _, rms_b, _ = budgets(b, D, nd)
    _, rms_1, _ = budgets(np.ones(len(b)), D, nd)
    return rms_b / rms_1 if rms_1 > 0 else float("nan")


def _order(b):
    return tuple(TASKS[i] for i in np.argsort(-np.asarray(b)))


def redistribution_report(path, *, min_fraction=MIN_BATCH_FRACTION, json_out=None,
                          n_boot=2000, seed=0):
    d, C, R, D, nr, nd = load(path)
    NB = len(C)
    cos, ok = cosine_cells(C, nr, nd)
    print("\n=== redistribution rule ===")
    print(f"payload {path}   batches {NB}   min usable batches per cell: "
          f"{max(1, int(math.ceil(min_fraction * NB)))}")
    unusable = int((~ok).sum())
    print(f"unusable (batch, i, j) cells: {unusable} of {ok.size}"
          + ("" if unusable == 0 else "  <- dropped from that cell's mean, not from the matrix"))
    nd_ok = np.isfinite(nd)
    nd_mean, _ = masked_mean(nd, nd_ok)
    print("mean ||d_j|| =", nd_mean, f" max/min = {nd_mean.max() / nd_mean.min():.2f}")
    lin0, rms0, mean0 = budgets(np.ones(3), D, nd)
    print(f"control b=(1,1,1): sum b||d|| = {lin0:.4f}   ||sum b d|| RMS = {rms0:.4f}  mean = {mean0:.4f}")

    rows = []
    for budget in ("coef", "grad"):
        for name, Cb, okb in (("cosine", cos, ok), ("dot", C, np.isfinite(C))):
            for Sname, S in (("{alf,web}", [0, 2]), ("all", [0, 1, 2])):
                for diag in (True, False):
                    b, c, q, _ = redistribute(Cb, okb, nd, S, budget=budget, diag=diag,
                                              min_fraction=min_fraction)
                    lin, rms, mean = budgets(b, D, nd)
                    rows.append((budget, name, Sname, diag, b, lin, rms, mean))
    print(f"{'budget':<7}{'stat':<8}{'S':<11}{'diag':<6}{'b':<26}{'sum b':>7}"
          f"{'sum b||d||':>12}{'RMS||sum bd||':>15}{'mean||sum bd||':>16}")
    for budget, name, Sname, diag, b, lin, rms, mean in rows:
        print(f"{budget:<7}{name:<8}{Sname:<11}{str(diag):<6}{str(np.round(b, 4)):<26}{b.sum():>7.3f}"
              f"{lin:>12.4f}{rms:>15.4f}{mean:>16.4f}")

    # The pre-registered configuration: grad budget, cosine matrix, all rows.
    b_main, c_main, q_main, counts = redistribute(cos, ok, nd, [0, 1, 2], budget="grad",
                                                  min_fraction=min_fraction)
    lin_m, rms_m, mean_m = budgets(b_main, D, nd)
    scale = uniform_match(b_main, D, nd)
    print("\npre-registered (grad budget, cosine, all rows, diagonal kept):")
    print(f"  c = {c_main.round(4)}   q = {q_main.round(4)}   b = {b_main.round(6)}")
    print(f"  sum_j q_j b_j = {float((q_main * b_main).sum()):.6f}   (1.000000 is the rule's invariant)")
    print(f"  linear budget {lin_m:.4f} vs control {lin0:.4f}  ({lin_m / lin0 - 1:+.2%})")
    print(f"  composite RMS {rms_m:.4f} vs control {rms0:.4f}  ({rms_m / rms0 - 1:+.2%})")
    print(f"  uniform arm matching that composite RMS: b = {scale:.6f} each")
    per_batch_lin = np.array([
        float((b_main * nd[n]).sum() / nd[n].sum()) for n in range(NB)
        if np.isfinite(nd[n]).all() and nd[n].sum() > 0
    ])
    print(f"  per-batch linear budget ratio: {per_batch_lin.min():.3f}-{per_batch_lin.max():.3f} "
          f"(the invariant holds on the MEAN, not on each batch)")

    # ---- bootstrap: the sample's own ranking, and agreement with a reference
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, NB, NB)
        try:
            boot.append(redistribute(cos[idx], ok[idx], nd[idx], [0, 1, 2],
                                     budget="grad", min_fraction=min_fraction, check=False)[0])
        except ValueError:
            continue
    boot = np.array(boot)
    measured = _order(b_main)
    orders = [_order(x) for x in boot]
    rank_repro = float(np.mean([o == measured for o in orders]))
    ref_agree = float(np.mean([o == REFERENCE_ORDER for o in orders]))
    print(f"\nbootstrap ({len(boot)} resamples of {NB} batches):")
    print(f"  measured ranking            : {' > '.join(measured)}")
    print(f"  rank_reproducibility        : {rank_repro:.3f}   <- is THIS measurement stable")
    print(f"  agreement_with_reference    : {ref_agree:.3f}   (reference {' > '.join(REFERENCE_ORDER)})")
    print(f"  mean b                      : {boot.mean(0).round(4)}")
    pct = np.percentile(boot, [5, 95], axis=0).T
    for j, t in enumerate(TASKS):
        print(f"    {t:<9} 5-95% = [{pct[j][0]:.3f}, {pct[j][1]:.3f}]   "
              f"P(b>1) = {(boot[:, j] > 1).mean():.3f}   P(b<1) = {(boot[:, j] < 1).mean():.3f}")
    # THE GO CONDITION GATES THE OUTPUT, it does not merely comment on it.
    # Printing "kappa = 0" while returning and saving a non-trivial b is how a
    # vector that failed its own admission test ends up pinned in a lock file
    # three commands later.
    approved = rank_repro >= RANK_REPRO_MIN
    b_approved = b_main if approved else np.ones_like(b_main)
    if not approved:
        print(f"  GO CONDITION NOT MET: rank_reproducibility {rank_repro:.3f} < "
              f"{RANK_REPRO_MIN} -> kappa = 0. approved_b = {b_approved.round(6)} "
              f"(run the control only); the candidate is reported for inspection "
              f"but must not be used as an arm.")

    print("\nleave-one-batch-out (grad budget, cosine, all rows):")
    loo = {}
    for n in range(NB):
        idx = [k for k in range(NB) if k != n]
        bn = redistribute(cos[idx], ok[idx], nd[idx], [0, 1, 2], budget="grad",
                          min_fraction=min_fraction, check=False)[0]
        loo[f"drop_b{n + 1}"] = [float(x) for x in bn]
        print(f"  drop b{n + 1}: b = {bn.round(4)}")

    if json_out:
        M, _ = masked_mean(cos, ok)
        payload = {
            "source": path,
            "checkpoint": d.get("checkpoint"),
            "tasks": TASKS,
            "n_batches": NB,
            "min_batches_required": max(1, int(math.ceil(min_fraction * NB))),
            "usable_batches_per_cell": counts.tolist(),
            "cosine_matrix": np.where(np.isfinite(M), M, None).tolist(),
            "opd_grad_norm_mean": nd_mean.tolist(),
            "q": q_main.tolist(),
            "c": c_main.tolist(),
            # Two fields, deliberately. "b" was BOTH the candidate and the thing
            # a reader would copy into a lock file, so a failed go condition had
            # nowhere to show up.
            "candidate_b": b_main.tolist(),
            "approved_b": b_approved.tolist(),
            "approved": bool(approved),
            "rank_reproducibility_min": RANK_REPRO_MIN,
            "sum_q_b": float((q_main * b_main).sum()),
            "budget_linear": {"control": lin0, "redistributed": lin_m},
            "budget_composite_rms": {"control": rms0, "redistributed": rms_m},
            "budget_composite_mean": {"control": mean0, "redistributed": mean_m},
            "per_batch_linear_ratio": per_batch_lin.tolist(),
            "uniform_arm_scale_rms": scale,
            "measured_order": list(measured),
            "reference_order": list(REFERENCE_ORDER),
            "rank_reproducibility": rank_repro,
            "agreement_with_reference": ref_agree,
            "bootstrap_mean_b": boot.mean(0).tolist(),
            "bootstrap_5_95": pct.tolist(),
            "leave_one_out": loo,
        }
        with open(json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {json_out}")
    return b_approved


def _flag_value(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


# One dispatch, at the bottom. It used to be two, with the plain-QP one sitting
# ABOVE the redistribution section -- so main() called helpers that the module
# had not defined yet and died with a NameError the moment those helpers moved.
if __name__ == "__main__":
    if "--redistribute" in sys.argv:
        b = redistribution_report(
            _payload_arg(),
            min_fraction=_flag_value("--min-batches", MIN_BATCH_FRACTION, float),
            json_out=_flag_value("--json"),
        )
        # Non-zero exit when the admission test failed, so a shell pipeline that
        # feeds this into a launch cannot walk past it.
        if np.allclose(b, 1.0):
            print("approved_b is uniform: nothing to redistribute", file=sys.stderr)
            sys.exit(3)
    else:
        main(_payload_arg())
