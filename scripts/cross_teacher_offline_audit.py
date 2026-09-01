#!/usr/bin/env python3
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
"""Audit the cross-teacher premise from the dumps a run already wrote.

    python3 scripts/cross_teacher_offline_audit.py
    python3 scripts/cross_teacher_offline_audit.py --section deadzone
    python3 scripts/cross_teacher_offline_audit.py --dir ~/sign_tokens/<run> --label mine

The wandb scalars say how large an intervention was and how concentrated; they
cannot say whether the quantity being gated carries information. That question
is answerable offline, from ``trainer.sign_token_dump_dir``, with no GPU and no
model load -- which matters because the arms that would answer it occupy both
cards for a day. Four sections, each a separate claim:

``deadzone``   Sweeps the sign deadzone over the 3-teacher event dump and reports,
               for each epsilon, the teacher probability mass a unanimity rule
               would reach AND the part of it a coin flip already explains. Reads
               ``sign_events_step*.jsonl`` (needs shift_off_lo / shift_off_hi).

``measures``   Compares candidate agreement measures on COVERAGE (mass reached),
               SIGNAL (does the off-task teacher predict the on-task teacher's
               target beyond base) and OUTCOME (does it predict the row's
               advantage). Reads ``sign_pair_events_step*.jsonl``.

``action``     Re-runs the outcome test at two coarser granularities: role-
               stratified (env_action vs format), and span level -- (step, dst,
               advantage, reward, row_len) reconstructs one turn's response,
               because ``turn`` is constant inside such a group.

``intent2``    The separate claim that off-task teachers help WHERE THE ON-TASK
               TEACHER DOES NOT: splits sampled tokens by p_on and asks whether
               p_source says anything about advantage inside each band.

``delta``      Whether log p_teacher - log p_base is worth anything AT ALL, which
               is a different question from whether it transfers across tasks.
               Puts the on-task and the off-task delta on the same sample and
               bootstraps the DIFFERENCE. Read this before concluding anything
               about the shift as a quantity: the cross-task nulls above do not
               licence a null on the on-task shift, and here it clears zero.

               A TRAP THIS SECTION EXISTS TO AVOID. delta_on and delta_src share
               the -log p_base term, so if log p_base correlates with the outcome
               -- it does, negatively, in the control -- then BOTH deltas inherit
               a spurious association from it. Only the partial test, holding
               L_base fixed, separates the teacher's own contribution.

``tokens``     The aggregated token tables (``sign_tokens_step*.jsonl``), including the
               WEIGHT-FREE schema a control writes -- ranked_by base_logit_push /
               kl_mass, direction_class base_up / base_down, no state labels.
               ``sign_token_scan.py`` cannot read that schema (it keys on state and
               effect_net), which is why this lives here. Says where the plain
               teacher-KL budget sits, splits it into hand-assigned token classes,
               and tracks how it moves across training.

``shared``      The shared FORMAT direction, which is a different quantity from
               agreement and has the opposite sign to the base LEVEL. Off-task
               teachers know nothing about the destination task, so their shift
               on its states estimates "where RL of this family goes regardless
               of task". Reports it with log p_base partialled out, split by
               period and by task, and also tests the SUBTRACTION idea (removing
               the shared direction from the on-task shift), which this data
               refutes. Read the splits before believing the pooled number.

THREE THINGS THIS DOES THAT READING THE JSONL DOES NOT.

**It centres the sign test against the marginals.** The teachers have very
different up/down tendencies -- P(shift > 0) ranges from 0.24 to 0.89 across
(dst, src) pairs -- so a raw "how often do the signs match" count mostly measures
that skew. The independence baseline is pq + (1-p)(1-q), not 0.5, and the phi
coefficient is reported beside the raw rate.

**It separates inherited from novel agreement.** Spearman(p_on, p_source) is
0.82, which reads as a strong result until p_base is partialled out and it goes
to 0.00-0.04. Every level-agreement number here is reported both ways, because
the raw one is the shared prior and says nothing about either task.

**Its intervals respect the clustering.** Candidates inside one position are not
independent, and neither are positions inside one step. Every CI is a cluster
bootstrap over (step, dst); a naive resample is several times too tight and would
turn most of these nulls into findings.

Spearman throughout: log p and p share ranks, so the denormal floor never enters
a correlation. ``--nboot`` trades runtime for CI resolution; the default is
enough to read a 95% interval to about three digits.
"""
import argparse
import collections
import glob
import json
import os
import re

import numpy as np

DEFAULT_ARMS = [
    ("ARM(sg1, mechanism ON)", "~/sign_tokens/opd_grpo_multitask_cross_teacher_klw_qwen3_1.7b_sg1"),
    ("CTL(xt1, mechanism OFF)", "~/sign_tokens/opd_grpo_multitask_cross_teacher_klw_qwen3_1.7b_xt1"),
]
FLOOR = 1e-30
# 3-teacher unanimity is one of 2 of the 8 equally likely sign triples.
CHANCE_3 = 0.25


# --------------------------------------------------------------------------- io
def load_jsonl(directory, stem):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, f"{stem}_step*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------- statistics
def rank(values):
    values = np.asarray(values, dtype=float)
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(rank(a), rank(b))


def partial_spearman(a, b, given):
    """Spearman(a, b | given): residualise both rank vectors on the given one."""
    ra, rb, rg = rank(a), rank(b), rank(given)
    if len(ra) < 5 or rg.std() == 0:
        return float("nan")
    design = np.vstack([np.ones_like(rg), rg]).T
    resid_a = ra - design @ np.linalg.lstsq(design, ra, rcond=None)[0]
    resid_b = rb - design @ np.linalg.lstsq(design, rb, rcond=None)[0]
    return pearson(resid_a, resid_b)


def cluster_bootstrap(stat, groups, rng, nboot):
    """Point estimate plus a 95% percentile interval, resampling whole clusters."""
    keys = list(groups.keys())
    point = stat(np.concatenate([groups[k] for k in keys])) if keys else float("nan")
    draws = []
    for _ in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        value = stat(np.concatenate([groups[keys[p]] for p in pick]))
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < 50:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def clusters_by(step, dst, indices):
    groups = collections.defaultdict(list)
    for i in indices:
        groups[(int(step[i]), str(dst[i]))].append(i)
    return {k: np.array(v) for k, v in groups.items()}


def report(label, triple, suffix=""):
    point, lo, hi = triple
    flag = "   <-- CI excludes 0" if np.isfinite(lo) and (lo > 0 or hi < 0) else ""
    print(f"      {label:44s}{point:>9.4f}  CI[{lo:+.4f},{hi:+.4f}]{suffix}{flag}")


# --------------------------------------------------------- agreement measures
def measure_sign(f):
    """The shipped premise: the two shifts share a sign."""
    return ((f["d_on"] * f["d_src"]) > 0).astype(float)


def measure_sign_magnitude(f):
    """Sign agreement scaled by the weakest voice -- what min|delta| would give,
    and the reason a deadzone is not needed to suppress noise-signed candidates."""
    agree = (f["d_on"] * f["d_src"]) > 0
    return np.where(agree, np.minimum(np.abs(f["d_on"]), np.abs(f["d_src"])), 0.0)


def _ratio(x, y):
    lo, hi = np.minimum(x, y), np.maximum(x, y)
    return np.where(hi > 0, lo / np.maximum(hi, FLOOR), 0.0)


def measure_level(f):
    """Agreement on the LEVEL rather than the shift: min(p)/max(p) in [0, 1].
    Reaches the head of the distribution, which no shift rule can."""
    return _ratio(f["p_on"], f["p_src"])


def measure_level_novel(f):
    """Level agreement with the part base already supplies removed."""
    return measure_level(f) * (1.0 - _ratio(f["p_on"], f["p_base"]))


MEASURES = [
    ("sign (shipped)", measure_sign),
    ("sign x min|delta|", measure_sign_magnitude),
    ("level r", measure_level),
    ("level x novelty", measure_level_novel),
]


def fields(rows):
    out = {
        "p_on": np.array([r["p_on"] for r in rows]),
        "p_src": np.array([r["p_source"] for r in rows]),
        "p_base": np.array([r["p_base"] for r in rows]),
        "p_stu": np.array([r["p_student"] for r in rows]),
        "d_on": np.array([r["delta_on_raw"] for r in rows]),
        "d_src": np.array([r["delta_source_raw"] for r in rows]),
        "adv": np.array([r["advantage"] for r in rows]),
        "samp": np.array([r["is_sampled"] for r in rows]),
        "step": np.array([r["step"] for r in rows]),
        "dst": np.array([r["dst"] for r in rows]),
    }
    return out


# ------------------------------------------------------------------- sections
def section_deadzone(directory):
    """Does a smaller deadzone reach more mass, and is the added mass signal?"""
    rows = [r for r in load_jsonl(directory, "sign_events") if r["stratum"] == "spread"]
    if len(rows) < 50:
        print(f"    sign_events spread rows = {len(rows)}: too few")
        return
    p_on = np.array([r["p_on"] for r in rows])
    p_base = np.array([r["p_base"] for r in rows])
    s_on = np.array([r["shift_on"] for r in rows])
    s_lo = np.array([r["shift_off_lo"] for r in rows])
    s_hi = np.array([r["shift_off_hi"] for r in rows])
    weakest = np.minimum(np.abs(s_on), np.minimum(np.abs(s_lo), np.abs(s_hi)))
    unanimous = (s_on != 0) & (s_lo * s_on > 0) & (s_hi * s_on > 0)
    total = p_on.sum()

    # one RMS unit in nats, recovered from the raw and standardised columns
    ok = (p_base > 0) & (p_on > 0) & (np.abs(s_on) > 1e-6)
    scale = (np.log(p_on[ok]) - np.log(p_base[ok])) / s_on[ok]
    scale = np.median(scale[np.isfinite(scale) & (scale > 0)])
    print(f"    spread rows = {len(rows)};  1 RMS unit ~ {scale:.3f} nats")
    print(f"      {'eps(RMS)':>9s}{'eps(nats)':>10s}{'agree mass':>12s}{'up':>9s}{'down':>9s}"
          f"{'chance':>9s}{'excess':>9s}{'purity':>8s}")
    for eps in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0):
        keep = weakest > eps
        if keep.sum() < 5:
            continue
        agree = keep & unanimous
        mass_agree = p_on[agree].sum() / total
        mass_keep = p_on[keep].sum() / total
        up = p_on[agree & (s_on > 0)].sum() / total
        down = p_on[agree & (s_on < 0)].sum() / total
        chance = mass_keep * CHANCE_3
        excess = mass_agree - chance
        purity = excess / mass_agree if mass_agree > 0 else float("nan")
        print(f"      {eps:9.3f}{eps * scale:10.4f}{mass_agree:12.4f}{up:9.4f}{down:9.4f}"
              f"{chance:9.4f}{excess:9.4f}{purity:8.2f}")
    print("      up/down is the budget a mass-conserving (Z=1) rule can move without")
    print("      taxing the head; they balance near eps ~ 0.03 RMS.")

    head = np.abs(s_on) < 0.01
    if head.any():
        print(f"    the unreachable head: |shift_on| < 0.01 RMS holds {p_on[head].sum() / total:.3f}"
              f" of mass, p_on median {np.median(p_on[head]):.4f},"
              f" p_base median {np.median(p_base[head]):.4f},"
              f" weakest shift exactly 0 in {(weakest[head] == 0).mean():.3f} of them")


def section_measures(directory, rng, nboot):
    rows = load_jsonl(directory, "sign_pair_events")
    spread = [r for r in rows if r["stratum"] == "spread"]
    if len(spread) < 100:
        print(f"    sign_pair_events spread rows = {len(spread)}: too few")
        return
    f = fields(spread)
    groups = clusters_by(f["step"], f["dst"], np.arange(len(spread)))
    total = f["p_on"].sum()
    print(f"    spread rows = {len(spread)}, sampled = {int(f['samp'].sum())},"
          f" clusters = {len(groups)}")

    print()
    print("    [signal] does the off-task teacher predict the on-task target beyond base?")
    report("Spearman(delta_on, delta_src)",
           cluster_bootstrap(lambda i: spearman(f["d_on"][i], f["d_src"][i]), groups, rng, nboot))
    report("Spearman(p_on, p_src)",
           cluster_bootstrap(lambda i: spearman(f["p_on"][i], f["p_src"][i]), groups, rng, nboot))
    report("  ^ partialling out p_base",
           cluster_bootstrap(lambda i: partial_spearman(f["p_on"][i], f["p_src"][i], f["p_base"][i]),
                             groups, rng, nboot))
    strong = np.where(np.minimum(np.abs(f["d_on"]), np.abs(f["d_src"])) > 0.62)[0]
    if len(strong) > 50:
        sub = clusters_by(f["step"], f["dst"], strong)
        report("Spearman(delta_on, delta_src) strong only",
               cluster_bootstrap(lambda i: spearman(f["d_on"][i], f["d_src"][i]), sub, rng, nboot),
               suffix=f"  n={len(strong)}")

    print()
    print("    [coverage] what each measure fires on (mass = share of sum p_on)")
    print(f"      {'measure':22s}{'fires':>9s}{'mass':>9s}{'p_on med':>11s}{'p_stu med':>11s}{'sampled':>9s}")
    for label, fn in MEASURES:
        score = fn(f)
        fired = score > 0
        if not fired.any():
            continue
        print(f"      {label:22s}{fired.mean():>9.4f}{f['p_on'][fired].sum() / total:>9.4f}"
              f"{np.median(f['p_on'][fired]):>11.2e}{np.median(f['p_stu'][fired]):>11.2e}"
              f"{f['samp'][fired].mean():>9.4f}")
    print(f"      {'(all candidates)':22s}{1.0:>9.4f}{1.0:>9.4f}"
          f"{np.median(f['p_on']):>11.2e}{np.median(f['p_stu']):>11.2e}{f['samp'].mean():>9.4f}")

    print()
    print("    [novelty] level agreement, restricted to where base does NOT agree")
    lvl = measure_level(f)
    base_agree = _ratio(f["p_on"], f["p_base"])
    for thr in (0.9, 0.5, 0.1):
        hit = lvl > thr
        print(f"      r>{thr}: mass {f['p_on'][hit].sum() / total:.4f}"
              f"   & base<0.5: {f['p_on'][hit & (base_agree < 0.5)].sum() / total:.4f}"
              f"   & base<0.1: {f['p_on'][hit & (base_agree < 0.1)].sum() / total:.4f}")

    print()
    print("    [outcome] on the tokens the student emitted (all strata; effect-biased)")
    fa = fields(rows)
    idx = np.where((fa["samp"] == 1) & (fa["adv"] != 0.0))[0]
    print(f"      n = {len(idx)}")
    if len(idx) >= 40:
        sub = clusters_by(fa["step"], fa["dst"], idx)
        for label, fn in MEASURES:
            score = fn(fa)
            report(f"Spearman({label}, adv)",
                   cluster_bootstrap(lambda i, s=score: spearman(s[i], fa["adv"][i]), sub, rng, nboot))
        print("      reference -- the ceiling for any teacher-probability measure:")
        for label, vec in (("p_on", fa["p_on"]), ("p_base", fa["p_base"]), ("p_src", fa["p_src"])):
            report(f"Spearman({label}, adv)",
                   cluster_bootstrap(lambda i, v=vec: spearman(v[i], fa["adv"][i]), sub, rng, nboot))


def _span_units(rows, role=None):
    """(step, dst, advantage, reward, row_len, src) -- turn is constant inside such
    a group, so one key is one turn's response as one source saw it."""
    units = collections.defaultdict(list)
    for r in rows:
        if r["is_sampled"] != 1 or r["advantage"] == 0.0:
            continue
        if role is not None and r["role"] != role:
            continue
        key = (r["step"], r["dst"], round(r["advantage"], 6), round(r["reward"], 4),
               r["row_len"], r["src"])
        units[key].append(r)
    return units


def _span_arrays(units, keys):
    logmean = lambda rs, k: float(np.mean([np.log(max(x[k], FLOOR)) for x in rs]))
    return {
        "L_on": np.array([logmean(units[k], "p_on") for k in keys]),
        "L_src": np.array([logmean(units[k], "p_source") for k in keys]),
        "L_base": np.array([logmean(units[k], "p_base") for k in keys]),
        "agree": np.array([float(np.mean([1.0 if x["delta_on_raw"] * x["delta_source_raw"] > 0 else 0.0
                                          for x in units[k]])) for k in keys]),
        "adv": np.array([k[2] for k in keys]),
        "step": np.array([k[0] for k in keys]),
        "dst": np.array([k[1] for k in keys]),
    }


def section_action(directory, rng, nboot):
    rows = load_jsonl(directory, "sign_pair_events")
    if not rows:
        print("    no sign_pair_events")
        return

    print("    [by role] token level, sampled and advantage != 0")
    for role in ("env_action", "tool_call", "format", "tag"):
        ev = [r for r in rows if r["is_sampled"] == 1 and r["advantage"] != 0.0 and r["role"] == role]
        if len(ev) < 30:
            print(f"      [{role}] n={len(ev)}: too few")
            continue
        f = fields(ev)
        groups = clusters_by(f["step"], f["dst"], np.arange(len(ev)))
        print(f"      [{role}] n={len(ev)}, clusters={len(groups)}")
        for label, fn in MEASURES[:1] + MEASURES[2:3]:
            score = fn(f)
            report(f"Spearman({label}, adv)",
                   cluster_bootstrap(lambda i, s=score: spearman(s[i], f["adv"][i]), groups, rng, nboot))
        report("Spearman(p_on, adv)",
               cluster_bootstrap(lambda i: spearman(f["p_on"][i], f["adv"][i]), groups, rng, nboot))
        report("partial Sp(p_on, p_src | p_base)",
               cluster_bootstrap(lambda i: partial_spearman(f["p_on"][i], f["p_src"][i], f["p_base"][i]),
                                 groups, rng, nboot))

    print()
    print("    [span level] one turn's response, sampled tokens aggregated")
    units = _span_units(rows)
    for minimum in (2, 3):
        keep = {k: v for k, v in units.items() if len(v) >= minimum}
        if len(keep) < 30:
            print(f"      min_tokens={minimum}: units={len(keep)}: too few")
            continue
        keys = list(keep.keys())
        a = _span_arrays(keep, keys)
        groups = clusters_by(a["step"], a["dst"], np.arange(len(keys)))
        print(f"      min_tokens={minimum}: units={len(keys)}, clusters={len(groups)}")
        report("partial Sp(L_on, L_src | L_base)",
               cluster_bootstrap(lambda i: partial_spearman(a["L_on"][i], a["L_src"][i], a["L_base"][i]),
                                 groups, rng, nboot))
        report("Spearman(agree rate, adv)",
               cluster_bootstrap(lambda i: spearman(a["agree"][i], a["adv"][i]), groups, rng, nboot))
        report("Spearman(L_on - L_base, adv)",
               cluster_bootstrap(lambda i: spearman(a["L_on"][i] - a["L_base"][i], a["adv"][i]),
                                 groups, rng, nboot))
        report("Spearman(L_src - L_base, adv)",
               cluster_bootstrap(lambda i: spearman(a["L_src"][i] - a["L_base"][i], a["adv"][i]),
                                 groups, rng, nboot))

    print()
    print("    [span level, env_action only]")
    units = _span_units(rows, role="env_action")
    if len(units) >= 30:
        keys = list(units.keys())
        a = _span_arrays(units, keys)
        groups = clusters_by(a["step"], a["dst"], np.arange(len(keys)))
        print(f"      units={len(keys)} (>=2 tokens: {sum(1 for v in units.values() if len(v) >= 2)}),"
              f" clusters={len(groups)}")
        report("partial Sp(L_on, L_src | L_base)",
               cluster_bootstrap(lambda i: partial_spearman(a["L_on"][i], a["L_src"][i], a["L_base"][i]),
                                 groups, rng, nboot))
        report("Spearman(agree rate, adv)",
               cluster_bootstrap(lambda i: spearman(a["agree"][i], a["adv"][i]), groups, rng, nboot))
        report("Spearman(L_src - L_base, adv)",
               cluster_bootstrap(lambda i: spearman(a["L_src"][i] - a["L_base"][i], a["adv"][i]),
                                 groups, rng, nboot))
    else:
        print(f"      units={len(units)}: too few")


def section_intent2(directory, rng, nboot):
    """Off-task teachers where the on-task teacher does not back what was emitted."""
    rows = load_jsonl(directory, "sign_pair_events")
    f = fields(rows)
    idx = np.where((f["samp"] == 1) & (f["adv"] != 0.0))[0]
    if len(idx) < 100:
        print(f"    n={len(idx)}: too few")
        return
    p_on, p_src, adv = f["p_on"][idx], f["p_src"][idx], f["adv"][idx]
    step, dst = f["step"][idx], f["dst"][idx]
    print(f"    n = {len(idx)}")
    print(f"      {'p_on band':>18s}{'n':>7s}{'adv mean':>11s}{'p_src med':>11s}  Spearman(p_src, adv)")
    edges = (0.0, 1e-4, 1e-2, 0.1, 0.5, 1.01)
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        sel = np.where((p_on >= lo_e) & (p_on < hi_e))[0]
        if len(sel) < 25:
            print(f"      {f'[{lo_e:g},{hi_e:g})':>18s}{len(sel):>7d}   too few")
            continue
        groups = clusters_by(step, dst, sel)
        point, lo, hi = cluster_bootstrap(lambda i: spearman(p_src[i], adv[i]), groups, rng, nboot)
        flag = "   <--" if np.isfinite(lo) and (lo > 0 or hi < 0) else ""
        print(f"      {f'[{lo_e:g},{hi_e:g})':>18s}{len(sel):>7d}{adv[sel].mean():>11.4f}"
              f"{np.median(p_src[sel]):>11.2e}   {point:+.4f} CI[{lo:+.4f},{hi:+.4f}]{flag}")

    low = np.where(p_on < 1e-2)[0]
    if len(low) >= 40:
        groups = clusters_by(step, dst, low)
        print("      on-task does not back the emitted token (p_on < 1e-2), split by p_src:")
        for thr in (1e-2, 0.1, 0.5):
            def delta(i, t=thr):
                high = p_src[i] >= t
                if high.sum() < 3 or (~high).sum() < 3:
                    return float("nan")
                return float(adv[i][high].mean() - adv[i][~high].mean())
            report(f"delta adv, p_src >= {thr:g}",
                   cluster_bootstrap(delta, groups, rng, nboot))


def section_delta(directory, rng, nboot):
    """Is the shift from base worth anything -- on-task versus off-task, paired."""
    rows = load_jsonl(directory, "sign_pair_events")
    ev = [r for r in rows if r["is_sampled"] == 1 and r["advantage"] != 0.0]
    if len(ev) < 100:
        print(f"    n={len(ev)}: too few")
        return
    log = lambda x: np.log(np.maximum(x, FLOOR))
    p_on = np.array([r["p_on"] for r in ev])
    p_src = np.array([r["p_source"] for r in ev])
    p_base = np.array([r["p_base"] for r in ev])
    adv = np.array([r["advantage"] for r in ev])
    d_on, d_src = log(p_on) - log(p_base), log(p_src) - log(p_base)
    step = np.array([r["step"] for r in ev])
    dst = np.array([r["dst"] for r in ev])
    groups = clusters_by(step, dst, np.arange(len(ev)))
    print(f"    [token level] n={len(ev)}, clusters={len(groups)}")
    report("Spearman(delta_on, adv)",
           cluster_bootstrap(lambda i: spearman(d_on[i], adv[i]), groups, rng, nboot))
    report("Spearman(delta_src, adv)",
           cluster_bootstrap(lambda i: spearman(d_src[i], adv[i]), groups, rng, nboot))
    report("paired difference",
           cluster_bootstrap(lambda i: spearman(d_on[i], adv[i]) - spearman(d_src[i], adv[i]),
                             groups, rng, nboot))
    print("      does the teacher back good actions above base?")
    for label, vec in (("delta_on", d_on), ("delta_src", d_src)):
        def gap(i, v=vec):
            pos, neg = adv[i] > 0, adv[i] < 0
            if pos.sum() < 5 or neg.sum() < 5:
                return float("nan")
            return float((v[i][pos] > 0).mean() - (v[i][neg] > 0).mean())
        report(f"P({label}>0|adv>0) - P(..|adv<0)", cluster_bootstrap(gap, groups, rng, nboot))

    units = {k: v for k, v in _span_units(rows).items() if len(v) >= 2}
    if len(units) < 30:
        print(f"    [span level] units={len(units)}: too few")
        return
    keys = list(units.keys())
    a = _span_arrays(units, keys)
    groups = clusters_by(a["step"], a["dst"], np.arange(len(keys)))
    D_on, D_src = a["L_on"] - a["L_base"], a["L_src"] - a["L_base"]
    print(f"    [span level] units={len(keys)}, clusters={len(groups)}")
    report("Spearman(Delta_on, adv)",
           cluster_bootstrap(lambda i: spearman(D_on[i], a["adv"][i]), groups, rng, nboot))
    report("Spearman(Delta_src, adv)",
           cluster_bootstrap(lambda i: spearman(D_src[i], a["adv"][i]), groups, rng, nboot))
    report("paired difference",
           cluster_bootstrap(lambda i: spearman(D_on[i], a["adv"][i]) - spearman(D_src[i], a["adv"][i]),
                             groups, rng, nboot))
    print("      holding L_base fixed -- the only form that separates the two:")
    for label, vec in (("Delta_on", D_on), ("Delta_src", D_src)):
        report(f"partial Sp({label}, adv | L_base)",
               cluster_bootstrap(lambda i, v=vec: partial_spearman(v[i], a["adv"][i], a["L_base"][i]),
                                 groups, rng, nboot))


def section_shared(directory, rng, nboot):
    """The shared format direction: is it worth using, and does subtracting it help?

    Levels and shifts disagree in sign here and conflating them is the trap:
    a high log p_base means a generic token (mildly bad), while a positive
    mean off-task shift means the RL-tuned models pushed that token up
    (mildly good). Only the second is what this section measures.
    """
    rows = load_jsonl(directory, "sign_pair_events")
    # One candidate is emitted twice, once per source; fold them so the mean over
    # off-task teachers is available. turn/position pin the candidate inside a row.
    cand = collections.defaultdict(list)
    for r in rows:
        if r["is_sampled"] != 1 or r["advantage"] == 0.0:
            continue
        key = (r["step"], r["dst"], r["position"], r["token_id"],
               round(r["advantage"], 6), round(r["reward"], 4), r["row_len"], r["turn"])
        cand[key].append(r)
    both = {k: v for k, v in cand.items() if len({x["src"] for x in v}) >= 2}
    print(f"    candidates {len(cand)} (with both off-task teachers present: {len(both)})")
    if len(both) < 60:
        print("    too few")
        return
    keys = list(both.keys())
    log = lambda x: np.log(max(x, FLOOR))
    L_base = np.array([log(both[k][0]["p_base"]) for k in keys])
    L_on = np.array([log(both[k][0]["p_on"]) for k in keys])
    L_src = np.array([float(np.mean([log(x["p_source"]) for x in both[k]])) for k in keys])
    adv = np.array([k[4] for k in keys])
    step = np.array([k[0] for k in keys])
    dst = np.array([k[1] for k in keys])
    s_gen = L_src - L_base                 # the shared direction
    d_on = L_on - L_base
    contrast = L_on - L_src                # = d_on - s_gen; base cancels exactly

    groups = clusters_by(step, dst, np.arange(len(keys)))
    print(f"    n={len(keys)}, clusters={len(groups)}")
    report("Spearman(s_shared, adv)",
           cluster_bootstrap(lambda i: spearman(s_gen[i], adv[i]), groups, rng, nboot))
    report("Spearman(log p_base, adv)  [the confound]",
           cluster_bootstrap(lambda i: spearman(L_base[i], adv[i]), groups, rng, nboot))
    print("      partialling out log p_base -- the form to believe:")
    report("partial Sp(s_shared, adv | log p_base)",
           cluster_bootstrap(lambda i: partial_spearman(s_gen[i], adv[i], L_base[i]), groups, rng, nboot))
    report("partial Sp(delta_on, adv | log p_base)",
           cluster_bootstrap(lambda i: partial_spearman(d_on[i], adv[i], L_base[i]), groups, rng, nboot))
    all3 = (2.0 * L_src + L_on) / 3.0 - L_base
    report("partial Sp(all-3 mean shift, adv | log p_base)",
           cluster_bootstrap(lambda i: partial_spearman(all3[i], adv[i], L_base[i]), groups, rng, nboot))

    print("      does SUBTRACTING the shared direction help? (negative = it hurts)")
    report("Spearman(delta_on - s_shared, adv)",
           cluster_bootstrap(lambda i: spearman(contrast[i], adv[i]), groups, rng, nboot))
    report("paired difference vs delta_on alone",
           cluster_bootstrap(lambda i: spearman(contrast[i], adv[i]) - spearman(d_on[i], adv[i]),
                             groups, rng, nboot))

    print("      splits -- the pooled number does not survive these:")
    stat = lambda i: partial_spearman(s_gen[i], adv[i], L_base[i])
    for label, mask in (("steps <=150", step <= 150), ("steps >150", step > 150)):
        sub = clusters_by(step, dst, np.where(mask)[0])
        if int(mask.sum()) >= 40:
            report(f"  {label}", cluster_bootstrap(stat, sub, rng, nboot), suffix=f"  n={int(mask.sum())}")
    for task in sorted(set(dst.tolist())):
        mask = dst == task
        if int(mask.sum()) < 40:
            continue
        sub = clusters_by(step, dst, np.where(mask)[0])
        report(f"  {task}", cluster_bootstrap(stat, sub, rng, nboot), suffix=f"  n={int(mask.sum())}")


# Hand-assigned token classes. The boundary is arbitrary but reproducible, and
# "content" is the residual bucket -- it still catches function-ish verbs like
# 'select'. Reported so the shares can be re-derived under a different split.
_DISCOURSE = {"Therefore", "Thus", "Since", "However", "So", "Then", "Given", "Because",
              "First", "Next", "Now", "Finally", "Also", "But", "Hence", "Additionally"}
_DISCOURSE |= {"\u0120" + w for w in list(_DISCOURSE)}
_FUNCTION = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "for", "with",
             "that", "this", "it", "on", "at", "as", "be", "will", "can", "have", "has",
             "not", "by", "from", "if", "was", "were", "do", "does", "there", "which",
             "but", "all", "more", "one", "its", "their", "some"}
_FUNCTION = {"\u0120" + w for w in _FUNCTION} | {"The", "A", "I", "\u0120I", "\u0120The"}
_STRUCT = re.compile(r"^(\u0120?[<>\[\]{}/\\|]|<\|)|(<th|</|>\u010a|im_end|endoftext)")
_PUNCT = re.compile(r"^[\u0120\u010a\s.,:;!?\"'`()\-\u2013\u2014_=+*&%$#@~^]+$")
_NUM = re.compile(r"^\u0120?\d+$")
_CLASSES = ["discourse", "structure", "punct", "number", "function", "content"]


def token_class(token):
    if token in _DISCOURSE:
        return "discourse"
    if _STRUCT.search(token):
        return "structure"
    if _PUNCT.match(token):
        return "punct"
    if _NUM.match(token):
        return "number"
    if token in _FUNCTION:
        return "function"
    return "content"


def section_tokens(directory, rng, nboot):
    """Where the plain teacher-KL budget sits, and how it moves."""
    files = sorted(glob.glob(os.path.join(directory, "sign_tokens_step*.jsonl")))
    if not files:
        print("    no sign_tokens files")
        return
    steps = sorted(int(re.search(r"step(\d+)", f).group(1)) for f in files)

    def load(step, scope, ranked_by):
        out = []
        with open(os.path.join(directory, f"sign_tokens_step{step:06d}.jsonl")) as h:
            for line in h:
                r = json.loads(line)
                if r["scope"] == scope and r.get("ranked_by") == ranked_by:
                    out.append(r)
        return out

    probe = load(steps[-1], "__pooled__", "kl_mass")
    if not probe:
        print("    no kl_mass rows -- this run's tables use a different schema")
        return
    scopes = ["__pooled__"] + [t for t in ("alfworld", "search", "webshop")
                               if load(steps[-1], t, "kl_mass")]
    late = [x for x in steps if x >= steps[-1] - 25]
    print(f"    steps {steps[0]}..{steps[-1]} ({len(steps)} files); late window {late}")
    print("    NOTE tables are truncated to top-64 per (scope, ranked_by); every share")
    print("         below is relative to that top-64, not to the whole step.")

    print()
    print("    [budget direction and class shares, late window]")
    print(f"      {'scope':12s}{'down frac':>10s}{'wgt p_stu':>10s}{'p>0.5':>8s}"
          + "".join(f"{c:>10s}" for c in _CLASSES))
    for scope in scopes:
        rows = [r for st in late for r in load(st, scope, "kl_mass")]
        if not rows:
            continue
        mass = np.array([abs(r["kl_mass"]) for r in rows])
        pstu = np.array([r["p_student_mean"] for r in rows])
        down = sum(abs(r["kl_mass"]) for r in rows if r["direction_class"] == "base_down")
        share = collections.defaultdict(float)
        for r in rows:
            share[token_class(r["token"])] += abs(r["kl_mass"])
        tot = sum(share.values()) or 1.0
        print(f"      {scope:12s}{down / mass.sum():10.3f}{(mass * pstu).sum() / mass.sum():10.3f}"
              f"{mass[pstu > 0.5].sum() / mass.sum():8.3f}"
              + "".join(f"{share[c] / tot:10.3f}" for c in _CLASSES))

    print()
    print("    [what the teacher asks for more / less of, late window, by base_logit_push]")
    for scope in [s for s in scopes if s != "__pooled__"]:
        acc = collections.defaultdict(lambda: [0.0, 0, 0.0])
        for st in late:
            for r in load(st, scope, "base_logit_push"):
                a = acc[r["token"]]
                a[0] += r["base_logit_push"]; a[1] += r["count"]; a[2] += r["p_student_mean"]
        n = max(len(late), 1)
        up = sorted((kv for kv in acc.items() if kv[1][0] > 0), key=lambda kv: -kv[1][0])[:8]
        dn = sorted((kv for kv in acc.items() if kv[1][0] < 0), key=lambda kv: kv[1][0])[:8]
        print(f"      --- {scope} ---")
        print(f"        {'up':>26s}{'push':>8s}     {'down':>26s}{'push':>8s}")
        for i in range(max(len(up), len(dn))):
            left = f"{up[i][0]!r:>26s}{up[i][1][0] / n:8.2f}" if i < len(up) else " " * 34
            right = f"{dn[i][0]!r:>26s}{dn[i][1][0] / n:8.2f}" if i < len(dn) else ""
            print(f"        {left}     {right}")

    print()
    print("    [how the budget moves across training, scope=__pooled__]")
    print(f"      {'step':>6s}{'down':>8s}{'wgt p_stu':>11s}{'top5':>8s}{'discourse':>11s}"
          f"{'content':>9s}   top-3 tokens")
    for st in steps[::max(1, len(steps) // 12)]:
        rows = load(st, "__pooled__", "kl_mass")
        if not rows:
            continue
        mass = np.array([abs(r["kl_mass"]) for r in rows])
        pstu = np.array([r["p_student_mean"] for r in rows])
        by = lambda cls: sum(abs(r["kl_mass"]) for r in rows if token_class(r["token"]) == cls)
        down = sum(abs(r["kl_mass"]) for r in rows if r["direction_class"] == "base_down")
        top3 = ", ".join(repr(r["token"]) for r in
                         sorted(rows, key=lambda x: -abs(x["kl_mass"]))[:3])
        print(f"      {st:6d}{down / mass.sum():8.3f}{(mass * pstu).sum() / mass.sum():11.3f}"
              f"{np.sort(mass)[::-1][:5].sum() / mass.sum():8.3f}{by('discourse') / mass.sum():11.3f}"
              f"{by('content') / mass.sum():9.3f}   {top3}")

    print()
    print("    [turnover of the top-32 by |kl_mass| (Jaccard)]")
    picks = [(steps[0], steps[len(steps) // 2]), (steps[len(steps) // 2], steps[-1]),
             (steps[0], steps[-1])]
    def top_set(st, scope, k=32):
        rows = load(st, scope, "kl_mass")
        return {r["token"] for r in sorted(rows, key=lambda x: -abs(x["kl_mass"]))[:k]}
    print(f"      {'scope':12s}" + "".join(f"{f'{a}->{b}':>14s}" for a, b in picks))
    for scope in scopes:
        vals = []
        for a, b in picks:
            A, B = top_set(a, scope), top_set(b, scope)
            vals.append(len(A & B) / len(A | B) if A | B else float("nan"))
        print(f"      {scope:12s}" + "".join(f"{v:14.3f}" for v in vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", action="append", default=None,
                        help="trainer.sign_token_dump_dir; repeatable. Defaults to the two "
                             "cross-teacher arms under ~/sign_tokens.")
    parser.add_argument("--label", action="append", default=None,
                        help="display name for the matching --dir")
    parser.add_argument("--section", default="all",
                        choices=["all", "deadzone", "measures", "action", "intent2", "delta", "shared", "tokens"])
    parser.add_argument("--nboot", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dir:
        labels = args.label or []
        arms = [(labels[i] if i < len(labels) else os.path.basename(d.rstrip("/")), d)
                for i, d in enumerate(args.dir)]
    else:
        arms = DEFAULT_ARMS

    rng = np.random.default_rng(args.seed)
    for label, directory in arms:
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            print(f"### {label}: no such directory {directory}")
            continue
        steps = sorted({r["step"] for r in load_jsonl(directory, "sign_pair_events")})
        print("=" * 104)
        print(f"### {label}")
        print(f"    {directory}")
        if steps:
            print(f"    collected steps {steps[0]}..{steps[-1]} ({len(steps)} files)")
        for name, fn in (("deadzone", section_deadzone), ("measures", section_measures),
                         ("action", section_action), ("intent2", section_intent2),
                         ("delta", section_delta),
                         ("shared", section_shared),
                         ("tokens", section_tokens)):
            if args.section not in ("all", name):
                continue
            print()
            print(f"  == {name} ==")
            if name == "deadzone":
                fn(directory)
            else:
                fn(directory, rng, args.nboot)
        print()


if __name__ == "__main__":
    main()
