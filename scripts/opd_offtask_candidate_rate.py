#!/usr/bin/env python3
"""Would a reward-agreeing off-task teacher exist where the on-task one conflicts?

The output-space multitask proposal (docs/opd_pushback_output_space_multitask.md)
replaces the on-task teacher's push with an off-task teacher's at tokens where the
on-task teacher opposes the reward. Whether that ever fires is decided by the
REPLACEMENT CANDIDATE RATE, and the exact quantity -- u_R . u_m over the student's
top-k -- needs the off-task teacher's forward, i.e. a GPU run.

This is the offline stand-in, on dumps that already exist. At the SAMPLED token,
with the descent convention of opd_task_diag (positive = push this logit UP):

    reward wants UP        iff  advantage > 0
    teacher t wants UP     iff  log p_t(a) > log p_s(a)          (see below)
    on-task conflict       iff  those two disagree
    replacement candidate  iff  at a conflicting token, some off-task teacher
                                agrees with the reward

WHY THE TEACHER CRITERION IS D-FREE, AND WHICH WAY IT ERRS. The exact descent
direction is g_t(a) = p_s(a) (D_t - f_t(a)) with f_t(a) = log p_s(a) - log p_t(a)
and D_t the per-token KL to teacher t. The dumps carry D only for the ON-TASK
teacher, so the off-task side uses f_t(a) < 0 in place of f_t(a) < D_t. Since
D_t >= 0, f_t(a) < 0 implies f_t(a) < D_t: the criterion UNDER-counts "pushes up",
so it cannot manufacture candidates that are not there.

THE NULL IS THE POINT. Teachers agree with each other on format, and
p_t(a) > p_s(a) has a base rate of its own, so a raw candidate rate of 50% means
nothing on its own. The null shuffles the sign of the advantage per token, which
breaks the reward-teacher association while leaving every marginal rate intact.
An excess near zero says the off-task teacher's direction carries no information
about the reward at the tokens the mechanism would act on.

    python3 scripts/opd_offtask_candidate_rate.py [dump_dir ...]
"""
import collections
import glob
import json
import math
import random
import sys

DEFAULT_RUNS = {
    "klw_sg1": "~/sign_tokens/opd_grpo_multitask_cross_teacher_klw_qwen3_1.7b_sg1",
    "klw_xt1": "~/sign_tokens/opd_grpo_multitask_cross_teacher_klw_qwen3_1.7b_xt1",
}
TASKS = ("alfworld", "search", "webshop")
N_NULL = 200


def load_tokens(pattern):
    """``{token key: {dst, A, on, m{src: bool}}}`` over the sampled tokens of one run."""
    toks = collections.defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("is_sampled"):
                    continue
                adv, p_s, p_on, p_src = (r.get("advantage"), r.get("p_student"),
                                         r.get("p_on"), r.get("p_source"))
                # A degenerate group carries advantage 0: the reward has no
                # direction there, so the conflict question is not defined.
                if not adv or not p_s or not p_on or not p_src:
                    continue
                key = (r["step"], r["dst"], r["turn"], r["position"])
                e = toks[key]
                e["dst"], e["A"] = r["dst"], adv > 0
                e["on"] = math.log(p_on) > math.log(p_s)
                e.setdefault("m", {})[r["src"]] = math.log(p_src) > math.log(p_s)
    return [e for e in toks.values() if "m" in e]


def rates(events, flip=None):
    """(conflicting tokens, those with >=1 agreeing off-task teacher)."""
    n = hit = 0
    for e in events:
        want_up = (not e["A"]) if (flip and flip.get(id(e))) else e["A"]
        if e["on"] == want_up:
            continue
        n += 1
        if any(m == want_up for m in e["m"].values()):
            hit += 1
    return n, hit


def main(dirs):
    runs = {d.rstrip("/").split("/")[-1]: d for d in dirs} if dirs else DEFAULT_RUNS
    rng = random.Random(0)
    for name, root in runs.items():
        events = load_tokens(f"{root.replace('~', __import__('os').path.expanduser('~'))}"
                             "/sign_pair_events_step*.jsonl")
        by_task = collections.defaultdict(list)
        for e in events:
            by_task[e["dst"]].append(e)
        print(f"\n=== {name} ===")
        print(f"{'receiver':<10}{'conflicts':>10}{'candidate rate':>16}{'shuffled-A null':>18}{'excess':>10}")
        for task in TASKS:
            ev = by_task.get(task)
            if not ev:
                continue
            n, hit = rates(ev)
            if n == 0:
                print(f"{task:<10}{0:>10}{'--':>16}")
                continue
            null = []
            for _ in range(N_NULL):
                flip = {id(e): rng.random() < 0.5 for e in ev}
                nn, hh = rates(ev, flip)
                null.append(hh / max(nn, 1))
            mu = sum(null) / len(null)
            obs = hit / n
            print(f"{task:<10}{n:>10}{obs * 100:>15.1f}%{mu * 100:>17.1f}%{(obs - mu) * 100:>+9.1f}pt")


if __name__ == "__main__":
    main(sys.argv[1:])
