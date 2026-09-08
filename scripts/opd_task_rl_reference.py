#!/usr/bin/env python3
"""Do different tasks' RL reference directions collapse onto one direction?

The output-space cross gate (docs/opd_output_space_cross_gate_design.md) compares
task j's OPD push against a per-task, per-role reference built from task i's RL
signal. If those references were all the same direction, the cross statistic would
carry no receiver-specific information -- it would gate the same tokens whichever
task it claims to protect. This checks that one degeneracy, and nothing else.

WHAT IT DOES NOT SHOW, because the question keeps being over-read:

  * It is NOT a usefulness check. References that differ can still carry nothing
    about cross-task interference; references that agree would still not make the
    cross statistic a restatement of the self gate, because the self gate uses the
    TOKEN'S OWN u_R and this uses an AGGREGATE (v_i = v_j does not give
    v_i . u_D,j,t = u_R,j,t . u_D,j,t).
  * The shared-token-id counts it prints are set overlaps, NOT inner-product
    strength. Two shared ids carrying large gradient outweigh two hundred
    carrying none. Nothing here weighs the common support.
  * The reference is built from the SAMPLED token only, i.e. the e_a term of
    u_R = pg_scale (e_a - p). Dropping -p removes the part supported on the whole
    top-k, so the support overlap printed here is a LOWER bound and the direction
    is the task-specific half. A real reference is broader and more shared.

The dumps are per (dst, src) pairs, so a token appears several times; it is
counted once, keyed by (step, dst, turn, position). Rows whose group was
degenerate carry advantage 0 and are dropped -- the reward has no direction there.

    python3 scripts/opd_task_rl_reference.py [dump_dir ...]
"""
import collections
import glob
import itertools
import json
import math
import os
import sys

DEFAULT_RUNS = (
    "~/sign_tokens/opd_grpo_multitask_cross_teacher_klw_qwen3_1.7b_sg1",
    "~/sign_tokens/opd_grpo_multitask_cross_teacher_klw_qwen3_1.7b_xt1",
)
TASKS = ("alfworld", "search", "webshop")


def references(root):
    """``{(task, role): {token_id: weight}}`` and the token count behind each."""
    seen = set()
    vec = collections.defaultdict(lambda: collections.defaultdict(float))
    n = collections.Counter()
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(root),
                                              "sign_pair_events_step*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("is_sampled") or not r.get("advantage"):
                    continue
                key = (r["step"], r["dst"], r["turn"], r["position"])
                if key in seen:
                    continue
                seen.add(key)
                vec[(r["dst"], r["role"])][r["token_id"]] += r["advantage"]
                n[(r["dst"], r["role"])] += 1
    return vec, n


def cosine(a, b):
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return float("nan")
    keys = a if len(a) < len(b) else b
    return sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys) / (na * nb)


def main(roots):
    for root in roots or DEFAULT_RUNS:
        vec, n = references(root)
        if not vec:
            print(f"\n=== {root}: no dumps ===")
            continue
        print(f"\n=== {os.path.basename(root.rstrip('/'))} ===")
        for role in sorted({c for _, c in vec}):
            present = {t: n[(t, role)] for t in TASKS if n[(t, role)]}
            if len(present) < 2:
                continue
            print(f"  role={role:<12} sampled tokens {present}")
            for a, b in itertools.combinations(sorted(present), 2):
                va, vb = vec[(a, role)], vec[(b, role)]
                shared, union = len(set(va) & set(vb)), len(set(va) | set(vb))
                print(f"      cos(v_{a}, v_{b}) = {cosine(va, vb):+.3f}"
                      f"    shared token ids {shared}/{union}")


if __name__ == "__main__":
    main(sys.argv[1:])
