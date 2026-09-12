"""CPU test for oci_saturated: classification, selection, and the arithmetic."""
import math, os, sys, types
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verl.trainer.ppo.oci_saturated import (
    classify_groups, select_saturated_injections, zero_injected_advantage, injection_metrics)

TOK, TURNS = 4, 2


def build(spec):
    """spec: {uid: [return, ...]} one entry per trajectory."""
    rew, uids, tuids = [], [], []
    for uid, rets in spec.items():
        for k, r in enumerate(rets):
            for _ in range(TURNS):
                row = [0.0] * TOK
                row[-1] = float(r)
                rew.append(row); uids.append(uid); tuids.append(f"{uid}:{k}")
    b = {"token_level_rewards": torch.tensor(rew)}
    nt = {"uid": np.array(uids), "traj_uid": np.array(tuids)}
    return types.SimpleNamespace(batch=b, non_tensor_batch=nt, meta_info={})


ok = True
b = build({"sat": [10.0] * 8, "stuck": [0.0] * 8, "live": [0.0] * 4 + [10.0] * 4})
g = classify_groups(b)
got = {k: v["status"] for k, v in g.items()}
good = got == {"sat": "saturated", "stuck": "stuck", "live": "live"}
ok &= good
print(("  OK  " if good else "  FAIL") + f" classification: {got}")

# a 7-judged group with an 8th reserved row: the verdict ignores the reserve
b2 = build({"g": [10.0] * 7 + [0.0]})
judged = np.array([True] * (7 * TURNS) + [False] * TURNS)
g2 = classify_groups(b2, judged_rows=judged)
good = g2["g"]["status"] == "saturated" and g2["g"]["n_judged"] == 7
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the reserved row is excluded from the verdict: {g2['g']['status']}, judged {g2['g']['n_judged']}")

# selection keeps only a failing candidate in a saturated group
cand = np.array([False] * (7 * TURNS) + [True] * TURNS)
keep = select_saturated_injections(b2, g2, candidate_rows=cand)
good = keep.sum() == TURNS and keep[-1]
ok &= good
print(("  OK  " if good else "  FAIL") + f" a failing candidate in a saturated group is kept ({int(keep.sum())} rows)")

# a candidate that also SUCCEEDS is not kept: the group would stay uniform
b3 = build({"g": [10.0] * 8})
g3 = classify_groups(b3, judged_rows=judged)
keep3 = select_saturated_injections(b3, g3, candidate_rows=cand)
good = keep3.sum() == 0
ok &= good
print(("  OK  " if good else "  FAIL") + " a candidate that also succeeds is rejected")

# and nothing is kept in a live or stuck group
for spec, lab in (({"g": [0.0] * 4 + [10.0] * 3 + [0.0]}, "live"), ({"g": [0.0] * 8}, "stuck")):
    bb = build(spec); gg = classify_groups(bb, judged_rows=judged)
    kk = select_saturated_injections(bb, gg, candidate_rows=cand)
    good = kk.sum() == 0
    ok &= good
    print(("  OK  " if good else "  FAIL") + f" nothing injected into a {lab} group")

# the manufactured advantage: +1/sqrt(7) and -sqrt(7), for any reward scale
print("\n  the advantage the injection manufactures:")
for s, f in ((1.0, 0.0), (10.0, 1.72), (10.0, 0.27)):
    r = np.array([s] * 7 + [f]); m = r.mean(); sd = r.std()
    As, Af = (s - m) / sd, (f - m) / sd
    good = abs(As - 1 / math.sqrt(7)) < 1e-9 and abs(Af + math.sqrt(7)) < 1e-9
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" R={s:5.2f}/{f:4.2f} -> A_success {As:+.4f}, A_failure {Af:+.4f}")

# arm B removes only the injected row's gradient
adv = torch.ones(8 * TURNS, TOK)
b2.batch["advantages"] = adv
n = zero_injected_advantage(b2, keep)
good = bool(n == TURNS
            and float(adv[-TURNS:].abs().sum()) == 0.0
            and float(adv[: 7 * TURNS].sum()) == 7 * TURNS * TOK)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" arm B zeroes {n} injected rows and leaves the seven successes untouched")

m = injection_metrics(g, np.zeros(len(g), dtype=bool), task="alfworld")
good = m["oci/groups_saturated/alfworld"] == 1 and abs(m["oci/live_frac/alfworld"] - 1/3) < 1e-9
ok &= good
print(("  OK  " if good else "  FAIL") + f" metrics: {m}")

# --- token mass by class: the number that sizes the mechanism ---------------
from verl.trainer.ppo.oci_saturated import token_mass_by_class

# a saturated group that finished in 1 turn against a stuck one that ran 5
spec4 = {"sat": [10.0] * 8, "stuck": [0.0] * 8}
turns = {"sat": 1, "stuck": 5}
rew4, uids4, tuids4 = [], [], []
for uid, rets in spec4.items():
    for k, r in enumerate(rets):
        for _ in range(turns[uid]):
            row = [0.0] * TOK
            row[-1] = float(r)
            rew4.append(row); uids4.append(uid); tuids4.append(f"{uid}:{k}")
n4 = len(rew4)
b4 = types.SimpleNamespace(
    batch={"token_level_rewards": torch.tensor(rew4),
           "responses": torch.zeros(n4, TOK, dtype=torch.long),
           "loss_mask": torch.ones(n4, TOK, dtype=torch.long),
           "attention_mask": torch.ones(n4, TOK, dtype=torch.long)},
    non_tensor_batch={"uid": np.array(uids4), "traj_uid": np.array(tuids4)},
    meta_info={})
tm = token_mass_by_class(b4, classify_groups(b4))
good = (abs(tm["oci/tokmass/saturated/group_share"] - 0.5) < 1e-9
        and abs(tm["oci/tokmass/saturated/token_share"] - 1 / 6) < 1e-9
        and abs(tm["oci/tokmass/dead_token_share"] - 1.0) < 1e-9)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" token mass splits where the group count does not: saturated is "
      f"{tm['oci/tokmass/saturated/group_share']:.0%} of groups but "
      f"{tm['oci/tokmass/saturated/token_share']:.0%} of tokens")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
