"""The mass probe's arithmetic: which cell a group lands in, and what is summed.

WHAT IT PROTECTS. The whole point of the measurement is that a degenerate group
is not gradient-free: the teacher KL reaches every row, and one badly formatted
row moves every advantage in its group (the +19.9 fingerprint). If stuck groups
with and without that were pooled, a stalled task's policy gradient would look
healthy, which is the reading the probe exists to prevent. So the split, the row
weights and the padding rule are checked here, with no model and no GPU.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "verl")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from verl.trainer.ppo import term_mass as tm  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


print("1. which cell a group lands in")
check(tm.group_class("live", 12.0) == "live", "a live group is live whatever its advantages are")
check(tm.group_class("stuck", 0.0) == "stuck_uniform",
      "all-fail with every advantage zero: the policy gradient is exactly nothing there")
check(tm.group_class("stuck", 19.946) == "stuck_mixed",
      "all-fail with the format channel carrying it is a different cell")
check(tm.group_class("saturated", 0.0) == "saturated_uniform"
      and tm.group_class("saturated", 5.0) == "saturated_mixed", "same split for all-success")
check(tm.group_class("stuck", 1e-15) == "stuck_uniform",
      "and the split is on a tolerance, not on exact float equality")

print("2. the sums")
# two groups of two rows, three response tokens each; one live, one stuck-uniform
rows = 4
mask = np.ones((rows, 3))
adv = np.array([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
w = np.array([0.5, 0.5, 0.25, 0.25])
groups = {"live1": {"rows": [0, 1], "status": "live", "task": "alfworld"},
          "stuck1": {"rows": [2, 3], "status": "stuck", "task": "search"}}
records = [{"row": 0, "tokens": 3, "pg_abs": 6.0, "pg_signed": -6.0, "kl": 1.0, "w": 0.5,
            "kl_row_coef": 1.0, "teacher_kl_coef": 0.01},
           {"row": 1, "tokens": 3, "pg_abs": 3.0, "pg_signed": 3.0, "kl": 2.0, "w": 0.5,
            "kl_row_coef": 1.0, "teacher_kl_coef": 0.01},
           {"row": 2, "tokens": 3, "pg_abs": 0.0, "pg_signed": 0.0, "kl": 4.0, "w": 0.25,
            "kl_row_coef": 1.0, "teacher_kl_coef": 0.01},
           {"row": 3, "tokens": 3, "pg_abs": 0.0, "pg_signed": 0.0, "kl": 4.0, "w": 0.25,
            "kl_row_coef": 1.0, "teacher_kl_coef": 0.01}]
out = tm.aggregate_term_mass(groups, records, row_mask=mask, advantages=adv,
                             task_weights=w, pg_loss_coef=1.0)
c = out["cells"]
check(set(c) == {"alfworld/live", "search/stuck_uniform"}, f"one cell per (task, class): {set(c)}")
check(abs(c["alfworld/live"]["pg_mass"] - (0.5 * 6.0 + 0.5 * 3.0)) < 1e-9,
      "PG mass is the row weight times the actor's own |pg| sum")
check(abs(c["alfworld/live"]["kl_mass"] - (0.5 * 1.0 + 0.5 * 2.0) * 0.01) < 1e-9,
      "KL mass carries the teacher coefficient")
check(abs(c["alfworld/live"]["pg_mass_driver"] - (0.5 * 6.0 + 0.5 * 3.0)) < 1e-9,
      "and the driver's own sum over |advantage| agrees -- the row mapping self-check")
check(c["search/stuck_uniform"]["pg_mass"] == 0.0
      and abs(c["search/stuck_uniform"]["kl_mass"] - (0.25 * 4 + 0.25 * 4) * 0.01) < 1e-9,
      "a uniform stuck group contributes no policy gradient and still carries teacher KL")

print("3. padding and missing rows")
w_pad = np.array([0.5, 0.5, 0.0, 0.25])
out_pad = tm.aggregate_term_mass(groups, records, row_mask=mask, advantages=adv,
                                 task_weights=w_pad, pg_loss_coef=1.0)
check(out_pad["padding_rows_skipped"] == 1
      and out_pad["cells"]["search/stuck_uniform"]["rows"] == 1,
      "a row with weight 0 is adjust_batch padding and is dropped, not counted")
out_missing = tm.aggregate_term_mass(groups, records[:3], row_mask=mask, advantages=adv,
                                     task_weights=w, pg_loss_coef=1.0)
check(out_missing["rows_missing_from_actor"] == 1,
      "a row the actor never reported is counted rather than silently dropped")

print("4. the report")
acc = {}
tm.add_batches(acc, out)
tm.add_batches(acc, out)
s = tm.summarise(acc)
check(s["batches"] == 2 and abs(s["pg_mass_actor_vs_driver"] - 1.0) < 1e-9,
      "batches accumulate and the actor/driver check is a ratio of 1")
check(abs(s["tasks"]["search"]["classes"]["stuck_uniform"]["pg_share"] or 0.0) == 0.0
      and s["tasks"]["search"]["pg_over_kl"] == 0.0,
      "a task whose groups are all uniform-stuck reads as pure teacher-driven")
check(abs(s["tasks"]["alfworld"]["pg_over_kl"] - (9.0 / 0.03)) < 1e-6,
      "and a live task's PG/KL is the ratio of the two masses")
check(any("stuck_uniform" in line for line in tm.format_report(s)),
      "the printed table names every class")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
