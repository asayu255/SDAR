"""The probe's per-trajectory rescue aggregation, and the bug that made it loop.

The first version of this aggregation was written inline in
_accumulate_oci_probe and used `n` as a loop variable. That method's `n` is the
batch counter, stored by `state["batches"] = n`, so the counter sat at the last
bucket's trajectory count, never reached n_batches, and the walkthrough_stepwise
probe ran on indefinitely -- thirteen batches, twelve of them printed as "batch 1".
The aggregation now lives in _aggregate_cand_records; this checks its arithmetic
and pins the method to a single assignment of `n`.
"""
import ast, inspect, os, sys, textwrap

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
import verl.trainer.ppo.opd_ray_trainer as ort  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


records = [
    {"traj": "a", "uid": "g1", "task": "heat some mug and put it in cabinet.", "plan_len": 6, "leading_k": 6, "ptr_final": 6},
    {"traj": "b", "uid": "g2", "task": "put two pencil in drawer.", "plan_len": 8, "leading_k": 3, "ptr_final": 5},
    {"traj": "c", "uid": "g3", "task": "put a mug in coffeemachine.", "plan_len": 4, "leading_k": 4, "ptr_final": 4},
    {"traj": "d", "uid": "g4", "task": "look at bowl under the desklamp.", "plan_len": 4, "leading_k": 0, "ptr_final": 1},
]
status = {"g1": "stuck", "g2": "stuck", "g3": "live", "g4": "stuck"}
ret = {"a": 10.0, "b": -0.1, "c": 9.9, "d": 0.0}
agg = ort._aggregate_cand_records(records, status, ret)

check([r["kind"] for r in records] == ["heat", "pick_two", "pick_and_place", "look_at"], "task kind is read off the task sentence")
check([r["solved"] for r in records] == [True, False, True, False], "solved means a positive return, penalties and zero are not")
s = agg["stuck/long_ge5"]
check((s["trajectories"], s["solved"], s["full_follow"]) == (2, 1, 1), "stuck/long_ge5 counts: two trajectories, one solved, one followed to the end")
check(abs(s["solve_rate"] - 0.5) < 1e-9 and abs(s["ptr_final_mean"] - 5.5) < 1e-9 and abs(s["plan_len_mean"] - 7.0) < 1e-9, "its rates and means")
check(agg["stuck/short_le4"]["trajectories"] == 1 and agg["stuck/short_le4"]["solved"] == 0, "the 4-line stuck plan is filed as short")
check(agg["live/pick_and_place"]["solve_rate"] == 1.0, "the live group is filed under its own class")
check(all(not k.startswith("_") for v in agg.values() for k in v), "no scratch fields leak into the payload")

# the regression guard for the actual bug
src = textwrap.dedent(inspect.getsource(ort.OPDRayTrainer._accumulate_oci_probe))
assigned = [node.lineno for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "n"]
check(len(assigned) == 1, f"_accumulate_oci_probe assigns its batch counter `n` exactly once (found {len(assigned)})")


def test_cand_records():
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
