"""walkthrough_stepwise: a per-turn progress line for the eighth rollout.

The whole-path probe rescued 12 of 34 stuck groups while the walkthrough itself
wins 30 of 30 games when replayed, and the followers stopped at four lines -- the
whole solution for the short task types, the midpoint of the long ones. With two
turns of history and a numbered block that says nothing about which line is next,
the student cannot tell where it is by step four. This mode keeps the block
byte-identical and adds one line, right before the turn prompt, naming the step
still owed. These checks are about exactly that: the pointer moves only on the
step actually taken, the line lands only on the candidate slot, and nowhere else
in the prompt changes.
"""
import glob, json, os, sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
import agent_system.environments.env_manager as em  # noqa: E402
from agent_system.environments.prompts import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


walk = ["go to countertop 1", "take mug 1 from countertop 1", "go to microwave 1",
        "heat mug 1 with microwave 1", "go to coffeemachine 1", "move mug 1 to coffeemachine 1"]

# --- the pure pieces ----------------------------------------------------------
check(em._guide_line(walk, 0).endswith("Your next action is step 1 of 6: go to countertop 1\n\n"), "pointer 0 names step 1")
check("Steps 1-3 of 6 are done. Your next action is step 4 of 6: heat mug 1 with microwave 1" in em._guide_line(walk, 3), "pointer 3 names what is done and step 4")
check("All 6 steps" in em._guide_line(walk, 6), "a finished path says so instead of indexing past the end")
check(em._guide_line([], 0) == "", "no walkthrough, no line")
check(em._advance_guide(walk, 0, "  Go To Countertop 1 ") == 1, "the pointer advances on the owed step (case and spaces ignored)")
check(em._advance_guide(walk, 1, "go to countertop 1") == 1, "a detour leaves the pointer on the step still owed")
check(em._advance_guide(walk, 6, "anything") == 6, "a finished pointer stays finished")
for name, tmpl in (("no-history", ALFWORLD_TEMPLATE_NO_HIS.format(current_observation="obs", admissible_actions="'look'")),
                   ("history", ALFWORLD_TEMPLATE.format(task_description="t", step_count=1, history_length=1,
                                                        action_history="h", current_step=2, current_observation="obs",
                                                        admissible_actions="'look'"))):
    out = em._insert_guide(tmpl, em._guide_line(walk, 2))
    i_line, i_anchor, i_adm = out.find("[Privileged Solution Path progress]"), out.find(em._GUIDE_ANCHOR), out.find("Your admissible actions")
    check(0 <= i_adm < i_line < i_anchor and out.count("[Privileged Solution Path progress]") == 1,
          f"{name} template: the line sits after the admissible actions and right before the turn prompt")
    check(out.replace(em._guide_line(walk, 2), "") == tmpl, f"{name} template: nothing else in the prompt changes")
check(em._insert_guide("no anchor here", "x") == "no anchor here", "no anchor, no insertion")

# --- the manager, end to end, on a real game ---------------------------------
games = sorted(glob.glob(os.path.expanduser("~/data/alfworld/json_2.1.1/train/pick_heat_then_place_in_recep*/*/game.tw-pddl")))
gf = next((g for g in games if len(json.load(open(g)).get("walkthrough") or []) >= 5), None)
if gf is None:
    print("SKIP: no long alfworld walkthrough on this host")
else:
    real_walk = json.load(open(gf))["walkthrough"]
    N = len(real_walk)

    class FakeEnvs:
        group_n, is_train = 2, True
        get_admissible_commands = [["look", "inventory"], ["look", "inventory"]]

        def reset(self):
            obs = ["You are in the middle of a room.\n\nYour task is to: heat some mug and put it somewhere."] * 2
            return obs, None, [{"extra.gamefile": gf, "won": False}] * 2

        def step(self, actions):
            return ([f"You did: {a}." for a in actions], None, [0.0, 0.0], [False, False],
                    [{"extra.gamefile": gf, "won": False}] * 2)

    cfg = SimpleNamespace(env=SimpleNamespace(history_length=2),
                          algorithm={"oci_sat": {"enable": True, "plan_corruption": "walkthrough_stepwise"}})
    em._WRONG_PLAN_CACHE.clear()
    m = em.AlfWorldEnvironmentManager(FakeEnvs(), lambda acts, adm: (list(acts), [1] * len(acts)), cfg)
    obs, _ = m.reset(kwargs=None)
    plain, cand = obs["text"]
    check("[Privileged Solution Path]" not in plain and "progress]" not in plain, "slot 0 (plain) carries neither the block nor the line")
    check(cand.startswith("[Privileged Solution Path]") and f"step 1 of {N}: {real_walk[0]}" in cand, "slot 1 (candidate) carries the block and 'step 1'")

    def step(actions):
        o, *_ = m.step(actions)
        return o["text"][1]

    t1 = step(["look", real_walk[0]])
    check(f"Your next action is step 2 of {N}: {real_walk[1]}" in t1, "taking step 1 moves the line to step 2")
    t2 = step(["look", "look"])
    check(f"Your next action is step 2 of {N}: {real_walk[1]}" in t2, "a detour keeps it on step 2")
    t3 = step(["look", real_walk[1]])
    check(f"Your next action is step 3 of {N}: {real_walk[2]}" in t3, "taking the owed step moves it to step 3")
    check(m._guide_ptr == [0, 2], f"the plain slot's pointer never moves ({m._guide_ptr})")
    blk = em._build_wrong_plan(gf, "walkthrough")
    check(t3.startswith(blk), "the block at the top is byte-identical to the walkthrough mode's")

    cfg.algorithm["oci_sat"]["plan_corruption"] = "walkthrough"
    em._WRONG_PLAN_CACHE.clear()
    m2 = em.AlfWorldEnvironmentManager(FakeEnvs(), lambda acts, adm: (list(acts), [1] * len(acts)), cfg)
    o2, _ = m2.reset(kwargs=None)
    check("progress]" not in o2["text"][1], "plain `walkthrough` mode adds no progress line")


def test_stepwise_guide():
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
