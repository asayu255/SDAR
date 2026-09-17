"""(a)'s k, counted in the environment managers for every row, and carried to the batch.

WHAT IT PROTECTS.
  * ALFWorld: a walkthrough step counts only when it is the next step AND the
    environment carried it out -- a row typing a later step from the wrong room
    ("Nothing happens.") must not score progress.
  * WebShop: a click counts only when it was on the page (the environment's own
    test), the results page is judged by what is on screen, and leaving the
    product clears its options.
  * Search: one step, and none at all for yes/no questions.
  * Wiring: with algorithm.progress_rank off, the managers write nothing and the
    rollout loop records nothing -- the batch keeps control's columns.
No model, no retriever, no WebShop index: fake environments throughout.
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "verl")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from agent_system.environments import progress as P  # noqa: E402
from agent_system.environments.oci_layout import answer_strings, is_yesno  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def config(on=True, n=2):
    return OmegaConf.create({
        "env": {"history_length": 0, "rollout": {"n": n}},
        "algorithm": {"progress_rank": {"enable": on},
                      "oci_sat": {"enable": False}, "oci_slots": {"enable": False},
                      "oci_rank": {"enable": False}},
    })


print("1. ALFWorld: the walkthrough pointer")
walk = ["go to desk 1", "take pencil 2 from desk 1", "go to shelf 2", "put pencil 2 in/on shelf 2"]
ptr = 0
ptr = P.advance_walkthrough(walk, ptr, "take pencil 2 from desk 1", "Nothing happens.")
check(ptr == 0, "a later step typed first does not count")
ptr = P.advance_walkthrough(walk, ptr, "go to desk 1", "You arrive at desk 1. On the desk 1, you see a pencil 2.")
check(ptr == 1, "the next step, carried out, counts")
ptr = P.advance_walkthrough(walk, ptr, "look", "You are facing the desk 1.")
check(ptr == 1, "an unrelated action in between leaves the pointer where it is")
ptr = P.advance_walkthrough(walk, ptr, "take pencil 2 from desk 1", "Nothing happens.")
check(ptr == 1, "the right step that the environment refused does not count")
ptr = P.advance_walkthrough(walk, ptr, "  Take Pencil 2 From Desk 1 ", "You pick up the pencil 2 from the desk 1.")
check(ptr == 2, "compared the way the document pointer compares (case, whitespace)")
check(P.advance_walkthrough([], 0, "go to desk 1", "ok") == 0, "no walkthrough, no progress")

print("2. WebShop: the goal record")
goal = {"asin": "B07ABC1234", "goal_options": ["black", "large"], "name": "a bag"}
landing = {"has_search_bar": True, "clickables": ["search"]}
results = {"has_search_bar": False, "clickables": ["back to search", "next >", "b07abc1234", "b09zzz0000"]}
results_other = {"has_search_bar": False, "clickables": ["back to search", "next >", "b09zzz0000"]}
item = {"has_search_bar": False, "clickables": ["back to search", "< prev", "description", "features",
                                                "reviews", "buy now", "black", "white", "large", "small"]}
sub = {"has_search_bar": False, "clickables": ["back to search", "< prev"]}
w = P.WebshopProgress(goal, landing)
check(w.total == 5 and w.k == 0, "K = search + open + 2 options + buy = 5")
w.step("search[a bag]", results_other)
check(w.k == 0, "a results page without the product is not the first step")
w.step("search[a black bag]", results)
check(w.k == 1, "a results page showing the product is")
w.step("click[b09zzz0000]", item)
check(w.k == 1, "opening another product is not progress")
w.step("click[back to search]", results)
w.step("click[b07abc1234]", item)
check(w.k == 2, "opening the goal product is")
w.step("click[black]", item)
check(w.k == 3, "a required option on it is")
w.step("click[description]", sub)
w.step("click[< prev]", item)
check(w.on_goal, "'< prev' from a description page returns to the product")
w.step("click[large]", item)
check(w.k == 4, "and the options chosen before the detour still count")
w.step("click[< prev]", results)
check(not w.on_goal and w.k == 4, "'< prev' from the product page leaves it; the best visit is kept")
w.step("click[buy now]", landing)
check(w.k == 4, "buying from the results page is not buying the goal")
w2 = P.WebshopProgress(goal, landing)
w2.step("search[bag]", results)
w2.step("click[b07abc1234]", item)
w2.step("click[black]", {"clickables": ["buy now", "large"]})   # the page after the click
w2.step("click[white]", {"clickables": ["buy now"]})             # not on the page any more
check(w2.k == 3, "a click on something that was not on the page does nothing")
w2.step("click[buy now]", landing)
check(w2.k == 4 and w2.bought, "buying the goal product counts, with whatever options it had")
w3 = P.WebshopProgress(goal, landing)
w3.step("search[bag]", results)
w3.step("click[b07abc1234]", item)
w3.step("click[black]", item)
w3.step("search[another bag]", results)
w3.step("click[b07abc1234]", item)
check(w3.selected == set() and w3.k == 3, "a new search clears the product's options; the best visit stays")
check(P.WebshopProgress(None, landing).total == 0, "no goal record, no sequence")
check(P.WebshopProgress({"asin": "B0X", "goal_options": {"color": "Red"}}, None).options == ["red"],
      "options given as a mapping are read by value, lower-cased like the document's clicks")

print("3. Search: one step")
kw = dict(answer_strings=answer_strings, is_yesno=is_yesno)
check(P.search_progress(True, {"target": ["Paris"]}, **kw) == (1, 1), "a returned result carried it")
check(P.search_progress(False, {"target": ["Paris"]}, **kw) == (0, 1), "none did")
check(P.search_progress(True, {"target": ["yes"]}, **kw) == (0, 0), "yes/no: no progress to count")
check(P.search_progress(True, {"target": []}, **kw) == (0, 0), "no answer: no progress to count")

print("4. the switch")
check(P.progress_on(config(True)) and not P.progress_on(config(False)), "algorithm.progress_rank.enable")
check(not P.progress_on(OmegaConf.create({"env": {}})), "absent means off")

print("5. ALFWorld manager")
from agent_system.environments.env_manager import (  # noqa: E402
    AlfWorldEnvironmentManager, SearchEnvironmentManager, WebshopEnvironmentManager)

tmp = tempfile.mkdtemp()
game = os.path.join(tmp, "pick_and_place", "trial_1")
os.makedirs(game)
with open(os.path.join(game, "game.tw-pddl"), "w") as f:
    json.dump({"walkthrough": walk, "pddl_problem": "(:init )"}, f)
gamefile = os.path.join(game, "game.tw-pddl")


class _AlfEnvs:
    is_train = True
    group_n = 2

    def __init__(self):
        self.get_admissible_commands = [["go to desk 1", "look"], ["go to desk 1", "look"]]

    def reset(self):
        obs = ["You are in a room. Your task is to: put a pencil on a shelf."] * 2
        return obs, None, [{"extra.gamefile": gamefile}, {"extra.gamefile": gamefile}]

    def step(self, actions):
        obs = ["Nothing happens." if a.startswith("take") else f"You did {a}." for a in actions]
        return obs, None, [0.0, 0.0], [False, False], [{"extra.gamefile": gamefile} for _ in actions]


def _alf_proj(texts, pools):
    return [t.lower() for t in texts], [1] * len(texts)


for on in (True, False):
    mgr = AlfWorldEnvironmentManager(_AlfEnvs(), _alf_proj, config(on))
    mgr.reset(None)
    _, _, _, infos = mgr.step(["go to desk 1", "take pencil 2 from desk 1"])
    if on:
        check([i.get("progress_k") for i in infos] == [1, 0] and infos[0].get("progress_total") == 4,
              "every row carries k of K; the refused step did not count")
        _, _, _, infos = mgr.step(["look", "go to desk 1"])
        check([i.get("progress_k") for i in infos] == [1, 1], "and the pointer persists across turns")
    else:
        check(all("progress_k" not in i for i in infos), "off: nothing is written")


print("6. WebShop manager")


class _WsEnvs:
    def __init__(self):
        self.t = 0

    def reset(self):
        obs = ["WebShop [SEP] Instruction: [SEP] buy a black large bag [SEP] Search"] * 2
        infos = [{"available_actions": landing, "goal": goal} for _ in range(2)]
        return obs, infos

    def step(self, actions):
        self.t += 1
        pages = {1: results, 2: item}
        page = pages.get(self.t, item)
        obs = ["WebShop [SEP] Instruction: [SEP] buy a black large bag [SEP] page"] * 2
        return obs, [0.0, 0.0], [False, False], [{"available_actions": page} for _ in actions]


for on in (True, False):
    mgr = WebshopEnvironmentManager(_WsEnvs(), lambda acts: (list(acts), [1] * len(acts)), config(on))
    mgr.reset(None)
    _, _, _, infos = mgr.step(["search[black bag]", "search[shoes]"])
    if on:
        check([i.get("progress_k") for i in infos] == [1, 1] and infos[0]["progress_total"] == 5,
              "the results page showing the product counts for both rows")
        _, _, _, infos = mgr.step(["click[b07abc1234]", "click[b09zzz0000]"])
        check([i.get("progress_k") for i in infos] == [2, 1], "only the row that opened the goal moves on")
    else:
        check(all("progress_k" not in i for i in infos), "off: nothing is written")


print("7. Search manager")
QUESTION = "what is the capital of france"


class _SearchEnvs:
    group_n = 2
    is_train = True

    def reset(self, kwargs=None):
        return [QUESTION] * 2, [{} for _ in range(2)]

    def step(self, actions):
        obs = ["<information> Paris is the capital of France. </information>", "<information> Lyon. </information>"]
        return obs, [0.0, 0.0], [False, False], [{"won": 0.0} for _ in actions]


for on in (True, False):
    mgr = SearchEnvironmentManager(_SearchEnvs(), lambda acts: (list(acts), [1] * len(acts)), config(on))
    mgr.reset([{"question": QUESTION, "ground_truth": {"target": ["Paris"]}}] * 2)
    _, _, _, infos = mgr.step(["<search> capital of france </search>"] * 2)
    if on:
        check([i.get("progress_k") for i in infos] == [1, 0] and infos[1]["progress_total"] == 1,
              "k is whether this row's own results carried the answer")
    else:
        check(all("progress_k" not in i for i in infos), "off: nothing is written")


print("8. the rollout loop records it")
import torch  # noqa: E402

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector  # noqa: E402
from verl import DataProto  # noqa: E402


def record(on, infos):
    c = TrajectoryCollector.__new__(TrajectoryCollector)
    c._progress_rank_on = on
    c._queue_row_for_prefetch = lambda *a: None
    batch = DataProto.from_single_dict({"input_ids": torch.zeros(2, 3, dtype=torch.long),
                                        "traj_uid": np.array(["a", "b"], dtype=object)})
    tbl, tinf = [[], []], [[], []]
    c._record_turn(batch=batch, active_idx=np.array([0, 1]), active_masks=np.array([True, True]),
                   infos=infos, traj_uid=np.array(["a", "b"], dtype=object),
                   total_batch_list=tbl, total_infos=tinf, batch_size=2)
    return tbl


tbl = record(True, [{"progress_k": 3, "progress_total": 6}, {}])
check(tbl[0][0]["progress_k"] == 3.0 and tbl[0][0]["progress_total"] == 6.0, "on: the row carries k and K")
check(np.isnan(tbl[1][0]["progress_k"]) and np.isnan(tbl[1][0]["progress_total"]),
      "a row whose task has no count gets NaN, so every row has the same columns")
tbl = record(False, [{"progress_k": 3, "progress_total": 6}, {}])
check("progress_k" not in tbl[0][0], "off: the batch keeps control's columns")

print("9. ALFWorld milestones: by object type, in any order")


def game_dir(task_type, obj, recep="", lamp="", sliced=False):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "traj_data.json"), "w") as f:
        json.dump({"task_type": task_type, "pddl_params": {
            "object_target": obj, "parent_target": recep, "toggle_target": lamp,
            "mrecep_target": "", "object_sliced": sliced}}, f)
    return os.path.join(d, "game.tw-pddl")


OKOBS = "You did it."
m = P.AlfworldMilestones(game_dir("pick_clean_then_place_in_recep", "Cup", "Microwave"))
check(m.total == 3 and m.k == 0, "clean-then-place: K = 3")
m.step("go to cabinet 2", OKOBS)
m.step("take cup 1 from cabinet 2", OKOBS)
check(m.k == 1, "the instance and the receptacle it came from do not matter")
m.step("move cup 1 to microwave 1", OKOBS)
check(m.k == 1, "placing an UNcleaned cup does not count")
m.step("take cup 1 from microwave 1", OKOBS)
m.step("clean cup 1 with sinkbasin 1", "Nothing happens.")
check(m.k == 1, "a treatment the environment refused does not count")
m.step("clean cup 1 with sinkbasin 1", OKOBS)
check(m.k == 2, "the treatment does")
m.step("move cup 1 to microwave 1", OKOBS)
check(m.k == 3, "and placing the treated cup completes it")

m = P.AlfworldMilestones(game_dir("look_at_obj_in_light", "AlarmClock", lamp="DeskLamp"))
m.step("use desklamp 1", OKOBS)
check(m.k == 1, "look-at: the lamp first counts (the walkthroughs do it in that order)")
m.step("take alarmclock 2 from dresser 1", OKOBS)
check(m.k == 2 and m.total == 2, "and taking the object completes it")
m = P.AlfworldMilestones(game_dir("look_at_obj_in_light", "AlarmClock", lamp="DeskLamp"))
m.step("use floorlamp 1", OKOBS)
check(m.k == 0, "a lamp of another type does not count")

m = P.AlfworldMilestones(game_dir("pick_two_obj_and_place", "Pencil", "Drawer"))
for a in ("take pencil 2 from desk 1", "move pencil 2 to drawer 1", "take pencil 2 from drawer 1"):
    m.step(a, OKOBS)
check(m.k == 2, "pick-two: the same instance taken twice is still one object")
m.step("take pencil 3 from desk 1", OKOBS)
check(m.k == 3, "a second instance is the third milestone")
m.step("put pencil 3 in/on drawer 1", OKOBS)
check(m.k == 4 and m.total == 4, "and 'put ... in/on' places it like 'move ... to'")

m = P.AlfworldMilestones(game_dir("pick_heat_then_place_in_recep", "Mug", "CoffeeMachine"))
m.step("take mug 3 from countertop 1", OKOBS)
m.step("heat mug 3 with microwave 1", OKOBS, won=True)
check(m.k == 3, "a game the environment declares won is at K (goals are checked by type)")

check(P.AlfworldMilestones(game_dir("pick_and_place_with_movable_recep", "Pen", "Bowl")).total == 0,
      "a task type outside ALFWorld's six has no milestones")
check(P.AlfworldMilestones(game_dir("pick_heat_then_place_in_recep", "Apple", "CounterTop", sliced=True)).total == 0,
      "nor does a sliced-object task")
check(P.AlfworldMilestones(None).total == 0 and P.alfworld_task("") == {},
      "nor a row without a game file (never a stray traj_data.json in the working directory)")

print("10. both ALFWorld counts reach the rows")
gamefile2 = game_dir("pick_and_place_simple", "Pencil", "Shelf")
with open(gamefile2, "w") as f:
    json.dump({"walkthrough": walk, "pddl_params": {}, "pddl_problem": "(:init )"}, f)


class _AlfEnvs2(_AlfEnvs):
    def reset(self):
        obs = ["You are in a room. Your task is to: put a pencil on a shelf."] * 2
        return obs, None, [{"extra.gamefile": gamefile2}, {"extra.gamefile": gamefile2}]

    def step(self, actions):
        obs = ["Nothing happens." if a.startswith("move") else f"You did {a}." for a in actions]
        return obs, None, [0.0, 0.0], [False, False], [{"extra.gamefile": gamefile2, "won": False}
                                                        for _ in actions]


for which in ("walkthrough", "milestone"):
    cfg = config(True)
    cfg.algorithm.progress_rank.alfworld_k = which
    mgr = AlfWorldEnvironmentManager(_AlfEnvs2(), _alf_proj, cfg)
    mgr.reset(None)
    # row 0 takes a DIFFERENT pencil straight away (the walkthrough goes to desk 1 first)
    _, _, _, infos = mgr.step(["take pencil 7 from shelf 3", "go to desk 1"])
    if which == "walkthrough":
        check([i["progress_k"] for i in infos] == [0, 1] and [i["progress_k_milestone"] for i in infos] == [1, 0],
              "walkthrough mode: progress_k is the pointer, the milestone count rides beside it")
        check(infos[0]["progress_total_milestone"] == 2 and infos[0]["progress_total"] == 4,
              "each with its own K")
    else:
        check([i["progress_k"] for i in infos] == [1, 0], "milestone mode: (a) ranks by the milestones")
cfg = config(True)
cfg.algorithm.progress_rank.alfworld_k = "typo"
try:
    P.alfworld_k_definition(cfg)
    check(False, "an unknown alfworld_k is refused")
except AssertionError:
    check(True, "an unknown alfworld_k is refused")

tbl = record(True, [{"progress_k": 3, "progress_total": 6, "progress_k_milestone": 2,
                     "progress_total_milestone": 3}, {}])
check(tbl[0][0]["progress_k_milestone"] == 2.0 and np.isnan(tbl[1][0]["progress_k_milestone"]),
      "the recorder carries the milestone columns too, NaN where a task has none")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
