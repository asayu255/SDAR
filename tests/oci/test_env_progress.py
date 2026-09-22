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


check(P.alfworld_k_definition(config(True)) == "milestone",
      "the milestone count is what (a) ranks ALFWorld by unless told otherwise")
for on in (True, False):
    _cfg = config(on)
    _cfg.algorithm.progress_rank.alfworld_k = "walkthrough"   # this section tests the pointer
    mgr = AlfWorldEnvironmentManager(_AlfEnvs(), _alf_proj, _cfg)
    mgr.reset(None)
    _, _, _, infos = mgr.step(["go to desk 1", "take pencil 2 from desk 1"])
    if on:
        check([i.get("progress_k") for i in infos] == [1, 0] and infos[0].get("progress_total") == 4,
              "every row carries k of K; the refused step did not count")
        _, _, _, infos = mgr.step(["look", "go to desk 1"])
        check([i.get("progress_k") for i in infos] == [1, 1], "and the pointer persists across turns")
        check([i.get("revisits") for i in infos] == [0, 0] and [i.get("progress_done_walkset") for i in infos] == [1, 1],
              "shadows: no repeat yet; both rows did one walkthrough line (the refused take did not count)")
        _, _, _, infos = mgr.step(["go to desk 1", "look"])
        check([i.get("revisits") for i in infos] == [1, 0], "row 0 repeated 'go to desk 1': revisits 1")
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
        check([i.get("committed") for i in infos] == [False, False] and [i.get("revisits") for i in infos] == [0, 0],
              "shadows: nobody bought yet, no repeated action")
        _, _, _, infos = mgr.step(["click[b07abc1234]", "click[buy now]"])
        check(infos[0]["revisits"] == 1 and infos[1]["committed"] is True and infos[0]["committed"] is False,
              "row 0 repeated a click (revisits 1); row 1's buy is the terminal action (committed), goal or not")
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

print("9b. ALFWorld 'arrived': at a target receptacle, holding the (treated) target object")
m = P.AlfworldMilestones(game_dir("pick_and_place_simple", "Mug", "Shelf"))
check(m.total == 2 and m.total_arrive == 3 and m.k_arrive == 0, "milestone_arrive: K + 1")
m.step("go to shelf 1", OKOBS)
check(m.k_arrive == 0, "standing at the target with empty hands is not it")
m.step("go to desk 1", OKOBS)
m.step("take mug 2 from desk 1", OKOBS)
check(m.k_arrive == 1 and not m.arrived, "holding it somewhere else is not it either")
m.step("go to shelf 3", "Nothing happens.")
check(not m.arrived, "a 'go to' the environment refused moves nobody")
m.step("go to shelf 3", OKOBS)
check(m.arrived and m.k_arrive == 2 and m.k == 1, "carried to ANY shelf: arrived; the base count is untouched")
m.step("go to desk 1", OKOBS)
check(m.k_arrive == 2, "and it stays (k never goes down)")

m = P.AlfworldMilestones(game_dir("pick_clean_then_place_in_recep", "Cup", "Microwave"))
m.step("take cup 1 from cabinet 2", OKOBS)
m.step("go to microwave 1", OKOBS)
check(not m.arrived and m.k_arrive == 1, "treatment task: an UNtreated cup at the target is off the path")
m.step("go to sinkbasin 1", OKOBS)
m.step("clean cup 1 with sinkbasin 1", OKOBS)
check(m.k_arrive == 2 and not m.arrived, "treated, not yet carried")
m.step("go to microwave 1", OKOBS)
check(m.arrived and m.k_arrive == 3 and m.total_arrive == 4, "the treated cup at the target: arrived")
m.step("move cup 1 to microwave 1", OKOBS)
check(m.k_arrive == 4 and m.k == 3, "placing completes both counts")

m = P.AlfworldMilestones(game_dir("pick_heat_then_place_in_recep", "Tomato", "CounterTop"))
m.step("take tomato 2 from countertop 1", OKOBS)
m.step("go to microwave 1", OKOBS)
m.step("heat tomato 2 with microwave 1", OKOBS)
m.step("go to coffeemachine 1", OKOBS)
check(not m.arrived, "a receptacle sharing the target's location is not known to be the target")
m.step("move tomato 2 to countertop 1", OKOBS)
check(m.arrived and m.k_arrive == m.total_arrive == 4, "but a counted placement implies arrived (placed => arrived)")

m = P.AlfworldMilestones(game_dir("pick_and_place_simple", "Tomato", "CounterTop"))
m.step("go to coffeemachine 1", OKOBS)
m.step("take tomato 2 from countertop 1", OKOBS)
check(m.arrived, "taking FROM a receptacle proves the rollout stands at it")
m = P.AlfworldMilestones(game_dir("pick_and_place_simple", "Mug", "Fridge"))
m.step("take mug 1 from desk 1", OKOBS)
m.step("go to countertop 1", OKOBS)
m.step("open fridge 1", OKOBS)
check(m.arrived, "so does opening it")

m = P.AlfworldMilestones(game_dir("look_at_obj_in_light", "Bowl", lamp="DeskLamp"))
m.step("go to dresser 1", "You arrive at dresser 1. On the dresser 1, you see a desklamp 1, and a pen 2.")
check(not m.arrived and m.total_arrive == 3, "look-at: at the lamp without the object is not it")
m.step("go to desk 1", "You arrive at desk 1. On the desk 1, you see a bowl 1.")
m.step("take bowl 1 from desk 1", OKOBS)
check(not m.arrived, "the lamp was at the PREVIOUS place")
m.step("go to dresser 1", "You arrive at dresser 1. On the dresser 1, you see a desklamp 1, and a pen 2.")
check(m.arrived and m.k_arrive == 2, "holding the bowl where the arrival text names the lamp: arrived")
m.step("use desklamp 1", OKOBS, won=True)
check(m.k_arrive == 3 and m.k == 2, "won is at K in both counts")
m = P.AlfworldMilestones(game_dir("look_at_obj_in_light", "Bowl", lamp="DeskLamp"))
m.step("go to desk 1", "You arrive at desk 1. On the desk 1, you see a bowl 1, and a desklamp 1.")
m.step("take bowl 1 from desk 1", OKOBS)
check(m.arrived, "object and lamp at the same place: taking it is arriving")

m = P.AlfworldMilestones(game_dir("pick_two_obj_and_place", "Pencil", "Drawer"))
m.step("take pencil 2 from desk 1", OKOBS)
m.step("go to drawer 1", OKOBS)
check(m.k_arrive == 2 and m.total_arrive == 5, "pick-two: one flag, K = 5")
m.step("move pencil 2 to drawer 1", OKOBS)
m.step("go to desk 1", OKOBS)
m.step("take pencil 3 from desk 1", OKOBS)
m.step("move pencil 3 to drawer 1", OKOBS)
check(m.k_arrive == 5, "and the full path reaches it")
m = P.AlfworldMilestones(game_dir("pick_heat_then_place_in_recep", "Mug", "CoffeeMachine"))
m.step("take mug 3 from countertop 1", OKOBS)
m.step("heat mug 3 with microwave 1", OKOBS, won=True)
check(m.k_arrive == m.total_arrive == 4, "a game won early is at K here too")
check(P.AlfworldMilestones(None).total_arrive == 0 and P.AlfworldMilestones(None).k_arrive == 0,
      "no milestones, no arrived")

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


for which in ("walkthrough", "milestone", "milestone_arrive", "walkthrough_set"):
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
    elif which == "milestone":
        check([i["progress_k"] for i in infos] == [1, 0] and infos[0]["progress_total"] == 2,
              "milestone mode: (a) ranks by the milestones")
        check([i["progress_k_arrive"] for i in infos] == [2, 0] and infos[0]["progress_total_arrive"] == 3,
              "and the count with 'arrived' is recorded beside it, so a run can see what it would split")
    elif which == "milestone_arrive":
        # row 0 took the pencil FROM a shelf, the target type: it is holding it at the target
        check([i["progress_k"] for i in infos] == [2, 0] and infos[0]["progress_total"] == 3
              and [i["progress_k_milestone"] for i in infos] == [1, 0] and infos[0]["progress_total_milestone"] == 2,
              "milestone_arrive mode: (a) ranks by milestones + arrived; the milestone columns keep the base count")
        check([i["progress_k_walkset"] for i in infos] == [0, 1] and infos[0]["progress_total_walkset"] == 4,
              "and the walkthrough-set count rides beside it in its own columns")
    else:
        # walkthrough_set: row 1's `go to desk 1` is a line of the set (type-normalised);
        # row 0's take from a SHELF is not in this walkthrough at all. K is the distinct
        # normalised line count, and the pointer / milestone columns keep their own values.
        check([i["progress_k"] for i in infos] == [0, 1] and infos[0]["progress_total"] == 4,
              "walkthrough_set mode: (a) ranks by the set of normalised walkthrough lines")
        check([i["progress_k_milestone"] for i in infos] == [1, 0] and infos[0]["progress_total_milestone"] == 2,
              "the milestone count still rides beside it")
cfg = config(True)
cfg.algorithm.progress_rank.alfworld_k = "typo"
try:
    P.alfworld_k_definition(cfg)
    check(False, "an unknown alfworld_k is refused")
except AssertionError:
    check(True, "an unknown alfworld_k is refused")

print("11. the walkthrough as a set (alfworld_k = walkthrough_set)")
check(P.normalize_alfworld_action("  Take Pan 1 From StoveBurner 2 ") == "take pan from stoveburner",
      "normalise: lowercase, single spaces, instance numbers gone")
check(P.normalize_alfworld_action("put cup 3 in/on shelf 12") == "put cup in/on shelf",
      "normalise keeps 'in/on' and strips a two-digit instance")
clean = ["go to cabinet 5", "take cup 1 from cabinet 5", "go to sinkbasin 1",
         "clean cup 1 with sinkbasin 1", "go to shelf 2", "put cup 1 in/on shelf 2"]
ws = P.AlfworldWalkSet(clean)
check(ws.total == 6 and ws.k == 0, "six distinct lines, none done")
ws.step("go to shelf 2", "You arrive at shelf 2.")
check(ws.k == 1, "ORDER-FREE: the fifth line counts first")
ws.step("go to cabinet 2", "You arrive at cabinet 2. On the cabinet 2, you see a cup 1.")
check(ws.k == 2, "TYPE-NORMALISED: cabinet 2 is the walkthrough's cabinet 5")
ws.step("take cup 1 from cabinet 2", "Nothing happens.")
check(ws.k == 2, "EXECUTED ONLY: a take the environment refused does not count")
ws.step("take cup 1 from cabinet 2", "You pick up the cup 1 from the cabinet 2.")
check(ws.k == 3, "...and counts once it is carried out")
ws.step("go to shelf 2", "You arrive at shelf 2.")
check(ws.k == 3, "a line done twice is one line")
ws.step("examine shelf 2", "On the shelf 2, you see nothing.")
check(ws.k == 3, "an action outside the walkthrough leaves k alone")
ws.step("look", "You are facing the shelf 2.", won=True)
check(ws.k == 6, "won => K, as the other counts hold")
two = ["go to desk 1", "take pencil 2 from desk 1", "go to shelf 1", "move pencil 2 to shelf 1",
       "go to desk 1", "take pencil 3 from desk 1", "go to shelf 1", "move pencil 3 to shelf 1"]
w2 = P.AlfworldWalkSet(two)
check(w2.total == 4, "DEDUPLICATED: pick_two's eight lines are four distinct ones")
w2.step("take pencil 3 from desk 1", "You pick up the pencil 3 from the desk 1.")
w2.step("take pencil 2 from desk 1", "You pick up the pencil 2 from the desk 1.")
check(w2.k == 1, "...so two takes of one type are one line (the pick_two cost, by design)")
check(P.AlfworldWalkSet([]).total == 0 and P.AlfworldWalkSet(None).k == 0, "no walkthrough, no progress")
e = P.AlfworldWalkSet(clean)
e.step("go to cabinet 5", "You arrive at cabinet 5.", won=True)
check(e.k == 6, "a win on the first line is still at K")
check("walkthrough_set" in P.ALFWORLD_K_DEFINITIONS, "the definition is registered")

tbl = record(True, [{"progress_k": 3, "progress_total": 6, "progress_k_milestone": 2,
                     "progress_total_milestone": 3}, {}])
check(tbl[0][0]["progress_k_milestone"] == 2.0 and np.isnan(tbl[1][0]["progress_k_milestone"]),
      "the recorder carries the milestone columns too, NaN where a task has none")
tbl = record(True, [{"progress_k": 3, "progress_total": 6, "progress_k_arrive": 3, "progress_total_arrive": 4},
                    {"progress_k": 1, "progress_total": 5, "task_score": 0.6}])
check(tbl[0][0]["progress_k_arrive"] == 3.0 and tbl[0][0]["progress_total_arrive"] == 4.0
      and np.isnan(tbl[1][0]["progress_k_arrive"]), "and the count with 'arrived', NaN where a task has none")
check(tbl[1][0]["task_score"] == 0.6 and np.isnan(tbl[0][0]["task_score"]),
      "WebShop's purchase score rides along too, NaN on the tasks that have none")
tbl = record(False, [{"progress_k_arrive": 3, "task_score": 0.6}, {}])
check("progress_k_arrive" not in tbl[0][0] and "task_score" not in tbl[0][0],
      "off: none of them, the batch keeps control's columns")

print("11. ProGPO's coverage D, a shadow beside k")
cov = P.ObservationCoverage("You are in a room.")
check(cov.d == 1, "the initial observation is in the set before the first action")
cov.step("You arrive at desk 1.")
cov.step("Nothing happens.")
cov.step("Nothing happens.")
cov.step("You arrive at desk 1.")
check(cov.d == 3, "a repeated observation adds nothing, a revisit included")
cov.step("you arrive at desk 1.")
cov.step("You arrive at desk 1. ")
check(cov.d == 5, "strings are compared exactly: case and whitespace are not normalised (ProGPO 8.2)")
cov.step(None)
cov.step("")
check(cov.d == 6, "an empty observation is one observation, None the same one")

for on in (True, False):
    _cfg = config(on)
    mgr = AlfWorldEnvironmentManager(_AlfEnvs(), _alf_proj, _cfg)
    mgr.reset(None)
    _, _, _, infos = mgr.step(["go to desk 1", "take pencil 2 from desk 1"])
    if on:
        check([i.get("coverage_d") for i in infos] == [2, 2], "ALFWorld: one new observation each")
        _, _, _, infos = mgr.step(["look", "take pencil 2 from desk 1"])
        check([i.get("coverage_d") for i in infos] == [3, 2],
              "a second 'Nothing happens.' is not new; 'You did look.' is")
    else:
        check(all("coverage_d" not in i for i in infos), "ALFWorld off: nothing is written")

for on in (True, False):
    mgr = WebshopEnvironmentManager(_WsEnvs(), lambda acts: (list(acts), [1] * len(acts)), config(on))
    mgr.reset(None)
    _, _, _, infos = mgr.step(["search[black bag]", "search[shoes]"])
    _, _, _, infos2 = mgr.step(["click[b07abc1234]", "click[b09zzz0000]"])
    if on:
        check([i.get("coverage_d") for i in infos] == [2, 2] and [i.get("coverage_d") for i in infos2] == [2, 2],
              "WebShop: the landing page, then one page string the simulator repeats")
    else:
        check(all("coverage_d" not in i for i in infos), "WebShop off: nothing is written")

for on in (True, False):
    mgr = SearchEnvironmentManager(_SearchEnvs(), lambda acts: (list(acts), [1] * len(acts)), config(on))
    mgr.reset([{"question": QUESTION, "ground_truth": {"target": ["Paris"]}}] * 2)
    _, _, _, infos = mgr.step(["<search> capital of france </search>"] * 2)
    _, _, _, infos2 = mgr.step(["<search> capital of france </search>"] * 2)
    if on:
        check([i.get("coverage_d") for i in infos] == [2, 2] and [i.get("coverage_d") for i in infos2] == [2, 2],
              "Search: the question, then the returned results -- the same results twice count once")
    else:
        check(all("coverage_d" not in i for i in infos), "Search off: nothing is written")

tbl = record(True, [{"progress_k": 1, "progress_total": 1, "coverage_d": 4}, {}])
check(tbl[0][0]["coverage_d"] == 4.0 and np.isnan(tbl[1][0]["coverage_d"]),
      "the recorder carries D, NaN where a row has none")
tbl = record(False, [{"coverage_d": 4}, {}])
check("coverage_d" not in tbl[0][0], "off: not even D reaches the batch")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
