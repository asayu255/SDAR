"""The progress-value records: WebShop's session count and the pv_* columns.

WHAT IT PROTECTS.
  * The switch: the managers count and the loop records when EITHER algorithm.progress_rank or
    algorithm.progress_value is on; with both off the rows carry exactly control's columns.
  * WebShop's session count reads the session the environment holds, not the clicks: black ->
    white -> large on {color: black, size: large} is ONE required option (the legacy count says
    two, and stays exactly as it was); list-form goals are matched as multisets; a buy counts only
    when the environment ended the episode with the goal product in the session.
  * The worker ships that session, and on the turn the environment ends it ships the session from
    BEFORE the action -- the environment resets itself after a purchase, so the live session is
    then another episode's.
  * webshop_k picks which count goes into progress_k; legacy (the default) is bit-identical.
  * Every pv_* column: turn t's "before" is turn t-1's "after", turn 0's is the reset state;
    stagnation counts turns since the count last went up.
  * pv_term: 1 won, 2 the task's own terminal action failed (WebShop buy, Search answer), 3 out of
    turns -- the multitask cap (recorded before it overwrites the dones), the environment's own step
    limit (ALFWorld, Search), or the rollout loop's last turn with no done at all.
No model, no GPU, no Ray, no WebShop index: fake environments throughout.
"""
import json
import os
import re
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "verl")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from agent_system.environments import progress as P  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def config(rank=False, value=True, n=2, max_steps=None, **pr):
    cfg = {"env": {"history_length": 0, "rollout": {"n": n}},
           "algorithm": {"progress_rank": dict({"enable": rank}, **pr),
                         "progress_value": {"enable": value},
                         "oci_sat": {"enable": False}, "oci_slots": {"enable": False},
                         "oci_rank": {"enable": False}}}
    if max_steps is not None:
        cfg["env"]["max_steps"] = max_steps
    return OmegaConf.create(cfg)


STEMS = ("hold", "inside", "at", "ongoal", "optnow", "evid")


def chain_ok(turns, reset_states):
    """Turn t's before == turn t-1's after, for k and every state stem; turn 0's == the reset state."""
    for i, start in enumerate(reset_states):
        prev = dict(start)
        for infos in turns:
            info = infos[i]
            if info["pv_k_before"] != prev["k"]:
                return False
            for s in STEMS:
                if f"pv_{s}_b" in info and info[f"pv_{s}_b"] != prev[s]:
                    return False
            prev = {"k": info["pv_k_after"]}
            prev.update({s: info[f"pv_{s}_a"] for s in STEMS if f"pv_{s}_a" in info})
    return True


def col(turns, key, i):
    return [infos[i][key] for infos in turns]


print("1. the switch")
check(P.progress_on(config(rank=False, value=True)), "progress_value.enable alone turns the counting on")
check(P.progress_on(config(rank=True, value=False)), "progress_rank.enable alone still does")
check(not P.progress_on(config(rank=False, value=False)), "both off: off")
check(not P.progress_on(OmegaConf.create({"algorithm": {"progress_value": {}}})), "a block without enable: off")
check(P.webshop_k_definition(config()) == "legacy", "webshop_k defaults to the legacy count")
check(P.webshop_k_definition(config(webshop_k="session")) == "session", "webshop_k=session is read")
try:
    P.webshop_k_definition(config(webshop_k="fixed"))
    check(False, "an unknown webshop_k is refused")
except AssertionError:
    check(True, "an unknown webshop_k is refused")
check(P.turn_cap(config(max_steps=15)) == 15 and P.turn_cap(config()) is None, "pv_cap is env.max_steps, None when unset")

print("2. WebShop: the session count against the legacy one")
GOAL = {"asin": "B07ABC1234", "goal_options": {"color": "black", "size": "large"}, "name": "a bag"}
landing = {"has_search_bar": True, "clickables": ["search"]}
results = {"has_search_bar": False, "clickables": ["back to search", "next >", "b07abc1234", "b09zzz0000"]}
results_other = {"has_search_bar": False, "clickables": ["back to search", "next >", "b09zzz0000"]}
item = {"has_search_bar": False, "clickables": ["back to search", "< prev", "description", "features",
                                                "reviews", "buy now", "black", "white", "large", "small"]}
S0 = {"asin": None, "options": {}}


def S(asin=None, **options):
    return {"asin": asin, "options": dict(options)}


w = P.WebshopProgress(GOAL, landing, state=S0)
check(w.total == 5 and w.k == 0 and w.k_session == 0 and not w.on_goal and w.opts_now == 0,
      "K = 5; nothing yet in either count; the landing session holds no product")
w.step("search[black bag]", results, state=S0, done=False)
w.step("click[b07abc1234]", item, state=S("b07abc1234"), done=False)
w.step("click[black]", item, state=S("b07abc1234", color="black"), done=False)
check(w.k == 3 and w.k_session == 3 and w.opts_now == 1, "found, opened, one option: 3 in both counts")
w.step("click[white]", item, state=S("b07abc1234", color="white"), done=False)
check(w.opts_now == 0 and w.k_session == 3, "white replaces black in the session: 0 options NOW, the best visit stays 1")
w.step("click[large]", item, state=S("b07abc1234", color="white", size="large"), done=False)
check(w.best_options == 2 and w.session_best_options == 1 and w.opts_now == 1,
      "black -> white -> large: the session holds ONE required option; the legacy count says two")
check(w.k == 4 and w.k_session == 3, "so legacy k 4, session k 3")
w.step("click[buy now]", landing, state=S("b07abc1234", color="white", size="large"), done=True)
check(w.k == 5 and w.k_session == 4 and w.bought and w.session_bought,
      "the buy of the goal product counts in both, but only the legacy count reaches K on a wrong option")

w = P.WebshopProgress(GOAL, landing, state=S0)
w.step("search[bag]", results, state=S0, done=False)
w.step("click[b09zzz0000]", item, state=S("b09zzz0000"), done=False)
w.step("click[black]", item, state=S("b09zzz0000", color="black"), done=False)
check(not w.on_goal and w.opts_now == 0 and w.k_session == 1,
      "options held on ANOTHER product are not the goal's options")
w.step("click [buy now]", landing, state=S("b09zzz0000", color="black"), done=False)
check(not w.session_bought and w.k_session == 1, "a buy the environment did not execute (no done) buys nothing")

w = P.WebshopProgress(GOAL, landing, state=S0)
w.step("search[bag]", results, state=S0, done=False)
w.step("click[b07abc1234]", item, state=S("b07abc1234"), done=False)
w.step("click [buy now]", item, state=S("b07abc1234"), done=False)
check(w.bought and w.k == 3 and not w.session_bought and w.k_session == 2,
      "'click [buy now]' (a space before the bracket): the environment ignores it and the session count "
      "with it; the legacy parse strips the space and counts a buy (kept as the runs recorded it)")

print("3. WebShop: list-form goals, matched as multisets")
lgoal = {"asin": "B07ABC1234", "goal_options": ["black", "large"]}
w = P.WebshopProgress(lgoal, landing, state=S0)
check(w._pairs is None and w.total == 5, "a list-form goal names no option")
w.step("search[bag]", results, state=S0, done=False)
w.step("click[b07abc1234]", item, state=S("b07abc1234", color="black", size="large"), done=False)
check(w.opts_now == 2, "both required values in the session: 2")
w.step("click[black]", item, state=S("b07abc1234", color="black", size="black"), done=False)
check(w.opts_now == 1 and w.session_best_options == 2, "black under two names is still one 'black' against one required")
dgoal = {"asin": "B07ABC1234", "goal_options": ["black", "black"]}
w = P.WebshopProgress(dgoal, landing, state=S0)
w.step("click[b07abc1234]", results, state=S("b07abc1234", color="black", trim="black"), done=False)
check(w.opts_now == 2, "a value required twice is matched twice when the session holds it twice")
w = P.WebshopProgress(lgoal, landing)
w.step("search[bag]", results)
w.step("click[b07abc1234]", item)
w.step("click[black]", item)
check(w._session is None and w.k_session == w.k == 3 and w.opts_now == 1 and w.on_goal,
      "no snapshot ever shipped: the session count falls back to the legacy one, the state to its reconstruction")

print("4. the worker ships the session; on the ending turn, the one from before the action")
from agent_system.environments.env_package.webshop import envs as WS  # noqa: E402


class _FakeTextEnv:
    """WebAgentTextEnv as far as the session goes: exact 'search'/'click' names, one value per option
    NAME, a buy ends the episode and the environment then resets itself into a random new session."""
    OPTION_OF = {"black": "color", "white": "color", "large": "size", "small": "size"}

    def __init__(self, goal):
        self.server = types.SimpleNamespace(user_sessions={}, goals=[goal], product_prices={},
                                            product_item_dict={})
        self.session, self._n = None, 0

    def reset(self, session=None):
        self.server.user_sessions.clear()
        self._n += 1
        self.session = str(session) if session is not None else f"random{self._n}"
        self.server.user_sessions[self.session] = {"goal": self.server.goals[0], "done": False,
                                                   "asin": None, "options": {}}
        return "obs", None

    def step(self, action):
        m = re.match(r"(.+)\[(.+)\]", action)
        name, arg = (m.group(1), m.group(2).lower()) if m else (action, None)
        s = self.server.user_sessions[self.session]
        done, reward = False, 0.0
        if name == "search" and arg:
            s["asin"], s["options"] = None, {}
        elif name == "click" and arg:
            if arg == "buy now":
                done = True
                good = (s["asin"] or "").lower() == "b07abc1234" and s["options"] == {"color": "black", "size": "large"}
                reward = 1.0 if good else 0.5
            elif arg in self.OPTION_OF:
                s["options"][self.OPTION_OF[arg]] = arg
            else:
                s["asin"] = arg.upper()
        if done:
            self.reset()
        return "state", reward, done, None

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": []}


def worker():
    wk = WS.WebshopWorker.__new__(WS.WebshopWorker)
    wk.env, wk._goal_order, wk._env_seed = _FakeTextEnv(GOAL), None, 0
    return wk


wk = worker()
_, rinfo = wk.reset(3)
check(rinfo["ws_state"] == {"asin": None, "options": {}}, "reset ships the landing session: no product, no options")
turns = [wk.step(a) for a in ("search[black bag]", "click[b07abc1234]", "click[black]", "click[large]")]
check(turns[1][3]["ws_state"] == {"asin": "b07abc1234", "options": {}}, "after a product click: its asin, lower-cased")
check(turns[3][3]["ws_state"] == {"asin": "b07abc1234", "options": {"color": "black", "size": "large"}},
      "after option clicks: one value per option name")
_, reward, done, binfo = wk.step("click[buy now]")
check(done and binfo["won"] and reward == 10.0, "the fake purchase of the right product and options wins")
check(wk.env.session != "3" and wk.env.server.user_sessions[wk.env.session]["asin"] is None,
      "the environment has reset itself: the live session is a new one")
check(binfo["ws_state"] == {"asin": "b07abc1234", "options": {"color": "black", "size": "large"}},
      "...yet the ending turn ships the session BEFORE the action -- the one that was bought")
w = P.WebshopProgress(GOAL, landing, state=rinfo["ws_state"])
pages = [results, item, item, item]
for (_, _, d, info), a, pg in zip(turns, ("search[black bag]", "click[b07abc1234]", "click[black]", "click[large]"), pages):
    w.step(a, pg, state=info["ws_state"], done=d)
w.step("click[buy now]", landing, state=binfo["ws_state"], done=done)
check(w.session_bought and w.k_session == 5 == w.total, "fed to the counter: a buy of the goal product, at K")
wk = worker()
wk.reset(3)
wk.step("click[b07abc1234]")
_, _, d2, info2 = wk.step("click [buy now]")
check(not d2 and info2["ws_state"]["asin"] == "b07abc1234", "a malformed buy: no done, the session is untouched")
wk.env.server.user_sessions.clear()
check(wk._session_state() is None, "a session that cannot be read is None, never an exception in the step")

print("5. WebShop manager: webshop_k, the pv_* chain, stagnation")
from agent_system.environments.env_manager import (  # noqa: E402
    AlfWorldEnvironmentManager, MultiTaskEnvironmentManager, SearchEnvironmentManager, WebshopEnvironmentManager)


class _ScriptedWs:
    """WebshopMultiProcessEnv's interface replaying, per turn and env, (page, session, done, won)."""

    def __init__(self, script, ship_state=True):
        self.script, self.t, self.ship = script, 0, ship_state
        self.n = len(script[0])

    def reset(self):
        obs = ["WebShop [SEP] Instruction: [SEP] buy a black large bag [SEP] Search"] * self.n
        infos = [{"available_actions": landing, "goal": GOAL} for _ in range(self.n)]
        if self.ship:
            for info in infos:
                info["ws_state"] = dict(S0)
        return obs, infos

    def step(self, actions):
        turn = self.script[self.t]
        self.t += 1
        obs = ["WebShop [SEP] Instruction: [SEP] buy a black large bag [SEP] page"] * self.n
        infos = [{"available_actions": pg, "won": won, "task_score": 1.0 if won else 0.0} for pg, _, _, won in turn]
        if self.ship:
            for info, (_, st, _, _) in zip(infos, turn):
                info["ws_state"] = st
        return obs, [10.0 if t[3] else 0.0 for t in turn], [t[2] for t in turn], infos


# env 0: search, open the goal, black, white, large, buy (wrong option: not won)
# env 1: search elsewhere, open another product, click an option on it, back, search again, open the goal
WS_ACTIONS = [("search[black bag]", "search[shoes]"), ("click[b07abc1234]", "click[b09zzz0000]"),
              ("click[black]", "click[black]"), ("click[white]", "click[back to search]"),
              ("click[large]", "search[black bag]"), ("click[buy now]", "click[b07abc1234]")]
WS_SCRIPT = [
    [(results, S0, False, False), (results_other, S0, False, False)],
    [(item, S("b07abc1234"), False, False), (item, S("b09zzz0000"), False, False)],
    [(item, S("b07abc1234", color="black"), False, False), (item, S("b09zzz0000", color="black"), False, False)],
    [(item, S("b07abc1234", color="white"), False, False), (landing, S0, False, False)],
    [(item, S("b07abc1234", color="white", size="large"), False, False), (results, S0, False, False)],
    [(landing, S("b07abc1234", color="white", size="large"), True, False), (item, S("b07abc1234"), False, False)],
]


def run_ws(cfg, script=WS_SCRIPT, ship=True, actions=WS_ACTIONS):
    mgr = WebshopEnvironmentManager(_ScriptedWs(script, ship), lambda acts: (list(acts), [1] * len(acts)), cfg)
    mgr.reset(None)
    return mgr, [mgr.step(list(a))[3] for a in actions]


_, legacy = run_ws(config(max_steps=15))
_, session = run_ws(config(max_steps=15, webshop_k="session"))
check(col(legacy, "progress_k", 0) == [1, 2, 3, 3, 4, 5] and col(session, "progress_k", 0) == [1, 2, 3, 3, 3, 4],
      "progress_k: legacy (the default) keeps the over-count, webshop_k=session the session's count")
check(col(legacy, "pv_k_after", 0) == col(session, "pv_k_after", 0) == [1, 2, 3, 3, 3, 4],
      "pv_k is the session count whatever webshop_k says")
check(chain_ok(legacy, [{"k": 0, "ongoal": 0, "optnow": 0}] * 2),
      "every turn's before is the previous turn's after; turn 0's is the landing page")
check(col(legacy, "pv_ongoal_a", 0) == [0, 1, 1, 1, 1, 1] and col(legacy, "pv_optnow_a", 0) == [0, 0, 1, 0, 1, 1],
      "env 0's session now: on the goal from the click on; options 1, 0 (white), 1 (large), 1 at the buy")
check(col(legacy, "pv_optnow_a", 1) == [0, 0, 0, 0, 0, 0] and col(legacy, "pv_k_after", 1) == [0, 0, 0, 0, 1, 2],
      "env 1: an option on another product is not progress; found and opened come late")
check(col(legacy, "pv_stag_before", 0) == [0, 0, 0, 0, 1, 2] and col(legacy, "pv_stag_before", 1) == [0, 1, 2, 3, 4, 0],
      "stagnation: turns since the count last went up, 0 at t = 0 and right after an increase")
check(col(legacy, "pv_term", 0) == [0, 0, 0, 0, 0, 2] and col(legacy, "pv_env_done", 0) == [0, 0, 0, 0, 0, 1]
      and col(legacy, "pv_won", 0) == [0] * 6, "env 0's buy ended the episode without a win: pv_term 2")
check(col(legacy, "pv_term", 1) == [0] * 6, "env 1 is still running: nothing is ended by the manager")
check(all(i["pv_cap"] == 15 and i["pv_K"] == 5 for t in legacy for i in t), "pv_cap = env.max_steps, pv_K = 5")
check(all(k not in i for t in legacy for i in t for k in ("pv_hold_a", "pv_evid_a", "pv_t")),
      "no ALFWorld or Search stems on a WebShop info, and no pv_t (the loop's)")
_, nostate = run_ws(config(max_steps=15), ship=False)
check(col(nostate, "pv_k_after", 0) == col(nostate, "progress_k", 0) == [1, 2, 3, 3, 4, 5],
      "a worker that ships no session: pv_k is the legacy count (k_session's fallback)")
try:
    run_ws(config(max_steps=15, webshop_k="session"), ship=False)
    check(False, "webshop_k=session without a shipped session is refused at reset")
except AssertionError:
    check(True, "webshop_k=session without a shipped session is refused at reset")
_, off = run_ws(config(rank=False, value=False, max_steps=15))
check(all(not any(k.startswith("pv_") for k in i) and "progress_k" not in i for t in off for i in t),
      "both switches off: the manager writes nothing")

print("6. ALFWorld manager: the milestones' current state, and how an episode ends")


def game_dir(task_type, obj, recep="", lamp=""):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "traj_data.json"), "w") as f:
        json.dump({"task_type": task_type, "pddl_params": {
            "object_target": obj, "parent_target": recep, "toggle_target": lamp,
            "mrecep_target": "", "object_sliced": False}}, f)
    gf = os.path.join(d, "game.tw-pddl")
    with open(gf, "w") as f:
        json.dump({"walkthrough": ["go to desk 1", "take pencil 2 from desk 1", "go to shelf 1",
                                   "move pencil 2 to shelf 1"], "pddl_problem": "(:init )"}, f)
    return gf


GF = game_dir("pick_and_place_simple", "Pencil", "Shelf")


class _AlfEnvs:
    is_train = True
    group_n = 2

    def __init__(self, dones, wons):
        self.get_admissible_commands = [["look"], ["look"]]
        self.dones, self.wons, self.t = dones, wons, 0

    def reset(self):
        obs = ["You are in a room. Your task is to: put a pencil on a shelf."] * 2
        return obs, None, [{"extra.gamefile": GF}, {"extra.gamefile": GF}]

    def step(self, actions):
        d, won = self.dones[self.t], self.wons[self.t]
        self.t += 1
        return ([f"You did {a}." for a in actions], None, [0.0, 0.0], list(d),
                [{"extra.gamefile": GF, "won": x} for x in won])


ALF_ACTIONS = [("go to desk 1", "look"), ("take pencil 2 from desk 1", "look"),
               ("go to shelf 1", "look"), ("move pencil 2 to shelf 1", "look")]
# env 0 wins on the placement; env 1 is ended by TextWorld's own step limit on the same turn
ALF_DONES = [(False, False), (False, False), (False, False), (True, True)]
ALF_WONS = [(False, False), (False, False), (False, False), (True, False)]
acfg = config(max_steps=50, alfworld_k="milestone_arrive")
mgr = AlfWorldEnvironmentManager(_AlfEnvs(ALF_DONES, ALF_WONS), lambda t, p: (list(t), [1] * len(t)), acfg)
mgr.reset(None)
alf = [mgr.step(list(a))[3] for a in ALF_ACTIONS]
check(col(alf, "pv_hold_a", 0) == [0, 1, 1, 0] and col(alf, "pv_at_a", 0) == [0, 0, 1, 1]
      and col(alf, "pv_inside_a", 0) == [0, 0, 0, 1],
      "env 0: holds the pencil from the take, stands at the shelf, then one pencil is inside it")
check(col(alf, "pv_k_after", 0) == col(alf, "progress_k", 0) == [0, 1, 2, 3] and alf[0][0]["pv_K"] == 3,
      "pv_k is the alfworld_k count (milestone_arrive: took, arrived, placed/won), with its K")
check(chain_ok(alf, [{"k": 0, "hold": 0, "inside": 0, "at": 0}] * 2), "the before/after chain from the reset state")
check(col(alf, "pv_stag_before", 1) == [0, 1, 2, 3], "env 1 never moves: stagnation 0, 1, 2, 3")
check(col(alf, "pv_term", 0) == [0, 0, 0, 1] and col(alf, "pv_term", 1) == [0, 0, 0, 3],
      "a win is 1; ALFWorld has no terminal action, so an environment done without a win is out of turns")
check(col(alf, "pv_env_done", 1) == [0, 0, 0, 1] and alf[3][0]["pv_won"] == 1, "pv_env_done and pv_won as the env said")
check(all(k not in i for t in alf for i in t for k in ("pv_ongoal_a", "pv_evid_a")), "no WebShop or Search stems")
look = P.AlfworldMilestones(game_dir("look_at_obj_in_light", "Bowl", lamp="DeskLamp"))
look.step("go to desk 1", "You arrive at desk 1. On the desk 1, you see a bowl 1, and a desklamp 1.")
check(look.current_state() == {"hold": 0, "inside": 0, "at": 1}, "look_at: 'at' is a lamp of the type being here")
look.step("take bowl 1 from desk 1", "You pick up the bowl 1.")
check(look.current_state()["hold"] == 1, "...and holding the bowl there")
treat = P.AlfworldMilestones(game_dir("pick_clean_then_place_in_recep", "Cup", "Microwave"))
treat.step("take cup 1 from cabinet 2", "ok")
check(treat.current_state()["hold"] == 0, "a treatment task: an untreated cup in hand is not 'hold'")
treat.step("clean cup 1 with sinkbasin 1", "ok")
check(treat.current_state()["hold"] == 1, "...a cleaned one is")

print("7. Search manager: evidence, and answer vs out of turns")
QUESTION = "what is the capital of france"
PARIS = "<information> Paris is the capital of France. </information>"


class _ScriptedSearch:
    group_n = 3
    is_train = True

    def __init__(self, script):
        self.script, self.t = script, 0

    def reset(self, kwargs=None):
        return [QUESTION] * 3, [{} for _ in range(3)]

    def step(self, actions):
        turn = self.script[self.t]
        self.t += 1
        return [o for o, _, _ in turn], [1.0 if w else 0.0 for _, _, w in turn], [d for _, d, _ in turn], \
            [{"won": w} for _, _, w in turn]


# env 0 finds it and answers right; env 1 answers wrong; env 2 searches until SearchEnv's own max_turns
SEARCH_ACTIONS = [("<search> capital of france </search>",) * 3,
                  ("<answer> Paris </answer>", "<answer> Lyon </answer>", "<search> france </search>")]
SEARCH_SCRIPT = [[(PARIS, False, False), ("<information> Lyon. </information>", False, False), (PARIS, False, False)],
                 [("", True, True), ("", True, False), ("", True, False)]]
scfg = config(n=3, max_steps=2, search_k="evidence_answered")
mgr = SearchEnvironmentManager(_ScriptedSearch(SEARCH_SCRIPT), lambda acts: (list(acts), [1] * len(acts)), scfg)
mgr.reset([{"question": QUESTION, "ground_truth": {"target": ["Paris"]}}] * 3)
srch = [mgr.step(list(a))[3] for a in SEARCH_ACTIONS]
check(col(srch, "pv_evid_a", 0) == [1, 1] and col(srch, "pv_evid_a", 1) == [0, 0], "evidence seen, as the rule reads it")
check(col(srch, "pv_k_after", 0) == [1, 2] and col(srch, "pv_k_after", 2) == [1, 1] and srch[0][0]["pv_K"] == 2,
      "pv_k is the search_k count (evidence_answered: seen, then answered after it)")
check(chain_ok(srch, [{"k": 0, "evid": 0}] * 3), "the before/after chain from the reset state")
check([srch[1][i]["pv_term"] for i in range(3)] == [1, 2, 3],
      "right answer 1, wrong answer 2, SearchEnv's own turn limit without an answer 3")
check(srch[1][2]["pv_env_done"] == 1 and srch[1][0]["pv_won"] == 1, "all three environment-done; only the first won")

print("8. the multitask cap: recorded before it overwrites the dones")


def run_multitask(cap, cfg, script=WS_SCRIPT):
    inner = WebshopEnvironmentManager(_ScriptedWs(script), lambda acts: (list(acts), [1] * len(acts)), cfg)
    mt = MultiTaskEnvironmentManager({"webshop": inner}, {"webshop": cap}, cfg)
    mt.reset([{"task_name": "webshop"}] * 2)
    out = []
    for a in WS_ACTIONS[:cap]:
        _, _, dones, infos = mt.step(list(a))
        out.append((list(dones), infos))
    return out


mcfg = config(max_steps=6)
mt = run_multitask(6, mcfg)
dones6, last = mt[-1]
check(dones6 == [True, True], "the cap turn ends both episodes")
check(last[1]["pv_term"] == 3 and last[1]["pv_cap_forced"] is True and last[1]["pv_env_done"] == 0,
      "env 1, still running at the cap: out of turns (3), the cap forced it, the environment did not end it")
check(last[0]["pv_term"] == 2 and last[0]["pv_cap_forced"] is False and last[0]["pv_env_done"] == 1,
      "env 0 bought on the cap turn: the environment's own verdict (2), not a timeout")
check(all(i["pv_cap_forced"] is False for _, infos in mt[:-1] for i in infos), "no cap before the cap turn")
WIN_SCRIPT = [list(t) for t in WS_SCRIPT]
WIN_SCRIPT[5] = [(landing, S("b07abc1234", color="black", size="large"), True, True), WS_SCRIPT[5][1]]
mt = run_multitask(6, mcfg, WIN_SCRIPT)
check(mt[-1][1][0]["pv_term"] == 1, "a win on the cap turn is a win")
mt = run_multitask(5, config(max_steps=5))
check([i["pv_term"] for i in mt[-1][1]] == [3, 3] and all(i["pv_cap_forced"] for i in mt[-1][1]),
      "a tighter cap ends both mid-way: both out of turns")
mt = run_multitask(6, config(rank=False, value=False, max_steps=6))
check(all("pv_cap_forced" not in i and "pv_term" not in i for _, infos in mt for i in infos),
      "both switches off: the multitask manager writes nothing either")

print("9. the rollout loop: every row carries the columns; the loop's own end is a timeout")
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector, mark_pv_timeouts  # noqa: E402
from verl import DataProto  # noqa: E402


def collector(on):
    c = TrajectoryCollector.__new__(TrajectoryCollector)
    c._progress_rank_on = on
    c._queue_row_for_prefetch = lambda *a: None
    return c


def record_turns(on, turns, active=None):
    """Record each turn's infos as the loop does; returns total_batch_list."""
    c = collector(on)
    n = len(turns[0])
    tuids = np.array([f"t{i}" for i in range(n)], dtype=object)
    tbl, tinf = [[] for _ in range(n)], [[] for _ in range(n)]
    for t, infos in enumerate(turns):
        am = np.ones(n, dtype=bool) if active is None else np.asarray(active[t], dtype=bool)
        batch = DataProto.from_single_dict({"input_ids": torch.zeros(n, 3, dtype=torch.long), "traj_uid": tuids,
                                            "active_masks": np.array(list(am), dtype=object)})
        c._record_turn(batch=batch, active_idx=np.nonzero(am)[0], active_masks=am, infos=infos, traj_uid=tuids,
                       total_batch_list=tbl, total_infos=tinf, batch_size=n)
    return tbl


tbl = record_turns(True, legacy)
check(all(set(P.PV_COLUMNS) <= set(r) for rows in tbl for r in rows), "on: every row has every pv_* column")
check([r["pv_t"] for r in tbl[0]] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "pv_t is the row's turn, 0 to T-1")
check([r["pv_k_before"] for r in tbl[0]] == [0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
      and [r["pv_optnow_b"] for r in tbl[0]][1:] == [r["pv_optnow_a"] for r in tbl[0]][:-1],
      "the rows carry the managers' before/after as floats")
check(all(np.isnan(r["pv_hold_a"]) and np.isnan(r["pv_evid_b"]) for rows in tbl for r in rows),
      "another task's stems are NaN on a WebShop row")
mark_pv_timeouts(tbl)
check(tbl[0][-1]["pv_term"] == 2.0 and tbl[1][-1]["pv_term"] == 3.0,
      "single-task: env 1 stopped at env.max_steps with no done is out of turns; env 0's failed buy stays 2")
check([r["pv_term"] for r in tbl[1][:-1]] == [0.0] * 5, "only the last turn is terminal")
# COMPACT_RECORD off: finished rows are appended too, inactive; the timeout goes on the last ACTIVE row
tbl = record_turns(True, alf[:2] + alf[1:2], active=[[True, True], [True, True], [False, True]])
if len(tbl[0]) == 3:   # the rows of finished trajectories were recorded (COMPACT_RECORD=0)
    mark_pv_timeouts(tbl)
    check(tbl[0][1]["pv_term"] == 3.0 and tbl[0][2]["pv_term"] == 0.0, "the last ACTIVE row is marked, not a finished one")
else:
    mark_pv_timeouts(tbl)
    check(tbl[0][-1]["pv_term"] == 3.0 and tbl[1][-1]["pv_term"] == 3.0, "the last recorded row of each is marked")
off_rows = record_turns(False, legacy)
empty_off = record_turns(False, [[{}, {}]])
check(set(off_rows[0][0]) == set(empty_off[0][0]) == {"input_ids", "traj_uid", "active_masks", "gamefile"},
      "both switches off: the rows carry exactly control's columns, whatever the infos hold")
before = [dict(r) for rows in off_rows for r in rows]
mark_pv_timeouts(off_rows)
check([dict(r) for rows in off_rows for r in rows] == before, "and the timeout pass touches nothing then")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
