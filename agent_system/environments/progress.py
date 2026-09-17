"""How far each trajectory got along its task's correct sequence: the k of (a).

WHAT IT IS FOR. A stuck group -- eight rollouts, eight failures -- has an
advantage of exactly zero, so GRPO learns nothing from it. (a) ranks those eight
by rule-based progress and writes the ranking into the advantage (see
verl/trainer/ppo/progress_rank.py). This module is the environment half: it
counts, for EVERY row rather than only for a document row, how many steps of the
correct sequence the trajectory has actually carried out.

    k   steps done so far, monotone within an episode
    K   steps in the correct sequence; 0 means "no sequence", and the trajectory
        then has no progress at all (its group is left alone, not ranked as 0)

THE SAME CORRECT SEQUENCES AS THE DOCUMENTS, per task:

  ALFWorld  TextWorld's own walkthrough (game.tw-pddl), 3553/3553 replayed. k
            moves past a step only when the action taken IS that step and the
            environment carried it out -- "Nothing happens." is ALFWorld's own
            failure line, the one its handcoded expert reads. Unrelated actions in
            between are allowed and leave k where it is.
  WebShop   the goal record: search -> open the goal product -> each required
            option -> buy. The query is free text, so the first step is judged by
            the SCREEN (the goal product was on a results page), and every click
            counts only when it was on the page -- the environment's own test for
            whether a click does anything.
  Search    one step: a returned result carried the answer, with the numeric-answer
            guard of oci_layout.evidence_in_text. Yes/no questions have no
            progress ("yes" is in almost any passage), and neither does a question
            with no answer to look for.

BESIDE k, A SHADOW: ProGPO's first-visit observation coverage D (2607.22724), the
closest published progress signal, counted on the same rollouts so the two can be
compared without a run of their own (see ObservationCoverage). Nothing ranks by it.

NOTHING HERE TOUCHES A REWARD OR AN OBSERVATION. The counts ride on the info
dict to the rollout loop and from there to the trainer as columns.
"""

import hashlib
import re
from typing import Iterable, Optional, Tuple

PROGRESS_K_INFO = "progress_k"
PROGRESS_TOTAL_INFO = "progress_total"
# The second ALFWorld count (milestones of the task type), recorded beside the
# first so the two definitions can be compared on the same rollouts. NaN on the
# other tasks' rows.
PROGRESS_K_MILESTONE_INFO = "progress_k_milestone"
PROGRESS_TOTAL_MILESTONE_INFO = "progress_total_milestone"
ALFWORLD_K_DEFINITIONS = ("walkthrough", "milestone")
# ProGPO's D, as of the row's turn: distinct observations seen, the first included.
COVERAGE_D_INFO = "coverage_d"


def alfworld_k_definition(config) -> str:
    """``algorithm.progress_rank.alfworld_k``: which ALFWorld count (a) ranks by."""
    try:
        cfg = (config.get("algorithm", {}) or {}).get("progress_rank", None) or {}
        name = str(cfg.get("alfworld_k", "milestone") or "milestone")
    except AttributeError:
        name = "milestone"
    assert name in ALFWORLD_K_DEFINITIONS, (
        f"algorithm.progress_rank.alfworld_k={name!r}; expected one of {ALFWORLD_K_DEFINITIONS}")
    return name


def progress_on(config) -> bool:
    """``algorithm.progress_rank.enable``: the managers count, the loop records.

    Off, nothing here runs and the batch has exactly control's columns.
    """
    try:
        cfg = (config.get("algorithm", {}) or {}).get("progress_rank", None)
        return bool(cfg is not None and cfg.get("enable", False))
    except AttributeError:
        return False


# ALFWorld's failure line (alfworld/agents/controller/base.py and oracle.py).
_ALFWORLD_NOTHING = "nothing happens"


def alfworld_executed(observation) -> bool:
    """Did the environment carry the action out?"""
    return _ALFWORLD_NOTHING not in str(observation or "").lower()


def advance_walkthrough(walk, ptr: int, action, observation) -> int:
    """The walkthrough pointer after one turn.

    Moves past step ``ptr`` only when the action taken is that step, compared the
    way the document rows' pointer compares it, AND the environment executed it.
    The second condition is what the document pointer does not need: a row
    following the path does not type a step before it is possible, but a plain
    row flailing in a stuck group can type ``take pencil 2 from desk 1`` from the
    wrong room, and that must not count as progress.
    """
    walk = list(walk or [])
    if ptr >= len(walk):
        return ptr
    if str(action).strip().lower() != str(walk[ptr]).strip().lower():
        return ptr
    return ptr + 1 if alfworld_executed(observation) else ptr


# --- ALFWorld, by milestones of the task type ---------------------------- #
#
# WHY A SECOND COUNT. The walkthrough pointer compares actions to the walkthrough
# word for word, so it is bound to that walkthrough's instance numbers and route.
# Measured at step 75: a game whose walkthrough reads "go to cabinet 5", "take cup 1
# from cabinet 2", ... is solved in seven actions by going straight to cabinet 2 and
# replaying the rest verbatim -- and the pointer stays at 0, because step 1 never
# matches and nothing after it can count. All eight winners of that group scored 0.
# A different receptacle or a different instance of the same object blocks the
# pointer for the rest of the episode.
#
# WHAT IT COUNTS. The task's own milestones, read off the actions the environment
# carried out, by object TYPE rather than instance, in any order:
#   pick_and_place_simple          took a target object; placed one in a target receptacle   K=2
#   look_at_obj_in_light           took a target object; used a lamp of the target type       K=2
#   pick_{clean,heat,cool}_then_*  took one; cleaned / heated / cooled one; placed a treated one K=3
#   pick_two_obj_and_place         took one; placed one; took a second instance; placed two   K=4
# The task type and targets come from traj_data.json beside the game file
# (pddl_params). A task type outside these six, or a sliced-object task (no
# walkthrough either), has no milestones: K = 0.

_ALF_TAKE = re.compile(r"^take (\S+) (\d+) from (\S+) (\d+)$")
_ALF_PLACE = re.compile(r"^(?:move (\S+) (\d+) to|put (\S+) (\d+) in/on) (\S+) (\d+)$")
_ALF_TREAT = re.compile(r"^(clean|heat|cool) (\S+) (\d+) with (\S+) (\d+)$")
_ALF_USE = re.compile(r"^use (\S+) (\d+)$")
_ALF_TREATMENT = {"pick_clean_then_place_in_recep": "clean",
                  "pick_heat_then_place_in_recep": "heat",
                  "pick_cool_then_place_in_recep": "cool"}
_ALF_TOTAL = {"pick_and_place_simple": 2, "look_at_obj_in_light": 2,
              "pick_clean_then_place_in_recep": 3, "pick_heat_then_place_in_recep": 3,
              "pick_cool_then_place_in_recep": 3, "pick_two_obj_and_place": 4}
_ALF_TASK_CACHE: dict = {}


def alfworld_task(gamefile) -> dict:
    """``{task_type, object, receptacle, lamp, sliced}`` from traj_data.json, cached; {} if unreadable."""
    import json
    import os

    key = str(gamefile or "")
    if not key:
        return {}
    if key in _ALF_TASK_CACHE:
        return _ALF_TASK_CACHE[key]
    out = {}
    d = key if os.path.isdir(key) else os.path.dirname(key)
    path = os.path.join(d, "traj_data.json")
    try:
        with open(path) as f:
            raw = json.load(f)
        pp = raw.get("pddl_params") or {}
        out = {"task_type": str(raw.get("task_type") or ""),
               "object": str(pp.get("object_target") or "").lower(),
               "receptacle": str(pp.get("parent_target") or "").lower(),
               "lamp": str(pp.get("toggle_target") or "").lower(),
               "sliced": bool(pp.get("object_sliced"))}
    except (OSError, ValueError, AttributeError):
        out = {}
    _ALF_TASK_CACHE[key] = out
    return out


class AlfworldMilestones:
    """One episode's milestones for its task type (see the block comment above)."""

    def __init__(self, gamefile):
        t = alfworld_task(gamefile)
        self.task_type = t.get("task_type", "")
        self.object = t.get("object", "")
        self.receptacle = t.get("receptacle", "")
        self.lamp = t.get("lamp", "")
        ok = (self.task_type in _ALF_TOTAL and self.object and not t.get("sliced")
              and (self.lamp if self.task_type == "look_at_obj_in_light" else self.receptacle))
        self.total = _ALF_TOTAL[self.task_type] if ok else 0
        self.treatment = _ALF_TREATMENT.get(self.task_type)
        self.took = set()        # target-object instances ever picked up
        self.treated = set()     # target-object instances cleaned / heated / cooled
        self.placed = set()      # target-object instances placed in a target receptacle
        self.used_lamp = False
        # The environment's own verdict. Some games are won before every milestone
        # above is reached: "heat some mug and put it in coffeemachine" is won the
        # moment a mug is heated when another mug already sits in the coffee
        # machine, because the goal is checked by type. A won episode is at K.
        self.won = False

    def step(self, action, observation, won: bool = False) -> None:
        if won:
            self.won = True
        if not self.total or not alfworld_executed(observation):
            return
        a = " ".join(str(action or "").strip().lower().split())
        m = _ALF_TAKE.match(a)
        if m:
            if m.group(1) == self.object:
                self.took.add(m.group(2))
            return
        m = _ALF_PLACE.match(a)
        if m:
            obj, idx = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            if obj == self.object and m.group(5) == self.receptacle:
                # A treatment task only counts a treated object going in.
                if not self.treatment or idx in self.treated:
                    self.placed.add(idx)
            return
        m = _ALF_TREAT.match(a)
        if m:
            if self.treatment and m.group(1) == self.treatment and m.group(2) == self.object:
                self.treated.add(m.group(3))
            return
        m = _ALF_USE.match(a)
        if m and m.group(1) == self.lamp:
            self.used_lamp = True

    @property
    def k(self) -> int:
        if not self.total:
            return 0
        if self.won:
            return self.total
        took = int(bool(self.took))
        if self.task_type == "pick_and_place_simple":
            return took + int(bool(self.placed))
        if self.task_type == "look_at_obj_in_light":
            return took + int(self.used_lamp)
        if self.treatment:
            return took + int(bool(self.treated)) + int(bool(self.placed))
        # pick_two_obj_and_place
        return took + int(len(self.placed) >= 1) + int(len(self.took) >= 2) + int(len(self.placed) >= 2)


# --- WebShop ------------------------------------------------------------- #

# engine.parse_action's pattern, restated so this module does not import the
# WebShop engine (and its search index) to parse a string.
_WS_ACTION = re.compile(r"(.+)\[(.+)\]")
_WS_BUY = "buy now"
_WS_BACK = "back to search"
_WS_PREV = "< prev"


def _ws_parse(action) -> Tuple[Optional[str], Optional[str]]:
    m = _WS_ACTION.match(str(action or "").strip())
    if m is None:
        return None, None
    name, arg = m.groups()
    return name.strip().lower(), arg.lower()


def _ws_clickables(avail) -> set:
    return set((avail or {}).get("clickables", []) or [])


def webshop_goal_steps(goal) -> Tuple[str, list]:
    """``(goal asin in clickable form, required option values)``; ('', []) if none."""
    if not goal:
        return "", []
    asin = str(goal.get("asin") or "").strip().lower()
    options = goal.get("goal_options") or {}
    values = options.values() if isinstance(options, dict) else options
    values = [str(v).strip().lower() for v in (values or []) if str(v).strip()]
    return asin, values


class WebshopProgress:
    """One episode's walk along the goal record, from actions and page states.

    k = [goal product was on a results page] + [goal product opened]
        + [most required options selected on it in one visit] + [bought it]
    K = 3 + number of required options -- the length of the document's own path
    (oci_layout.webshop_document_lines), so the two agree on what "all of it" is.

    WHAT IS APPROXIMATE. Options are counted as the required VALUES clicked on the
    goal product since it was opened; the session keeps one value per option
    NAME, which this cannot see, so selecting a required value and then another
    value of the same option still counts the first. Rare, and only ever an
    over-count of one within a group whose ranking it shifts by one step.
    """

    def __init__(self, goal, avail=None):
        self.asin, self.options = webshop_goal_steps(goal)
        self.total = (3 + len(self.options)) if self.asin else 0
        self.found = False
        self.opened = False
        self.on_goal = False
        self.selected = set()
        self.best_options = 0
        self.bought = False
        self._avail = avail
        self._see(avail)

    def _see(self, avail) -> None:
        # On a results page the product links ARE clickables, in lowercase.
        if self.asin and self.asin in _ws_clickables(avail):
            self.found = True

    def step(self, action, avail_after) -> None:
        """Fold in one turn: the action taken, and the page it led to."""
        before = _ws_clickables(self._avail)
        name, arg = _ws_parse(action)
        if name == "search" and arg:
            # The environment runs a search whatever page it is on, and a search
            # clears the session's product and options.
            self.on_goal = False
            self.selected = set()
        elif name == "click" and arg is not None and arg in before and arg != "search":
            if arg == self.asin:
                self.on_goal = True
                self.opened = True
                self.selected = set()
            elif arg == _WS_BACK:
                self.on_goal = False
                self.selected = set()
            elif arg == _WS_PREV and _WS_BUY in before:
                # "< Prev" from the ITEM page goes back to the results; from a
                # description / features / reviews sub page (no buy button) it
                # returns to the item page, which keeps the product.
                self.on_goal = False
                self.selected = set()
            elif arg == _WS_BUY:
                if self.on_goal:
                    self.bought = True
            elif self.on_goal and arg in self.options:
                self.selected.add(arg)
                self.best_options = max(self.best_options, len(self.selected))
        self._avail = avail_after
        self._see(avail_after)

    @property
    def k(self) -> int:
        if not self.total:
            return 0
        return int(self.found) + int(self.opened) + self.best_options + int(self.bought)


# --- Search -------------------------------------------------------------- #

def search_progress(evidence_seen: bool, target, *, answer_strings, is_yesno) -> Tuple[int, int]:
    """``(k, K)`` for one Search row: K is 1 when the question can be judged."""
    if not list(answer_strings(target)) or is_yesno(target):
        return 0, 0
    return int(bool(evidence_seen)), 1


def put_progress(infos: Iterable, ks, totals, *, k_key: str = PROGRESS_K_INFO,
                 total_key: str = PROGRESS_TOTAL_INFO) -> None:
    """Write (k, K) into each info dict, in place."""
    for info, k, n in zip(infos, ks, totals):
        if isinstance(info, dict):
            info[k_key] = int(k)
            info[total_key] = int(n)


# --- ProGPO's coverage, as a shadow -------------------------------------- #

def _observation_digest(observation) -> bytes:
    text = "" if observation is None else str(observation)
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()


class ObservationCoverage:
    """ProGPO's first-visit observation coverage of one trajectory, D.

    D = |{o_1, ..., o_{T+1}}|: how many distinct observations the trajectory has
    seen, the initial one included. ProGPO's progress is P = (D - 1) / T with T
    the actions executed (their Proposition 4.1(i)); the trainer takes T as the
    trajectory's turn rows. As their Section 8.2 specifies, observations are the
    exact strings the environment emitted -- no case, whitespace or tokenisation
    rule, the terminal observation treated like any other -- compared through a
    128-bit digest, so a trajectory keeps 16 bytes per distinct observation
    rather than the pages themselves.

    A SHADOW. It is recorded beside k so the two progress signals can be compared
    on the same rollouts (shadow/coverage/* metrics, the group records); no
    advantage is ever computed from it.
    """

    __slots__ = ("_seen",)

    def __init__(self, initial_observation):
        self._seen = {_observation_digest(initial_observation)}

    def step(self, observation) -> None:
        self._seen.add(_observation_digest(observation))

    @property
    def d(self) -> int:
        return len(self._seen)


def put_coverage(infos: Iterable, coverages) -> None:
    """Write each trajectory's D into its info dict, in place."""
    for info, cov in zip(infos, coverages):
        if isinstance(info, dict) and cov is not None:
            info[COVERAGE_D_INFO] = int(cov.d)
