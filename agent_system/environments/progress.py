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
# The milestones plus "arrived" (alfworld_k = milestone_arrive), also always recorded:
# a run ranking by one count measures, on its own rollouts, what the other would split.
PROGRESS_K_ARRIVE_INFO = "progress_k_arrive"
PROGRESS_TOTAL_ARRIVE_INFO = "progress_total_arrive"
# The walkthrough as a SET of type-normalised lines (alfworld_k = walkthrough_set),
# also always recorded: order-free and instance-free, executed-only (see
# AlfworldWalkSet). A rollout that solves the game by another route or another
# instance still scores the lines it did carry out.
PROGRESS_K_WALKSET_INFO = "progress_k_walkset"
PROGRESS_TOTAL_WALKSET_INFO = "progress_total_walkset"
ALFWORLD_K_DEFINITIONS = ("walkthrough", "milestone", "milestone_arrive", "walkthrough_set")
# SEARCH, TWO COUNTS, both always recorded (see search_progress_answered):
#   evidence           k = 1 once a returned result carried the answer          K = 1
#   evidence_answered  0 not seen / 1 seen, no answer / 2 seen and answered     K = 2
# plus the per-row flag "search_answered": has the rollout sent an <answer> the
# environment took. The flag is what makes "pushed up yet never answered" countable.
SEARCH_K_DEFINITIONS = ("evidence", "evidence_answered")
PROGRESS_K_SEARCH_EVIDENCE_INFO = "progress_k_search_evidence"
PROGRESS_TOTAL_SEARCH_EVIDENCE_INFO = "progress_total_search_evidence"
PROGRESS_K_SEARCH_ANSWERED_INFO = "progress_k_search_answered"
PROGRESS_TOTAL_SEARCH_ANSWERED_INFO = "progress_total_search_answered"
SEARCH_ANSWERED_INFO = "search_answered"
# SHADOW COLUMNS, every task, never ranked by (see RevisitCounter and the notes below):
#   revisits               actions the rollout had already taken once, taken again
#   committed              the rollout sent its task's TERMINAL action: Search's
#                          <answer>, WebShop's buy (ALFWorld has none: nothing is
#                          written and the row reads NaN)
#   progress_done_walkset  ALFWorld: walkthrough lines actually carried out, WITHOUT
#                          the won => K rule the ranked counts apply
#   searches               Search: queries the environment has sent to the retriever
#                          so far. Counted from the env's own tool call (tool_calling
#                          with a parsed <search> query), not from the turn count: the
#                          turn that ends the episode -- an answer, or the turn cap,
#                          where SearchEnv.step returns done before any tool runs --
#                          sends none, and neither does a turn with no <search> block.
REVISITS_INFO = "revisits"
COMMITTED_INFO = "committed"
PROGRESS_DONE_WALKSET_INFO = "progress_done_walkset"
SEARCHES_INFO = "searches"
# WebShop: the episode's goal cannot pay 1.0 even when bought correctly -- the environment's
# option-matching bug, found by the worker at reset (envs.goal_capped); None when unknown. Carried on
# every row so progress_rank can keep such groups out of the tied-group shares its Beta is fitted to
# (algorithm.progress_rank.beta_exclude_capped). It changes no reward and no ranking.
GOAL_CAPPED_INFO = "goal_capped"
# WebShop: the episode's goal NUMBER, its position in the environment's unshuffled goal list
# (envs.goal_order), the same on every worker; None when unknown. The session index a reset is given
# names a different goal on every worker seed, so only this number finds the same goal in another group.
GOAL_ID_INFO = "goal_id"
# ...and its price bound (the "price lower than" in the instruction; 1000000 when there is none). It is
# drawn per worker seed, so one goal number can come with different bounds: the pair is the instruction.
GOAL_PRICE_INFO = "goal_price_upper"
# ...and the worker seed, which also drew the catalog's prices (engine.generate_product_prices), so one goal
# number and bound can come with another price for its product: (env seed, goal number) is the environment.
# Plus the goal product's price in it.
ENV_SEED_INFO = "env_seed"
GOAL_PRODUCT_PRICE_INFO = "goal_product_price"
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


def search_k_definition(config) -> str:
    """``algorithm.progress_rank.search_k``: which Search count (a) ranks by."""
    try:
        cfg = (config.get("algorithm", {}) or {}).get("progress_rank", None) or {}
        name = str(cfg.get("search_k", "evidence") or "evidence")
    except AttributeError:
        name = "evidence"
    assert name in SEARCH_K_DEFINITIONS, (
        f"algorithm.progress_rank.search_k={name!r}; expected one of {SEARCH_K_DEFINITIONS}")
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


# --- Revisits, every task ---------------------------------------------------- #
#
# WHY. The saturated-group term ranks a game's winners by TURN COUNT. On the 2026-09-22
# validation dumps (3,681 winners of 90 games) a winner's 13.8 turns were 4.4 repeats
# of an action it had already taken, 4.0 first visits off the walkthrough and the
# walkthrough's own lines; failures (all at the 50-turn cap) were 59% repeats. Ranking
# winners by repeats alone orders 18% of same-game winner pairs differently from turns,
# and needs no walkthrough matcher, so it is the candidate to compare against turns.
# Recorded, never ranked by: the next run's records answer "what would change".
#
# WHAT COUNTS. The action string the manager sent to the environment, lower-cased and
# whitespace-collapsed, instance numbers KEPT ("go to cabinet 1" twice is a revisit,
# "go to cabinet 2" is not), executed or not (a refused repeat is still a wasted turn).
# A walkthrough that legitimately repeats a line (pick_two's second "go to") is not
# excused here; the analysis can subtract it, the count cannot know it.

class RevisitCounter:
    """How many of a rollout's actions repeated one it had already taken."""

    __slots__ = ("_seen", "revisits")

    def __init__(self):
        self._seen = set()
        self.revisits = 0

    def step(self, action) -> None:
        a = " ".join(str(action or "").strip().lower().split())
        if not a:
            return
        if a in self._seen:
            self.revisits += 1
        else:
            self._seen.add(a)


# --- ALFWorld, by the walkthrough as a set -------------------------------- #
#
# WHY A FOURTH COUNT. The milestones count RESULTS (took / treated / placed / lamp),
# 2-4 of them, and the runs' stuck groups are tied at k = 0 on 76-81% of them. Yet
# 54% of a failure's actions are `go to`, executed and grammatical -- the one thing
# a failure does right, and the milestones never look at it. The walkthrough names
# those moves. Its pointer form (advance_walkthrough) is bound to the reference
# route and its instance numbers -- all eight winners of one group scored 0 -- so
# this count takes the walkthrough as a SET:
#   order-free       ALFWorld's goals are conjunctions (look_at is isToggled AND
#                    holds); the real dependencies (nothing is placed before it is
#                    taken) are PDDL preconditions the environment enforces with
#                    "Nothing happens.", so demanding the reference ORDER was
#                    stricter than the game itself;
#   type-normalised  `cabinet 5` and `cabinet 2` are one line, the way the
#                    milestones and the goal check compare by type;
#   deduplicated     normalising merges lines; K is the distinct count.
# The one thing kept strict is the executed check: with the order gone, a
# failure's invalid `take` (47% of them) would otherwise advance the count.
#
# WHAT IT CHANGES. K grows from 2-4 to 3-8 (mean 5.2), so one line is worth less
# of the (k - mean)/K score; pick_two grows least (its two takes of one type merge)
# and keeps the most weight per line -- watch that in the records. WHAT IT DOES
# NOT CHANGE: a stuck group whose eight rollouts all reached the destination and
# none the object's receptacle is still tied, at 1 instead of 0 (the 2026-09-21
# continuation probe: 53/56 and 0/56). Its target is the mixed groups' failures
# and the stuck groups the milestones already split -- resolution, not rescue.

_ALF_INSTANCE = re.compile(r"\s+\d+\b")


def normalize_alfworld_action(action) -> str:
    """The action by TYPE: lowercase, single spaces, every instance number gone.

    ``take pan 1 from stoveburner 2`` -> ``take pan from stoveburner``.
    """
    a = " ".join(str(action or "").strip().lower().split())
    return _ALF_INSTANCE.sub("", a)


class AlfworldWalkSet:
    """One episode's coverage of its walkthrough as a set of normalised lines.

    ``total`` is the number of distinct normalised lines (0 without a
    walkthrough, and the row then has no progress at all); ``k`` the number of
    them the rollout has carried out, in any order, counted only when the
    environment executed the action. A won episode is at K, as the other counts
    hold, so a rollout that wins by another route is not scored below one that
    lost along the reference one.
    """

    def __init__(self, walk):
        lines = [normalize_alfworld_action(w) for w in (walk or [])]
        self.lines = tuple(dict.fromkeys(l for l in lines if l))   # distinct, first-seen order
        self._set = frozenset(self.lines)
        self.total = len(self.lines)
        self.done = set()
        self.won = False

    def step(self, action, observation, won: bool = False) -> None:
        if won:
            self.won = True
        if not self.total or not alfworld_executed(observation):
            return
        a = normalize_alfworld_action(action)
        if a in self._set:
            self.done.add(a)

    @property
    def k(self) -> int:
        if not self.total:
            return 0
        if self.won:
            return self.total
        return len(self.done)

    @property
    def raw_done(self) -> int:
        """Lines carried out, with no won => K: what a winner actually did of the
        walkthrough. Recorded as progress_done_walkset; the ranked count is ``k``."""
        return len(self.done)


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
# "PLACED" MEANS IN THE RECEPTACLE AT ONCE (2026-09-28): the milestone counts the most
# target objects the rollout has had inside ONE receptacle instance at the same time,
# not every instance it has ever put down. The goal checks two objects in the same
# receptacle at the same moment, and a replay of 2,813 validation trajectories against
# the game's facts found 13 of 609 pick_two failures that the ever-count put at K (K+1
# with arrived), level with a win: 12 put one object in, took it back out and put the
# other in, and 1 put one each into drawer 1 and drawer 2. A take from a receptacle
# removes the object from it; nothing else can (put/move need the object in hand).
# Every other task type wins, and ends, on its first counted placement, so the two
# counts agree there.
# The task type and targets come from traj_data.json beside the game file
# (pddl_params). A task type outside these six, or a sliced-object task (no
# walkthrough either), has no milestones: K = 0.
#
# ONE MORE, UNDER alfworld_k = milestone_arrive: "arrived" -- the rollout stood at a
# receptacle of the TARGET type (pddl_params parent_target; for look_at, where a lamp
# of the target type is) while holding an object of the target type, TREATED if the
# task has a treatment. K grows by one for every task type.
#   WHY. It is the state the last milestone needs: nothing can be placed (or looked at
#   under the lamp) from anywhere else, so every win passes through it -- except the
#   wins the environment grants early by type (2-4% of wins, all treatment tasks),
#   which the won => K rule below already covers -- and a won episode is still at K
#   (657 of 657). Replayed on 1,372 validation trajectories (2026-09-19),
#   winners reached it before losers in 24 of 27 (alfworld-only) and 21 of 25
#   (multitask) mixed games -- the best of the candidates tried; "stood where the
#   object is" and "saw the object" were not added, because winners take the object
#   one turn after seeing it and the count would only repeat "took". What it
#   separates: "carried it there and failed to place it" from "took it and wandered",
#   which the base count ties at the same k.
#   WHY TREATED. An untreated object at the target is off the path -- placing it does
#   not count either (see `placed`) -- so marking it would pay for skipping the
#   treatment. Order is still free: the flag is independent of the others.
#   HOW IT IS READ, from actions the environment carried out, no game state needed:
#   the receptacles the rollout is known to stand at are the last "go to X" plus any
#   receptacle it has since taken from, placed in, opened, closed or treated with
#   (several receptacles can share a location: "go to coffeemachine 1" then "take
#   tomato 2 from countertop 1" is a legal pair); in hand is the last object taken and
#   not yet put down; a lamp is "here" when the arrival text names one or it was used.
#   A counted placement sets the flag too, so placed => arrived and k never skips it.
#   Checked against the game's own facts (agent location, receptacles there, object in
#   hand, isclean/ishot/iscool) on those replays: set on the same turn in 863 of 864
#   trajectories, one turn late in one, never missed and never set without the facts.
#   WHAT TO EXPECT. Small. Same-game failures tied at k >= 1 mostly share the flag:
#   it split 3% of such pairs (alfworld-only, 35 validations of one checkpoint) and 1 of
#   6 (multitask); it bites in pick_two at k = 1 and in treatment tasks at k = 2, and
#   never in look_at (0 of 57 look_at failures at k >= 1 held the object at a lamp).
#   In pick_two it is often set at the take itself: an object taken from a receptacle of
#   the target type is already held at the target, so tied failures there mostly share it.

_ALF_TAKE = re.compile(r"^take (\S+) (\d+) from (\S+) (\d+)$")
_ALF_PLACE = re.compile(r"^(?:move (\S+) (\d+) to|put (\S+) (\d+) in/on) (\S+) (\d+)$")
_ALF_TREAT = re.compile(r"^(clean|heat|cool) (\S+) (\d+) with (\S+) (\d+)$")
_ALF_USE = re.compile(r"^use (\S+) (\d+)$")
_ALF_GOTO = re.compile(r"^go to (\S+) (\d+)$")
_ALF_OPEN_CLOSE = re.compile(r"^(?:open|close) (\S+) (\d+)$")
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
        self.placed = set()      # target-object instances ever placed in a target receptacle
        # target-object instances inside each receptacle instance of the target type now,
        # and the most there have been in one of them at once (the counted milestone)
        self._inside = {}
        self.placed_at_once = 0
        self.used_lamp = False
        # "arrived" (alfworld_k = milestone_arrive; tracked always, counted only there).
        self.arrived = False
        self._here = set()       # receptacle types the rollout is known to stand at
        self._lamp_here = False  # a lamp of the target type is where it stands
        self._holding = None     # (object type, instance) in hand
        self._lamp_re = re.compile(r"\b" + re.escape(self.lamp) + r" \d+\b") if self.lamp else None
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
        self._milestone(a)
        self._locate(a, observation)

    def _milestone(self, a: str) -> None:
        m = _ALF_TAKE.match(a)
        if m:
            if m.group(1) == self.object:
                self.took.add(m.group(2))
                if m.group(3) == self.receptacle:
                    self._inside.get(m.group(4), set()).discard(m.group(2))
            return
        m = _ALF_PLACE.match(a)
        if m:
            obj, idx = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            if obj == self.object and m.group(5) == self.receptacle:
                # A treatment task only counts a treated object going in.
                if not self.treatment or idx in self.treated:
                    self.placed.add(idx)
                    here = self._inside.setdefault(m.group(6), set())
                    here.add(idx)
                    self.placed_at_once = max(self.placed_at_once, len(here))
                    self.arrived = True      # placed => arrived
            return
        m = _ALF_TREAT.match(a)
        if m:
            if self.treatment and m.group(1) == self.treatment and m.group(2) == self.object:
                self.treated.add(m.group(3))
            return
        m = _ALF_USE.match(a)
        if m and m.group(1) == self.lamp:
            self.used_lamp = True

    def _locate(self, a: str, observation) -> None:
        """Where the rollout stands and what it holds, after one executed action; then "arrived"."""
        m = _ALF_GOTO.match(a)
        if m:
            self._here = {m.group(1)}
            self._lamp_here = bool(m.group(1) == self.lamp or (
                self._lamp_re is not None and self._lamp_re.search(str(observation or "").lower())))
        else:
            m = _ALF_TAKE.match(a)
            if m:
                self._holding = (m.group(1), m.group(2))
                self._here.add(m.group(3))
            else:
                m = _ALF_PLACE.match(a)
                if m:
                    self._holding = None
                    self._here.add(m.group(5))
                else:
                    m = _ALF_TREAT.match(a) or _ALF_OPEN_CLOSE.match(a)
                    if m:
                        self._here.add(m.group(4) if m.re is _ALF_TREAT else m.group(1))
                    else:
                        m = _ALF_USE.match(a)
                        if m and m.group(1) == self.lamp:
                            self._lamp_here = True
        held = self._holding
        if held is None or held[0] != self.object or (self.treatment and held[1] not in self.treated):
            return
        if self._lamp_here if self.task_type == "look_at_obj_in_light" else (self.receptacle in self._here):
            self.arrived = True

    @property
    def k(self) -> int:
        if not self.total:
            return 0
        if self.won:
            return self.total
        took = int(bool(self.took))
        placed1 = int(self.placed_at_once >= 1)
        if self.task_type == "pick_and_place_simple":
            return took + placed1
        if self.task_type == "look_at_obj_in_light":
            return took + int(self.used_lamp)
        if self.treatment:
            return took + int(bool(self.treated)) + placed1
        # pick_two_obj_and_place: both objects in one receptacle at the same time
        return took + placed1 + int(len(self.took) >= 2) + int(self.placed_at_once >= 2)

    @property
    def total_arrive(self) -> int:
        """K under alfworld_k = milestone_arrive: one more than ``total``."""
        return self.total + 1 if self.total else 0

    @property
    def k_arrive(self) -> int:
        if not self.total:
            return 0
        if self.won:
            return self.total_arrive
        return self.k + int(self.arrived)


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


def search_progress_answered(evidence_seen: bool, answered: bool, target, *,
                             answer_strings, is_yesno) -> Tuple[int, int]:
    """``(k, K)`` for one Search row, counting the answer as the second stage.

    The correct sequence is retrieve, then answer: k = 1 once a returned result
    carried the answer, 2 once the rollout has also sent an <answer> after that
    (right or wrong), K = 2. An answer sent WITHOUT the evidence stays at 0, so a
    guess is never paid for; seeing the evidence and never answering stays at 1,
    below every rollout that did both.

    WHY A THIRD LEVEL, NOT A TWO-LEVEL "seen AND answered". The two forms order every
    pair of rollouts the same way but one: "seen, never answered" against "not seen".
    Three levels keep today's order there (the first stage counts, as "took" does in
    ALFWorld and "found" in WebShop) and add "answered > not answered" among the
    rollouts that saw it -- which today's K = 1 ties, e.g. a stuck group whose eight
    rollouts all saw the answer. Whether ranking "seen, never answered" above "not
    seen" feeds a search-to-the-cap habit is not settled (2026-09-22: pushed up 2%
    vs pushed down 9% never answered in stuck groups at step 150, 16% vs 0% among
    mixed-group failures); SEARCH_ANSWERED_INFO is recorded so any run can measure it.

    Unjudgeable questions (no answer strings, yes/no) have K = 0 under both counts.
    """
    if not list(answer_strings(target)) or is_yesno(target):
        return 0, 0
    seen = bool(evidence_seen)
    return int(seen) + int(seen and bool(answered)), 2


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
