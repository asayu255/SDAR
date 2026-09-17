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

NOTHING HERE TOUCHES A REWARD OR AN OBSERVATION. The counts ride on the info
dict to the rollout loop and from there to the trainer as columns.
"""

import re
from typing import Iterable, Optional, Tuple

PROGRESS_K_INFO = "progress_k"
PROGRESS_TOTAL_INFO = "progress_total"


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


def put_progress(infos: Iterable, ks, totals) -> None:
    """Write (k, K) into each info dict, in place."""
    for info, k, n in zip(infos, ks, totals):
        if isinstance(info, dict):
            info[PROGRESS_K_INFO] = int(k)
            info[PROGRESS_TOTAL_INFO] = int(n)
