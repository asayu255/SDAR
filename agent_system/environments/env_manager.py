# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Sequence, Tuple, Dict, Union, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np
import atexit
from functools import partial
import json
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.oci_layout import (
    OCI_DOC_KEY, OCI_PLAIN_KEY, OCI_ROLE_KEY, PLAN_FOOTER, PLAN_HEADER, PLAN_LEAD,
    rank_on as _oci_rank_on, rank_tasks as _oci_rank_tasks,
    ROLE_DOC, ROLE_FOREIGN, render_document,
    search_document_lines as _search_document_lines,
    search_rescue_document_lines as _search_rescue_document_lines,
    search_progress_line as _search_progress_line,
    search_doc_mode as _slots_search_doc, has_second_doc as _slots_second_doc,
    search_flow_document_lines as _search_flow_document_lines,
    search_flow_path as _slots_search_flow_path,
    advance_route as _advance_route, search_route_line as _search_route_line,
    contains_answer as _contains_answer, evidence_in_text as _evidence_in_text,
    answer_strings as _answer_strings, is_numeric_answer as _is_numeric_answer,
    ROLE_DOC_B, SEARCH_RULE_LEAD, SEARCH_PROGRESS_LEAD, SEARCH_FLOW_LEAD,
    webshop_document_lines as _webshop_document_lines,
    doc_mode as _slots_doc_mode, doc_stepwise as _slots_doc_stepwise,
    foreign_prompt as _slots_foreign_prompt, foreign_task as _slots_foreign_task,
    alfworld_foreign_obs as _slots_foreign_alfworld_obs,
    slot_role as _slots_role, slots_on as _slots_on)
from agent_system.memory import SimpleMemory, SearchMemory


# ---------------------------------------------------------------------------
# PRIVILEGED_WRONG_PLAN: the instance's expert plan with its REQUIREMENT removed.
# ---------------------------------------------------------------------------
#
# WHAT IT IS FOR. A saturated group -- eight of eight rollouts at max return --
# has zero advantage, and that is GRPO behaving correctly: the baseline equals
# the return, so the correct update is zero. Injecting a failure changes the
# baseline and manufactures an advantage of +1/sqrt(7) on each of the seven
# successes and -sqrt(7) on the injected row, independent of the reward scale.
# Whether that helps is the open question; this switch produces the failure to
# inject.
#
# WHY A CORRUPTED PLAN AND NOT THE BASE POLICY. The shaping coefficient on an
# injected row is A*gamma*rho/(rho+gamma)^2 with rho = pi(a|x)/pi(a|x,z), and it
# goes to ZERO as rho does. A failure the unconditioned student would never
# produce carries no gradient. base's failures come from different weights, so
# their rho is small by construction. A failure produced by the SAME weights
# under a different prompt at least has a chance of staying near rho ~ 1.
#
# WHY DROPPING THE REQUIREMENT AND NOT SOMETHING ELSE. Every remaining step must
# stay groundable in THIS game, otherwise the policy ignores the plan, and then
# there is neither a failure nor a measurable rho. Dropping one step leaves every
# other step naming an object and a place the game actually has; the episode runs
# to the end and scores zero because the requirement was never met.
#
#   clean / heat / cool / slice   drop that transformation
#   look_at_obj_in_light          drop ToggleObject
#   pick_and_place*               drop the final PutObject
#
# Measured over 600 sampled games this covers 100% of them, and the block is 80
# tokens at p50 (217 max) against an alfworld p99 turn prompt of 947 and a 4096
# ceiling, so max_model_len does not move.
#
# THE SWITCH DOES NOT VALIDATE ITSELF. That the corrupted plan actually makes the
# student fail, and that rho stays measurable, are both open and must be measured
# before this is used to train anything.
_WRONG_PLAN_CACHE = {}
# Every `go to X` the scene offers, captured the first time a game is seen.
# The detour needs somewhere IRRELEVANT to send the student, and the plan only
# names the places that matter -- where the object is, the tool, the
# destination. Sending it round those three is a guided tour of the solution.
_SCENE_RECEPS = {}


def _scene_receptacles(gamefile, admissible):
    """The scene's `go to` targets, cached per game. Numbered, so no grounding
    step is needed, and admissible by construction."""
    import re

    key = str(gamefile)
    if key not in _SCENE_RECEPS and admissible:
        found = []
        for a in admissible:
            m = re.fullmatch(r"go to ([a-z]+ \d+)", str(a).strip())
            if m and m.group(1) not in found:
                found.append(m.group(1))
        if found:
            _SCENE_RECEPS[key] = found
    return _SCENE_RECEPS.get(key, [])
_REQUIREMENT_ACTIONS = ("CleanObject", "HeatObject", "CoolObject",
                        "SliceObject", "ToggleObject")


# THE PREFIX TRAVELS ON THE OBSERVATION, NOT IN A MODULE-LEVEL DICT.
#
# The row builder needs to know what was prepended, to record where the block
# sits in the tokenised prompt. The first version kept a dict keyed by env slot
# and the loop read it by the row's GLOBAL batch index. Those are different
# numbers. Under TASK_BALANCE_INTERLEAVE (the default) the generation batch is
# laid out alf0, search0, webshop0, alf1, ... at the PROMPT level and each prompt
# is then repeated group_n times contiguously, so alfworld local slot i is global
# row 24*(i//8) + i%8 on a three-task batch of 8. Looking up the global index in
# a locally-keyed dict matched for exactly ONE group of fifteen: four more landed
# on another group's candidate (a different game, so the startswith check
# rejected it) and ten were past the end of the dict. Fourteen candidate
# trajectories went unmarked -- judged as part of their group and trained as
# plain rows with the corrupted plan still in the prompt -- and because one group
# WAS marked, the "at least one candidate" assert passed.
#
# The dict was also shared with the validation manager, which writes '' into it
# for every slot it builds, so a validation step could blank the marks for the
# next batch's turn 0.
#
# A key on the observation dict has neither problem: MultiTaskEnvironmentManager
# ._merge_observations reorders EVERY key from each manager's local order into
# global order using _task_indices, and the validation manager has its own
# observations.
OCI_PREFIX_KEY = "oci_prefix"


def _oci_candidate_row(i: int, envs, config=None) -> bool:
    """Is env slot ``i`` the one that gets the corrupted plan?

    ALFWorld seeds worker i with ``seed + i // group_n``, so the group_n
    consecutive workers of a group all hold the SAME game; taking the last one
    leaves the other group_n-1 as untouched plain rollouts of that game and needs
    no second generation pass.

    THE SWITCH IS A CONFIG KEY, NOT AN ENVIRONMENT VARIABLE. It used to be
    PRIVILEGED_WRONG_PLAN, copying how PRIVILEGED_SKILLS and PRIVILEGED_PLAN are
    done here, and that has a failure mode this project has already paid for: a
    variable exported in the launching shell does not reach a setsid'd process
    over ssh, so an arm ran as a plain student for thirty minutes and reported as
    the arm. It also cannot be pinned by the intent lock, so the arm's identity
    was not fixed anywhere. _copy_config_for_task copies the WHOLE config onto
    each task's manager, so the manager can read algorithm.oci_sat directly and
    the lock pins both the switch and the corruption mode.

    ASKS THE ENVS, NOT THE CONFIG, FOR THE GROUP SIZE. config.env.rollout.n is
    the TRAINING group size on a shared config object, so the validation manager
    -- built with group_n=1, is_train=False -- matched it too and one alfworld
    instance in eight was VALIDATED with a corrupted plan, i.e. the arm's own
    success-rate measurement was contaminated by the arm.
    """
    if not _oci_switch_on(config):
        return False
    if envs is None or not getattr(envs, "is_train", False):
        return False
    try:
        g = int(getattr(envs, "group_n", 0))
    except (TypeError, ValueError):
        return False
    return g >= 2 and (i % g) == (g - 1)


def _oci_cfg(config):
    """``algorithm.oci_sat`` off the manager's own config, or None."""
    if config is None:
        return None
    try:
        return config.algorithm.get("oci_sat", None)
    except Exception:
        return None


def _oci_switch_on(config) -> bool:
    cfg = _oci_cfg(config)
    return bool(cfg is not None and cfg.get("enable", False))


def _assert_one_privileged_arm(config) -> None:
    """Two privileged inputs in one prompt is two interventions, and rho removes one.

    ``oci_sat`` writes a plan into one slot per group; ``oci_slots`` writes a
    walkthrough into the document slot and another task's prompt into the foreign
    one. Both are stripped for the shaped term by a SINGLE verified replacement
    per row, so a prompt carrying both cannot be re-scored on the prompt that
    exists at test time -- and the two arms also disagree about which slot is
    which.
    """
    if _oci_switch_on(config) and _slots_on(config):
        raise ValueError(
            "algorithm.oci_sat.enable and algorithm.oci_slots.enable are both on. "
            "They are two different privileged inputs into the same alfworld "
            "prompt and two different slot layouts; pick one arm.")


def _oci_detour(config) -> int:
    """Extra ``go to`` steps inserted into a corrupted path. 0 = off."""
    cfg = _oci_cfg(config)
    try:
        return max(0, int((cfg or {}).get("detour_steps", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _oci_delay_turns(config) -> int:
    """Turns the `delay` tour must consume before the true path begins.

    Counted in TURNS, not lines: the episode cap is 50 turns, so a tour of 50
    executable lines means perfect compliance never reaches the tail.
    """
    cfg = _oci_cfg(config)
    try:
        return max(1, int((cfg or {}).get("delay_turns", 50) or 50))
    except (TypeError, ValueError):
        return 50


def _oci_plan_mode(config) -> str:
    cfg = _oci_cfg(config)
    mode = str((cfg or {}).get("plan_corruption", "misdirect") or "misdirect")
    if mode not in PLAN_CORRUPTIONS:
        raise ValueError(
            f"algorithm.oci_sat.plan_corruption={mode!r}; expected one of "
            f"{PLAN_CORRUPTIONS}. 'misdirect' permutes the navigation targets, "
            "'delay' prefixes the true path with a tour that spends the turn "
            "budget, 'intact' prints the true path from high_pddl (unnumbered), "
            "'walkthrough' prints TextWorld's own numbered solution, "
            "'walkthrough_stepwise' / 'delay_stepwise' add a per-turn progress "
            "line to those blocks. 'drop' "
            "removed the requirement step and is gone: it was refuted "
            "(cand_fail_rate 1/15)."
        )
    return mode


def _wrong_plan_prefix(task: str, gamefile, config=None, admissible=None) -> str:
    """The instance's plan, corrupted per ``plan_corruption``, or '' when off.

    ``admissible`` is this slot's admissible-action list. It is read once per
    game to learn which receptacles the SCENE offers, which is where the detour
    sends the student -- the plan only names the places that matter. Cached, so
    the block is the same text on every turn of an episode rather than drifting
    with whatever happens to be reachable now.
    """
    if not _oci_switch_on(config):
        return ""
    return _plan_block(task, gamefile, _oci_plan_mode(config), admissible,
                       _oci_detour(config), _oci_delay_turns(config))


def _plan_block(task: str, gamefile, mode: str, admissible=None,
                n_detour: int = 0, turns: int = 50) -> str:
    """The block itself, for a caller that already knows which mode it wants.

    Split out of ``_wrong_plan_prefix`` for the ten-slot layout, whose document
    slot asks for one specific mode (the game's own walkthrough) while
    ``algorithm.oci_sat`` is off -- so the switch that gates the other arm cannot
    be the gate here.
    """
    if task != "alfworld" or not gamefile:
        return ""
    needs_scene = n_detour or mode in ("delay", "delay_stepwise")
    scene = _scene_receptacles(gamefile, admissible) if needs_scene else []
    key = (str(gamefile), mode, n_detour, turns)
    if key not in _WRONG_PLAN_CACHE:
        block = _build_wrong_plan(str(gamefile), mode, n_detour, scene, turns)
        # DO NOT CACHE A BLOCK BUILT WITHOUT THE SCENE. The pool comes from the
        # admissible actions, so an early call that has none would otherwise
        # pin an empty tour for the rest of training.
        if block or not needs_scene or scene:
            _WRONG_PLAN_CACHE[key] = block
        return block
    return _WRONG_PLAN_CACHE[key]


PLAN_CORRUPTIONS = ("misdirect", "intact", "delay", "walkthrough", "walkthrough_stepwise", "delay_stepwise")

# The block, in the environment's own words. Both modes emit the SAME text apart
# from the numbered lines -- no word anywhere says whether the path is right.
# Defined in oci_layout beside the WebShop and Search document builders, so the
# three tasks cannot drift into three different wrappers: the strip, the
# numbered-line reader and the progress line all assume one shape.
_PLAN_HEADER = PLAN_HEADER
_PLAN_FOOTER = PLAN_FOOTER
_PLAN_LEAD = PLAN_LEAD

# ALFWorld's own command grammar, from the installed package:
#   go to {recep} | take {obj} from {recep} | put {obj} in/on {recep}
#   open {recep}  | close {recep}           | use {obj}
#   heat {obj} with {microwave} | cool {obj} with {fridge}
#   clean {obj} with {cleaner}  | slice {obj} with {knife}
_PLAN_TOOL = {"HeatObject": "microwave", "CoolObject": "fridge",
              "CleanObject": "sinkbasin", "SliceObject": "knife"}


def _plan_lines(steps) -> List[str]:
    """PDDL high-level steps -> lines in the environment's action vocabulary.

    WHY NOT THE PDDL SYMBOLS. The first design printed ``GotoLocation(dresser)``
    / ``PickupObject(alarmclock)``. Those symbols appear nowhere in the
    environment, whose admissible actions are strings like ``go to dresser 1``.
    The block was asking a policy never trained on the notation to ground it.

    EVERY LOCATION REFERENCE FOLLOWS THE WALK. An action's receptacle is taken
    from the last ``go to``, not from the PDDL argument. Without this the
    permutation broke the path's own preconditions: ``go to diningtable`` was
    followed by ``clean apple with sinkbasin``, which cannot be taken from the
    diningtable. 57% of corrupted paths carried such a step, and 100% once the
    detour was inserted between a ``go to`` and the action that needed it. A
    path that contradicts itself is not a misdirection, it is noise, and the
    generations show the student ignoring it outright.

    The consequence for the tool steps is deliberate and worth stating: the
    environment's grammar fixes each one (``clean {obj} with {cleaner}`` takes a
    sinkbasin), so a coherent walk that ends somewhere else yields ``clean apple
    with diningtable`` -- self-consistent as a plan, refused by the parser. The
    alternative is a line that is grammatical and contradicts the walk. This
    picks coherence, because the walk is what the student is being asked to
    follow.

    ``NoOp`` and anything unmapped is dropped: the environment has no command
    for it.
    """
    out, here = [], None
    for act, args in steps:
        a0 = args[0] if args else None
        if act == "GotoLocation" and a0:
            here = a0
            out.append(f"go to {a0}")
        elif act == "PickupObject" and a0:
            out.append(f"take {a0} from {here}" if here else f"take {a0}")
        elif act == "PutObject" and a0:
            # The walk, not the PDDL destination.
            out.append(f"put {a0} in/on {here}" if here else f"put {a0}")
        elif act == "ToggleObject" and a0:
            out.append(f"use {here or a0}")
        elif act in _PLAN_TOOL and a0:
            out.append(f"{act[:-6].lower()} {a0} with {here or _PLAN_TOOL[act]}")
        elif act == "OpenObject" and a0:
            out.append(f"open {here or a0}")
        elif act == "CloseObject" and a0:
            out.append(f"close {here or a0}")
    return out


def _misdirect(steps):
    """Send every receptacle-valued slot to a different one from the same plan.

    THE NAVIGATION IS THE INFORMATIVE HALF. An expert plan's value here is
    knowing which receptacle holds the object and in what order to visit things;
    that is exactly what the task description cannot tell the student. The
    design this replaces removed the plan's REQUIREMENT step instead, and the
    requirement is the one step the task text already states -- 92% of 853
    sampled games name it ("put a CLEAN mug", "HEAT an egg", "TURNING ON a
    light"). Measured: the candidate solved the task 14 times in 15.

    EVERY RECEPTACLE SLOT, not just GotoLocation. Leaving `put X in/on R`
    correct keeps the one line that states the goal, and the task text names
    that destination too.

    THE TOOLS ARE LEFT ALONE. The grammar fixes each one -- `clean {obj} with
    {cleaner}` takes a sinkbasin and nothing else -- so every possible swap
    yields a line the environment cannot execute, and an impossible line does
    not mislead, it announces that the block is unreliable. The tool also leaks
    nothing the task text does not ("the WASHED apple" -> sinkbasin).

    Returns ``None`` when the plan names fewer than two receptacles, which is
    0.3% of games: there is nowhere to send anything.
    """
    recs = []
    for act, args in steps:
        if act == "GotoLocation" and args and args[0] not in recs:
            recs.append(args[0])
        if act == "PutObject" and len(args) > 1 and args[1] not in recs:
            recs.append(args[1])
    if len(recs) < 2:
        return None
    swap = {r: recs[(i + 1) % len(recs)] for i, r in enumerate(recs)}
    out = []
    for act, args in steps:
        if act == "GotoLocation" and args:
            out.append((act, [swap[args[0]]] + list(args[1:])))
        elif act == "PutObject" and len(args) > 1:
            out.append((act, [args[0], swap[args[1]]] + list(args[2:])))
        else:
            out.append((act, list(args)))
    return out


# ALFWorld's openable receptacles. Taken from the walkthroughs themselves: over
# 1200 sampled games the only nouns that ever follow `open` are these five, and
# the grammar gates examineReceptacle on an `openable(r:receptacle)` predicate,
# so openability is a property of the TYPE, not of the instance.
_OPENABLE_RECEPS = ("cabinet", "drawer", "fridge", "microwave", "safe")

_TW_PDDL_CACHE = {}


def _tw_pddl(gamefile: str):
    """``(walkthrough, banned_nouns)`` for this instance, cached per game.

    WHY THE WALKTHROUGH. ``game.tw-pddl`` carries the ground-truth action
    sequence IN THE ENVIRONMENT'S OWN NUMBERED VOCABULARY -- ``['go to desk 1',
    'take pencil 2 from desk 1', 'go to shelf 2', 'use desklamp 1']`` -- present
    in 300 of 300 sampled games, median length 6. traj_data.json's high_pddl,
    which the other two modes read, is the same plan with the instance numbers
    stripped, so a path built from it has to be grounded by the student. Printed
    from the walkthrough the path is executable as written, which is what the
    delay tail needs: the tour in front of it must be the only reason the task
    is not solved.

    WHY BANNED NOUNS. The tour must not show the student the object it needs.
    ``go to R`` already reveals R's contents -- GotoLocation.feedback is "You
    arrive at {r.name}. #examineReceptacle.feedback#" -- and 73% of games hold
    more than one instance of the target type, so keeping the tour off the
    walkthrough's own receptacles is not enough. The ``:init`` facts give
    ``(inReceptacle <ObjId> <RecepId>)`` and the text name of anything in
    ALFWorld is the lowercased type prefix of its id (``Pencil_bar__plus_01...``
    -> ``pencil``, ``GarbageCan_bar_...`` -> ``garbagecan``), so the receptacles
    holding a needed object type can be read off without mapping PDDL ids to
    instance numbers: ban the NOUN and every instance of it goes.
    """
    import json as _json
    import os
    import re

    key = str(gamefile)
    if key in _TW_PDDL_CACHE:
        return _TW_PDDL_CACHE[key]

    d, cand = key, None
    for _ in range(4):
        probe = os.path.join(d, "game.tw-pddl") if os.path.isdir(d) else None
        if probe and os.path.exists(probe):
            cand = probe
            break
        d = os.path.dirname(d)
    if not cand:
        _TW_PDDL_CACHE[key] = ([], set())
        return _TW_PDDL_CACHE[key]
    try:
        with open(cand) as fh:
            raw = _json.load(fh)
        walk = [str(a) for a in (raw.get("walkthrough") or [])]
        problem = str(raw.get("pddl_problem") or "")
    except Exception:
        _TW_PDDL_CACHE[key] = ([], set())
        return _TW_PDDL_CACHE[key]

    def noun(ident):
        return ident.split("_bar_")[0].lower()

    init = problem[problem.index("(:init"):] if "(:init" in problem else ""
    # The object nouns the path touches, and so must stay out of sight.
    needed = set()
    for line in walk:
        tok = line.split()
        if len(tok) > 1 and tok[0] in ("take", "move", "put", "clean", "heat",
                                       "cool", "slice", "use", "open", "close"):
            needed.add(tok[1])
    banned = {noun(r) for o, r in
              re.findall(r"\(inReceptacle\s+(\S+)\s+(\S+?)\)", init)
              if noun(o) in needed}
    # Plus every receptacle the path itself names: the object's location, the
    # tool, the destination. A tour of those three is the search the task needs.
    words = {w for line in walk for w in line.split()}
    banned |= {n for n in (noun(r) for r in
                           re.findall(r"\(receptacleAtLocation\s+(\S+)\s+", init))
               if n in words}
    _TW_PDDL_CACHE[key] = (walk, banned)
    return _TW_PDDL_CACHE[key]


# --- walkthrough_stepwise: where the student is in the path, every turn ------
#
# WHY. Shown the whole walkthrough, the eighth rollout rescued 12 of 34 stuck groups
# -- real (chance 2.4%) but far below the 100% a compliant student gets: replayed in
# the live env the walkthrough wins 30 of 30 games. How far it followed says where it
# breaks: of 60 candidate trajectories 17 followed exactly 4 lines and only 3 went
# past 4, and 4 lines is the whole solution for pick_and_place and look_at but the
# midpoint of heat/cool/clean/pick_two -- the share of 4-step tasks (33%) is about
# the rescue rate (35%). The prompt carries two turns of history, and the block is
# numbered but says nothing about which line is next, so by step four the student
# cannot tell from its own prompt where it is.
#
# WHAT. (A) A pointer per slot that advances only when the action actually taken is
# the next line, so a deviation leaves it on the step still owed; (B) the line
# rendered right before "Now it's your turn to take an action.", where the action
# is chosen, instead of only in the block at the top of a ~600-token prompt. The
# block itself is byte-identical to `walkthrough`, so the two probes differ in this
# line and nothing else.
_GUIDE_ANCHOR = "Now it's your turn to take an action."


_STEPWISE_MODES = ("walkthrough_stepwise", "delay_stepwise")


def _oci_stepwise(config) -> bool:
    return _oci_switch_on(config) and _oci_plan_mode(config) in _STEPWISE_MODES


def _block_lines(block: str):
    """The numbered lines of the block actually shown -- what the pointer walks.

    Read off the rendered block rather than rebuilt from the walkthrough, so the
    progress line can never disagree with the path on screen: for
    walkthrough_stepwise these ARE the walkthrough, for delay_stepwise they are the
    tour followed by the walkthrough.
    """
    import re
    return re.findall(r"^\d+\. (.+)$", block or "", flags=re.M)


def _guide_line(walk, ptr: int) -> str:
    """The progress line for a slot whose pointer is at ``ptr``."""
    n = len(walk)
    if n == 0:
        return ""
    if ptr >= n:
        return f"[Privileged Solution Path progress] All {n} steps of the path are done.\n\n"
    done = "" if ptr == 0 else (f"Step 1 of {n} is done. " if ptr == 1 else f"Steps 1-{ptr} of {n} are done. ")
    return (f"[Privileged Solution Path progress] {done}"
            f"Your next action is step {ptr + 1} of {n}: {walk[ptr]}\n\n")


def _advance_guide(walk, ptr: int, action) -> int:
    """Advance past a step only when the action taken IS that step."""
    if ptr < len(walk) and str(action).strip().lower() == str(walk[ptr]).strip().lower():
        return ptr + 1
    return ptr


def _insert_guide(obs: str, line: str) -> str:
    """Put ``line`` immediately before the turn prompt; untouched if that is ambiguous."""
    if not line or obs.count(_GUIDE_ANCHOR) != 1:
        return obs
    return obs.replace(_GUIDE_ANCHOR, line + _GUIDE_ANCHOR, 1)


# Search's turn prompt opens with its own sentence; the progress line goes in the
# same place for the same reason (the line is read where the action is chosen).
_SEARCH_GUIDE_ANCHOR = "Now it's your turn to respond for the current step."


def _search_query(text: str) -> str:
    """The query a response asked for, or '' -- for comparing the rescue row's
    queries with its eight plain siblings' (a query only a row that has READ the
    answer could write is what no string rule can catch)."""
    import re as _re
    m = _re.search(r"<search>(.*?)</search>", str(text or ""), flags=_re.S)
    return m.group(1).strip() if m else ""


def _insert_search_guide(obs: str, line: str) -> str:
    if not line or obs.count(_SEARCH_GUIDE_ANCHOR) != 1:
        return obs
    return obs.replace(_SEARCH_GUIDE_ANCHOR, line + _SEARCH_GUIDE_ANCHOR, 1)


def _delay_lines(walk, banned, scene_recs, min_turns: int):
    """A tour that cannot be refuted, then the true path behind it.

    WHAT THE MISDIRECT MEASUREMENTS SAID. The student does read the block --
    shown the true path it solves 21.1 points more often than its plain
    siblings, and at turn 0 it restates the corrupted path in its reasoning and
    executes line 1 verbatim. What it will not do is keep following a path the
    environment has contradicted: one ``Nothing happens.``, or one sight of the
    object somewhere the path did not mention, and it reverts to searching and
    solves the task. Every misdirect line is a refutable claim about where
    something is, so the corruption buys a few turns and then evaporates:
    cand_fail_rate 22.2% against a 50% bar, and the candidate still beat its
    plain siblings by 5.3 points.

    WHAT THIS DOES INSTEAD. Failure in ALFWorld has exactly one currency. The
    return is binary, an inadmissible action costs a turn and no penalty, and
    nothing in the action set is irreversible -- so the only way a block can
    turn a success into a failure is to spend the 50-turn budget. The tour
    spends it with lines that assert nothing: every line is executable, and
    every observation it draws ("You arrive at cabinet 3. The cabinet 3 is
    closed.") is consistent with a path that never claimed anything was there.
    There is nothing for the student to catch.

    The tail is the true path, unaltered, so this mode is exactly the already
    measured ``intact`` arm plus a prefix -- and the prefix is sized in TURNS,
    not lines, so perfect compliance exhausts the budget before the tail is
    reached and failure is certain by construction. Partial compliance to step k
    leaves 50-k turns, against the ~20 an unaided episode takes.

    Motif per receptacle: ``go to`` then ``open``/``close`` where the type is
    openable, else ``examine`` -- both supported by the grammar
    (CloseObject.feedback, examineReceptacle.feedback) and both leaving the
    scene as they found it, so a second pass draws the same observation as the
    first. One pass over the un-banned receptacles covers the budget in 41% of
    games (median 15 instances, 2 passes); the rest sweep back and forth.
    """
    pool = [r for r in scene_recs if r.rsplit(" ", 1)[0] not in banned]
    if len(pool) < 3:
        # 3% of games: a scene too small to avoid everything the object type
        # touches. Fall back to keeping the tour off the path's own receptacles.
        named = {w for line in walk for w in line.split()}
        pool = [r for r in scene_recs if r.rsplit(" ", 1)[0] not in named]
    if not pool:
        return None
    tour, rounds = [], 0
    while len(tour) < min_turns and rounds <= 64:
        order = pool if rounds % 2 == 0 else list(reversed(pool))
        for r in order:
            if len(tour) >= min_turns:
                break
            tour.append(f"go to {r}")
            if r.rsplit(" ", 1)[0] in _OPENABLE_RECEPS:
                tour.extend([f"open {r}", f"close {r}"])
            else:
                tour.append(f"examine {r}")
        rounds += 1
    return tour[:min_turns] + list(walk)


def _add_detour(steps, n_detour: int, scene_recs=None):
    """Insert ``n_detour`` extra ``go to`` steps that go nowhere useful.

    WHY LENGTH. The generations say what a wrong path actually costs. The
    student FOLLOWS it -- shown a path that puts the mug in the microwave it
    emits ``take mug from microwave 1``, and after ``Nothing happens.`` it emits
    the same action again -- but an inadmissible action is not punished: the
    projection's validity check looks for ``<action>`` and ``<think>`` tags and
    nothing else. A wrong line costs one turn and no penalty, and the return is
    BINARY (0 or 10, no partial credit), so the only way a wrong path can turn a
    success into a failure is to spend the 50-turn budget. Measured: 19.9 turns
    per candidate on the true path, 22.9 on the permuted one, 82% still solving.

    WHY IRRELEVANT RECEPTACLES. A first version drew the detour from the
    receptacles THE PLAN NAMES -- which are exactly the places that matter: the
    object's location, the tool, the destination. Walking round those three is
    not waste, it is the search the task requires, so the padding would have
    helped. The scene offers about 23 `go to` targets and the plan names three;
    the other twenty hold nothing relevant, so a step there is spent and returns
    nothing. They come from the admissible actions, so they are real, numbered
    and executable, and a tour of twenty distinct places reads as a systematic
    search rather than a three-way loop.

    Sized to exhaust the budget rather than to approach it: with a follow rate f
    the waste is f*n_detour, and 22.9 + f*n_detour > 50 needs n_detour > 27/f.
    f is not known, so under-shooting buys nothing -- the cost of a long list is
    tokens, and the budget has thousands spare.
    """
    if n_detour <= 0:
        return steps
    plan_recs = set()
    for act, args in steps:
        if act == "GotoLocation" and args:
            plan_recs.add(args[0])
        if act == "PutObject" and len(args) > 1:
            plan_recs.add(args[1])
    # A scene name is "cabinet 3"; a plan name is "cabinet". Match on the noun.
    pool = [r for r in (scene_recs or []) if r.rsplit(" ", 1)[0] not in plan_recs]
    if not pool:
        return steps

    last_real = max((i for i, (a, _) in enumerate(steps)
                     if a in ("PutObject", "ToggleObject") or a in _PLAN_TOOL),
                    default=len(steps) - 1)
    out, k, placed = [], 0, 0
    per_pass = max(1, n_detour // max(last_real + 1, 1))
    for idx, (act, args) in enumerate(steps):
        if idx <= last_real:
            for _ in range(per_pass):
                if placed >= n_detour:
                    break
                prev = out[-1][1][0] if (out and out[-1][0] == "GotoLocation" and out[-1][1]) else None
                cand = None
                for _try in range(len(pool)):
                    c = pool[k % len(pool)]
                    k += 1
                    if c != prev:
                        cand = c
                        break
                if cand is None:
                    break
                out.append(("GotoLocation", [cand]))
                placed += 1
        out.append((act, list(args)))
    return out


def _build_wrong_plan(gamefile: str, mode: str = "misdirect", n_detour: int = 0,
                      scene_recs=None, delay_turns: int = 50) -> str:
    """The instance's expert path, as a block shown to the student.

    ``intact`` prints the true path; ``misdirect`` permutes its receptacles.
    The two produce byte-identical text apart from the numbered lines, which is
    the point: no word labels the path as right or wrong, so the two arms differ
    in the path and in nothing else.
    """
    import json as _json
    import os

    if mode not in PLAN_CORRUPTIONS:
        raise ValueError(f"plan_corruption={mode!r}; expected one of {PLAN_CORRUPTIONS}")

    if mode in ("walkthrough", "walkthrough_stepwise"):
        # THE CORRECT DOCUMENT, AS STRONG AS IT GETS. game.tw-pddl's walkthrough is
        # TextWorld's own solution in the environment's numbered vocabulary
        # (`take mug 1 from countertop 2`), executable as printed -- unlike
        # `intact`, which prints high_pddl's unnumbered plan and leaves the student
        # to ground every noun. This is the mode for measuring whether showing the
        # solution RESCUES a group whose seven plain rollouts all failed; the
        # measurement is only worth anything if the document is the best one we
        # have. Known gap: 3.5% of walkthroughs (clean/cool/heat types) stop
        # before the final placement.
        walk, _ = _tw_pddl(gamefile)
        if not walk:
            return ""
        body = "\n".join(f"{i + 1}. {l}" for i, l in enumerate(walk))
        return f"{_PLAN_HEADER}\n{_PLAN_LEAD}\n{body}\n{_PLAN_FOOTER}\n\n"

    if mode in ("delay", "delay_stepwise"):
        walk, banned = _tw_pddl(gamefile)
        if not walk or not scene_recs:
            return ""
        lines = _delay_lines(walk, banned, list(scene_recs), delay_turns)
        if not lines:
            return ""
        body = "\n".join(f"{i + 1}. {l}" for i, l in enumerate(lines))
        return f"{_PLAN_HEADER}\n{_PLAN_LEAD}\n{body}\n{_PLAN_FOOTER}\n\n"

    d, cand = gamefile, None
    for _ in range(4):
        probe = os.path.join(d, "traj_data.json") if os.path.isdir(d) else None
        if probe and os.path.exists(probe):
            cand = probe
            break
        d = os.path.dirname(d)
    if not cand:
        return ""
    try:
        with open(cand) as fh:
            raw = _json.load(fh).get("plan", {}).get("high_pddl", [])
    except Exception:
        return ""

    steps = []
    for st in raw:
        da = st.get("discrete_action", {}) or {}
        act = da.get("action")
        if act:
            steps.append((act, [a for a in (da.get("args") or []) if a]))
    if len(steps) < 3:
        return ""

    if mode == "misdirect":
        steps = _misdirect(steps)
        if steps is None:
            return ""
        steps = _add_detour(steps, n_detour, scene_recs)
    lines = _plan_lines(steps)
    if not lines:
        return ""

    body = "\n".join(f"{i + 1}. {l}" for i, l in enumerate(lines))
    return f"{_PLAN_HEADER}\n{_PLAN_LEAD}\n{body}\n{_PLAN_FOOTER}\n\n"


from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        self._oci_roles = []
        self._oci_plains = []
        self._oci_docs = []
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self._probe_flush()
        self.tasks = obs
        # The question and its accepted answers, for the document slot. Unlike
        # the other two tasks nothing has to be dug out of the environment: the
        # dataset row IS the problem, and it arrives here as the reset kwargs.
        _kw = list(kwargs) if kwargs is not None else []
        self.problems = [dict(k) if isinstance(k, dict) else {} for k in _kw]
        self.problems += [{}] * max(0, len(obs) - len(self.problems))

        # THE RULE, PER ROW (search_doc=answer_rule). `_evidence_seen` flips the
        # turn a returned result carries the answer; `_answer_early` records a row
        # that wrote the answer -- in its thinking or in a query -- before that.
        # Both are read by the progress line and by the probe dump; neither
        # touches the reward, which stays exactly what control trains on.
        n = len(obs)
        self._evidence_seen = [False] * n
        self._answer_early = [False] * n
        # Where each route-document row stands in its route (expert_flow only).
        self._route_ptr = [0] * n
        self._probe_reset = int(getattr(self, "_probe_reset", -1)) + 1
        self._probe_rows = [self._probe_new_row(i) for i in range(n)] if self._probe_dir else []
        if self._probe_rows and not getattr(self, "_probe_atexit", False):
            # Rows the environment never marked done are written at the NEXT reset,
            # and the last batch of a probe has none -- without this its records
            # (the rows that spent the turn budget, i.e. the failures) are lost.
            atexit.register(self._probe_flush)
            self._probe_atexit = True

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy(),
            OCI_ROLE_KEY: list(self._oci_roles),
            OCI_PLAIN_KEY: list(self._oci_plains),
            OCI_DOC_KEY: list(self._oci_docs),
        }

        return observations, infos

    def document_block(self, i: int, slot: str = "a") -> str:
        """The block that answers question ``i``, or '' when it cannot be built.

        Same wrapper and numbering as the other two tasks.
        ``answer_only``    the query and the answer; nothing stops the slot from
                           writing the answer at once, which is the hole this task
                           has and the other two do not.
        ``answer_rule``    the same two lines under the rule that a returned result
                           must carry the answer first (SEARCH_RULE_LEAD).
        ``progress_only``  no answer at all: only the per-turn verdict on whether a
                           result has carried it (SEARCH_PROGRESS_LEAD).
        ``slot`` picks the variant: "a" is the row the arm would ship, "b" the
        measurement-only row generated beside it.
        """
        problems = getattr(self, "problems", None) or []
        p = problems[i] if i < len(problems) else {}
        # getattr: the accessor is also called on a bare manager (tests/oci/test_documents.py),
        # where no config has been attached and the historical block is what is asked for.
        mode = _slots_search_doc(getattr(self, "config", None), slot=slot)
        if mode == "expert_flow":
            # Someone else's verified route, then the answer it ends at, under the
            # same rule. '' for a question with no verified route.
            return render_document(
                _search_flow_document_lines(p.get("question"), p.get("ground_truth"),
                                            _slots_search_flow_path(getattr(self, "config", None))),
                lead=SEARCH_FLOW_LEAD)
        if mode in ("answer_rule", "progress_only"):
            show = mode == "answer_rule"
            return render_document(
                _search_rescue_document_lines(p.get("question"), p.get("ground_truth"),
                                              show_answer=show),
                lead=SEARCH_RULE_LEAD if show else SEARCH_PROGRESS_LEAD)
        if mode == "none":
            return ""
        return render_document(_search_document_lines(p.get("question"), p.get("ground_truth")))

    def step(self, text_actions: List[str]):
        # ORDER MATTERS. What the row wrote is judged against the state it was in
        # when it wrote it, so the "wrote it early" test runs before this turn's
        # results are folded in.
        self._note_written(text_actions)
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })
        self._note_returned(next_obs)
        self._note_route(text_actions)
        self._probe_note_turn(text_actions, next_obs, rewards, dones, infos)

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy(),
            OCI_ROLE_KEY: list(self._oci_roles),
            OCI_PLAIN_KEY: list(self._oci_plains),
            OCI_DOC_KEY: list(self._oci_docs),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    # --- the rule, and the probe's record of it ---------------------------- #

    def _answers(self, i):
        problems = getattr(self, "problems", None) or []
        p = problems[i] if i < len(problems) else {}
        return p.get("ground_truth")

    def _question(self, i):
        problems = getattr(self, "problems", None) or []
        p = problems[i] if i < len(problems) else {}
        return p.get("question")

    def _note_written(self, text_actions) -> None:
        """A row that writes the answer before a result carried it breaks the rule."""
        seen = getattr(self, "_evidence_seen", None)
        if seen is None:
            return
        for i, text in enumerate(text_actions[:len(seen)]):
            if not seen[i] and _contains_answer(text, self._answers(i)):
                self._answer_early[i] = True

    def _note_returned(self, next_obs) -> None:
        """The progress line's only state: has a returned result carried the answer?

        Guarded for bare-number answers, which a passage can hold by accident --
        see oci_layout.evidence_in_text.
        """
        seen = getattr(self, "_evidence_seen", None)
        if seen is None:
            return
        for i, obs in enumerate(list(next_obs)[:len(seen)]):
            if not seen[i] and _evidence_in_text(obs, self._answers(i), self._question(i)):
                seen[i] = True

    def _route_slot(self, i):
        """'a' / 'b' when row i wears a route document, else None."""
        envs = getattr(self, "envs", None)
        role = _slots_role(i, envs, self.config, foreign=False,
                           second_doc=_slots_second_doc(self.config))
        slot = "a" if role == ROLE_DOC else ("b" if role == ROLE_DOC_B else None)
        if slot and _slots_search_doc(self.config, slot=slot) == "expert_flow":
            return slot
        return None

    def _note_route(self, text_actions) -> None:
        """Move each route row's pointer past the step it just ran."""
        ptrs = getattr(self, "_route_ptr", None)
        if ptrs is None:
            return
        for i in range(min(len(ptrs), len(text_actions))):
            slot = self._route_slot(i)
            if not slot:
                continue
            lines = _block_lines(self.document_block(i, slot=slot))
            ptrs[i] = _advance_route(lines, ptrs[i], _search_query(text_actions[i]),
                                     bool(self._evidence_seen[i]))

    @property
    def _probe_dir(self) -> str:
        """Where to write per-episode records, or '' (the default) for nowhere.

        Off unless SEARCH_PROBE_DUMP names a directory: this is a measurement
        hook for run_search_rescue_probe_qwen3.sh, not part of training.
        """
        return os.environ.get("SEARCH_PROBE_DUMP", "").strip()

    def _probe_new_row(self, i: int) -> dict:
        problems = getattr(self, "problems", None) or []
        p = problems[i] if i < len(problems) else {}
        envs = getattr(self, "envs", None)
        try:
            group_n = int(getattr(envs, "group_n", 0)) or 1
        except (TypeError, ValueError):
            group_n = 1
        gt = p.get("ground_truth")
        # pid and reset identify the batch: `group` is a position inside one
        # reset of one worker, so it repeats across batches and across workers.
        role = int(_slots_role(i, envs, self.config, foreign=False,
                               second_doc=_slots_second_doc(self.config)))
        slot = "a" if role == ROLE_DOC else ("b" if role == ROLE_DOC_B else None)
        answers = _answer_strings(gt)
        return {"pid": os.getpid(), "reset": int(getattr(self, "_probe_reset", 0)),
                "env": i, "group": i // group_n, "role": role,
                "question": p.get("question"), "answers": answers,
                "data_source": p.get("data_source") or p.get("task_name"),
                # Which variant this row wore, and whether it was SHOWN anything: a
                # yes/no question has no document (the rule is a string test and
                # "yes" is in any passage), so its slot ran plain and must not be
                # scored as a rescue.
                "variant": _slots_search_doc(self.config, slot=slot) if slot else None,
                "has_document": bool(self.document_block(i, slot=slot)) if slot else False,
                # For the split the rescue rate has to be read in: a one-word or
                # bare-number answer is both easier to write and easier to match
                # by accident.
                "answer_words": min((len(str(a).split()) for a in answers), default=0),
                "answer_numeric": bool(_is_numeric_answer(gt)),
                "turns": [], "won": None, "open": True}

    def _probe_note_turn(self, text_actions, next_obs, rewards, dones, infos) -> None:
        rows = getattr(self, "_probe_rows", None)
        if not rows:
            return
        obs_list, done_list = list(next_obs), list(dones)
        for i, row in enumerate(rows):
            if not row.get("open") or i >= len(obs_list):
                continue
            text = str(text_actions[i]) if i < len(text_actions) else ""
            gt, question = self._answers(i), self._question(i)
            row["turns"].append({
                "action": text,
                "query": _search_query(text),
                "wrote_answer": bool(_contains_answer(text, gt)),
                # Both tests: the guarded one drives the progress line and the
                # rule, the raw one bounds how often the guard mattered.
                "info_has_answer": bool(_evidence_in_text(obs_list[i], gt, question)),
                "info_has_answer_raw": bool(_contains_answer(obs_list[i], gt)),
                "info": str(obs_list[i]),
            })
            if bool(done_list[i]):
                info = infos[i] if i < len(infos) else {}
                row["won"] = float(info.get("won", 0.0)) if isinstance(info, dict) else None
                self._probe_write(i)

    def _probe_write(self, i: int) -> None:
        rows = getattr(self, "_probe_rows", None)
        if not rows or i >= len(rows) or not rows[i].get("open"):
            return
        row = rows[i]
        row["open"] = False
        row["evidence_seen"] = bool(self._evidence_seen[i])
        row["answer_early"] = bool(self._answer_early[i])
        ptrs = getattr(self, "_route_ptr", None)
        row["route_ptr"] = int(ptrs[i]) if ptrs is not None and i < len(ptrs) else None
        row["n_turns"] = len(row["turns"])
        path = os.path.join(self._probe_dir, f"search_rollouts.{os.getpid()}.jsonl")
        os.makedirs(self._probe_dir, exist_ok=True)
        if not getattr(self, "_probe_announced", False):
            # SEARCH_PROBE_DUMP is read where the manager runs, not where the
            # launcher exported it; one line in the log says it arrived.
            print(f"[search_probe] writing rollout records to {path}", flush=True)
            self._probe_announced = True
        with open(path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _probe_flush(self) -> None:
        """Write out rows the environment never marked done (turn budget spent)."""
        rows = getattr(self, "_probe_rows", None)
        for i in range(len(rows or [])):
            self._probe_write(i)
        self._probe_rows = []

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []
        # Rebuilt every turn, like alfworld's: the document slot's progress line
        # depends on what has come back so far.
        self._oci_roles = []
        self._oci_plains = []
        self._oci_docs = []
        _envs = getattr(self, "envs", None)
        _rank = _oci_rank_on(self.config) and "search" in _oci_rank_tasks(self.config)
        _second = _slots_second_doc(self.config)
        _seen = getattr(self, "_evidence_seen", None) or [False] * len(text_obs)

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )

            # What the plain student would have been asked at this turn, kept
            # before anything privileged is added -- the prompt the document
            # slot's tokens are re-scored on.
            plain_obs = obs_i
            # foreign=False: search has no foreign slot (oci_layout.has_foreign_slot),
            # so the last slots of the group are the document rows and the other
            # eight are ordinary rollouts -- exactly the eight control trains.
            _role = _slots_role(i, _envs, self.config, foreign=False, second_doc=_second)
            self._oci_roles.append(_role)
            if _role in (ROLE_DOC, ROLE_DOC_B):
                _slot = "a" if _role == ROLE_DOC else "b"
                _blk = self.document_block(i, slot=_slot)
                if _blk:
                    obs_i = _blk + obs_i
                    _mode = _slots_search_doc(self.config, slot=_slot)
                    if _mode == "expert_flow":
                        _ptrs = getattr(self, "_route_ptr", None) or [0] * len(text_obs)
                        obs_i = _insert_search_guide(
                            obs_i, _search_route_line(_block_lines(_blk), _ptrs[i], bool(_seen[i])))
                    elif _mode in ("answer_rule", "progress_only"):
                        obs_i = _insert_search_guide(obs_i, _search_progress_line(bool(_seen[i])))
            self._oci_plains.append(
                plain_obs if (_role in (ROLE_DOC, ROLE_DOC_B) and obs_i != plain_obs) else "")
            # Whole document, no progress line: the rank scorer does not act.
            _doc_blk = self.document_block(i) if _rank else ""
            self._oci_docs.append(_doc_blk + plain_obs if _doc_blk else "")

            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        # Refilled by every build_text_obs call; see OCI_PREFIX_KEY.
        self._oci_prefixes = []
        # The same, for the ten-slot layout (algorithm.oci_slots): what each slot
        # is for, and the render it would have had with no privileged text in it.
        # Both travel on the observation dict, for the reason OCI_PREFIX_KEY does.
        self._oci_roles = []
        self._oci_plains = []
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        _assert_one_privileged_arm(self.config)
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # walkthrough_stepwise: every slot starts at step 1 of its path.
        self._guide_ptr = [0] * len(text_obs)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs,
                OCI_PREFIX_KEY: list(self._oci_prefixes),
                OCI_ROLE_KEY: list(self._oci_roles),
                OCI_PLAIN_KEY: list(self._oci_plains),
                OCI_DOC_KEY: list(self._oci_docs)}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        if (_oci_stepwise(self.config) or _slots_doc_stepwise(self.config)) and getattr(self, 'gamefile', None):
            _envs = getattr(self, 'envs', None)
            ptrs = list(getattr(self, "_guide_ptr", None) or [0] * len(actions))
            for i, act in enumerate(actions):
                # The single-candidate arm marks one slot per group; the ten-slot
                # layout marks the document slot. reset() asserts the two are not
                # both on, so at most one of these can be true for a given run.
                if (_oci_candidate_row(i, _envs, self.config)
                        or _slots_role(i, _envs, self.config) == ROLE_DOC):
                    shown = self._oci_prefixes[i] if i < len(self._oci_prefixes) else ""
                    ptrs[i] = _advance_guide(_block_lines(shown), ptrs[i], act)
            self._guide_ptr = ptrs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs,
                             OCI_PREFIX_KEY: list(self._oci_prefixes),
                             OCI_ROLE_KEY: list(self._oci_roles),
                             OCI_PLAIN_KEY: list(self._oci_plains),
                             OCI_DOC_KEY: list(self._oci_docs)}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def document_block(self, i: int) -> str:
        """The block that solves env ``i``'s game: TextWorld's own walkthrough.

        The same accessor the other two managers expose, so a caller that shows
        documents does not branch on the task. '' when this manager has no
        gamefile yet, or the game has no walkthrough.
        """
        gamefiles = getattr(self, "gamefile", None) or []
        gf = gamefiles[i] if i < len(gamefiles) else None
        return _plan_block("alfworld", gf, "walkthrough")

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.

        Also fills ``self._oci_prefixes``: one entry per slot IN THIS MANAGER'S
        OWN ORDER, holding the corrupted-plan block prepended to that slot's
        observation or '' when none was. The caller puts it on the observation
        dict under OCI_PREFIX_KEY so the multitask merge reorders it with
        everything else -- see the note on OCI_PREFIX_KEY for what happened when
        it travelled by env-slot index instead.

        With the ten-slot layout on (algorithm.oci_slots) it fills two more lists
        the same way: ``self._oci_roles``, what each slot is for, and
        ``self._oci_plains``, the render THIS turn would have had with no
        privileged text in it -- kept because the shaped term re-scores the
        special slots' response tokens on exactly that prompt, and rebuilding it
        later from a stored history would not be the same string.
        """
        self._oci_prefixes = []
        self._oci_roles = []
        self._oci_plains = []
        # The same turn with this instance's document in front of it, for the
        # rank scorer. Built for EVERY row, not just a special slot: the scorer
        # asks how likely each ordinary rollout is under the self that knows the
        # answer, so every row needs the conditioning. '' when the switch is off
        # or the document cannot be built.
        self._oci_docs = []
        _rank = (_oci_rank_on(self.config)
                 and "alfworld" in _oci_rank_tasks(self.config))
        _envs = getattr(self, 'envs', None)
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            # What the plain student would have been asked at this turn. Kept as
            # it is here, before anything privileged is added, because it is the
            # prompt the special slots' tokens are re-scored on.
            plain_obs = obs
            _gamefile = self.gamefile[i] if getattr(self, 'gamefile', None) else None
            _role = _slots_role(i, _envs, self.config)
            self._oci_roles.append(_role)
            if _role == ROLE_DOC:
                # This game's own solution path, and (stepwise) the one line that
                # says which step it still owes -- the difference between 35% and
                # 88-95% of stuck groups solved.
                _oci_pre = _plan_block('alfworld', _gamefile, _slots_doc_mode(self.config),
                                       admissible_actions[i])
                obs = _oci_pre + obs
                if _oci_pre and _slots_doc_stepwise(self.config):
                    _ptrs = getattr(self, "_guide_ptr", None) or [0] * len(text_obs)
                    obs = _insert_guide(obs, _guide_line(_block_lines(_oci_pre), _ptrs[i]))
            elif _role == ROLE_FOREIGN:
                _oci_pre = ""
                _ftask = _slots_foreign_task(self.config)
                if _ftask == "alfworld":
                    # THIS GAME'S OBSERVATION, ANOTHER GAME'S GOAL. The slot acts
                    # in the right room with the right vocabulary toward the
                    # wrong target, so what it writes is an ordinary alfworld
                    # response that the real goal's prompt can re-score
                    # (rho near 1) -- unlike the WebShop prompt below, whose
                    # rows sat at rho = e^-40 and trained nothing.
                    obs = _slots_foreign_alfworld_obs(obs, self.tasks[i], _gamefile)
                else:
                    # ANOTHER TASK'S PROMPT, not a document about this game: the
                    # slot is told nothing about where it is, so every action it
                    # produces is inadmissible here and the episode spends its
                    # turn budget. Chosen from the gamefile, so a game is shown
                    # the same one every time it is drawn.
                    obs = _slots_foreign_prompt(_ftask, _gamefile)
            else:
                _oci_pre = (_wrong_plan_prefix('alfworld', _gamefile, self.config,
                                               admissible_actions[i])
                            if _oci_candidate_row(i, _envs, self.config) else "")
                obs = _oci_pre + obs
                if _oci_pre and _oci_stepwise(self.config):
                    _ptrs = getattr(self, "_guide_ptr", None) or [0] * len(text_obs)
                    obs = _insert_guide(obs, _guide_line(_block_lines(_oci_pre), _ptrs[i]))
            self._oci_prefixes.append(_oci_pre)
            # Only where the render actually differs. '' means "nothing to strip",
            # which is what every plain slot wants and what the single-candidate
            # arm wants too -- that one is stripped by its prefix instead.
            self._oci_plains.append(
                plain_obs if (_role in (ROLE_DOC, ROLE_FOREIGN) and obs != plain_obs) else "")
            # WHOLE DOCUMENT, NO PROGRESS LINE. The scorer does not act, so it
            # needs no pointer -- and the pointer only advances on an exact
            # action match, which an ordinary rollout does not give.
            _doc_blk = self.document_block(i) if _rank else ""
            self._oci_docs.append(_doc_blk + plain_obs if _doc_blk else "")

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Training success over the ORDINARY slots, with the special ones split out.

        The ten-slot layout adds two rollouts per group that are not the student
        being measured: one is shown the answer, the other is shown another task's
        prompt. Folding them into ``success_rate`` would move the training curve by
        construction -- up by the document slot's rescues, down by the foreign
        slot's certain failures -- and that curve is what this arm is compared
        against control on. They get their own keys instead, where they say what
        each slot is doing: the document slot's rate is its rescue rate, and the
        foreign slot's should stay at zero.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        _envs = getattr(self, 'envs', None)
        roles = [_slots_role(i, _envs, self.config) for i in range(len(total_batch_list))]
        if not any(r in (ROLE_DOC, ROLE_FOREIGN) for r in roles):
            return super().success_evaluator(*args, **kwargs)

        plain = defaultdict(list)
        special = {ROLE_DOC: defaultdict(list), ROLE_FOREIGN: defaultdict(list)}
        for bs in range(len(total_batch_list)):
            self._process_batch(bs, total_batch_list, total_infos,
                                special.get(roles[bs], plain))
        out = {key: np.array(value) for key, value in plain.items()}
        out["oci_doc_success_rate"] = np.array(special[ROLE_DOC]["success_rate"])
        out["oci_foreign_success_rate"] = np.array(special[ROLE_FOREIGN]["success_rate"])
        return out

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        # The goal records this episode is scored against, carried from the
        # workers (see WebshopWorker.reset). Kept for the whole episode: the
        # document slot needs them on every turn and reset is the only place
        # they appear.
        self.goals = [(info or {}).get('goal') for info in (infos or [])]
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        # The GOAL TEXT, not just the screen. The cross gate takes
                        # a GRPO group's prompt identity from its turn-0 anchor
                        # (opd_cross_gate.prompt_side), and format_obs above keeps
                        # only the parts that FOLLOW the instruction -- on
                        # WebShop's landing page that is the single token
                        # "'Search'" for every task. So every shopping goal hashed
                        # to ONE key on ONE side: the reference never reached
                        # min_prompts, WebShop was never a valid receiver, and its
                        # own OPD stayed eligible for attenuation on behalf of
                        # tasks that could not reciprocate. Reproduced on CPU:
                        # 5 distinct goals -> 1 anchor -> 1 key.
                        # self.tasks is what extract_task pulled out of the same
                        # observation, so it is available here and is stable
                        # across steps for a given episode seed.
                        'anchor': [f"{t} [SEP] {o}" for t, o in zip(self.tasks, obs)],
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks

    def document_block(self, i: int) -> str:
        """The block that solves env ``i``'s goal, or '' when it cannot be built.

        Same wrapper, numbering and signature as the other two managers', so
        whatever shows a document does not have to know which task it came from.
        Which query the first line uses is the builder's decision and is not
        repeated here -- a second default is a second thing to keep in step.
        """
        goals = getattr(self, "goals", None) or []
        goal = goals[i] if i < len(goals) else None
        return render_document(_webshop_document_lines(goal))
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs


def _normalize_multitask_name(task_name: str) -> str:
    task_name = str(task_name).lower()
    if "alfworld" in task_name:
        return "alfworld"
    if "webshop" in task_name:
        return "webshop"
    if "search" in task_name:
        return "search"
    raise ValueError(f"Unsupported multitask task_name: {task_name}")


def _plain_container(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


class MultiTaskEnvironmentManager(EnvironmentManagerBase):
    """Route a mixed batch to existing task-specific environment managers."""

    def __init__(self, managers: Dict[str, EnvironmentManagerBase], task_max_steps: Dict[str, int], config):
        self.managers = managers
        self.task_max_steps = {task: int(task_max_steps[task]) for task in managers}
        self.config = config
        self._task_indices = {}
        self._task_steps = {}
        self._task_done = {}
        self._last_obs_by_task = {}
        self._last_infos_by_task = {}

    def rewind_games(self):
        """Rewind every task that has a problem cycle. See EnvironmentManagerBase."""
        out = {}
        for task, manager in self.managers.items():
            info = manager.rewind_games()
            if info:
                out[task] = info
        return out

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        if kwargs is None:
            raise ValueError("multitask environment requires env_kwargs with task_name for every sample.")
        if isinstance(kwargs, np.ndarray):
            kwargs = kwargs.tolist()

        task_to_items = defaultdict(list)
        for idx, item_kwargs in enumerate(kwargs):
            if item_kwargs is None or "task_name" not in item_kwargs:
                raise ValueError("Every multitask env_kwargs entry must contain task_name.")
            task = _normalize_multitask_name(item_kwargs["task_name"])
            if task not in self.managers:
                raise ValueError(f"No environment manager configured for task_name={task}.")
            task_to_items[task].append((idx, item_kwargs))

        self._task_indices = {}
        self._task_steps = {}
        self._task_done = {}
        task_obs = {}
        task_infos = {}

        task_kwargs_by_task = {}
        for task, items in task_to_items.items():
            indices, task_kwargs = zip(*items)
            self._task_indices[task] = list(indices)
            task_kwargs_by_task[task] = list(task_kwargs)

        # Reset all task managers concurrently (env construction/reset is
        # I/O-bound), mirroring the parallel step() so the rollout doesn't
        # serialize per-task startup.
        reset_results = {}
        if len(task_kwargs_by_task) == 1:
            (task, task_kwargs), = task_kwargs_by_task.items()
            reset_results[task] = self.managers[task].reset(task_kwargs)
        else:
            with ThreadPoolExecutor(max_workers=len(task_kwargs_by_task)) as executor:
                futures = {
                    executor.submit(self.managers[task].reset, task_kwargs): task
                    for task, task_kwargs in task_kwargs_by_task.items()
                }
                for future in as_completed(futures):
                    reset_results[futures[future]] = future.result()

        for task in task_kwargs_by_task:
            indices = self._task_indices[task]
            obs, infos = reset_results[task]
            for info in infos:
                info["task_name"] = task
            self._task_steps[task] = 0
            self._task_done[task] = np.zeros(len(indices), dtype=bool)
            self._last_obs_by_task[task] = obs
            self._last_infos_by_task[task] = infos
            task_obs[task] = obs
            task_infos[task] = infos

        observations = self._merge_observations(task_obs, len(kwargs))
        infos = self._merge_infos(task_infos, len(kwargs))
        return observations, infos

    def task_row_indices(self) -> Dict[str, List[int]]:
        """Which batch rows belong to which task, as decided by the last reset.

        The rollout loop needs this to advance one task at a time: the row sets
        are disjoint, so two callers holding different tasks write to different
        rows of every shared array.
        """
        return {task: list(indices) for task, indices in self._task_indices.items()}

    def step(self, text_actions: List[str], tasks: Optional[Sequence[str]] = None):
        """Step the environments.

        ``tasks`` restricts the step to those tasks. The returned arrays are
        still batch-shaped, but **only the selected tasks' rows carry meaning**
        -- the rest hold the neutral fill of the merge helpers (None obs, 0
        reward, False done, None info), not that task's real state. A caller
        that passes ``tasks`` must therefore read only the rows it asked for.
        This is what lets each task advance on its own turn counter: alfworld
        runs to 50 turns without dragging search's finished rows behind it, and
        one task's env.step overlaps another's generate instead of the whole
        batch standing still together.

        Per-task state (``_task_steps`` / ``_task_done`` / ``_last_obs_by_task``)
        is keyed by task and each key is touched by one caller, so concurrent
        calls for different tasks do not race. Two concurrent calls naming the
        SAME task would, and nothing here defends against that.
        """
        if not self._task_indices:
            raise RuntimeError("MultiTaskEnvironmentManager.step called before reset.")
        if tasks is not None:
            unknown = [task for task in tasks if task not in self._task_indices]
            if unknown:
                raise ValueError(f"step() asked for tasks not in this batch: {unknown}")
            wanted = set(tasks)
        else:
            wanted = None

        task_obs = {}
        task_rewards = {}
        task_dones = {}
        task_infos = {}

        # Tasks that still need a real environment step (others are short-circuited).
        active_tasks = {}
        for task, indices in self._task_indices.items():
            if wanted is not None and task not in wanted:
                continue
            if self._task_done[task].all() or self._task_steps[task] >= self.task_max_steps[task]:
                task_obs[task] = self._last_obs_by_task[task]
                task_rewards[task] = np.zeros(len(indices), dtype=np.float32)
                task_dones[task] = np.ones(len(indices), dtype=bool)
                task_infos[task] = self._done_infos(task)
                continue
            active_tasks[task] = [text_actions[idx] for idx in indices]

        # Step active tasks concurrently. Each manager.step is independent and
        # I/O-bound (HTTP for search, Ray/subprocess IPC for alfworld/webshop),
        # so threads overlap them and the per-turn barrier becomes ~max(task)
        # instead of sum(task). Bookkeeping below runs in the main thread to
        # avoid races on the shared state dicts.
        stepped = {}
        if len(active_tasks) == 1:
            (task, actions), = active_tasks.items()
            stepped[task] = self.managers[task].step(actions)
        elif active_tasks:
            with ThreadPoolExecutor(max_workers=len(active_tasks)) as executor:
                futures = {
                    executor.submit(self.managers[task].step, actions): task
                    for task, actions in active_tasks.items()
                }
                for future in as_completed(futures):
                    stepped[futures[future]] = future.result()

        for task in active_tasks:
            indices = self._task_indices[task]
            obs, rewards, dones, infos = stepped[task]
            rewards = np.asarray(rewards).reshape(-1)
            dones = np.asarray(dones).reshape(-1).astype(bool)

            self._task_steps[task] += 1
            if self._task_steps[task] >= self.task_max_steps[task]:
                dones = np.ones(len(indices), dtype=bool)

            for info in infos:
                info["task_name"] = task

            self._task_done[task] = np.logical_or(self._task_done[task], dones)
            self._last_obs_by_task[task] = obs
            self._last_infos_by_task[task] = infos

            task_obs[task] = obs
            task_rewards[task] = rewards
            task_dones[task] = dones
            task_infos[task] = infos

        observations = self._merge_observations(task_obs, len(text_actions))
        rewards = self._merge_arrays(task_rewards, len(text_actions), dtype=np.float32)
        dones = self._merge_arrays(task_dones, len(text_actions), dtype=bool)
        infos = self._merge_infos(task_infos, len(text_actions))
        return observations, rewards, dones, infos

    def _done_infos(self, task: str) -> List[Dict]:
        infos = []
        for info in self._last_infos_by_task[task]:
            done_info = dict(info)
            done_info["task_name"] = task
            done_info["is_action_valid"] = to_numpy(True)
            infos.append(done_info)
        return infos

    def _merge_observations(self, task_obs: Dict[str, Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        keys = set()
        for obs in task_obs.values():
            keys.update(obs.keys())

        merged = {}
        for key in keys:
            values = [None] * batch_size
            has_values = False
            for task, obs in task_obs.items():
                obs_values = obs.get(key)
                if obs_values is None:
                    continue
                has_values = True
                for idx, value in zip(self._task_indices[task], obs_values):
                    values[idx] = value
            merged[key] = values if has_values else None
        return merged

    def _merge_infos(self, task_infos: Dict[str, List[Dict]], batch_size: int) -> List[Dict]:
        merged = [None] * batch_size
        for task, infos in task_infos.items():
            for idx, info in zip(self._task_indices[task], infos):
                merged[idx] = info
        return merged

    def _merge_arrays(self, task_values: Dict[str, np.ndarray], batch_size: int, dtype) -> np.ndarray:
        merged = np.zeros(batch_size, dtype=dtype)
        for task, values in task_values.items():
            for idx, value in zip(self._task_indices[task], values):
                merged[idx] = value
        return merged

    @staticmethod
    def _slice_optional_batch(value, indices):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value[indices]
        if isinstance(value, np.ndarray):
            return value[indices]
        return [value[idx] for idx in indices]

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        episode_rewards = kwargs.get("episode_rewards")
        episode_lengths = kwargs.get("episode_lengths")
        success = defaultdict(list)

        for task, indices in self._task_indices.items():
            task_total_infos = [total_infos[idx] for idx in indices]
            task_total_batch_list = [total_batch_list[idx] for idx in indices]
            task_episode_rewards = self._slice_optional_batch(episode_rewards, indices)
            task_episode_lengths = self._slice_optional_batch(episode_lengths, indices)
            task_success = self.managers[task].success_evaluator(
                total_infos=task_total_infos,
                total_batch_list=task_total_batch_list,
                episode_rewards=task_episode_rewards,
                episode_lengths=task_episode_lengths,
            )
            success["success_rate"].extend(task_success["success_rate"].tolist())
            success[f"{task}_success_rate"].extend(task_success["success_rate"].tolist())
            for key, value in task_success.items():
                if key == "success_rate":
                    continue
                success[f"{task}_{key}"].extend(value.tolist())

        return {key: np.array(value) for key, value in success.items()}

    def close(self) -> None:
        for manager in self.managers.values():
            manager.close()


def _get_multitask_tasks(config) -> List[str]:
    multitask_cfg = config.env.get("multitask", {})
    tasks = _plain_container(multitask_cfg.get("tasks", ["alfworld", "search", "webshop"]))
    return [_normalize_multitask_name(task) for task in tasks]


def _get_multitask_task_max_steps(config, tasks: List[str]) -> Dict[str, int]:
    defaults = {"alfworld": 50, "search": 4, "webshop": 15}
    multitask_cfg = config.env.get("multitask", {})
    configured = _plain_container(multitask_cfg.get("max_steps", {}))
    if configured:
        defaults.update({task: int(value) for task, value in configured.items()})
    return {task: defaults[task] for task in tasks}


def _get_multitask_per_task_batch_size(config, tasks: List[str], is_train: bool) -> int:
    if is_train:
        data_task_balance = config.data.get("task_balance", {})
        per_task_batch_size = data_task_balance.get("per_task_batch_size", None)
        total_batch_size = config.data.train_batch_size
    else:
        multitask_cfg = config.env.get("multitask", {})
        per_task_batch_size = multitask_cfg.get("val_per_task_batch_size", None)
        total_batch_size = config.data.val_batch_size

    if per_task_batch_size is None:
        if total_batch_size is None:
            raise ValueError("multitask val_batch_size must be set when val_per_task_batch_size is not configured.")
        if int(total_batch_size) % len(tasks) != 0:
            raise ValueError(f"multitask batch size {total_batch_size} is not divisible by {len(tasks)} tasks.")
        per_task_batch_size = int(total_batch_size) // len(tasks)
    return int(per_task_batch_size)


def _get_multitask_val_batch_sizes(config, tasks: List[str]):
    """``(sizes, default)`` for validation: how many rows each task's batch holds.

    ``val_per_task_batch_size`` may be one number, as it always was, or a mapping
    naming the tasks that differ -- the same shape ``max_steps`` and
    ``history_length`` already take. Anything unnamed keeps ``data.val_batch_size``,
    so adding an entry for one task cannot silently resize another.

    A task's size is not free of its scoring. alfworld draws its episodes from a
    seeded game cycle indexed by position within its environment manager, and the
    manager is built at this size: change it and the run plays different games.
    search does not care -- every row carries its own question and ground truth --
    which is the task the size is worth changing for.
    """
    multitask_cfg = config.env.get("multitask", {})
    configured = _plain_container(multitask_cfg.get("val_per_task_batch_size", None))
    if isinstance(configured, dict):
        unknown = [task for task in configured if task not in tasks]
        if unknown:
            raise ValueError(f"val_per_task_batch_size names tasks not in this run: {unknown} (configured: {tasks})")
        sizes = {task: int(size) for task, size in configured.items()}
        for task, size in sizes.items():
            if size <= 0:
                raise ValueError(f"val_per_task_batch_size[{task!r}] must be positive, got {size}")
        default = int(config.data.val_batch_size)
        return {task: sizes.get(task, default) for task in tasks}, default
    uniform = _get_multitask_per_task_batch_size(config, tasks, is_train=False)
    return {task: uniform for task in tasks}, uniform


def _size_for(per_task_batch_size, task: str) -> int:
    if isinstance(per_task_batch_size, dict):
        return int(per_task_batch_size[task])
    return int(per_task_batch_size)


def _get_multitask_task_history_length(config, task: str):
    multitask_cfg = config.env.get("multitask", {})
    configured = _plain_container(multitask_cfg.get("history_length", {}))
    if isinstance(configured, dict) and task in configured:
        return int(configured[task])
    return config.env.history_length


def _copy_config_for_task(config, env_name: str, max_steps: int, task: str = None):
    task_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    task_config.env.env_name = env_name
    task_config.env.max_steps = max_steps
    if task is not None:
        task_config.env.history_length = _get_multitask_task_history_length(config, task)
    return task_config


def _build_multitask_manager(config, tasks: List[str], task_max_steps: Dict[str, int], per_task_batch_size, group_n: int, is_train: bool, seed: int, resources_per_worker: Dict):
    """``per_task_batch_size`` is one number for every task, or a mapping."""
    managers = {}

    for task in tasks:
        env_num = _size_for(per_task_batch_size, task)
        if task == "alfworld":
            from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection

            alf_config_path = os.path.join(os.path.dirname(__file__), "env_package/alfworld/configs/config_tw.yaml")
            env_kwargs = {"eval_dataset": config.env.alfworld.eval_dataset}
            task_config = _copy_config_for_task(config, "alfworld/AlfredTWEnv", task_max_steps[task], task=task)
            _envs = build_alfworld_envs(
                alf_config_path,
                seed,
                env_num,
                group_n,
                resources_per_worker=resources_per_worker,
                is_train=is_train,
                env_kwargs=env_kwargs,
            )
            managers[task] = AlfWorldEnvironmentManager(_envs, partial(alfworld_projection), task_config)
        elif task == "search":
            from agent_system.environments.env_package.search import build_search_envs, search_projection

            task_config = _copy_config_for_task(config, "search", task_max_steps[task], task=task)
            _envs = build_search_envs(
                seed=seed,
                env_num=env_num,
                group_n=group_n,
                is_train=is_train,
                env_config=task_config.env,
            )
            managers[task] = SearchEnvironmentManager(_envs, partial(search_projection), task_config)
        elif task == "webshop":
            from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection

            if config.env.webshop.use_small:
                file_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_shuffle_1000.json")
                attr_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_ins_v2_1000.json")
            else:
                file_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_shuffle.json")
                attr_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_ins_v2.json")
            env_kwargs = {
                "observation_mode": "text",
                "num_products": None,
                "human_goals": config.env.webshop.human_goals,
                "file_path": file_path,
                "attr_path": attr_path,
            }
            task_config = _copy_config_for_task(config, "Webshop", task_max_steps[task], task=task)
            _envs = build_webshop_envs(
                seed=seed,
                env_num=env_num,
                group_n=group_n,
                is_train=is_train,
                env_kwargs=env_kwargs,
                resources_per_worker=resources_per_worker,
            )
            managers[task] = WebshopEnvironmentManager(_envs, partial(webshop_projection), task_config)
        else:
            raise ValueError(f"Unsupported multitask task: {task}")

    return MultiTaskEnvironmentManager(managers=managers, task_max_steps=task_max_steps, config=config)


class LazyEnvManager:
    """Defer env construction until something actually needs the environments.

    The val envs are Ray actors that sit idle from startup until the first
    validation (``trainer.test_freq`` steps in), and with ``test_freq <= 0`` they are
    never touched at all; the train envs are likewise dead weight in a
    ``trainer.val_only`` run. In multitask that is 252 of 492 actors either way, each
    holding its own copy of the environment's data, which is the difference between
    fitting in host RAM and being killed by the OOM killer.

    Everything else is unchanged: the same builder runs with the same arguments, just
    later, so a rollout sees the same games and seeds it would have seen before.
    """

    def __init__(self, builder):
        self._builder = builder
        self._envs = None

    def materialize(self):
        if self._envs is None:
            self._envs = self._builder()
        return self._envs

    def __getattr__(self, name):
        # only reached for names that are not instance attributes, i.e. never for
        # _builder / _envs, so this cannot recurse
        return getattr(self.materialize(), name)


# Tasks a second validation manager may be built for. A second manager scores the
# same episodes only when a row's episode comes entirely from that row -- search
# passes each env its own question and ground_truth at reset, and its `_rng` is
# constructed and never used, so two managers are interchangeable.
#
# alfworld is not on this list and must not be: AlfworldEnvs seeds worker i with
# `seed + i // group_n`, so which game a row plays is a function of its position
# *within its manager*. Split across two and every row plays a different game --
# which is the one thing the per-checkpoint-process design exists to prevent, and
# it would not raise.
#
# webshop is left off for the same reason until someone checks it: its envs are
# Ray actors handed goals at construction, and "probably fine" is not the standard
# for a scoring path.
PIPELINEABLE_VAL_TASKS = ("search",)


def get_val_batch_sizes(config):
    """``(sizes, default)`` for validation batching, or ``None`` for a non-multitask run.

    The trainer's loader needs the same numbers the environment managers are
    built with -- a batch of 252 rows handed to a manager holding 126
    environments would index past the end -- so both read them from here.
    """
    if config.env.env_name.lower() != "multitask":
        return None
    tasks = _get_multitask_tasks(config)
    return _get_multitask_val_batch_sizes(config, tasks)


def build_val_env_manager(config, tasks: List[str]):
    """A second validation env manager, restricted to `tasks`.

    Same arguments as the one make_envs builds, so the environments are the ones
    that batch would have seen anyway -- only fewer tasks, because a manager for
    tasks it will never be handed costs 126 Ray actors each.

    Lazy, like the primary: a run that never routes a batch here never builds it.
    """
    for task in tasks:
        if task not in PIPELINEABLE_VAL_TASKS:
            raise ValueError(
                f"{task!r} cannot have a second validation manager (see PIPELINEABLE_VAL_TASKS); "
                f"allowed: {list(PIPELINEABLE_VAL_TASKS)}"
            )
    all_tasks = _get_multitask_tasks(config)
    unknown = [task for task in tasks if task not in all_tasks]
    if unknown:
        raise ValueError(f"tasks not configured for this run: {unknown} (configured: {all_tasks})")

    def _build():
        return _build_multitask_manager(
            config=config,
            tasks=list(tasks),
            task_max_steps=_get_multitask_task_max_steps(config, all_tasks),
            per_task_batch_size=_get_multitask_val_batch_sizes(config, all_tasks)[0],
            group_n=1,
            is_train=False,
            seed=config.env.seed + 1000,
            resources_per_worker=OmegaConf.to_container(config.env.resources_per_worker, resolve=True),
        )

    return LazyEnvManager(_build)


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if config.env.env_name.lower() == "multitask":
        tasks = _get_multitask_tasks(config)
        task_max_steps = _get_multitask_task_max_steps(config, tasks)
        train_per_task_batch_size = _get_multitask_per_task_batch_size(config, tasks, is_train=True)
        val_batch_sizes = _get_multitask_val_batch_sizes(config, tasks)[0]
        if train_per_task_batch_size * len(tasks) != int(config.data.train_batch_size):
            raise ValueError(
                "multitask train batch mismatch: "
                f"{train_per_task_batch_size} * {len(tasks)} != {config.data.train_batch_size}"
            )
        # Validation supports two layouts:
        #   - per-task batches: val_batch_size == val_per_task_batch_size, the task-sorted
        #     test parquet yields one single-task batch per task (each task is evaluated
        #     in its own rollout pass)
        #   - mixed batch: val_batch_size == val_per_task_batch_size * len(tasks)
        #
        # Neither is checked once the sizes differ by task: the batches then come
        # from TaskBatchSampler, which groups by task by construction rather than
        # by the sizes lining up, and val_batch_size is only the default for the
        # tasks the config does not name.
        val_batch_size = int(config.data.val_batch_size)
        uniform = set(val_batch_sizes.values())
        if len(uniform) == 1:
            val_per_task_batch_size = uniform.pop()
            if val_batch_size not in (val_per_task_batch_size, val_per_task_batch_size * len(tasks)):
                raise ValueError(
                    "multitask val batch mismatch: val_batch_size must be "
                    f"{val_per_task_batch_size} (per-task validation batches) or "
                    f"{val_per_task_batch_size * len(tasks)} (single mixed batch), got {val_batch_size}"
                )

        def _build_train_envs():
            managers = _build_multitask_manager(
                config=config,
                tasks=tasks,
                task_max_steps=task_max_steps,
                per_task_batch_size=train_per_task_batch_size,
                group_n=group_n,
                is_train=True,
                seed=config.env.seed,
                resources_per_worker=resources_per_worker,
            )
            if "webshop" in tasks:
                import time

                time.sleep(train_per_task_batch_size * group_n * 0.1)
            return managers

        def _build_val_envs():
            managers = _build_multitask_manager(
                config=config,
                tasks=tasks,
                task_max_steps=task_max_steps,
                per_task_batch_size=val_batch_sizes,
                group_n=1,
                is_train=False,
                seed=config.env.seed + 1000,
                resources_per_worker=resources_per_worker,
            )
            if "webshop" in tasks:
                import time

                time.sleep(val_batch_sizes["webshop"] * 0.1)
            return managers

        return LazyEnvManager(_build_train_envs), LazyEnvManager(_build_val_envs)
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        projection_f = partial(search_projection)
        envs = LazyEnvManager(lambda: SearchEnvironmentManager(
            build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env),
            projection_f, config))
        val_envs = LazyEnvManager(lambda: SearchEnvironmentManager(
            build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env),
            projection_f, config))
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = LazyEnvManager(lambda: GymCardEnvironmentManager(
            build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker),
            projection_f, config))
        val_envs = LazyEnvManager(lambda: GymCardEnvironmentManager(
            build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker),
            projection_f, config))
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        projection_f = partial(alfworld_projection)
        envs = LazyEnvManager(lambda: AlfWorldEnvironmentManager(
            build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker),
            projection_f, config))
        val_envs = LazyEnvManager(lambda: AlfWorldEnvironmentManager(
            build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker),
            projection_f, config))
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        projection_f = partial(sokoban_projection)
        envs = LazyEnvManager(lambda: SokobanEnvironmentManager(
            build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker),
            projection_f, config))
        val_envs = LazyEnvManager(lambda: SokobanEnvironmentManager(
            build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker),
            projection_f, config))
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        projection_f = partial(webshop_projection)

        def _build_train_envs():
            import time
            managers = WebshopEnvironmentManager(
                build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker),
                projection_f, config)
            time.sleep(config.data.train_batch_size * group_n * 0.1)  # wait for the envs to be ready
            return managers

        def _build_val_envs():
            import time
            managers = WebshopEnvironmentManager(
                build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker),
                projection_f, config)
            time.sleep(config.data.val_batch_size * 0.1)  # wait for the envs to be ready
            return managers

        envs = LazyEnvManager(_build_train_envs)
        val_envs = LazyEnvManager(_build_val_envs)
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        projection_f = partial(appworld_projection)
        envs = LazyEnvManager(lambda: AppWorldEnvironmentManager(
            build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker),
            projection_f, config))
        val_envs = LazyEnvManager(lambda: AppWorldEnvironmentManager(
            build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker),
            projection_f, config))
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
