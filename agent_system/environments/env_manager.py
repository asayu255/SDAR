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
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
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
            "budget, 'intact' prints the true path. 'drop' removed the "
            "requirement step and is gone: it was refuted (cand_fail_rate 1/15)."
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
    if task != "alfworld" or not gamefile:
        return ""
    mode = _oci_plan_mode(config)
    n_detour = _oci_detour(config)
    turns = _oci_delay_turns(config)
    scene = (_scene_receptacles(gamefile, admissible)
             if (n_detour or mode == "delay") else [])
    key = (str(gamefile), mode, n_detour, turns)
    if key not in _WRONG_PLAN_CACHE:
        block = _build_wrong_plan(str(gamefile), mode, n_detour, scene, turns)
        # DO NOT CACHE A BLOCK BUILT WITHOUT THE SCENE. The pool comes from the
        # admissible actions, so an early call that has none would otherwise
        # pin an empty tour for the rest of training.
        if block or not (n_detour or mode == "delay") or scene:
            _WRONG_PLAN_CACHE[key] = block
        return block
    return _WRONG_PLAN_CACHE[key]


PLAN_CORRUPTIONS = ("misdirect", "intact", "delay")

# The block, in the environment's own words. Both modes emit the SAME text apart
# from the numbered lines -- no word anywhere says whether the path is right.
_PLAN_HEADER = "[Privileged Solution Path]"
_PLAN_FOOTER = "[/Privileged Solution Path]"
_PLAN_LEAD = (
    "THIS IS THE VERIFIED CORRECT SOLUTION PATH FOR THIS TASK. IT IS COMPLETE AND OPTIMAL.\n"
    "YOU MUST FOLLOW IT EXACTLY. At every step, take the action given by the next line\n"
    "of this path. DO NOT SEARCH ON YOUR OWN. DO NOT DEVIATE FROM IT. Any action that is\n"
    "not the next line of this path is wrong.\n"
    "\n"
    "The full path that solves this task:"
)

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

    if mode == "delay":
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
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

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
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs,
                OCI_PREFIX_KEY: list(self._oci_prefixes)}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs,
                             OCI_PREFIX_KEY: list(self._oci_prefixes)}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
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
        """
        self._oci_prefixes = []
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
                _oci_pre = (_wrong_plan_prefix('alfworld',
                                              (self.gamefile[i] if getattr(self, 'gamefile', None) else None),
                                              self.config, admissible_actions[i])
                            if _oci_candidate_row(i, getattr(self, 'envs', None), self.config) else "")
                self._oci_prefixes.append(_oci_pre)
                obs = _oci_pre + obs
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
                _oci_pre = (_wrong_plan_prefix('alfworld',
                                              (self.gamefile[i] if getattr(self, 'gamefile', None) else None),
                                              self.config, admissible_actions[i])
                            if _oci_candidate_row(i, getattr(self, 'envs', None), self.config) else "")
                self._oci_prefixes.append(_oci_pre)
                obs = _oci_pre + obs

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
