# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Ten rollouts per group, eight of them trained: which slot is shown what.

THE LAYOUT (``algorithm.oci_slots``). Every group on a configured task runs
``group_n`` rollouts of the SAME problem, and the last two of them are not
ordinary rollouts:

    slots 0 .. group_n-4   plain      the ordinary student, always trained
    slot  group_n-3        reserve    ordinary too; trained unless a special
                                      slot takes its place
    slot  group_n-2        document   shown this game's own solution path
    slot  group_n-1        foreign    shown ANOTHER TASK'S prompt instead of its own

THE VERDICT IS READ OFF THE ORDINARY ONES -- plain + reserve, which at
``group_n=10`` are exactly the eight rollouts a control group trains. So a group
that is live here is live there, with the same eight returns in the same
statistic, and the special slots are only ever spent on a group control would
have thrown away:

    live       -> plain + reserve     (identical to control)
    stuck      -> plain + document    when the document rollout actually solved it
    saturated  -> plain + foreign     when the foreign prompt actually failed

Eight trajectories are trained either way. The other two are dropped before the
batch is padded, so the group size, the advantage's denominator and the row count
of an optimizer step stay what control's are -- which is the whole reason the
generation is ten and not nine. The selection itself is
``verl/trainer/ppo/oci_slots.py``; this module only says which slot is which and
what the foreign slot is shown.

WHY THE FOREIGN SLOT IS ANOTHER TASK'S PROMPT AND NOT A FAILURE-INDUCING
DOCUMENT. Five block designs were measured on saturated groups -- the only place
the slot is ever used -- at the control's step 300, and none reached 12%: the
uncorrupted path 0/25, misdirection 2/26, a 50-turn tour in front of the true
path 3/26, and that same tour with a per-turn progress line 3/27 (the line that
raised the document slot's rescue rate from 35% to 88-95%). In a saturated group
the student ignores a path it does not need: the pointer advanced 1.4 lines of
~56 on average and no trajectory followed one to the end. A prompt from another
task removes the question instead of arguing it -- every action it produces is
inadmissible here, so the episode spends its turn budget and ends at the
environment's failure return. It is not a privileged input, and it says nothing
about the game that could be right or wrong.

WHY ITS ACTIONS COST NO PENALTY. The WebShop prompt asks for exactly the
``<think>``/``<action>`` shape the alfworld projection checks, so the rows are
FORMAT-valid and inadmissible, which is what an ordinary failure is here; a
Search prompt would emit ``<search>`` tags instead and collect the invalid-action
penalty, which would put those rows BELOW an ordinary failure and change the
group's floor rather than sit on it.
"""

import zlib

from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS

# What a row's slot is for. Travels as the `oci_role` column; ROLE_NONE is every
# row of a run without the layout, and every row of a task it does not cover.
ROLE_NONE, ROLE_PLAIN, ROLE_RESERVE, ROLE_DOC, ROLE_FOREIGN = 0, 1, 2, 3, 4
ROLE_NAMES = {ROLE_NONE: "none", ROLE_PLAIN: "plain", ROLE_RESERVE: "reserve",
              ROLE_DOC: "document", ROLE_FOREIGN: "foreign"}

# Observation-dict keys, alongside OCI_PREFIX_KEY in env_manager. They travel on
# the observation because the multitask merge reorders every key from a manager's
# local slot order into global row order -- a side channel keyed by env slot and
# read by the global row index marked one group of fifteen and passed its own
# "at least one is marked" assert.
OCI_ROLE_KEY = "oci_role"
OCI_PLAIN_KEY = "oci_plain"

# What the document slot may print. Both print the game's own walkthrough, which
# replays 30/30 in this environment; the stepwise variant adds the one line that
# says which step is owed, and that line is the difference between 35% and 88-95%
# of stuck groups rescued.
DOC_MODES = ("walkthrough", "walkthrough_stepwise")

# Whose prompt the foreign slot shows. One entry, because the choice is not free:
# see the note on the invalid-action penalty above.
FOREIGN_TASKS = ("webshop",)


def slots_cfg(config):
    """``algorithm.oci_slots`` off whatever config object the caller holds, or None."""
    if config is None:
        return None
    try:
        return config.algorithm.get("oci_slots", None)
    except Exception:
        return None


def slots_on(config) -> bool:
    cfg = slots_cfg(config)
    return bool(cfg is not None and cfg.get("enable", False))


def doc_mode(config) -> str:
    mode = str((slots_cfg(config) or {}).get("doc_mode", "walkthrough_stepwise")
               or "walkthrough_stepwise")
    if mode not in DOC_MODES:
        raise ValueError(
            f"algorithm.oci_slots.doc_mode={mode!r}; expected one of {DOC_MODES}. "
            "The document slot prints the game's own solution path: 'walkthrough' "
            "prints it once, 'walkthrough_stepwise' adds the per-turn line naming "
            "the step still owed.")
    return mode


def doc_stepwise(config) -> bool:
    return slots_on(config) and doc_mode(config).endswith("_stepwise")


def foreign_task(config) -> str:
    task = str((slots_cfg(config) or {}).get("foreign_task", "webshop") or "webshop")
    if task not in FOREIGN_TASKS:
        raise ValueError(
            f"algorithm.oci_slots.foreign_task={task!r}; expected one of {FOREIGN_TASKS}.")
    return task


def slot_tasks(config):
    """Tasks whose groups carry the layout. Others run plain rollouts only."""
    cfg = slots_cfg(config) or {}
    return tuple(cfg.get("tasks", ["alfworld"]) or ["alfworld"])


def used_per_group(group_n: int) -> int:
    """How many trajectories a group trains: everything but the two special slots."""
    return max(int(group_n) - 2, 0)


def role_for_slot(slot: int, group_n: int) -> int:
    """What the slot at position ``slot`` of a group of ``group_n`` is for.

    Below four the layout does not exist: it needs two ordinary rollouts to read
    a verdict from and two more for the special slots.
    """
    g = int(group_n)
    if g < 4:
        return ROLE_NONE
    j = int(slot) % g
    if j == g - 1:
        return ROLE_FOREIGN
    if j == g - 2:
        return ROLE_DOC
    if j == g - 3:
        return ROLE_RESERVE
    return ROLE_PLAIN


def slot_role(i: int, envs, config=None) -> int:
    """The role of env slot ``i`` in the manager that holds ``envs``.

    ASKS THE ENVS FOR THE GROUP SIZE, NOT THE CONFIG, and only marks a TRAINING
    manager -- ``config.env.rollout.n`` is the training group size on a shared
    config object, so reading it here once got one alfworld instance in eight
    VALIDATED with a privileged block, i.e. the arm's own success measurement was
    contaminated by the arm.
    """
    if not slots_on(config):
        return ROLE_NONE
    if envs is None or not getattr(envs, "is_train", False):
        return ROLE_NONE
    try:
        g = int(getattr(envs, "group_n", 0))
    except (TypeError, ValueError):
        return ROLE_NONE
    return role_for_slot(i, g)


# --------------------------------------------------------------------------- #
# the foreign slot's prompt
# --------------------------------------------------------------------------- #

# REAL WEBSHOP TRAINING INSTRUCTIONS, sampled once from the goal set this repo's
# runs actually use (use_small=True, human_goals=False -> 6910 synthetic goals)
# and from its TRAIN range (goals 500+), so nothing here is a goal the WebShop
# validation scores. Kept as text rather than read from the WebShop data at
# runtime: an alfworld-only run has no WebShop server, and a prompt that depends
# on one would differ between runs that do and do not load it.
WEBSHOP_TRAIN_INSTRUCTIONS = (
    "Find me wide leg, slim fit, straight leg, loose fit women's shorts with long sleeve, elastic waist, high waist, tummy control, short sleeve with color: blue, and size: x-large, and price lower than 20.00 dollars",
    "Find me loose fit women's shorts with elastic closure, faux fur with color: red, and size: 3x-large, and price lower than 60.00 dollars",
    "Find me slim fit, loose fit men's tuxedo shirts with long sleeve, short sleeve, contrast color, classic fit for teen girls with color: a-bk2, and size: xx-large, and price lower than 30.00 dollars",
    "Find me officially licensed men's t-shirts & tanks with needle sleeve, classic fit with color: navy, and fit type: women, and size: medium, and price lower than 50.00 dollars",
    "Find me machine wash men's pants with relaxed fit with color: grey, and size: 44w x 32l, and price lower than 80.00 dollars",
    "Find me machine washable men's t-shirts with short sleeve for tumble dry with color: 36 pack mix, and size: 7x-large, and price lower than 90.00 dollars",
    "Find me high power, non slip binoculars & scopes for bird watching with style: 10x50 uhd binoculars, and price lower than 190.00 dollars",
    "Find me slim fit, straight leg men's pants with elastic waist, long sleeve, relaxed fit for everyday wear with color: green, and size: small, and price lower than 30.00 dollars",
    "Find me straight leg, machine washable men's jeans with color: coal grey, and size: 32w x 32l, and price lower than 80.00 dollars",
    "Find me slim fit, loose fit women's tops, tees & blouses with long sleeve, short sleeve with color: a01#blue, and size: 4x-large, and price lower than 30.00 dollars",
    "Find me machine wash men's dress shirts with cotton spandex, classic fit, short sleeve with color: white, and size: small, and price lower than 80.00 dollars",
    "Find me machine washable living room sets for living room with color: beige, and price lower than 60.00 dollars",
    "Find me wash cold, machine wash women's suiting & blazers with button closure, polyester spandex, long sleeve with color: cream | off white, and size: x-large, and price lower than 60.00 dollars",
    "Find me machine washable window coverings for living room with color: dusty blush, and size: 52\"w x 84\"l, and price lower than 40.00 dollars",
    "Find me butt lifting, light weight women's shorts with high waist, tummy control with color: blue, and size: small, and price lower than 50.00 dollars",
    "Find me quick drying, moisture wicking women's activewear with long sleeve with color: b-grey-thumbhole, and size: xx-large, and price lower than 40.00 dollars",
    "Find me day comfort, anti slip, non slip women's oxfords with high heel, closed toe, ankle strap, memory foam, rubber sole with color: black, and size: 6, and price lower than 60.00 dollars",
    "Find me space saving decorative mirrors for living room with color: accent bronze, and price lower than 180.00 dollars",
)

# WebShop's landing page, rendered the way WebshopEnvironmentManager renders it:
# format_obs keeps the ' [SEP] '-separated parts that FOLLOW the instruction (on
# the landing page that is "Search" alone), and format_avail_actions turns
# {'has_search_bar': True, 'clickables': ['search']} into these two lines. Both
# were read off a live WebShop env built with this repo's own env_kwargs, and
# tests/oci/test_slots.py holds this string against the manager's own output.
_WEBSHOP_LANDING_OBSERVATION = "'Search'"
_WEBSHOP_LANDING_ACTIONS = "'search[<your query>]',\n'click[search]',"


def game_key(gamefile) -> str:
    """The part of a gamefile path that is the same on every host.

    The data sits under a different root per host (``/home/...`` on one,
    ``/opt1/...`` on another), so a key that included the root would show the
    same game a different foreign prompt on each. Which prompt it is does not
    change the outcome -- any of them fails -- but a run that is meant to be
    reproducible elsewhere should draw the same text, so the key is the path
    from the split directory down: ``train/<task>/<trial>/game.tw-pddl``.
    """
    parts = [p for p in str(gamefile).replace("\\", "/").split("/") if p]
    return "/".join(parts[-4:])


# --------------------------------------------------------------------------- #
# the document slot's block, for each task that has one
# --------------------------------------------------------------------------- #

# The block, in the environment's own words, and the SAME wrapper for every task
# so the strip, the numbered-line reader and the progress line do not have to
# care which task produced it. ALFWorld's path comes from TextWorld's own
# walkthrough (env_manager._build_wrong_plan); the two below are built here.
PLAN_HEADER = "[Privileged Solution Path]"
PLAN_FOOTER = "[/Privileged Solution Path]"
PLAN_LEAD = (
    "THIS IS THE VERIFIED CORRECT SOLUTION PATH FOR THIS TASK. IT IS COMPLETE AND OPTIMAL.\n"
    "YOU MUST FOLLOW IT EXACTLY. At every step, take the action given by the next line\n"
    "of this path. DO NOT SEARCH ON YOUR OWN. DO NOT DEVIATE FROM IT. Any action that is\n"
    "not the next line of this path is wrong.\n"
    "\n"
    "The full path that solves this task:"
)


def render_document(lines) -> str:
    """The numbered block, or '' when there is no path to print.

    Numbered because that is what ``_block_lines`` reads back and what the
    per-turn progress line counts against; the wrapper is byte-identical across
    tasks so one strip handles all three.
    """
    lines = [str(line).strip() for line in (lines or []) if str(line).strip()]
    if not lines:
        return ""
    body = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    return f"{PLAN_HEADER}\n{PLAN_LEAD}\n{body}\n{PLAN_FOOTER}\n\n"


def webshop_document_lines(goal, query: str = "name") -> list:
    """The actions that buy this goal's own product, in WebShop's action grammar.

    WHAT MAKES IT CORRECT. The goal record names the product (``asin``) and the
    options the reward checks, so buying THAT product with THOSE options is the
    winning trajectory by construction. Replayed in a live env it reached reward
    1.0 on 286 of 300 goals; the other 14 top out at 0.80-0.86 because the reward's
    fuzzy option matching does not credit some of their own options, and no
    trajectory through the target can fix that. A document slot on those goals
    simply fails and its group keeps the reserve.

    ``query`` picks the first line, and the default is the leakier one on
    purpose. ``name`` searches the product's full title, which puts the target on
    results page 1 for 300 of 300 goals and wins 58 of 60 replayed documents;
    ``instruction`` searches the goal's own instruction text -- the string the
    student's prompt already contains -- and wins 52 of 60, losing six goals
    whose product never appears on page 1. The title is not a meaningful extra
    leak: the next line is ``click[<asin>]``, which names the product outright,
    and the block's whole purpose is to be the best document available, as
    ALFWorld's is TextWorld's own walkthrough.
    """
    if not goal:
        return []
    asin = str(goal.get("asin") or "").strip().lower()
    if not asin:
        return []
    if query == "name":
        q = goal.get("name") or goal.get("instruction_text") or goal.get("query")
    else:
        q = goal.get("instruction_text") or goal.get("query") or goal.get("name")
    if not q:
        return []
    options = goal.get("goal_options") or {}
    values = options.values() if isinstance(options, dict) else options
    lines = [f"search[{str(q).strip()}]", f"click[{asin}]"]
    lines += [f"click[{str(v).strip().lower()}]" for v in values if str(v).strip()]
    lines.append("click[buy now]")
    return lines


def search_document_lines(question, target) -> list:
    """One grounded search, then the answer, in the Search task's action grammar.

    THE SECOND LINE IS THE LEAK AND IT IS WHAT MAKES THE RESCUE CERTAIN: the
    reward is exact match on ``<answer>``, so a document that only proposed a
    query would rescue nothing on the questions this slot exists for. The first
    line is what keeps the trajectory honest -- searching the question verbatim
    returns passages containing the answer for 65 of 100 nq and 55 of 96
    hotpotqa questions, so on those the answer is supported by the context the
    student can see rather than produced from nowhere.

    ``target`` may be a string, a list, or a numpy array of accepted answers;
    the reward accepts any of them, so the first is printed.
    """
    if question is None or target is None:
        return []
    if isinstance(target, dict):
        target = target.get("target")
    if isinstance(target, (list, tuple)) or hasattr(target, "tolist"):
        target = list(target)
        target = target[0] if target else None
    q, a = str(question or "").strip(), str(target or "").strip()
    if not q or not a:
        return []
    return [f"<search> {q} </search>", f"<answer> {a} </answer>"]


def foreign_prompt(task: str, key) -> str:
    """The turn-0 prompt of another task, verbatim, for the slot that must fail.

    ``key`` (the gamefile) picks which one, through a stable hash of its
    host-independent part (see ``game_key``) rather than ``hash()``, whose salt
    differs per process -- a rollout worker and a test would otherwise disagree
    about what a game was shown.
    """
    if task not in FOREIGN_TASKS:
        raise ValueError(f"foreign_task={task!r}; expected one of {FOREIGN_TASKS}")
    pool = WEBSHOP_TRAIN_INSTRUCTIONS
    idx = zlib.crc32(game_key(key).encode("utf-8")) % len(pool)
    return WEBSHOP_TEMPLATE_NO_HIS.format(
        task_description=pool[idx],
        current_observation=_WEBSHOP_LANDING_OBSERVATION,
        available_actions=_WEBSHOP_LANDING_ACTIONS,
    )
