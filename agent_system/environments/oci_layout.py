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
    slot  group_n-1        foreign    shown a prompt it cannot win on: another
                                      task's (foreign_task=webshop) or this game
                                      with another game's goal (foreign_task=alfworld)

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

WHAT THE FIRST RUN SAID ABOUT THAT CHOICE (2026-09-15, alfworld-only, 150
steps). The foreign row is re-scored on the observation it stood in for, and a
WebShop response under an alfworld observation has rho = e^-40 (median of the
per-token log-ratio -39.7 on the steps where only foreign rows were injected).
The shaped coefficient is zero there, so the row's own tokens trained NOTHING;
its whole effect was to sit in the group statistic as a zero-return sample --
which a virtual floor (algorithm.oci_floor) gives without generating fifty
turns or paying the extra forward. It failed 352/352 and taught nothing.

THE SECOND CHOICE: THE SAME TASK, ANOTHER GAME'S GOAL. ``foreign_task=alfworld``
keeps this game's observation and swaps only the goal sentence for another
train game's, chosen with a different target object. The student then pursues
the wrong goal in the right room, with the right vocabulary: what it writes is
an ordinary alfworld response, so under the real goal most of its tokens have
rho near 1 and a clipped policy-gradient term can suppress exactly the actions
that are plausible under either goal. Whether it fails often enough, and what
its rho is, are what run_alfworld_oci_slots_probe_qwen3.sh measures.
"""

import html
import re
import unicodedata
import zlib

from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS

# What a row's slot is for. Travels as the `oci_role` column; ROLE_NONE is every
# row of a run without the layout, and every row of a task it does not cover.
ROLE_NONE, ROLE_PLAIN, ROLE_RESERVE, ROLE_DOC, ROLE_FOREIGN = 0, 1, 2, 3, 4
# A SECOND DOCUMENT ROW, FOR MEASUREMENT ONLY. Two rescue documents differ in one
# thing -- whether the answer is printed -- and the honest comparison is on the
# SAME group: same question, same eight siblings, same sampling. This row is
# never kept for training; the probe reads it and the selection drops it.
ROLE_DOC_B = 5
ROLE_NAMES = {ROLE_NONE: "none", ROLE_PLAIN: "plain", ROLE_RESERVE: "reserve",
              ROLE_DOC: "document", ROLE_FOREIGN: "foreign"}

# Observation-dict keys, alongside OCI_PREFIX_KEY in env_manager. They travel on
# the observation because the multitask merge reorders every key from a manager's
# local slot order into global row order -- a side channel keyed by env slot and
# read by the global row index marked one group of fifteen and passed its own
# "at least one is marked" assert.
OCI_ROLE_KEY = "oci_role"
OCI_PLAIN_KEY = "oci_plain"
# The render THIS turn would have had with the instance's document in front of
# the observation, for the rank scorer (algorithm.oci_rank). It travels beside
# the other two for the same reason: the multitask merge reorders every key.
OCI_DOC_KEY = "oci_doc"

# What the document slot may print. Both print the game's own walkthrough, which
# replays 30/30 in this environment; the stepwise variant adds the one line that
# says which step is owed, and that line is the difference between 35% and 88-95%
# of stuck groups rescued.
DOC_MODES = ("walkthrough", "walkthrough_stepwise")

# What the foreign slot is shown. "webshop": another task's turn-0 prompt in
# place of the observation (the first run). "alfworld": THIS game's observation
# with ANOTHER game's goal sentence -- see the paragraph above on why the first
# choice trained nothing.
FOREIGN_TASKS = ("webshop", "alfworld")


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


def rank_cfg(config):
    """``algorithm.oci_rank`` off whatever config object the caller holds, or None."""
    if config is None:
        return None
    try:
        return config.algorithm.get("oci_rank", None)
    except Exception:
        return None


def rank_on(config) -> bool:
    cfg = rank_cfg(config)
    return bool(cfg is not None and cfg.get("enable", False))


def rank_tasks(config):
    """Tasks whose rows carry the document-conditioned render."""
    cfg = rank_cfg(config) or {}
    return tuple(cfg.get("tasks", ["alfworld"]) or ["alfworld"])


def rank_self_check_rows(config) -> int:
    """Rows per batch the rank probe re-scores to check its privileged score;
    0 (off) unless the switch is on. When > 0 the rollout also stores each row's
    document prompt text, which only the self-check reads."""
    if not rank_on(config):
        return 0
    cfg = rank_cfg(config) or {}
    try:
        return max(0, int(cfg.get("self_check_rows", 0) or 0))
    except (TypeError, ValueError):
        return 0


def slot_tasks(config):
    """Tasks whose groups carry the layout. Others run plain rollouts only."""
    cfg = slots_cfg(config) or {}
    return tuple(cfg.get("tasks", ["alfworld"]) or ["alfworld"])


def has_foreign_slot(task) -> bool:
    """Whether this task's layout spends a slot on another task's prompt.

    The foreign slot exists to put a plausible FAILURE in a saturated group. For
    search there is nothing for it to wear: another task's prompt inside a search
    episode produces actions the projection refuses, so the row would spend its
    turns and hand the group what the virtual floor gives for free. Search's
    layout is eight ordinary rollouts and the document row, and nothing else.
    """
    return str(task) != "search"


def used_per_group(group_n: int, foreign: bool = True, second_doc: bool = False) -> int:
    """How many trajectories a group trains: everything but the special slots."""
    specials = (2 if foreign else 1) + (1 if second_doc else 0)
    return max(int(group_n) - specials, 0)


def role_for_slot(slot: int, group_n: int, foreign: bool = True, second_doc: bool = False) -> int:
    """What the slot at position ``slot`` of a group of ``group_n`` is for.

    Below four the layout does not exist: it needs two ordinary rollouts to read
    a verdict from and two more for the special slots (three without the foreign
    one, which search does not have -- see has_foreign_slot). ``second_doc`` adds
    the measurement-only variant row above (ROLE_DOC_B).
    """
    g = int(group_n)
    specials = (2 if foreign else 1) + (1 if second_doc else 0)
    if g < specials + 2:
        return ROLE_NONE
    j = int(slot) % g
    if foreign and j == g - 1:
        return ROLE_FOREIGN
    last = g - 1 if not foreign else g - 2
    if j == last:
        return ROLE_DOC
    if second_doc and j == last - 1:
        return ROLE_DOC_B
    if j == last - (2 if second_doc else 1):
        return ROLE_RESERVE
    return ROLE_PLAIN


def slot_role(i: int, envs, config=None, foreign: bool = True, second_doc: bool = False) -> int:
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
    return role_for_slot(i, g, foreign=foreign, second_doc=second_doc)


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


def render_document(lines, lead: str = PLAN_LEAD) -> str:
    """The numbered block, or '' when there is no path to print.

    Numbered because that is what ``_block_lines`` reads back and what the
    per-turn progress line counts against; the wrapper is byte-identical across
    tasks so one strip handles all three. ``lead`` is the only part that varies:
    Search's rule document tells the slot to FIND the answer rather than to
    replay a path (see SEARCH_RULE_LEAD), and its default leaves every existing
    caller's bytes unchanged.
    """
    lines = [str(line).strip() for line in (lines or []) if str(line).strip()]
    if not lines:
        return ""
    body = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    return f"{PLAN_HEADER}\n{lead}\n{body}\n{PLAN_FOOTER}\n\n"


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


# --------------------------------------------------------------------------- #
# the search document that shows the answer AND makes finding it the condition
# --------------------------------------------------------------------------- #
#
# WHY A SECOND SEARCH DOCUMENT. ``search_document_lines`` above prints the answer
# and nothing stops the slot from writing it on turn one: Search's reward reads
# only the final <answer> string and never checks that a search happened, so an
# answer-only rollout always scores. What that rescue row then trains is "copy
# the answer out of the prompt", which the plain prompt never contains.
# ALFWorld and WebShop do not have this hole -- their environments refuse an
# action whose preconditions are unmet, so a document can only be replayed by
# actually executing every step.
#
# WHAT THIS MODE DOES. It shows the answer, and the run's own success test for
# the row becomes: the answer string must not appear in anything the row WRITES
# until a result it RECEIVED from <search> contains it. A row that writes it
# early is discarded, so the only trained rescue rows are ones that searched,
# read, and then answered -- with the answer present in the retrieved context,
# which is exactly the state the plain student is in when it succeeds.
SEARCH_RULE_LEAD = (
    "THIS IS THE VERIFIED CORRECT ANSWER FOR THIS QUESTION, AND THE RULE THAT MAKES IT\n"
    "COUNT. The answer is only earned if you FIND it: a result returned to you inside\n"
    "<information> </information> must contain it BEFORE you write it. Until that\n"
    "happens, DO NOT WRITE THE ANSWER ANYWHERE -- not in your thinking, not inside a\n"
    "<search> query. A trajectory that writes it early is thrown away and teaches\n"
    "nothing. Search, read what comes back, search again with a different query if it\n"
    "is not there, and answer once it is.\n"
    "\n"
    "The path that solves this task:"
)

# THE VARIANT THAT PRINTS NOTHING BUT THE VERDICT. Even under the rule, a row
# that has read the answer can write a query only someone who knows it would
# write ("president 1861 1865 assassinated" for Lincoln), and no string test
# catches that -- what such a row then teaches may not survive the plain prompt.
# This variant withholds the answer and keeps only the per-turn line saying
# whether a result has carried it, which is the direct counterpart of ALFWorld's
# progress pointer. It cannot help the row choose a query, which is exactly why
# the two are generated side by side on the same group.
SEARCH_PROGRESS_LEAD = (
    "YOU ARE BEING TOLD WHEN YOU HAVE FOUND THE ANSWER, NOT WHAT IT IS. A checker\n"
    "that knows the correct answer reads every result returned to you, and the\n"
    "progress line below says whether one of them contains it. Search, read the\n"
    "results, and search again with a different query until the line says a result\n"
    "carries the answer. Then answer with what that result gives.\n"
    "\n"
    "The path that solves this task:"
)

SEARCH_DOC_MODES = ("answer_only", "answer_rule", "progress_only")


def search_doc_mode(config, slot: str = "a") -> str:
    """Which Search document a document slot shows.

    ``answer_only`` is the historical block (query, then the answer) and stays
    the default so a rerun of an older arm is byte-identical. ``slot="b"`` reads
    the measurement-only second row, which is "none" unless a probe asks for it.
    """
    cfg = slots_cfg(config) or {}
    key, default = ("search_doc", "answer_only") if slot == "a" else ("search_doc_b", "none")
    mode = str(cfg.get(key, default) or default)
    allowed = SEARCH_DOC_MODES if slot == "a" else SEARCH_DOC_MODES + ("none",)
    if mode not in allowed:
        raise ValueError(f"algorithm.oci_slots.{key}={mode!r}; expected one of {allowed}")
    return mode


def has_second_doc(config) -> bool:
    """Whether the layout generates the measurement-only variant row."""
    return search_doc_mode(config, slot="b") != "none"


def answer_strings(target) -> list:
    """Every accepted answer, as stripped strings.

    The reward accepts any of them, so the rule has to watch all of them: a row
    that writes an alias early has still written the answer.
    """
    if target is None:
        return []
    if isinstance(target, dict):
        target = target.get("target")
    if isinstance(target, (list, tuple)) or hasattr(target, "tolist"):
        target = list(target) if not hasattr(target, "tolist") else list(target.tolist())
    else:
        target = [target]
    return [str(t).strip() for t in target if str(t).strip()]


def fold_text(s) -> str:
    """Lowercase word stream: accents decomposed away, entities undone.

    The corpus stores text decomposed (NFD: "Lo\\u0308w") while a dataset answer
    is composed, so a naive comparison misses every accented answer.
    """
    s = unicodedata.normalize("NFKD", html.unescape(str(s)))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()


def contains_answer(text, target) -> bool:
    """Whole-word containment of ANY accepted answer ("5" is not in "1950")."""
    hay = f" {fold_text(text)} "
    for a in answer_strings(target):
        folded = fold_text(a)
        if folded and f" {folded} " in hay:
            return True
    return False


# Words a question shares with almost any passage; dropped before asking whether
# a passage is about this question at all.
_QUESTION_STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or is was were are be been am "
    "what which who whom whose when where why how did do does done has have had "
    "it its this that these those as into about many much first last name named".split())


def question_words(question) -> set:
    """The content words of a question, folded."""
    return {w for w in fold_text(question).split()
            if w not in _QUESTION_STOPWORDS and len(w) > 2}


def is_numeric_answer(target) -> bool:
    """Every accepted answer is a bare number: "1931", "5", "1861 1865"."""
    answers = [fold_text(a) for a in answer_strings(target)]
    return bool(answers) and all(a and all(tok.isdigit() for tok in a.split()) for a in answers)


def evidence_in_text(text, target, question=None) -> bool:
    """Does this passage carry the answer, rather than merely the same characters?

    THE FALSE POSITIVE THIS GUARDS. A year or a small integer turns up in
    passages that have nothing to do with the question: in the 300-question
    sample, 21 of 103 annotated nq questions get a top-3 passage that holds the
    answer string while holding none of the annotated evidence. For a bare-number
    answer the passage must therefore also carry a content word of the question.
    Everything else is judged by the answer alone, which is what the rescue row's
    rule and the progress line read.
    """
    if not contains_answer(text, target):
        return False
    if question is None or not is_numeric_answer(target):
        return True
    words = question_words(question)
    if not words:
        return True
    return bool(words & set(fold_text(text).split()))


def is_yesno(target) -> bool:
    """Yes/no questions have no document in this mode.

    The rule is a string test, and "yes" appears in almost any passage, so the
    progress line would read "found" before the first search. HotpotQA's
    comparison questions are 8 of 156 in a seed-0 sample of the training data.
    """
    answers = [fold_text(a) for a in answer_strings(target)]
    return bool(answers) and all(a in ("yes", "no") for a in answers)


def search_rescue_document_lines(question, target, show_answer: bool = True) -> list:
    """The first query to run, and what to write once a result carries the answer.

    Line 1 is the question verbatim -- the query that returns a passage holding
    the answer for 79% of the annotated nq questions and 43% of the hotpotqa
    bridge ones, and it contains no part of the answer, so running it can never
    break the rule. Line 2 is the answer when ``show_answer``; without it the row
    is told only that a result will be recognised, which is the progress_only
    variant (SEARCH_PROGRESS_LEAD).
    """
    answers = answer_strings(target)
    q = str(question or "").strip()
    if not q or not answers or is_yesno(target):
        return []
    last = (f"<answer> {answers[0]} </answer>" if show_answer
            else "<answer> what that result gives </answer>")
    return [f"<search> {q} </search>", last]


def search_progress_line(found: bool) -> str:
    """Where the slot stands: has a result carried the answer yet?

    ALFWorld's pointer advances when the action taken IS the next line; here the
    environment's own returns decide, which is the same idea on the only state
    Search exposes. The line sits where the action is chosen, for the reason the
    walkthrough pointer does: a slot that cannot tell where it is stops
    following the block (35% rescued without the line, 88-95% with it).
    """
    if found:
        return ("[Privileged Solution Path progress] A result you received contains the answer. "
                "Do step 2 now: write it inside <answer> </answer>.\n\n")
    return ("[Privileged Solution Path progress] No result you have received contains the answer yet. "
            "Do step 1: search. Do not write the answer anywhere until a result carries it.\n\n")


def foreign_prompt(task: str, key) -> str:
    """The turn-0 prompt of another task, verbatim, for the slot that must fail.

    ``key`` (the gamefile) picks which one, through a stable hash of its
    host-independent part (see ``game_key``) rather than ``hash()``, whose salt
    differs per process -- a rollout worker and a test would otherwise disagree
    about what a game was shown.
    """
    if task not in FOREIGN_TASKS:
        raise ValueError(f"foreign_task={task!r}; expected one of {FOREIGN_TASKS}")
    if task == "alfworld":
        raise ValueError("foreign_task='alfworld' edits this game's own observation; "
                         "call alfworld_foreign_obs(obs, real_task, gamefile) instead")
    pool = WEBSHOP_TRAIN_INSTRUCTIONS
    idx = zlib.crc32(game_key(key).encode("utf-8")) % len(pool)
    return WEBSHOP_TEMPLATE_NO_HIS.format(
        task_description=pool[idx],
        current_observation=_WEBSHOP_LANDING_OBSERVATION,
        available_actions=_WEBSHOP_LANDING_ACTIONS,
    )


# --------------------------------------------------------------------------- #
# the foreign slot, second design: this game, another game's goal
# --------------------------------------------------------------------------- #

# Goal sentences exactly as the environment prints them after "Your task is
# to: ", read off 120 train games (110 distinct), with the target object each
# names. The object is what the chooser avoids: a goal that shares the real
# goal's object could be satisfied on the way ("put a clean mug in fridge"
# solves "put a mug in fridge"), and then the slot would not fail.
ALFWORLD_TRAIN_TASKS = (
    ('clean some cup and put it in shelf.', 'Cup'),
    ('clean some dishsponge and put it in cabinet.', 'DishSponge'),
    ('clean some fork and put it in sidetable.', 'Fork'),
    ('clean some kettle and put it in stoveburner.', 'Kettle'),
    ('clean some knife and put it in countertop.', 'Knife'),
    ('clean some lettuce and put it in fridge.', 'Lettuce'),
    ('clean some mug and put it in coffeemachine.', 'Mug'),
    ('clean some potato and put it in microwave.', 'Potato'),
    ('cool some bowl and put it in microwave.', 'Bowl'),
    ('cool some lettuce and put it in countertop.', 'Lettuce'),
    ('cool some mug and put it in coffeemachine.', 'Mug'),
    ('cool some pot and put it in cabinet.', 'Pot'),
    ('cool some tomato and put it in garbagecan.', 'Tomato'),
    ('cool some tomato and put it in microwave.', 'Tomato'),
    ('cool some winebottle and put it in cabinet.', 'WineBottle'),
    ('examine the alarmclock with the desklamp.', 'AlarmClock'),
    ('examine the cellphone with the desklamp.', 'CellPhone'),
    ('examine the pillow with the desklamp.', 'Pillow'),
    ('examine the watch with the desklamp.', 'Watch'),
    ('find two bowl and put them in coffeetable.', 'Bowl'),
    ('find two cellphone and put them in bed.', 'CellPhone'),
    ('find two cellphone and put them in dresser.', 'CellPhone'),
    ('find two cup and put them in countertop.', 'Cup'),
    ('find two lettuce and put them in fridge.', 'Lettuce'),
    ('find two newspaper and put them in armchair.', 'Newspaper'),
    ('find two remotecontrol and put them in ottoman.', 'RemoteControl'),
    ('find two saltshaker and put them in countertop.', 'SaltShaker'),
    ('find two saltshaker and put them in drawer.', 'SaltShaker'),
    ('find two soapbar and put them in shelf.', 'SoapBar'),
    ('find two soapbottle and put them in countertop.', 'SoapBottle'),
    ('find two soapbottle and put them in toilet.', 'SoapBottle'),
    ('find two spraybottle and put them in dresser.', 'SprayBottle'),
    ('find two statue and put them in dresser.', 'Statue'),
    ('find two tissuebox and put them in drawer.', 'TissueBox'),
    ('heat some apple and put it in garbagecan.', 'Apple'),
    ('heat some cup and put it in cabinet.', 'Cup'),
    ('heat some cup and put it in countertop.', 'Cup'),
    ('heat some egg and put it in fridge.', 'Egg'),
    ('heat some egg and put it in garbagecan.', 'Egg'),
    ('heat some mug and put it in cabinet.', 'Mug'),
    ('heat some mug and put it in coffeemachine.', 'Mug'),
    ('heat some plate and put it in shelf.', 'Plate'),
    ('heat some tomato and put it in countertop.', 'Tomato'),
    ('heat some tomato and put it in sidetable.', 'Tomato'),
    ('look at book under the desklamp.', 'Book'),
    ('look at laptop under the desklamp.', 'Laptop'),
    ('look at pillow under the desklamp.', 'Pillow'),
    ('put a alarmclock in sidetable.', 'AlarmClock'),
    ('put a book in sofa.', 'Book'),
    ('put a butterknife in drawer.', 'ButterKnife'),
    ('put a cellphone in desk.', 'CellPhone'),
    ('put a clean butterknife in countertop.', 'ButterKnife'),
    ('put a clean cloth in cart.', 'Cloth'),
    ('put a clean cloth in drawer.', 'Cloth'),
    ('put a clean egg in diningtable.', 'Egg'),
    ('put a clean fork in diningtable.', 'Fork'),
    ('put a clean fork in drawer.', 'Fork'),
    ('put a clean kettle in diningtable.', 'Kettle'),
    ('put a clean knife in diningtable.', 'Knife'),
    ('put a clean soapbar in garbagecan.', 'SoapBar'),
    ('put a clean soapbar in toilet.', 'SoapBar'),
    ('put a clean tomato in countertop.', 'Tomato'),
    ('put a cool apple in microwave.', 'Apple'),
    ('put a cool bowl in countertop.', 'Bowl'),
    ('put a cool bowl in shelf.', 'Bowl'),
    ('put a cool egg in microwave.', 'Egg'),
    ('put a cool lettuce in countertop.', 'Lettuce'),
    ('put a cool mug in cabinet.', 'Mug'),
    ('put a cool mug in coffeemachine.', 'Mug'),
    ('put a cool pan in countertop.', 'Pan'),
    ('put a cool pan in diningtable.', 'Pan'),
    ('put a cool pan in stoveburner.', 'Pan'),
    ('put a cool pot in stoveburner.', 'Pot'),
    ('put a cool potato in microwave.', 'Potato'),
    ('put a cool tomato in microwave.', 'Tomato'),
    ('put a creditcard in drawer.', 'CreditCard'),
    ('put a creditcard in shelf.', 'CreditCard'),
    ('put a dishsponge in cabinet.', 'DishSponge'),
    ('put a glassbottle in countertop.', 'Glassbottle'),
    ('put a hot apple in diningtable.', 'Apple'),
    ('put a hot cup in fridge.', 'Cup'),
    ('put a hot mug in cabinet.', 'Mug'),
    ('put a hot mug in coffeemachine.', 'Mug'),
    ('put a pen in sidetable.', 'Pen'),
    ('put a pencil in dresser.', 'Pencil'),
    ('put a remotecontrol in armchair.', 'RemoteControl'),
    ('put a soapbottle in garbagecan.', 'SoapBottle'),
    ('put a soapbottle in toilet.', 'SoapBottle'),
    ('put a statue in coffeetable.', 'Statue'),
    ('put a tissuebox in sidetable.', 'TissueBox'),
    ('put a toiletpaper in toiletpaperhanger.', 'ToiletPaper'),
    ('put a tomato in microwave.', 'Tomato'),
    ('put some book on sofa.', 'Book'),
    ('put some box on armchair.', 'Box'),
    ('put some candle on cabinet.', 'Candle'),
    ('put some cd on diningtable.', 'CD'),
    ('put some pen on shelf.', 'Pen'),
    ('put some pencil on shelf.', 'Pencil'),
    ('put some remotecontrol on armchair.', 'RemoteControl'),
    ('put some tissuebox on coffeetable.', 'TissueBox'),
    ('put some toiletpaper on toiletpaperhanger.', 'ToiletPaper'),
    ('put two creditcard in armchair.', 'CreditCard'),
    ('put two peppershaker in drawer.', 'PepperShaker'),
    ('put two peppershaker in shelf.', 'PepperShaker'),
    ('put two remotecontrol in sofa.', 'RemoteControl'),
    ('put two saltshaker in cabinet.', 'SaltShaker'),
    ('put two spraybottle in cabinet.', 'SprayBottle'),
    ('put two spraybottle in countertop.', 'SprayBottle'),
    ('put two toiletpaper in countertop.', 'ToiletPaper'),
    ('put two wateringcan in shelf.', 'WateringCan'),
)


def alfworld_goal_from_gamefile(gamefile):
    """``(task_type, object, receptacle)`` read off the game directory's name.

    ALFWorld names the directory ``<type>-<Object>-<movable>-<Receptacle>-<scene>``
    (``pick_two_obj_and_place-PepperShaker-None-Drawer-301``), and that name is
    the only place the target object is spelled the way the goal pool spells it.
    Empty strings when the path is not shaped that way.
    """
    parts = [p for p in str(gamefile or "").replace("\\", "/").split("/") if p]
    name = parts[-3] if len(parts) >= 3 else ""
    fields = name.split("-")
    if len(fields) < 5:
        return ("", "", "")
    return (fields[0], fields[1], fields[3])


def alfworld_foreign_task(gamefile, real_task: str) -> str:
    """Another train game's goal sentence for this game, with a different object.

    Deterministic in the game (crc32 of its host-independent key, as the WebShop
    chooser), so a game is shown the same wrong goal every time it is drawn and
    on every host. Goals naming the real target object are excluded even when the
    directory name could not be parsed, by the object's lowercase spelling in
    the sentence.
    """
    _, obj, _ = alfworld_goal_from_gamefile(gamefile)
    real = str(real_task or "").strip()
    low = obj.lower()
    pool = [s for s, o in ALFWORLD_TRAIN_TASKS
            if s != real and (not low or (o.lower() != low and low not in s))]
    if not pool:
        raise ValueError(f"no foreign goal left for {gamefile!r} (object {obj!r})")
    return pool[zlib.crc32(game_key(gamefile).encode("utf-8")) % len(pool)]


def alfworld_foreign_obs(obs: str, real_task: str, gamefile) -> str:
    """This turn's rendered observation with every mention of the real goal
    sentence replaced by the foreign one.

    Every mention, because the sentence appears both in the template's own
    "Your task is to:" line and inside the turn-0 observation the history
    quotes, and a slot that saw the real goal anywhere would be an ordinary
    rollout. Raises when the sentence is not there at all: the plain render is
    what the slot is re-scored on, and a render that did not change would give
    rho = 1 on a row that is in fact plain.
    """
    real = str(real_task or "").strip()
    if not real or real not in obs:
        raise ValueError("the real goal sentence is not in this observation; the foreign "
                         "slot cannot be built from it")
    return obs.replace(real, alfworld_foreign_task(gamefile, real))
