"""CPU test: the correct-document for WebShop and Search, and the one wrapper.

WHY THE WRAPPER IS SHARED. The document slot's machinery reads the block back --
``_block_lines`` for the per-turn progress line, the render diff for the strip --
and all of it assumes ALFWorld's shape: a header, the lead, numbered lines, a
footer. A second task that printed its own shape would be shown a block nothing
downstream could count or remove, and the failure would look like "the student
ignored the document" rather than like a format mismatch. So the two builders
here produce LINES and the shared renderer produces the block.

WHAT MAKES EACH DOCUMENT CORRECT, and where it stops:
  * WebShop -- the goal record names the product and the options the reward's
    matcher checks, so buying that product with those options wins by
    construction. Measured by replay: 286 of 300 goals reach reward 1.0, and the
    other 14 cannot be won through the target at all.
  * Search -- the reward is exact match on <answer>, so the answer line makes the
    rescue certain. The search line in front of it is what keeps the answer
    supported by context the student can see (65/100 nq, 55/96 hotpotqa).
"""
import os, sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

import agent_system.environments.env_manager as em  # noqa: E402
from agent_system.environments import oci_layout as ol  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


GOAL = {
    "asin": "B09HX5CD2D",
    "name": "CSU Cleveland State University Vikings Fleece Drawstring Shorts Heather Charcoal",
    "query": "men's shorts",
    "instruction_text": "i need some draw string shorts that are official cleveland university, and price lower than 60.00 dollars",
    "goal_options": {"color": "Heather Charcoal", "size": "Small"},
    "price_upper": 60.0,
}

# --- 1. WebShop: the actions that buy this goal's own product ----------------
lines = ol.webshop_document_lines(GOAL)
check(lines == [f"search[{GOAL['name']}]", "click[b09hx5cd2d]",
                "click[heather charcoal]", "click[small]", "click[buy now]"],
      f"search, click the product, click each option, buy ({len(lines)} lines)")
check(all(x == x.lower() for x in lines[1:]),
      "the clickables are lowercased, which is how the env lists them")
check(ol.webshop_document_lines(GOAL, query="instruction")[0] == f"search[{GOAL['instruction_text']}]",
      "query='instruction' searches the student's own prompt text instead -- 52/60 replayed "
      "documents win against 58/60 for the title, because six products never reach page 1")
human = dict(GOAL, goal_options=["heather charcoal", "small"])
check(ol.webshop_document_lines(human) == lines,
      "goal_options as a LIST (human goals) gives the same lines as a dict (synthetic goals)")
check(ol.webshop_document_lines(dict(GOAL, goal_options={})) ==
      [f"search[{GOAL['name']}]", "click[b09hx5cd2d]", "click[buy now]"],
      "a goal with no options is search, click, buy")
check(ol.webshop_document_lines(None) == [] and ol.webshop_document_lines({"name": "x"}) == [],
      "no goal, or a goal with no asin: no document rather than a broken one")
check(ol.webshop_document_lines(dict(GOAL, name=None))[0] == f"search[{GOAL['instruction_text']}]"
      and ol.webshop_document_lines(dict(GOAL, name=None, instruction_text=None))[0]
      == f"search[{GOAL['query']}]",
      "a goal with no title falls back through instruction -> query")

# --- 2. Search: one grounded search, then the answer -------------------------
check(ol.search_document_lines("who wrote hamlet", "William Shakespeare")
      == ["<search> who wrote hamlet </search>", "<answer> William Shakespeare </answer>"],
      "the two lines are in the task's own <search>/<answer> grammar")
for label, target in (("list", ["William Shakespeare", "Shakespeare"]),
                      ("dict from the parquet", {"target": ["William Shakespeare"]}),
                      ("numpy array", np.array(["William Shakespeare", "Shakespeare"], dtype=object))):
    got = ol.search_document_lines("who wrote hamlet", target)
    check(got and got[1] == "<answer> William Shakespeare </answer>",
          f"a target given as a {label} prints the first accepted answer")
check(ol.search_document_lines("q", None) == [] and ol.search_document_lines(None, "a") == []
      and ol.search_document_lines("q", []) == [],
      "a missing question or answer gives no document")

# --- 3. one wrapper for every task -------------------------------------------
block = ol.render_document(lines)
check(block.startswith(ol.PLAN_HEADER) and block.rstrip().endswith(ol.PLAN_FOOTER)
      and ol.PLAN_LEAD in block,
      "the block carries the same header, lead and footer ALFWorld's does")
check(em._PLAN_HEADER is ol.PLAN_HEADER and em._PLAN_LEAD is ol.PLAN_LEAD,
      "and env_manager prints THAT text rather than a second copy of it")
check(em._block_lines(block) == lines,
      "the numbered-line reader gets back exactly the lines it was given -- this is what "
      "the per-turn progress line counts against")
check(ol.render_document([]) == "" and ol.render_document(None) == "",
      "no lines, no block (an empty wrapper would be a document that says nothing)")
check(em._block_lines(ol.render_document(ol.search_document_lines("q", "a")))
      == ["<search> q </search>", "<answer> a </answer>"],
      "Search's angle brackets survive the numbering and the read-back")

# --- 4. the managers' accessors ----------------------------------------------
wm = em.WebshopEnvironmentManager.__new__(em.WebshopEnvironmentManager)
wm.goals = [GOAL, None]
check(em._block_lines(wm.document_block(0)) == lines and wm.document_block(1) == "",
      "the WebShop manager builds the block from the goal its worker handed back")
check(wm.document_block(5) == "", "and asking past the end is '' rather than an exception")

sm = em.SearchEnvironmentManager.__new__(em.SearchEnvironmentManager)
sm.problems = [{"question": "who wrote hamlet", "ground_truth": {"target": ["William Shakespeare"]}}, {}]
check(em._block_lines(sm.document_block(0))[1] == "<answer> William Shakespeare </answer>"
      and sm.document_block(1) == "",
      "the Search manager builds the block from the reset kwargs")

am = em.AlfWorldEnvironmentManager.__new__(em.AlfWorldEnvironmentManager)
am.gamefile = [None]
check(am.document_block(0) == "" and am.document_block(3) == "",
      "the ALFWorld accessor has the same signature and the same empty answer")

import glob, json  # noqa: E402
_games = sorted(glob.glob(os.path.expanduser(
    "~/data/alfworld/json_2.1.1/train/pick_heat_then_place_in_recep*/*/game.tw-pddl")))
_gf = next((g for g in _games if len(json.load(open(g)).get("walkthrough") or []) >= 5), None)
if _gf is None:
    print("  SKIP  no alfworld game files on this host; the third accessor is unchecked")
else:
    em._WRONG_PLAN_CACHE.clear()
    am.gamefile = [_gf]
    check(em._block_lines(am.document_block(0)) == json.load(open(_gf))["walkthrough"],
          "and on a real game it returns TextWorld's own walkthrough, line for line")


def test_documents():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
