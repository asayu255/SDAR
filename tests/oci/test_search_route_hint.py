"""The Search route document that never names the answer (search_doc=route_hint).

WHY IT EXISTS. expert_flow prints a verified route AND the answer under the rule that a
returned result must carry it first; on the step-150 probe 46% of its rescue rows on stuck
groups wrote the answer before any result carried it. route_hint prints the same kind of
route and never the answer: what to look up, and -- once a result carries the answer --
which result it is, never what it says.

WHAT IS CHECKED HERE, with no model and no retriever:
  1 the block          the verified queries in order; no answer in the block or its lead
  2 the refusals       a query naming an answer the question does not name; yes/no; no route
  3 the position       which "Doc k:" of a returned block carries the answer
  4 the line           names the next query, then the search and the Doc, never the answer
  5 the live manager   only the document slot sees it; the pointer moves; the position is
                       the one the row actually received; expert_flow is untouched
  6 the probe dump     the record carries where the answer first came back
"""
import json, os, sys, tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

from omegaconf import OmegaConf  # noqa: E402

import agent_system.environments.oci_layout as ol  # noqa: E402
from agent_system.environments.env_manager import SearchEnvironmentManager  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


QUESTION = "what is the closest airport to white sulphur springs west virginia?"
ANSWER = "Greenbrier Valley Airport"
TARGET = {"target": [ANSWER]}
ROUTE = ["closest airport Lewisburg West Virginia", "Greenbrier County airport"]
# The environment's own format: a JSON string inside <information>, escaped newlines.
NO_EVIDENCE = ('<information>{"result": "Doc 1: \\"Lewisburg, West Virginia\\"\\nLewisburg is a city '
               'in Greenbrier County.\\nDoc 2: \\"White Sulphur Springs\\"\\nA resort town."}</information>')
EVIDENCE = ('<information>{"result": "Doc 1: \\"Lewisburg, West Virginia\\"\\nLewisburg is a city '
            'in Greenbrier County.\\nDoc 2: \\"Greenbrier Valley Airport\\"\\nGreenbrier Valley Airport '
            'is a public airport three miles north of Lewisburg.\\nDoc 3: \\"Greenbrier County\\"\\n'
            'A county."}</information>')

FLOW_FILE = os.path.join(tempfile.gettempdir(), f"route_hints_{os.getpid()}.json")
json.dump({"flows": {
    "0": {"question": QUESTION, "answers": [ANSWER], "hit": True, "queries": ROUTE},
    "1": {"question": "a route that did not reach the answer", "answers": ["x"], "hit": False,
          "queries": ["something"]},
    "2": {"question": "a route that names the answer", "answers": [ANSWER], "hit": True,
          "queries": [f"{ANSWER} runway"]},
    "3": {"question": "which opened first, Greenbrier Valley Airport or Yeager Airport?",
          "answers": [ANSWER], "hit": True, "queries": ["Greenbrier Valley Airport opened"]},
    "4": {"question": "is the airport public?", "answers": ["yes"], "hit": True,
          "queries": ["Greenbrier Valley Airport public"]},
}}, open(FLOW_FILE, "w"))

try:
    ol._SEARCH_FLOWS.clear()
    print("1. the block")
    lines = ol.search_route_hint_document_lines(QUESTION, TARGET, FLOW_FILE)
    check(lines == [f"<search> {q} </search>" for q in ROUTE], "the verified queries in order")
    block = ol.render_document(lines, lead=ol.SEARCH_ROUTE_HINT_LEAD)
    check(ANSWER not in block and "<answer>" not in block and "CORRECT ANSWER IS" not in block,
          "neither the block nor its lead names the answer, and no <answer> action is printed")
    check(block.startswith(ol.PLAN_HEADER) and "\n1. <search>" in block and "\n2. <search>" in block,
          "same wrapper and numbering as the other documents")

    print("2. the refusals")
    check(ol.search_route_hint_document_lines("a route that names the answer", TARGET, FLOW_FILE) == [],
          "a query that names an answer the question does not name is refused")
    check(ol.search_route_hint_document_lines(
        "which opened first, Greenbrier Valley Airport or Yeager Airport?", TARGET, FLOW_FILE)
        == ["<search> Greenbrier Valley Airport opened </search>"],
          "one that names an answer the question already names is kept (it tells nothing new)")
    check(ol.search_flow_document_lines(
        "which opened first, Greenbrier Valley Airport or Yeager Airport?", TARGET, FLOW_FILE) == [],
          "while expert_flow keeps its stricter rule unchanged")
    check(ol.search_route_hint_document_lines("is the airport public?", {"target": ["yes"]}, FLOW_FILE)
          == [], "no document for a yes/no question")
    check(ol.search_route_hint_document_lines("a route that did not reach the answer", {"target": ["x"]},
                                              FLOW_FILE) == []
          and ol.search_route_hint_document_lines("not in the file", TARGET, FLOW_FILE) == [],
          "none for a route that did not hit, or a question without one")

    print("3. the position")
    check(ol.evidence_position(EVIDENCE, TARGET, QUESTION) == 2, "Doc 2 carries it")
    check(ol.evidence_position(NO_EVIDENCE, TARGET, QUESTION) == 0, "no Doc does: 0")
    check(ol.evidence_in_text(EVIDENCE, TARGET, QUESTION)
          and not ol.evidence_in_text(NO_EVIDENCE, TARGET, QUESTION),
          "and the block-level test agrees on both")

    print("4. the line")
    L = ol.search_route_hint_document_lines(QUESTION, TARGET, FLOW_FILE)
    first = ol.search_route_line(L, 0, False)
    check("Your next action is search 1 of 2: <search> closest airport Lewisburg" in first,
          "at the start it names the first query")
    found = ol.search_route_line(L, 1, True, where=(1, 2))
    check("Doc 2 returned by your search 1 contains the answer" in found and ANSWER not in found,
          "once a result carries it: which search, which Doc -- never what it says")
    check("A result returned by your search 1 contains" in ol.search_route_line(L, 1, True, where=(1, 0)),
          "an unsplittable block names the search alone")
    check(ol.search_route_line(L, 1, True) == ol.search_route_line(L, 1, True, where=None)
          and "Now write it inside <answer> </answer>" in ol.search_route_line(L, 1, True),
          "without a position the line is expert_flow's, byte for byte")

    print("5. the live manager")

    class _Envs:
        """Canned results: rows in evidence_for get the answer back on their second search."""
        is_train = True

        def __init__(self, n, group_n, evidence_for):
            self.n, self.group_n, self.evidence_for, self.turns = n, group_n, set(evidence_for), 0

        def reset(self, kwargs=None):
            return [QUESTION] * self.n, [{} for _ in range(self.n)]

        def step(self, actions):
            self.turns += 1
            obs, rewards, dones, infos = [], [], [], []
            for i, act in enumerate(actions):
                answered = "<answer>" in str(act)
                searched = "<search>" in str(act)
                ev = i in self.evidence_for and self.turns >= 2 and searched
                obs.append("" if answered else (EVIDENCE if ev else NO_EVIDENCE))
                won = 1.0 if (answered and ANSWER in str(act)) else 0.0
                rewards.append(won)
                dones.append(answered or self.turns >= 4)
                infos.append({"won": won, "tool_calling": searched and not answered,
                              "tool_input": [str(act)] if searched and not answered else [None]})
            return obs, rewards, dones, infos

    def manager(group_n, evidence_for, search_doc, search_doc_b="none"):
        config = OmegaConf.create({
            "env": {"history_length": 4, "rollout": {"n": group_n}},
            "algorithm": {"oci_slots": {"enable": True, "tasks": ["search"], "search_doc": search_doc,
                                        "search_doc_b": search_doc_b, "search_flow_path": FLOW_FILE,
                                        "doc_mode": "walkthrough_stepwise", "foreign_task": "webshop"},
                          "oci_rank": {"enable": False}},
        })
        return SearchEnvironmentManager(_Envs(group_n, group_n, evidence_for),
                                        lambda acts: (list(acts), [1] * len(acts)), config)

    KW = [{"question": QUESTION, "ground_truth": TARGET}]
    check([ol.role_for_slot(i, 10, foreign=False) for i in range(10)]
          == [ol.ROLE_PLAIN] * 8 + [ol.ROLE_RESERVE, ol.ROLE_DOC],
          "ten slots without a second document: eight plain, one reserve (a fresh plain baseline), "
          "one document")
    dump = tempfile.mkdtemp(prefix="route_hint_dump_")
    os.environ["SEARCH_PROBE_DUMP"] = dump
    m = manager(10, evidence_for={0, 9}, search_doc="route_hint")
    t0 = m.reset(KW * 10)[0]["text"]
    check("Your next action is search 1 of 2: <search> closest airport Lewisburg West Virginia" in t0[9]
          and ANSWER not in t0[9], "the document row starts pointed at query 1, with no answer anywhere")
    check(all(ol.PLAN_HEADER not in t0[i] for i in range(9)), "the eight plain rows and the reserve see nothing")
    acts = ["<think> go </think><search> white sulphur springs airport </search>"] * 10
    acts[9] = "<think> step 1 </think><search> closest airport Lewisburg West Virginia </search>"
    t1 = m.step(acts)[0]["text"]
    check("Search 1 of 2 is done. Your next action is search 2 of 2: <search> Greenbrier County airport"
          in t1[9], "after running query 1 the pointer names query 2")
    acts[9] = "<think> step 2 </think><search> Greenbrier County airport </search>"
    t2 = m.step(acts)[0]["text"]
    guide = t2[9].split(ol.PLAN_FOOTER)[0] + "".join(
        ln for ln in t2[9].splitlines(True) if "[Privileged Solution Path progress]" in ln)
    check("Doc 2 returned by your search 2 contains the answer" in t2[9],
          "the answer came back in Doc 2 of the row's SECOND search, and the line says exactly that")
    check(ANSWER not in guide, "the privileged text still never names the answer")
    check(m._evidence_where[9] == (2, 2) and m._evidence_where[0] == (2, 2) and m._evidence_where[1] is None,
          "the position is tracked for every row that received it, from the row's own searches")
    acts[9] = f"<think> Doc 2 says it </think><answer> {ANSWER} </answer>"
    m.step(acts)
    m._probe_flush()
    recs = [json.loads(line) for f in os.listdir(dump) for line in open(os.path.join(dump, f))]
    doc = [r for r in recs if r["role"] == ol.ROLE_DOC]
    check(len(doc) == 1 and doc[0]["variant"] == "route_hint" and doc[0]["has_document"]
          and doc[0]["evidence_where"] == [2, 2] and doc[0]["won"] == 1.0 and not doc[0]["answer_early"],
          "the probe record carries the variant, the position, the win, and a kept rule")
    os.environ.pop("SEARCH_PROBE_DUMP", None)

    print("6. route_line: the pointer line alone")
    m3 = manager(10, evidence_for={9}, search_doc="route_line")
    t5 = m3.reset(KW * 10)[0]["text"]
    check(ol.PLAN_HEADER not in t5[9] and "THIS IS A SEARCH ROUTE" not in t5[9]
          and "Your next action is search 1 of 2: <search> closest airport Lewisburg West Virginia" in t5[9]
          and t5[9].index("[Privileged Solution Path progress]") > t5[9].index("Your question:"),
          "no block in front of the prompt; the pointer line sits inside it, after the question")
    check(m3.document_block(9, slot="a") != "" and ANSWER not in t5[9],
          "the row still counts as documented, and nothing names the answer")
    acts3 = ["<think> go </think><search> white sulphur springs airport </search>"] * 10
    acts3[9] = "<search> closest airport Lewisburg West Virginia </search>"
    t6 = m3.step(acts3)[0]["text"]
    acts3[9] = "<search> Greenbrier County airport </search>"
    t7 = m3.step(acts3)[0]["text"]
    check("Your next action is search 2 of 2" in t6[9] and "Doc 2 returned by your search 2 contains the answer"
          in t7[9], "the pointer moves and names the place exactly as route_hint's does")

    print("7. sdar_skills: SDAR's own skill text")
    from verl.trainer.ppo.rlsd_utils import SkillProvider
    sp = SkillProvider(skills_dir=os.path.join(REPO, "skills", "search"))
    for ds, q in (("nq", QUESTION), ("hotpotqa", "Which band was founded first, Hole or The Wolfhounds?")):
        mine = ol.sdar_search_skill_text(ds, q, os.path.join(REPO, "skills", "search"))
        check(mine and mine == sp.get_privileged_info_from_data_source(ds, q),
              f"{ds}: the same skill text SDAR's SkillProvider gives")
    check("### TASK: direct_retrieval" in ol.sdar_search_skill_text("nq", QUESTION)
          and "### TASK: multi_hop_reasoning" in ol.sdar_search_skill_text("hotpotqa", "x"),
          "nq gets direct_retrieval, hotpotqa multi_hop_reasoning, both after the general skills")
    m4 = manager(10, evidence_for={9}, search_doc="sdar_skills")
    t8 = m4.reset([{"question": QUESTION, "ground_truth": TARGET, "data_source": "nq"}] * 10)[0]["text"]
    check(t8[9].startswith(ol.SDAR_SKILL_HEADER) and "### GENERAL SKILLS ###" in t8[9]
          and "[Privileged Solution Path progress]" not in t8[9] and ANSWER not in t8[9],
          "the document row reads SDAR's header and skills, no pointer, no answer")
    check(all(ol.SDAR_SKILL_HEADER not in t8[i] for i in range(9)), "the plain rows and the reserve see nothing")

    print("8. the one-sentence nudges (doc_sentence / doc_sentence_b)")
    cfg8 = OmegaConf.create({
        "env": {"history_length": 4, "rollout": {"n": 10}},
        "algorithm": {"oci_slots": {"enable": True, "tasks": ["search"], "search_doc": "route_hint",
                                    "search_doc_b": "route_hint", "search_flow_path": FLOW_FILE,
                                    "doc_sentence": "explore", "doc_sentence_b": "none",
                                    "doc_mode": "walkthrough_stepwise", "foreign_task": "webshop"},
                      "oci_rank": {"enable": False}},
    })
    m5 = SearchEnvironmentManager(_Envs(10, 10, {8, 9}), lambda acts: (list(acts), [1] * len(acts)), cfg8)
    t9 = m5.reset(KW * 10)[0]["text"]
    EXPLORE = ol.DOC_SENTENCES["search"]["explore"]
    blk_a, blk_b = m5.document_block(9, slot="a"), m5.document_block(8, slot="b")
    check(EXPLORE in t9[9] and EXPLORE not in t9[8] and all(EXPLORE not in t9[i] for i in range(8)),
          "only the document row carries the exploration sentence; the second document row and the rest do not")
    check(blk_a.index(EXPLORE) > blk_a.index("2. <search>") and blk_a.rstrip().endswith(ol.PLAN_FOOTER)
          and ol.render_document(ol.search_route_hint_document_lines(QUESTION, TARGET, FLOW_FILE),
                                 lead=ol.SEARCH_ROUTE_HINT_LEAD) == blk_b,
          "it is the block's last line, after the numbered queries; the other row's block is unchanged")
    import re as _re
    check(_re.findall(r"^\d+\. (.+)$", blk_a, flags=_re.M) == _re.findall(r"^\d+\. (.+)$", blk_b, flags=_re.M),
          "the numbered lines the pointer walks are identical with and without the sentence")
    check("Your next action is search 1 of 2" in t9[9] and "Your next action is search 1 of 2" in t9[8],
          "both rows still get the pointer line")
    check(ol.with_doc_sentence("", EXPLORE) == "" and ol.with_doc_sentence(blk_b, "") == blk_b,
          "no block or no sentence: unchanged")
    try:
        ol.doc_sentence_key(OmegaConf.create({"algorithm": {"oci_slots": {"doc_sentence": "exploit"}}}))
        check(False, "an unknown sentence key is refused")
    except ValueError:
        check(True, "an unknown sentence key is refused")
    check(set(ol.DOC_SENTENCES) == {"alfworld", "webshop", "search"}
          and all(ANSWER not in v for d in ol.DOC_SENTENCES.values() for v in d.values()),
          "one explore and one efficiency sentence per task")
    check(all(set(d) == {"explore", "efficiency", "progress"} for d in ol.DOC_SENTENCES.values())
          and all("you have not" in d["progress"] for d in ol.DOC_SENTENCES.values()),
          "and the revised stuck-group sentence (progress) for every task")

    m2 = manager(10, evidence_for={8, 9}, search_doc="answer_rule", search_doc_b="expert_flow")
    t3 = m2.reset(KW * 10)[0]["text"]
    acts2 = ["<think> go </think><search> closest airport Lewisburg West Virginia </search>"] * 10
    m2.step(acts2)
    t4 = m2.step(acts2)[0]["text"]
    check("A result you received contains the answer. Now write it inside <answer> </answer>" in t4[8]
          and ANSWER in t3[8], "expert_flow keeps its own answer and its own line")
finally:
    os.remove(FLOW_FILE)
    ol._SEARCH_FLOWS.clear()

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
