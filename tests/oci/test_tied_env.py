"""CPU test: the environment side of algorithm.tied_opsd (tied-group self-distillation).

WHAT MUST HOLD.
  1. Search's state pointer: a chain step is done when the name the next query uses has come back,
     whichever query brought it; a yes/no route owes the first page not seen; a found answer owes it.
  2. WebShop's state pointer: search bar -> the search line; the goal product listed -> its click;
     the goal's item page -> the first required option the session lacks, then buy; anything else
     -> mismatch (None).
  3. ALFWorld's manager in tied mode: every row carries both teacher renders (stuck side: progress
     sentence; saturated side: efficiency sentence; both with the short lead and the state pointer's
     line), a refused action does not move the line, a wrong object gives a mismatch (no line,
     tied_match 0), and the prompt the policy sees is untouched.
  4. Search's manager in tied mode, on a real verified route: the line moves when the bridge name
     comes back and names the result once the answer does; tied_evidence follows.
  5. The rollout's two edits with the real tokenizer: plain -> stuck render (the existing one), and
     stuck render -> saturated render (the new, small one); applying both reproduces the saturated
     render token for token.
"""
import glob, json, os, sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alf_sim import Sim, REPO  # noqa: E402,F401

from omegaconf import OmegaConf  # noqa: E402

from agent_system.environments import oci_layout as ol  # noqa: E402
from agent_system.environments.alf_pointer import norm, _GO, _TAKE  # noqa: E402
from agent_system.environments.search_pointer import SearchStatePointer  # noqa: E402
from agent_system.environments.webshop_pointer import WebshopStatePointer  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


PROGRESS = {t: ol.DOC_SENTENCES[t]["progress"] for t in ol.TIED_TASKS}
EFFICIENCY = {t: ol.DOC_SENTENCES[t]["efficiency"] for t in ol.TIED_TASKS}

print("1. search pointer")
Q = "What is the capital of the archipelago where Djoi\u00e8zi is located?"
L = ["<search> Djoi\u00e8zi </search>", "<search> Comoros </search>"]
sp = SearchStatePointer(L, Q)
check(sp.targets[0] == {"comoros"} and sp.ptr(False) == 0, "chain: step 1 fetches the name step 2 uses")
sp.observe("Doc 1: \"Djoi\u00e8zi\"\\nDjoi\u00e8zi is a village on Moh\u00e9li in the Comoros.", query="village Djoiezi")
check(sp.ptr(False) == 1, "the name came back from a paraphrased query: step 2 is owed")
check(sp.ptr(True) == 2, "a found answer owes the answer")
sp2 = SearchStatePointer(L, Q)
sp2.observe("Doc 1: \"Moheli\"\\nan island", query="Djoi\u00e8zi")
check(sp2.ptr(False) == 0, "the route's own query that brought nothing back does not count")
sp3 = SearchStatePointer(["<search> alpha beta gamma </search>", "<search> alpha beta </search>"], "q?")
sp3.observe("nothing", query="alpha beta gamma delta")
check(sp3.targets[0] == set() and sp3.ptr(False) == 1, "a next query with no new word: the old query rule")
yn = SearchStatePointer(["<search> James Agee </search>", "<search> Nelly Sachs </search>"],
                        "Have James Agee and Nelly Sachs both won a Nobel Prize in Literature?",
                        titles=["James Agee", "Nelly Sachs"])
check(yn.ptr(False, {"nelly sachs"}) == 0 and yn.ptr(False, {"james agee"}) == 1
      and yn.ptr(True, {"james agee", "nelly sachs"}) == 2, "yes/no: the first page not seen yet")

print("2. webshop pointer")
lines = ["search[i want a red large shirt]", "click[b0abc]", "click[red]", "click[large]", "click[buy now]"]
wp = WebshopStatePointer(lines, "B0ABC", {"color": "red", "size": "large"})
check(wp.owed({"has_search_bar": True, "clickables": []}, None) == (0, lines[0]), "search page: the search")
check(wp.owed({"has_search_bar": False, "clickables": ["b0abc", "b0zzz", "next >"]}, {}) == (1, lines[1]),
      "results listing the goal: its click")
item = {"has_search_bar": False, "clickables": ["red", "blue", "large", "small", "buy now", "< prev"]}
check(wp.owed(item, {"asin": "B0ABC", "options": {}}) == (2, "click[red]")
      and wp.owed(item, {"asin": "B0ABC", "options": {"color": "red"}}) == (3, "click[large]")
      and wp.owed(item, {"asin": "B0ABC", "options": {"color": "red", "size": "large"}}) == (4, "click[buy now]"),
      "goal item page: the first missing option, then buy")
check(wp.owed(item, {"asin": "B0ZZZ", "options": {}}) is None
      and wp.owed({"has_search_bar": False, "clickables": ["< prev", "b0zzz"]}, {"asin": ""}) is None
      and wp.owed({"has_search_bar": False, "clickables": ["< prev"]}, {"asin": "B0ABC"}) is None,
      "another product, a list without the goal, a sub page: mismatch")
check(wp.line(item, {"asin": "B0ABC", "options": {}}).startswith(
    "[Privileged Solution Path progress] Steps 1-2 of 5 are done. Your next action is step 3 of 5: click[red]"),
      "the line has the other tasks' wording")

print("3. alfworld manager, tied mode")
import agent_system.environments.env_manager as em  # noqa: E402

heat = sorted(glob.glob(os.path.expanduser(
    "~/data/alfworld/json_2.1.1/train/pick_heat_then_place_in_recep*/*/game.tw-pddl")))
gf = next((g for g in heat if len(json.load(open(g)).get("walkthrough") or []) >= 4
           and _GO.match(norm(json.load(open(g))["walkthrough"][0]))
           and _TAKE.match(norm(json.load(open(g))["walkthrough"][1]))), None)
alf_docs = None
if gf is None:
    print("SKIP: no heat game on this host")
else:
    walk = [norm(a) for a in json.load(open(gf))["walkthrough"]]
    N = 4

    class SimEnvs:
        group_n, is_train = N, True

        def __init__(self):
            self.sims = [Sim(walk, extra_objects={"mug 9": "countertop 9"}) for _ in range(N)]

        @property
        def get_admissible_commands(self):
            return [sorted(s.admissible()) for s in self.sims]

        def reset(self):
            obs = ["You are in the middle of a room.\n\nYour task is to: heat some apple and put it somewhere."] * N
            return obs, None, [{"extra.gamefile": gf, "won": False} for _ in range(N)]

        def step(self, actions):
            obs = [s.do(a) for s, a in zip(self.sims, actions)]
            return obs, None, [0.0] * N, [False] * N, [{"extra.gamefile": gf, "won": False} for _ in range(N)]

    cfg = OmegaConf.create({"env": {"history_length": 2, "rollout": {"n": N}},
                            "algorithm": {"tied_opsd": {"enable": True}}})
    m = em.AlfWorldEnvironmentManager(SimEnvs(), lambda acts, adm: (list(acts), [1] * len(acts)), cfg)
    o, _ = m.reset(kwargs=None)
    f, s_, mt = o[ol.OCI_DOC_KEY], o[ol.OCI_DOC_S_KEY], o[ol.TIED_MATCH_KEY]
    check(all(f) and all(s_) and mt == [1] * N, "every row carries both renders and is placed")
    check(PROGRESS["alfworld"] in f[0] and EFFICIENCY["alfworld"] not in f[0]
          and EFFICIENCY["alfworld"] in s_[0] and PROGRESS["alfworld"] not in s_[0]
          and ol.PLAN_LEAD_SHORT in f[0] and "YOU MUST FOLLOW IT EXACTLY" not in f[0],
          "stuck render: progress sentence; saturated render: efficiency sentence; short lead")
    line0 = f"Your next action is step 1 of {len(walk)}: {walk[0]}"
    check(line0 in f[0] and line0 in s_[0] and "[Privileged Solution Path" not in o["text"][0],
          "both renders carry the state pointer's line; the policy's prompt has no document")
    check(f[0].replace(ol.tied_block("alfworld", m.document_block(0), "f"), "") ==
          s_[0].replace(ol.tied_block("alfworld", m.document_block(0), "s"), ""),
          "the two renders differ only in the block")
    o1, *_ = m.step([walk[1], walk[0], "go to countertop 9", walk[0]])   # row 0: take typed from nowhere
    check(line0 in o1[ol.OCI_DOC_KEY][0]
          and f"Your next action is step 2 of {len(walk)}: {walk[1]}" in o1[ol.OCI_DOC_KEY][1],
          "a refused take leaves row 0 at step 1; row 1 arrived and owes the take")
    o2, *_ = m.step(["look", walk[1], "take mug 9 from countertop 9", "look"])
    check(o2[ol.TIED_MATCH_KEY][2] == 0 and "[Privileged Solution Path progress]" not in o2[ol.OCI_DOC_KEY][2]
          and o2[ol.TIED_MATCH_KEY][1] == 1,
          "row 2 holds a mug in an apple task: mismatch, no line; row 1 placed")
    alf_docs = (o2["text"][1], o2[ol.OCI_DOC_KEY][1], o2[ol.OCI_DOC_S_KEY][1])

print("4. search manager, tied mode, on a real route")
from agent_system.environments.env_manager import SearchEnvironmentManager  # noqa: E402

FLOW = "/opt1/ohara/data/qa_annotations/route_hints_4500/route_hints_v2.json"
if not os.path.exists(FLOW):
    print("SKIP: no route file on this host")
else:
    TARGET = {"target": ["Moroni"]}
    BRIDGE = "Doc 1: \"Djoi\u00e8zi\"\\nDjoi\u00e8zi is a village on the island of Moh\u00e9li in the Comoros."
    EVID = "Doc 1: \"Moh\u00e9li\"\\nx\\nDoc 2: \"Comoros\"\\nThe capital of the Comoros is Moroni."

    class _Envs:
        is_train, group_n = True, 2

        def __init__(self):
            self.turn = 0

        def reset(self, kwargs=None):
            return [Q] * 2, [{} for _ in range(2)]

        def step(self, actions):
            self.turn += 1
            outs = [BRIDGE if self.turn == 1 else EVID, "Doc 1: \"Nothing\"\\nnothing"]
            infos = [{"won": 0.0, "tool_calling": True, "tool_input": [str(a)]} for a in actions]
            return outs, [0.0, 0.0], [False, False], infos

    cfg = OmegaConf.create({"env": {"history_length": 4, "rollout": {"n": 2}},
                            "algorithm": {"tied_opsd": {"enable": True},
                                          "oci_slots": {"enable": False, "search_flow_path": FLOW}}})
    sm = SearchEnvironmentManager(_Envs(), lambda acts: (list(acts), [1] * len(acts)), cfg)
    so = sm.reset([{"question": Q, "ground_truth": TARGET, "data_source": "hotpotqa"}] * 2)[0]
    check(all(so[ol.OCI_DOC_KEY]) and so[ol.TIED_MATCH_KEY] == [1, 1] and so[ol.TIED_EVID_KEY] == [0, 0]
          and "Your next action is search 1 of 2: <search> Djoi" in so[ol.OCI_DOC_KEY][0]
          and PROGRESS["search"] in so[ol.OCI_DOC_KEY][0] and EFFICIENCY["search"] in so[ol.OCI_DOC_S_KEY][0]
          and "Moroni" not in so[ol.OCI_DOC_KEY][0],
          "turn 1: route, sentences, search 1 owed, no answer anywhere")
    s1 = sm.step(["<think>x</think><search> where is Djoiezi village </search>"] * 2)[0]
    check("Search 1 of 2 is done. Your next action is search 2 of 2: <search> Comoros" in s1[ol.OCI_DOC_KEY][0]
          and "Your next action is search 1 of 2" in s1[ol.OCI_DOC_KEY][1],
          "row 0 got the bridge name back (from its own query): search 2 owed; row 1 did not")
    s2 = sm.step(["<think>x</think><search> Comoros capital </search>"] * 2)[0]
    check("Doc 2 returned by your search 2 contains the answer" in s2[ol.OCI_DOC_KEY][0]
          and s2[ol.TIED_EVID_KEY] == [1, 0]
          and "Moroni" not in s2[ol.OCI_DOC_KEY][0].split(ol.PLAN_FOOTER)[0]
          and "Moroni" not in [l for l in s2[ol.OCI_DOC_KEY][0].split("\n") if "progress]" in l][0],
          "the answer came back: the line names the result, never the answer (only the row's own "
          "history holds it); evidence flag 1")

print("5. the rollout's two edits, real tokenizer")
CKPT = "/opt1/ohara/offline_ladder/probe_hf/klwctl_step300"
if alf_docs is None or not os.path.isdir(CKPT):
    print("SKIP: no alfworld render or no tokenizer")
else:
    import torch
    from transformers import AutoTokenizer
    from agent_system.multi_turn_rollout.rollout_loop import _oci_render_edit, TIED_S_WIDTH
    from verl.trainer.ppo.oci_reachability import splice_span
    tok = AutoTokenizer.from_pretrained(CKPT)
    plain, doc_f, doc_s = alf_docs
    kw = {}
    msgs = [{"role": "user", "content": plain}]
    e1 = _oci_render_edit(tok, msgs, doc_f, 4096, kw, prompt_window=4096)
    msgs_f = [{"role": "user", "content": doc_f}]
    e2 = _oci_render_edit(tok, msgs_f, doc_s, TIED_S_WIDTH, kw, prompt_window=4096)
    render = lambda c: tok.encode(tok.apply_chat_template([{"role": "user", "content": c}],
                                                          add_generation_prompt=True, tokenize=False),
                                  add_special_tokens=False)
    ids_p, ids_f, ids_s = render(plain), render(doc_f), render(doc_s)
    check(e1 is not None and e2 is not None and len(e2[2]) <= TIED_S_WIDTH,
          f"both edits exist; the second is small ({len(e2[2]) if e2 else None} tokens)")
    if e1 and e2:
        W, R = 4096, 4
        def row(ids):
            x = torch.full((1, W + R), 0, dtype=torch.long)
            m_ = torch.zeros((1, W + R), dtype=torch.long)
            x[0, W - len(ids):W] = torch.tensor(ids)
            m_[0, W - len(ids):W] = 1
            x[0, W:] = 7
            m_[0, W:] = 1
            return x, m_
        x, am = row(ids_p)
        r1 = torch.zeros((1, 4096), dtype=torch.long)
        r1[0, :len(e1[2])] = torch.tensor(e1[2])
        x1, am1, _ = splice_span(x, am, torch.tensor([e1[0]]), torch.tensor([e1[1]]), r1,
                                 torch.tensor([len(e1[2])]), 0, response_length=R)
        live1 = x1[0, :W][am1[0, :W].bool()].tolist()
        r2 = torch.zeros((1, TIED_S_WIDTH), dtype=torch.long)
        r2[0, :len(e2[2])] = torch.tensor(e2[2])
        x2, am2, _ = splice_span(x1, am1, torch.tensor([e2[0]]), torch.tensor([e2[1]]), r2,
                                 torch.tensor([len(e2[2])]), 0, response_length=R)
        live2 = x2[0, :W][am2[0, :W].bool()].tolist()
        check(live1 == ids_f and live2 == ids_s and x2[0, W:].tolist() == [7] * R,
              "plain -> stuck render -> saturated render, token for token; the response untouched")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
