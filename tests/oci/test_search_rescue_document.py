"""The Search rescue document that shows the answer AND makes finding it the rule.

WHY THE RULE EXISTS. Search's reward reads only the final <answer> string and
never checks that a search happened, so the older document (search_doc=
answer_only) is rescued by being copied: the row can answer on turn one and
score. ALFWorld and WebShop have no such hole -- their environments refuse an
action whose preconditions are unmet -- which is why only Search needs this.

WHAT IS CHECKED HERE, with no model and no retriever:
  1 the block          two numbered lines, the rule's lead, nothing for yes/no
  2 the answer test    whole word, accents folded, entities undone
  3 the progress line  which of the two states it names, and where it sits
  4 the rule, live     a row that writes the answer early is marked; one that
                       waits until a returned result carries it is not
  5 the slots          only the document slot's prompt carries the block
  6 the probe dump     one JSON line per episode, with what the analysis reads
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
EVIDENCE = ('Doc 1(Title: "Greenbrier Valley Airport") Greenbrier Valley Airport is a public '
            "airport three miles north of Lewisburg in Greenbrier County, West Virginia.")
NO_EVIDENCE = 'Doc 1(Title: "Lewisburg, West Virginia") Lewisburg is a city in West Virginia.'

print("1. the block")
lines = ol.search_rescue_document_lines(QUESTION, {"target": [ANSWER]})
check(lines == [f"<search> {QUESTION} </search>", f"<answer> {ANSWER} </answer>"],
      "the question as the first query, the answer as the second line")
check(ol.search_rescue_document_lines(QUESTION, {"target": ["yes"]}) == []
      and ol.search_rescue_document_lines(QUESTION, {"target": ["no"]}) == [],
      "no block for a yes/no question -- 'yes' is in almost any passage, so the rule cannot read it")
check(ol.search_rescue_document_lines("", {"target": [ANSWER]}) == []
      and ol.search_rescue_document_lines(QUESTION, None) == [],
      "no question or no answer -> no block")
block = ol.render_document(lines, lead=ol.SEARCH_RULE_LEAD)
check(block.startswith(ol.PLAN_HEADER) and block.rstrip().endswith(ol.PLAN_FOOTER)
      and "\n1. <search>" in block and "\n2. <answer>" in block,
      "same wrapper and numbering as the other two tasks")
check("DO NOT WRITE THE ANSWER ANYWHERE" in block and ol.PLAN_LEAD not in block,
      "the rule's lead replaces the walkthrough lead")
check(ol.render_document(["go to cabinet 1"]).startswith(f"{ol.PLAN_HEADER}\n{ol.PLAN_LEAD}\n1. "),
      "the default lead is untouched, so alfworld's block is byte-identical")

print("2. the answer test")
check(ol.contains_answer(EVIDENCE, {"target": [ANSWER]}), "found in a passage that holds it")
check(not ol.contains_answer(NO_EVIDENCE, {"target": [ANSWER]}), "not found in one that does not")
check(not ol.contains_answer("released in 1950", {"target": ["5"]})
      and ol.contains_answer("a 5 day week", {"target": ["5"]}),
      "whole word: '5' is not inside '1950'")
check(ol.contains_answer("Joachim Löw was born in 1960", {"target": ["Joachim Löw"]}),
      "the corpus stores accents decomposed; the answer is composed")
check(ol.contains_answer("Ben & Jerry's is an ice cream company", {"target": ["Ben &amp; Jerry's"]}),
      "dataset entities are undone before comparing")
check(ol.contains_answer("the film was Kiss and Tell", {"target": ["wrong", "Kiss and Tell"]}),
      "any accepted answer counts -- an alias written early still breaks the rule")
check(ol.is_yesno({"target": ["yes"]}) and not ol.is_yesno({"target": [ANSWER]}),
      "yes/no questions are recognised")

print("3. the progress line")
waiting, found = ol.search_progress_line(False), ol.search_progress_line(True)
check("No result you have received contains the answer yet" in waiting and "Do step 1" in waiting,
      "before: search, and do not write it")
check("contains the answer" in found and "Do step 2" in found,
      "after: write it")


class _Envs:
    """Canned search results: the first query returns evidence, the rest do not."""
    is_train = True

    def __init__(self, n, group_n, evidence_for):
        self.group_n = group_n
        self.n = n
        self.evidence_for = evidence_for
        self.turns = 0

    def reset(self, kwargs=None):
        return [QUESTION] * self.n, [{} for _ in range(self.n)]

    def step(self, actions):
        self.turns += 1
        obs, rewards, dones, infos = [], [], [], []
        for i, act in enumerate(actions):
            answered = "<answer>" in str(act)
            obs.append(EVIDENCE if (i in self.evidence_for and not answered) else NO_EVIDENCE)
            won = 1.0 if answered else 0.0
            rewards.append(won)
            dones.append(answered or self.turns >= 4)
            infos.append({"won": won})
        return obs, rewards, dones, infos


def _manager(group_n=9, n=None, evidence_for=(), search_doc="answer_rule", search_doc_b="none",
             search_flow_path=""):
    n = group_n if n is None else n
    config = OmegaConf.create({
        "env": {"history_length": 2, "rollout": {"n": group_n}},
        "algorithm": {"oci_slots": {"enable": True, "tasks": ["search"], "search_doc": search_doc,
                                    "search_doc_b": search_doc_b,
                                    "search_flow_path": search_flow_path,
                                    "doc_mode": "walkthrough_stepwise", "foreign_task": "webshop"},
                      "oci_rank": {"enable": False}},
    })
    envs = _Envs(n, group_n, set(evidence_for))
    return SearchEnvironmentManager(envs, lambda acts: (list(acts), [1] * len(acts)), config)


KW = [{"question": QUESTION, "ground_truth": {"target": [ANSWER]}}]

print("4. the rule, on a live manager")
# Search has no foreign slot, so the LAST slot of the group is the document row
# and the other eight are ordinary rollouts -- the eight control trains.
check(not ol.has_foreign_slot("search") and ol.has_foreign_slot("alfworld"),
      "only search drops the foreign slot")
check([ol.role_for_slot(i, 9, foreign=False) for i in range(9)]
      == [ol.ROLE_PLAIN] * 7 + [ol.ROLE_RESERVE, ol.ROLE_DOC],
      "nine slots: seven plain, one reserve, one document")
check([ol.role_for_slot(i, 10) for i in range(10)]
      == [ol.ROLE_PLAIN] * 7 + [ol.ROLE_RESERVE, ol.ROLE_DOC, ol.ROLE_FOREIGN],
      "and alfworld's ten-slot layout is unchanged")
check(ol.used_per_group(9, foreign=False) == 8 and ol.used_per_group(10) == 8,
      "either way a group trains eight trajectories, as control does")
DOC_SLOT = [i for i in range(9) if ol.role_for_slot(i, 9, foreign=False) == ol.ROLE_DOC][0]
m = _manager(group_n=9, evidence_for={0, 1, DOC_SLOT})
m.reset(KW * 9)
early = [""] * 9
early[0] = f"<think> the answer is {ANSWER} </think><search> {QUESTION} </search>"
early[1] = f"<think> let me look </think><search> {ANSWER} </search>"
early[2] = f"<think> let me look </think><search> {QUESTION} </search>"
early[DOC_SLOT] = f"<think> I must search first </think><search> {QUESTION} </search>"
m.step(early)
check(m._answer_early[0] and m._answer_early[1],
      "writing the answer in the thinking, or inside the query, before it came back is marked")
check(not m._answer_early[2] and not m._answer_early[DOC_SLOT],
      "a query that does not name the answer is not")
check(m._evidence_seen[0] and m._evidence_seen[1],
      "a row that broke the rule still has its own evidence state tracked")
check(m._evidence_seen[DOC_SLOT] and not m._evidence_seen[2],
      "a row whose result carried the answer is the only one whose progress flips")
second = [""] * 9
second[2] = f"<answer> {ANSWER} </answer>"
second[DOC_SLOT] = f"<answer> {ANSWER} </answer>"
m.step(second)
check(m._answer_early[2] and not m._answer_early[DOC_SLOT],
      "answering without ever seeing it breaks the rule; answering after a result carried it does not")

print("5. what each slot is shown")
m2 = _manager(group_n=9, evidence_for={DOC_SLOT})
obs, _ = m2.reset(KW * 9)
texts = obs["text"]
check(ol.PLAN_HEADER in texts[DOC_SLOT] and ANSWER in texts[DOC_SLOT],
      "the document slot sees the block and the answer")
check(all(ol.PLAN_HEADER not in texts[i] and ANSWER not in texts[i]
          for i in range(9) if i != DOC_SLOT),
      "no other slot does -- the other eight are ordinary rollouts")
check(ol.search_progress_line(False) in texts[DOC_SLOT],
      "and the waiting progress line, where the turn prompt starts")
check(obs[ol.OCI_ROLE_KEY][DOC_SLOT] == ol.ROLE_DOC
      and obs[ol.OCI_PLAIN_KEY][DOC_SLOT] and not obs[ol.OCI_PLAIN_KEY][0],
      "the row carries its role and the plain render the strip needs")
after = m2.step([f"<search> {QUESTION} </search>"] * 9)[0]["text"]
check(ol.search_progress_line(True) in after[DOC_SLOT],
      "once a result carries the answer the line flips to 'write it'")
m3 = _manager(group_n=9, search_doc="answer_only")
check(ANSWER in m3.reset(KW * 9)[0]["text"][DOC_SLOT],
      "answer_only still prints the old block, so an older arm reruns unchanged")

print("6. the probe dump")
with tempfile.TemporaryDirectory() as tmp:
    os.environ["SEARCH_PROBE_DUMP"] = tmp
    try:
        m4 = _manager(group_n=9, evidence_for={DOC_SLOT})
        m4.reset(KW * 9)
        acts = [""] * 9
        acts[0] = f"<answer> {ANSWER} </answer>"
        acts[DOC_SLOT] = f"<search> {QUESTION} </search>"
        m4.step(acts)
        m4.step([f"<answer> {ANSWER} </answer>"] * 9)
        m4._probe_flush()
        path = os.path.join(tmp, f"search_rollouts.{os.getpid()}.jsonl")
        rows = [json.loads(l) for l in open(path)]
    finally:
        os.environ.pop("SEARCH_PROBE_DUMP", None)
check(len(rows) == 9, f"one record per row of the group ({len(rows)})")
by_env = {r["env"]: r for r in rows}
check(all(r["question"] == QUESTION and r["answers"] == [ANSWER] and r["group"] == 0 for r in rows),
      "each record names the question it answers and the group it belongs to")
check(by_env[DOC_SLOT]["role"] == ol.ROLE_DOC and by_env[0]["role"] != ol.ROLE_DOC,
      "and which slot it was")
check(by_env[0]["answer_early"] and not by_env[0]["evidence_seen"] and by_env[0]["won"] == 1.0,
      "a row that answered from the prompt alone: scored, rule broken, nothing retrieved")
check(by_env[DOC_SLOT]["evidence_seen"] and not by_env[DOC_SLOT]["answer_early"]
      and by_env[DOC_SLOT]["won"] == 1.0,
      "the rescue row: searched, the result carried the answer, then answered -- rule kept")
check(all("info_has_answer" in t and "wrote_answer" in t for t in by_env[DOC_SLOT]["turns"]),
      "per turn, what the row wrote and what came back")

print("7. the arm's own config check")
from verl.trainer.ppo import oci_slots  # noqa: E402


def _run_config(tasks, search_doc):
    return OmegaConf.create({
        "algorithm": {"oci_slots": {"enable": True, "tasks": tasks, "search_doc": search_doc,
                                    "gamma": 0.1, "special_loss": "shaped"},
                      "oci_sat": {"enable": False}, "oci_floor": {"enable": False},
                      "filter_groups": {"enable": False}},
        "env": {"rollout": {"n": 9}},
        "actor_rollout_ref": {"actor": {"pg_loss_coef": 1.0, "normalize_loss_by_task": True}},
    })


def _refused(tasks, search_doc):
    try:
        oci_slots.check_config(_run_config(tasks, search_doc))
        return False
    except AssertionError:
        return True


check(not _refused(["search"], "answer_rule"), "search is allowed with the rule document")
check(_refused(["search"], "answer_only"),
      "and refused without it -- an answer-only block is rescued by being copied")
check(not _refused(["alfworld"], "answer_only"), "alfworld is unchanged")
check(_refused(["webshop"], "answer_rule"), "webshop still has no verified document")

print("8. the analysis over a dump")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from analyze_search_rescue import summarise  # noqa: E402


def _rec(pid, reset, group, env, role, won, evidence, early, source="nq", answers=(ANSWER,)):
    return {"pid": pid, "reset": reset, "group": group, "env": env, "role": role, "won": won,
            "evidence_seen": evidence, "answer_early": early, "n_turns": 2,
            "data_source": source, "answers": list(answers), "question": QUESTION, "turns": []}


recs = []
# a stuck group: every ordinary row failed, five of eight had the answer returned
for i in range(8):
    recs.append(_rec(1, 0, 0, i, 1, 0.0, i < 5, False))
recs.append(_rec(1, 0, 0, 8, ol.ROLE_DOC, 1.0, True, False))          # rescued, rule kept
# a second stuck group whose document row copied the answer out of the prompt
for i in range(8):
    recs.append(_rec(1, 0, 1, i, 1, 0.0, False, False, source="hotpotqa"))
recs.append(_rec(1, 0, 1, 8, ol.ROLE_DOC, 1.0, False, True, source="hotpotqa"))
# a live group, and a saturated one
for i in range(8):
    recs.append(_rec(1, 1, 0, i, 1, 1.0 if i < 3 else 0.0, True, False))
recs.append(_rec(1, 1, 0, 8, ol.ROLE_DOC, 1.0, True, False))
for i in range(8):
    recs.append(_rec(2, 0, 0, i, 1, 1.0, True, False))
recs.append(_rec(2, 0, 0, 8, ol.ROLE_DOC, 0.0, True, False))

s = summarise(recs)
check(s["groups"] == 4 and s["classes"] == {"stuck": 2, "live": 1, "saturated": 1},
      f"groups keyed by (pid, reset, group): {s['classes']}")
check(s["plain"]["stuck"]["groups"] == 2 and s["plain"]["stuck"]["row_success"] == 0.0,
      "the class comes from the eight ordinary rows alone")
check(s["plain"]["stuck"]["rows_with_evidence"] == round(5 / 16, 3)
      and s["plain"]["stuck"]["groups_with_any_evidence"] == 0.5,
      "and the stuck rows that DID retrieve the answer are counted -- question 1")
check(s["rescue"]["stuck"]["scored"] == 1.0
      and s["rescue"]["stuck"]["scored_and_kept_rule"] == 0.5
      and s["rescue"]["stuck"]["broke_rule"] == 0.5,
      "the copied rescue scores but does not keep the rule -- question 2")
check(s["by_source"]["nq"]["groups"] == 3 and s["by_source"]["nq"]["stuck"] == round(1 / 3, 3)
      and s["by_source"]["hotpotqa"]["stuck"] == 1.0,
      "split by dataset, because nq and hotpotqa fail differently")
# the same stuck groups, now with the second variant's row beside the first
recs_b = list(recs)
recs_b.append(_rec(1, 0, 0, 7, ol.ROLE_DOC_B, 0.0, True, False))   # found it, answered wrong
recs_b.append(_rec(1, 0, 1, 7, ol.ROLE_DOC_B, 1.0, True, False, source="hotpotqa"))
for r in recs_b:
    if r["role"] == ol.ROLE_DOC:
        r["variant"] = "answer_rule"
    elif r["role"] == ol.ROLE_DOC_B:
        r["variant"] = "progress_only"
    r["has_document"] = r["role"] in (ol.ROLE_DOC, ol.ROLE_DOC_B)
sb = summarise(recs_b)["rescue_by_variant_on_stuck"]
check(sb["answer_rule"]["rows"] == 2 and sb["progress_only"]["rows"] == 2,
      "both variants are scored on the same stuck groups")
check(sb["answer_rule"]["scored_and_kept_rule"] == 0.5
      and sb["progress_only"]["scored_and_kept_rule"] == 0.5,
      "and their rescue rates are reported side by side")
check(summarise([_rec(1, 0, 0, 8, ol.ROLE_DOC, 0.0, False, False, answers=())]
                + [_rec(1, 0, 0, i, 1, 0.0, False, False) for i in range(8)]
                )["rescue"]["stuck"]["no_document"] == 1,
      "a yes/no question has no document row to score")

print("9. the selection, on a search batch laid out by the manager")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from verl import DataProto  # noqa: E402
from verl.trainer.ppo.oci_slots import select_rollouts  # noqa: E402


def _batch(n_groups=2, group_n=9, returns=None, plan_len=None, second_doc=False):
    """One row per rollout, marked the way the search manager marks them."""
    n = n_groups * group_n
    slots = [i % group_n for i in range(n)]
    roles = [ol.role_for_slot(s, group_n, foreign=False, second_doc=second_doc) for s in slots]
    rets = returns if returns is not None else [0.0] * n
    plens = plan_len if plan_len is not None else [
        7 if r in (ol.ROLE_DOC, ol.ROLE_DOC_B) else 0 for r in roles]
    return DataProto.from_dict(
        tensors={"oci_role": torch.tensor(roles, dtype=torch.long),
                 "oci_slot": torch.tensor(slots, dtype=torch.long),
                 "oci_plan_len": torch.tensor(plens, dtype=torch.long)},
        non_tensors={
            "uid": np.array([f"q{i // group_n}" for i in range(n)], dtype=object),
            "traj_uid": np.array([f"t{i}" for i in range(n)], dtype=object),
            "episode_rewards": np.array(rets, dtype=object),
            "task_name": np.array(["search"] * n, dtype=object),
        })


# group 0 is stuck and its document row solved it; group 1 is live.
rets = [0.0] * 9 + [0.0] * 9
rets[8] = 1.0                      # the document row of group 0
rets[9 + 0] = 1.0                  # one ordinary row of group 1
keep, injected, metrics = select_rollouts(_batch(returns=rets), tasks=["search"], group_n=9)
check(int(keep.sum()) == 16 and int(metrics["oci_slots/trained_per_group"]) == 8,
      f"eight trajectories per group train, as control does ({int(keep.sum())} of 18)")
check(bool(injected[8]) and int(injected.sum()) == 1,
      "the stuck group's document row is the one injected")
check(not keep[7] and keep[8],
      "it replaces the reserve row, so the group is seven plain plus the rescue")
check(keep[9 + 8] is np.False_ or not keep[9 + 8],
      "the live group's document row is dropped -- its eight ordinary rows already differ")
check(metrics.get("oci_slots/doc_rescue_rate/search") == 1.0
      and "oci_slots/foreign_fail_rate/search" not in metrics,
      "the rescue rate is reported; there is no foreign slot to report on")
try:
    select_rollouts(_batch(returns=rets), tasks=["alfworld"], group_n=9)
    check(False, "a batch marked search must not pass as alfworld")
except AssertionError:
    check(True, "rows carrying a role off the configured task are refused")

print("10. the document row's prompt is reconstructible")
import zlib  # noqa: E402

import agent_system.multi_turn_rollout.rollout_loop as rl  # noqa: E402


class _Tok:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def _ids(self, text):
        toks = text.replace("<|im_start|>", " <|im_start|> ").replace("<|im_end|>", " <|im_end|> ").split()
        return [(zlib.crc32(t.encode()) % 50000) + 1 for t in toks]

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def __call__(self, text, return_tensors="pt", add_special_tokens=False, **kw):
        ids = torch.tensor([self._ids(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


m5 = _manager(group_n=9, evidence_for={DOC_SLOT})
obs5, _ = m5.reset(KW * 9)
col = rl.TrajectoryCollector(
    config=OmegaConf.create({
        "data": {"max_prompt_length": 4096, "truncation": "left",
                 "return_raw_chat": False, "apply_chat_template_kwargs": {}},
        "env": {"rollout": {"n": 9}},
        "algorithm": {"oci_slots": {"enable": True, "tasks": ["search"],
                                    "search_doc": "answer_rule", "foreign_task": "webshop"}},
    }),
    tokenizer=_Tok(), processor=None)
gb = DataProto.from_dict(
    tensors={"dummy": torch.zeros(9)},
    non_tensors={"raw_prompt": np.array([[{"role": "user", "content": "x"}]] * 9, dtype=object),
                 "data_source": np.array(["nq"] * 9, dtype=object),
                 "task_name": np.array(["search"] * 9, dtype=object)})
rows5 = [col.preprocess_single_sample(item=i, gen_batch=gb, obs=obs5) for i in range(9)]
check(int(rows5[DOC_SLOT]["oci_candidate"]) == 1 and int(rows5[DOC_SLOT]["oci_plan_len"]) > 0
      and int(rows5[DOC_SLOT]["oci_plan_repl_len"]) > 0,
      "the document row's prompt is ONE replacement away from the plain one -- "
      "the block at the head and the progress line before the turn prompt together")
check(all(int(rows5[i]["oci_candidate"]) == 0 and int(rows5[i]["oci_plan_len"]) == 0
          for i in range(9) if i != DOC_SLOT),
      "and no ordinary row has anything to strip")
check(all(int(rows5[i]["oci_role"]) == ol.role_for_slot(i, 9, foreign=False) for i in range(9)),
      "every row records the role its slot predicts, which is what the selection asserts")

print("11. the variant that withholds the answer, generated beside the first")
b_lines = ol.search_rescue_document_lines(QUESTION, {"target": [ANSWER]}, show_answer=False)
b_block = ol.render_document(b_lines, lead=ol.SEARCH_PROGRESS_LEAD)
check(ANSWER not in b_block and "what that result gives" in b_block,
      "progress_only prints no answer anywhere in the block")
check("A checker" in b_block and "NOT WHAT IT IS" in b_block,
      "it says a checker is reading the results, which is all the row is told")
check([ol.role_for_slot(i, 10, foreign=False, second_doc=True) for i in range(10)]
      == [ol.ROLE_PLAIN] * 7 + [ol.ROLE_RESERVE, ol.ROLE_DOC_B, ol.ROLE_DOC],
      "ten slots: eight ordinary rollouts and the two rescue rows")
check(ol.used_per_group(10, foreign=False, second_doc=True) == 8,
      "still eight trained trajectories a group")
DOC_A, DOC_B = 9, 8
m6 = _manager(group_n=10, evidence_for={DOC_A}, search_doc_b="progress_only")
obs6, _ = m6.reset(KW * 10)
t6 = obs6["text"]
check(ANSWER in t6[DOC_A] and ol.PLAN_HEADER in t6[DOC_A],
      "slot 9 wears the document that shows the answer")
check(ol.PLAN_HEADER in t6[DOC_B] and ANSWER not in t6[DOC_B],
      "slot 8 wears the one that does not")
check(ol.search_progress_line(False) in t6[DOC_B],
      "and both carry the same per-turn verdict line")
check(all(ol.PLAN_HEADER not in t6[i] for i in range(8)),
      "the eight ordinary rows see nothing")

print("12. the guard on bare-number answers")
YEAR_Q = "when was alka-seltzer launched?"
check(ol.is_numeric_answer({"target": ["1931"]}) and not ol.is_numeric_answer({"target": [ANSWER]}),
      "a bare-number answer is recognised")
check(ol.evidence_in_text("Alka-Seltzer was launched in 1931.", {"target": ["1931"]}, YEAR_Q),
      "a passage about the question that carries the year counts")
check(not ol.evidence_in_text("The treaty of 1931 ended the war in Chaco.",
                              {"target": ["1931"]}, YEAR_Q),
      "one that merely contains the same digits does not")
check(ol.contains_answer("The treaty of 1931 ended the war in Chaco.", {"target": ["1931"]}),
      "the raw test still sees it -- the record keeps both, so the guard's size is measurable")
check(ol.evidence_in_text(EVIDENCE, {"target": [ANSWER]}, QUESTION),
      "a worded answer is judged by the answer alone")

print("13. the selection with both rescue rows")
rets2 = [0.0] * 20
rets2[9] = 1.0        # the answer_rule row of group 0 solved it
rets2[8] = 1.0        # so did the progress_only row
keep2, inj2, met2 = select_rollouts(
    _batch(n_groups=2, group_n=10, returns=rets2, second_doc=True),
    tasks=["search"], group_n=10, second_doc=True)
check(int(keep2.sum()) == 16 and int(met2["oci_slots/trained_per_group"]) == 8,
      f"eight trajectories a group still train ({int(keep2.sum())} of 20)")
check(bool(inj2[9]) and not bool(inj2[8]) and int(inj2.sum()) == 1,
      "the shipped variant is the one injected; the second row is measurement only")
check(not keep2[8], "and it is dropped rather than trained")

print("14. the route written by a stronger model")
FLOW_FILE = os.path.join(tempfile.gettempdir(), f"flows_{os.getpid()}.json")
json.dump({"flows": {
    "0": {"question": QUESTION, "answers": [ANSWER], "hit": True,
          "queries": ["closest airport Lewisburg West Virginia", "Greenbrier County airport"]},
    "1": {"question": "a question whose route did not work", "answers": ["x"], "hit": False,
          "queries": ["something"]},
    "2": {"question": "a route that names the answer", "answers": [ANSWER], "hit": True,
          "queries": [f"{ANSWER} runway"]},
}}, open(FLOW_FILE, "w"))
try:
    ol._SEARCH_FLOWS.clear()
    lines = ol.search_flow_document_lines(QUESTION, {"target": [ANSWER]}, FLOW_FILE)
    check(lines == ["<search> closest airport Lewisburg West Virginia </search>",
                    "<search> Greenbrier County airport </search>",
                    f"<answer> {ANSWER} </answer>"],
          "the verified queries in order, then the answer they end at")
    check(ol.search_flow_document_lines("a question whose route did not work",
                                        {"target": ["x"]}, FLOW_FILE) == [],
          "a route that did not return the answer is not shown")
    check(ol.search_flow_document_lines("a route that names the answer",
                                        {"target": [ANSWER]}, FLOW_FILE) == [],
          "and one whose query names the answer is refused here too, not only when built")
    check(ol.search_flow_document_lines("a question that is not in the file",
                                        {"target": [ANSWER]}, FLOW_FILE) == [],
          "a question with no route has no document")
    m7 = _manager(group_n=10, evidence_for={9}, search_doc="answer_rule",
                  search_doc_b="expert_flow", search_flow_path=FLOW_FILE)
    t7 = m7.reset(KW * 10)[0]["text"]
    check("closest airport Lewisburg West Virginia" in t7[8] and ANSWER in t7[8],
          "the second slot wears the route, with the answer under the rule")
    check("closest airport Lewisburg" not in t7[9] and ANSWER in t7[9],
          "the first still wears the plain answer document")
    check(ol.search_progress_line(False) in t7[8],
          "and the route row gets the same per-turn verdict line")
finally:
    os.remove(FLOW_FILE)
    ol._SEARCH_FLOWS.clear()

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
