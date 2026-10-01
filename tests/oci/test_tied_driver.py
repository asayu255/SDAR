"""CPU test: the OPD+GRPO driver's tied_opsd hook (_tied_opsd_columns) end to end, with fake worker groups.

WHAT MUST HOLD.
  - groups: stuck rows get side 1 and w_f, saturated rows side 2 and w_s, live rows nothing;
  - the teacher prompt of each distilled row is the recorded edit applied to its own prompt -- the stuck render
    for stuck rows, the stuck render AND the sentence edit for saturated rows;
  - the support per side: stuck rows are scored by the teacher's own top-k (compute_ref_topk_log_prob); saturated
    rows on the STUDENT's top-k (compute_actor_topk_ids on the plain prompt), then the teacher is read at exactly
    those ids (compute_ref_logprob_at_ids);
  - the token marks (no tags, no special tokens) and the columns the actor reads, row for row;
  - a row whose state pointer could not place it (tied_match 0) is not distilled.
"""
import os, sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


CKPT = "/opt1/ohara/offline_ladder/probe_hf/klwctl_step300"
if not os.path.isdir(CKPT):
    print("SKIP: no tokenizer")
    sys.exit(0)

from transformers import AutoTokenizer  # noqa: E402

from verl import DataProto  # noqa: E402
from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer  # noqa: E402
from verl.trainer.ppo.tied_opsd import SIDE_SAT, SIDE_STUCK, UNSCORED_LP  # noqa: E402

tok = AutoTokenizer.from_pretrained(CKPT)
PAD = int(tok.pad_token_id)
P, L, K = 8, 16, 4
resp_ids = tok.encode("<action>go to sinkbasin 1</action>", add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
assert len(resp_ids) <= L

# rows: g_stuck (2 traj x 2 rows), g_sat (2 traj x 2 rows), g_live (2 traj x 1 row), one unplaced stuck row
rows = []
for g, wins in (("g_stuck", (0, 0)), ("g_sat", (10, 10)), ("g_live", (10, 0))):
    for j, r in enumerate(wins):
        n_rows = 1 if g == "g_live" else 2
        for t in range(n_rows):
            rows.append(dict(uid=g, traj=f"{g}_{j}", rew=r, match=1))
rows[1]["match"] = 0                                   # g_stuck's first trajectory, its second turn: no line
N = len(rows)
prompt = [11, 12, 13, 14, 15, 16]                       # live prompt tokens (left-padded to P)
ids = torch.full((N, P + L), PAD, dtype=torch.long)
am = torch.zeros((N, P + L), dtype=torch.long)
for i in range(N):
    ids[i, P - len(prompt):P] = torch.tensor(prompt)
    am[i, P - len(prompt):P] = 1
    ids[i, P:P + len(resp_ids)] = torch.tensor(resp_ids)
    am[i, P:P + len(resp_ids)] = 1
pos = torch.clamp(torch.cumsum(am, dim=1) - 1, min=0)
W = 8
doc_repl = torch.zeros((N, W), dtype=torch.long)
doc_repl[:, :3] = torch.tensor([101, 102, 103])         # the stuck render replaces live token 1 ("12") with 3 tokens
s_repl = torch.zeros((N, 4), dtype=torch.long)
s_repl[:, :2] = torch.tensor([201, 202])                # the saturated render then replaces doc-render token 2 ("102")
tensors = {
    "input_ids": ids, "attention_mask": am, "position_ids": pos, "responses": ids[:, P:].clone(),
    "oci_doc_off": torch.full((N,), 1), "oci_doc_len": torch.full((N,), 1), "oci_doc_repl": doc_repl,
    "oci_doc_repl_len": torch.full((N,), 3),
    "tied_s_off": torch.full((N,), 2), "tied_s_len": torch.full((N,), 1), "tied_s_repl": s_repl,
    "tied_s_repl_len": torch.full((N,), 2),
    "tied_match": torch.tensor([r["match"] for r in rows]), "tied_evid": torch.zeros(N, dtype=torch.long),
}
non = {"uid": np.array([r["uid"] for r in rows], dtype=object),
       "traj_uid": np.array([r["traj"] for r in rows], dtype=object),
       "episode_rewards": np.array([float(r["rew"]) for r in rows], dtype=object),
       "task_name": np.array(["alfworld"] * N, dtype=object)}
batch = DataProto.from_dict(tensors=tensors, non_tensors=non)

calls = {}


class SelfTeacher:
    def compute_ref_topk_log_prob(self, tb):
        calls["topk"] = tb
        n = len(tb)
        return DataProto.from_dict(tensors={"teacher_topk_ids": torch.full((n, L, K), 7, dtype=torch.long),
                                            "teacher_topk_logprobs": torch.full((n, L, K), -1.0)})

    def compute_ref_logprob_at_ids(self, tb):
        calls["at_ids"] = tb
        return DataProto.from_dict(tensors={"teacher_lp_at_ids": torch.full((len(tb), L, K), -2.0)})

    def copy_weights_from_actor(self):
        calls["copy"] = calls.get("copy", 0) + 1
        return [(1, 1.0, 1.0)]

    def load_ref_from_actor_checkpoint(self, path):
        return [False]


class Actor:
    def compute_actor_topk_ids(self, pb):
        calls["student"] = pb
        return DataProto.from_dict(tensors={"student_topk_ids": torch.full((len(pb), L, K), 9, dtype=torch.long)})


tr = OPDGRPORayTrainer.__new__(OPDGRPORayTrainer)
tr.config = OmegaConf.create({"trainer": {"default_local_dir": "/nonexistent"},
                              "algorithm": {"tied_opsd": {"enable": True, "tasks": ["alfworld"], "topk": K,
                                                          "refresh_every": 50, "retention": 0.8, "guard": True}}})
tr.tokenizer = tok
tr.self_teacher_wg = SelfTeacher()
tr.actor_rollout_wg = Actor()
tr.global_steps = 1
tr._post_rollout_token_budget = 0
metrics = {}
tr._tied_opsd_columns(batch, tr.config.algorithm.tied_opsd, metrics)
b = batch.batch
side, w = b["tied_side"].tolist(), b["tied_w"].tolist()
stuck_i = [i for i, r in enumerate(rows) if r["uid"] == "g_stuck"]
sat_i = [i for i, r in enumerate(rows) if r["uid"] == "g_sat"]
live_i = [i for i, r in enumerate(rows) if r["uid"] == "g_live"]
check(all(side[i] == SIDE_STUCK for i in stuck_i) and all(side[i] == SIDE_SAT for i in sat_i)
      and all(side[i] == 0 for i in live_i), f"sides from the group outcomes ({side})")
check(all(w[i] > 0 for i in stuck_i if rows[i]["match"]) and w[1] == 0 and all(w[i] > 0 for i in sat_i)
      and all(w[i] == 0 for i in live_i), f"weights: stuck and saturated rows, not the unplaced or live rows ({w})")
check(abs(metrics["tied/alfworld/m"] - (1 / (0.5 ** 0.5 + 1e-6) * 0.5)) < 1e-3 or metrics["tied/alfworld/m"] > 0,
      f"M from the live group ({metrics['tied/alfworld/m']:.3f})")
ids_c, lp_c = b["tied_topk_ids"], b["tied_topk_lp"]
check(all(int(ids_c[i, 0, 0]) == 7 and float(lp_c[i, 0, 0]) == -1.0 for i in stuck_i if w[i] > 0),
      "stuck rows: the teacher's own top-k and its log-probs")
check(all(int(ids_c[i, 0, 0]) == 9 and float(lp_c[i, 0, 0]) == -2.0 for i in sat_i),
      "saturated rows: the student's top-k, the teacher read at those ids")
check(all(int(ids_c[i].abs().sum()) == 0 and float(lp_c[i, 0, 0]) == UNSCORED_LP for i in live_i + [1]),
      "rows not distilled: no ids, the unscored log-prob")
check(int(calls["at_ids"].batch["gather_ids"][0, 0, 0]) == 9 and len(calls["at_ids"]) == len(sat_i)
      and len(calls["student"]) == len(sat_i) and len(calls["topk"]) == sum(1 for i in stuck_i if w[i] > 0),
      "the student's ids reach the teacher; each call got only its side's rows")
live = lambda tb, j: tb.batch["input_ids"][j][:-L][tb.batch["attention_mask"][j][:-L].bool()].tolist()
check(live(calls["topk"], 0) == [11, 101, 102, 103, 13, 14, 15, 16],
      f"stuck teacher prompt = the stuck render ({live(calls['topk'], 0)})")
check(live(calls["at_ids"], 0) == [11, 101, 201, 202, 103, 13, 14, 15, 16],
      f"saturated teacher prompt = the stuck render with its sentence edit ({live(calls['at_ids'], 0)})")
check(live(calls["student"], 0) == prompt, "the student's top-k comes from its own plain prompt")
check(torch.equal(calls["at_ids"].batch["responses"][0], ids[sat_i[0], P:]),
      "the responses scored are the student's own")
tokm = b["tied_tok"]
kept = tok.decode([t for t, m_ in zip(resp_ids, tokm[sat_i[0], :len(resp_ids)].tolist()) if m_])
check("<" not in kept and ">" not in kept and "go to sinkbasin 1" in kept, f"token marks ({kept!r})")
check(metrics["tied/teacher_from_step"] == 0.0 and calls.get("copy", 0) == 0, "step 1: the initial teacher, no copy")
tr.global_steps = 51
tr._tied_opsd_columns(batch, tr.config.algorithm.tied_opsd, {})
check(calls.get("copy", 0) == 1 and tr._tied_teacher_from == 50, "step 51: one copy from the actor (teacher = step 50)")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
