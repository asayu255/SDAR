"""CPU test: the document-conditioned score path, end to end without a GPU.

WHAT IS CHECKED, in the order a row travels:

1. the env manager builds the document-conditioned render for EVERY row, not
   just a special slot, and it is the ordinary render with the whole document in
   front of it -- no progress line, because the scorer does not act;
2. the rollout loop records the edit that turns the ordinary prompt INTO that
   render, and the edit RECONSTRUCTS it token for token;
3. splicing it back in on the driver reproduces the document render even when it
   no longer fits the prompt window the batch was padded to (the window is
   widened first; a row that is left untouched is reported, never scored);
4. the aggregation is a length-normalised mean per trajectory;
5. the live-group AUC is what it claims: 1.0 for a perfect ranker, 0.5 for a
   tie, 0.0 for a reversed one -- so a scorer that cannot order a live group
   shows up as 0.5 rather than as a number that flatters it.
"""
import glob, json, os, sys, zlib
from types import SimpleNamespace

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import agent_system.environments.env_manager as em  # noqa: E402
from agent_system.environments import oci_layout as ol  # noqa: E402
from verl.trainer.ppo import oci_rank as orank  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def cfg(**over):
    rank = {"enable": True, "tasks": ["alfworld"]}
    rank.update(over.pop("rank", {}))
    slots = {"enable": False}
    slots.update(over.pop("slots", {}))
    return SimpleNamespace(env=SimpleNamespace(history_length=2),
                           data=SimpleNamespace(max_prompt_length=4096),
                           algorithm={"oci_rank": rank, "oci_slots": slots})


# --- 1. the manager renders the document for every row -----------------------
games = sorted(glob.glob(os.path.expanduser(
    "~/data/alfworld/json_2.1.1/train/pick_heat_then_place_in_recep*/*/game.tw-pddl")))
gf = next((g for g in games if len(json.load(open(g)).get("walkthrough") or []) >= 5), None)
if gf is None:
    print("SKIP: no alfworld walkthrough on this host")
else:
    class FakeEnvs:
        group_n, is_train = 4, True
        get_admissible_commands = [["look", "inventory"]] * 4

        def reset(self):
            obs = ["You are in the middle of a room.\n\nYour task is to: heat some mug and put it somewhere."] * 4
            return obs, None, [{"extra.gamefile": gf, "won": False}] * 4

        def step(self, actions):
            return ([f"You did: {a}." for a in actions], None, [0.0] * 4, [False] * 4,
                    [{"extra.gamefile": gf, "won": False}] * 4)

    em._WRONG_PLAN_CACHE.clear()
    m = em.AlfWorldEnvironmentManager(FakeEnvs(), lambda a, adm: (list(a), [1] * len(a)), cfg())
    obs, _ = m.reset(kwargs=None)
    text, docs = obs["text"], obs[ol.OCI_DOC_KEY]
    block = em._build_wrong_plan(gf, "walkthrough")
    check(len(docs) == 4 and all(d for d in docs),
          f"every row carries a document render ({sum(1 for d in docs if d)}/4)")
    check(all(d == block + t for d, t in zip(docs, text)),
          "and it is the ordinary render with the whole document in front of it")
    check(all("[Privileged Solution Path progress]" not in d for d in docs),
          "with no per-turn progress line")
    o1 = m.step(["look"] * 4)[0]
    check(all(d == block + t for d, t in zip(o1[ol.OCI_DOC_KEY], o1["text"])),
          "on later turns too, against that turn's own render")

    off = cfg(rank={"enable": False})
    m2 = em.AlfWorldEnvironmentManager(FakeEnvs(), lambda a, adm: (list(a), [1] * len(a)), off)
    check(all(d == "" for d in m2.reset(kwargs=None)[0][ol.OCI_DOC_KEY]),
          "switch off: nothing is rendered and nothing is recorded")

# --- 2. the edit the rollout records reconstructs that render -----------------
from omegaconf import OmegaConf  # noqa: E402

import agent_system.multi_turn_rollout.rollout_loop as rl  # noqa: E402


class _Tok:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        toks = text.replace("<|im_start|>", " <|im_start|> ").replace("<|im_end|>", " <|im_end|> ").split()
        return [(zlib.crc32(t.encode()) % 50000) + 1 for t in toks]


tok = _Tok()
plain_content = "obs: a room with a mug\nNow it's your turn to take an action."
doc_content = "DOC line one\nDOC line two\n\n" + plain_content
messages = [{"role": "user", "content": plain_content}]
edit = rl._oci_render_edit(tok, messages, doc_content, 4096, {}, prompt_window=4096)
check(edit is not None, "the document edit is recordable from the two renders")
if edit:
    off_, take_, repl_ = edit
    ids_plain = tok.encode(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    ids_doc = tok.encode(tok.apply_chat_template([{"role": "user", "content": doc_content}],
                                                 add_generation_prompt=True, tokenize=False))
    rebuilt = ids_plain[:off_] + list(repl_) + ids_plain[off_ + take_:]
    check(rebuilt == ids_doc, "and splicing it into the plain prompt reproduces the document render exactly")
    check(len(repl_) > take_, f"the document makes the prompt longer ({take_} -> {len(repl_)} tokens)")

    # --- 3. the driver splice, with the window widened ----------------------
    RESP = 3
    P = len(ids_plain)                      # the batch's prompt window: no slack
    width = P + RESP
    ids = torch.zeros(1, width, dtype=torch.long)
    am = torch.zeros(1, width, dtype=torch.long)
    pos = torch.zeros(1, width, dtype=torch.long)
    ids[0, :P] = torch.tensor(ids_plain); am[0, :P] = 1; pos[0, :P] = torch.arange(P)
    ids[0, P:] = torch.tensor([7, 8, 9]); am[0, P:] = 1; pos[0, P:] = torch.arange(P, P + RESP)
    repl_col = torch.zeros(1, 4096, dtype=torch.int32)
    repl_col[0, :len(repl_)] = torch.tensor(list(repl_), dtype=torch.int32)

    from verl.protocol import DataProto
    b = DataProto.from_dict(tensors={
        "input_ids": ids, "attention_mask": am, "position_ids": pos,
        "responses": ids[:, P:].clone(),
        "oci_doc_off": torch.tensor([off_]), "oci_doc_len": torch.tensor([take_]),
        "oci_doc_repl": repl_col, "oci_doc_repl_len": torch.tensor([len(repl_)])})
    out, spliced = orank.with_document(b, pad_token_id=0)
    check(bool(spliced[0]), "a row whose document does not fit the batch's window is still spliced "
                            "(the window is widened first)")
    live = out.batch["input_ids"][0][out.batch["attention_mask"][0].bool()]
    check(live[:len(ids_doc)].tolist() == ids_doc and live.numel() == len(ids_doc) + RESP,
          "and the spliced prompt IS the document render, with the response copied through")
    check(orank.document_rows(b).tolist() == [True]
          and orank.document_rows(DataProto.from_dict(tensors={
              "oci_doc_len": torch.tensor([0])})).tolist() == [False],
          "a row with no recorded edit is never counted as scored")

# --- 4. aggregation ----------------------------------------------------------
lp = torch.tensor([[-1.0, -3.0, 0.0], [-2.0, -2.0, -2.0]])
msk = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
sc = orank.trajectory_scores(lp, msk, np.array(["a", "a"]))
check(abs(sc["a"] - (-10.0 / 5.0)) < 1e-9,
      f"a trajectory's score is its length-normalised mean log-prob ({sc['a']:.3f})")
sc2 = orank.trajectory_scores(lp, msk, np.array(["a", "b"]))
check(abs(sc2["a"] + 2.0) < 1e-9 and abs(sc2["b"] + 2.0) < 1e-9,
      "and it is per trajectory, not per row")

# --- 5. the AUC says what it claims ------------------------------------------
check(orank.live_auc([(1.0, True), (0.0, False)]) == 1.0, "a perfect ranker scores AUC 1.0")
check(orank.live_auc([(0.0, True), (1.0, False)]) == 0.0, "a reversed one scores 0.0")
check(orank.live_auc([(1.0, True), (1.0, False)]) == 0.5, "ties score 0.5")
check(orank.live_auc([(1.0, True), (1.0, True)]) is None,
      "a group with no loser is not scored at all, rather than counted as perfect")
check(orank.live_auc([(3.0, True), (2.0, True), (1.0, False), (4.0, False)]) == 0.5,
      "and a scorer that orders no better than chance comes out at 0.5")


def test_rank():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
