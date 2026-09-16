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

    def __call__(self, text, return_tensors="pt", add_special_tokens=False, **kw):
        ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


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

    # --- 3b. the two self-checks, on the same row ---------------------------
    grow_doc = int(out.batch["input_ids"].shape[1]) - width
    null_b, null_spl = orank.with_document(orank.null_edit(b), pad_token_id=0, min_grow=grow_doc)
    check(bool(null_spl[0]) and tuple(null_b.batch["input_ids"].shape) == tuple(out.batch["input_ids"].shape),
          f"self-check 1: a null edit splices, widened exactly as far as the document was (+{grow_doc})")
    live_n = null_b.batch["attention_mask"][0].bool()
    check(orank.prompt_tokens(null_b, [0]) == [ids_plain]
          and null_b.batch["input_ids"][0][live_n].tolist() == ids_plain + [7, 8, 9]
          and null_b.batch["position_ids"][0][live_n].tolist() == list(range(P + RESP)),
          "and gives back the plain prompt, the response and the positions unchanged")
    doc_text = tok.apply_chat_template([{"role": "user", "content": doc_content}],
                                       add_generation_prompt=True, tokenize=False)
    W = int(out.batch["input_ids"].shape[1]) - RESP
    direct_b, too_long = orank.direct_render(out, [doc_text], tok, 0, W)
    live_d, live_s = direct_b.batch["attention_mask"][0].bool(), out.batch["attention_mask"][0].bool()
    check(not too_long and orank.prompt_tokens(direct_b, [0]) == orank.prompt_tokens(out, [0]) == [ids_doc]
          and direct_b.batch["input_ids"][0][live_d].tolist() == out.batch["input_ids"][0][live_s].tolist()
          and direct_b.batch["position_ids"][0][live_d].tolist() == out.batch["position_ids"][0][live_s].tolist(),
          "self-check 2: the document prompt tokenized from its own text equals the splice "
          "(prompt, response and positions)")
    plain_text_render = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    wrong_b, _ = orank.direct_render(out, [plain_text_render], tok, 0, W)
    check(orank.prompt_tokens(wrong_b, [0]) != orank.prompt_tokens(out, [0]),
          "and it can fail: the plain prompt's text does not equal the spliced document prompt")
    _, tl = orank.direct_render(out, [doc_text], tok, 0, len(ids_doc) - 1)
    check(tl == [0], "a text that does not fit the window is reported, never truncated")
    cmp_ = orank.compare_scores([-1.0, -2.0], [-1.0, -2.5], [2.0, 5.0])
    check(abs(cmp_["max_abs_row_sum_diff"] - 0.5) < 1e-12 and abs(cmp_["max_abs_token_mean_diff"] - 0.1) < 1e-12,
          f"score comparison per row and per token ({cmp_})")

# --- 4. trajectories: token-weighted scores and the gain against plain --------
recs = orank.build_trajectories(
    uids=["g", "g"], tuids=["a", "a"], turn_steps=[1, 0], returns=[10.0, 10.0],
    tasks=["alfworld"] * 2, gamefiles=["", ""], actions=["take x", "go to y"],
    sums={"plain": np.array([-4.0, -2.0]), "privileged": np.array([-1.0, -1.0])},
    counts=np.array([2.0, 2.0]))
r = recs["a"]
check(abs(r["score"]["plain"] + 1.5) < 1e-12 and abs(r["score"]["privileged"] + 0.5) < 1e-12,
      f"a trajectory's score is its token-weighted mean over all rows ({r['score']['plain']}, {r['score']['privileged']})")
check(abs(r["score"]["privileged_gain"] - 1.0) < 1e-12,
      "and privileged_gain is the mean per-token log q - log pi against plain")
check(r["actions"] == ["go to y", "take x"] and r["turns"] == 2,
      "rows are put back in turn order before anything reads the actions")

# --- 5. the checks say what they claim ----------------------------------------
check(orank.auc([1.0], [0.0]) == 1.0 and orank.auc([0.0], [1.0]) == 0.0
      and orank.auc([1.0], [1.0]) == 0.5 and orank.auc([1.0], []) is None,
      "AUC: 1 perfect, 0 reversed, 0.5 tie, None without both kinds")


def synth(groups):
    """groups: {uid: [(won, turns, {scorer: score})]} -> trajectory records."""
    out = {}
    for u, rows in groups.items():
        for k, (won, turns, sc) in enumerate(rows):
            t = f"{u}{k}"
            out[t] = {"traj": t, "uid": u, "task": "alfworld", "gamefile": "", "ret": 10.0 if won else 0.0,
                      "won": won, "turns": turns, "tokens": 1.0, "score": dict(sc), "actions": [], "_rows": []}
    return out


# A scorer that only reads length: losers run to the cap, winners are short.
live = {f"L{j}": [(True, 10 + j, {"len": -(10 + j)}), (True, 14 + j, {"len": -(14 + j)}),
                  (False, 50, {"len": -50}), (False, 50, {"len": -50})] for j in range(6)}
rs = synth(live)
st_ = orank.classify_groups(rs)
raw = orank.group_auc_stats(rs, st_, "len", n_boot=200)
adj = orank.group_auc_stats(rs, st_, "len", adjust="turns", n_boot=200)
check(raw["auc_mean"] == 1.0 and adj["auc_mean"] == 0.5,
      f"a length-only scorer is perfect raw ({raw['auc_mean']}) and near chance once turn count is "
      f"removed ({adj['auc_mean']:.2f}) -- the confound the first probe ran into")
check(raw["spearman_score_turns_mean"] is not None and raw["spearman_score_turns_mean"] < -0.9,
      "and its score-turns correlation is reported, strongly negative")

# first divergence: same prompt, one row each; the scorer prefers the winner's row
def div_recs(prefer_winner):
    rows_sum = {"plain": np.zeros(8), "privileged": np.zeros(8)}
    cnt = np.ones(8)
    recs = {}
    layout = [("w1", True, ["go a", "take x"]), ("w2", True, ["go a", "take x"]),
              ("l1", False, ["go a", "open b"]), ("l2", False, ["go a", "open b"])]
    i = 0
    for t, won, acts in layout:
        rows = []
        for step, act in enumerate(acts):
            rows.append((step, i, act))
            if step == 1:
                rows_sum["privileged"][i] = (1.0 if won else -1.0) * (1 if prefer_winner else -1)
            i += 1
        recs[t] = {"traj": t, "uid": "D", "won": won, "turns": 2, "score": {}, "_rows": rows,
                   "actions": acts, "ret": 10.0 if won else 0.0, "task": "alfworld", "gamefile": ""}
    return recs, rows_sum, cnt


for pref, want in ((True, 1.0), (False, 0.0)):
    rr, sums_, cnt_ = div_recs(pref)
    dv = orank.divergence_accuracy(rr, orank.classify_groups(rr), sums_, cnt_, "privileged", n_boot=50)
    check(dv["accuracy_mean"] == want and dv["pairs"] == 4
          and dv["per_group"] == [{"uid": "D", "turn": 1, "pairs": 4, "hits": 4.0 * want}],
          f"first-divergence accuracy is {want} when the scorer {'prefers' if pref else 'rejects'} the "
          f"winners' rows at the turn the actions split, kept per group for pooling ({dv})")
rr, sums_, cnt_ = div_recs(True)
dg = orank.divergence_accuracy(rr, orank.classify_groups(rr), sums_, cnt_, "privileged_gain", n_boot=50)
check(dg["accuracy_mean"] == 1.0, "and the gain form reads the same rows minus plain")

# walkthrough progress and the stuck-group diagnostic
walk = ["go to a", "take x from a", "go to b", "put x in b"]
cov, lcs = orank.walkthrough_progress(["go to a", "look", "take x from a", "go to c"], walk)
check(abs(cov - 0.5) < 1e-12 and abs(lcs - 0.5) < 1e-12,
      f"walkthrough progress: cover {cov}, lcs {lcs} for a trajectory that did the first two steps")
stuck = {}
for k, n_done in enumerate([0, 1, 2, 3]):
    t = f"S{k}"
    stuck[t] = {"traj": t, "uid": "S", "won": False, "turns": 50, "tokens": 1.0, "ret": 0.0,
                "task": "alfworld", "gamefile": "g", "actions": walk[:n_done] + ["look"] * 3,
                "score": {"privileged": float(n_done)}, "_rows": []}
dd = orank.degenerate_diagnostics(stuck, orank.classify_groups(stuck), "privileged", walk_of=lambda gf: walk)
check(dd["stuck"]["groups"] == 1 and dd["stuck"]["spearman_score_walk_cover_mean"] > 0.9,
      f"inside a stuck group, a scorer that tracks walkthrough progress shows it ({dd['stuck']})")


# --- 6. the three bugs the first rank probe had --------------------------------
# (1) a padding copy must not take its source trajectory out with it
tu = ["a", "a", "a", "b", "b"]
cand = [True, True, False, True, True]   # row 2 is adjust_batch's copy of an "a" row
okk = [True, True, False, True, False]   # row 4 is a real "b" row that could not be conditioned
score_m, dropped_ = orank.select_scored_rows(tu, cand, okk)
check(score_m.tolist() == [True, True, False, False, False] and dropped_ == ["b"],
      "bug 1: a padding copy never drops the trajectory it duplicates; a real row that fails still does")

# (2) rank correlation with ties
check(orank._spearman([3, 3, 3, 3, 3], [0.1, 0.5, 0.2, 0.9, 0.3]) is None,
      "bug 2: a side that does not vary has no rank correlation (the argsort ranking gave 0.5)")
rho_t = orank._spearman([0, 0, 0, 1, 1], [5, 4, 3, 2, 1])
check(rho_t is not None and abs(rho_t + 0.8660254037844386) < 1e-9,
      f"and tied values share their average rank: {rho_t:.4f} (scipy -0.8660; the old ranking gave -1.0)")
check(orank._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
      and orank._spearman([1.0, 2.0, float("nan")], [1, 2, 3]) is None,
      "a monotone pair is 1.0, and a NaN gives None")

# (3) a group's class from ALL its trajectories, not the scored ones
st_all = orank.group_status_from_rows(["G", "G", "G", "G", "H", "H"],
                                      ["w1", "w1", "l1", "l2", "h1", "h2"],
                                      [10.0, 10.0, 0.0, 0.0, 10.0, 10.0], [True] * 6)
check(st_all == {"G": "live", "H": "saturated"}, f"bug 3: classes read from every real trajectory ({st_all})")
check(orank.group_status_from_rows(["G", "G"], ["w", "l"], [10.0, 0.0], [True, False]) == {"G": "saturated"},
      "and a padding copy or special row does not count as one of the group's trajectories")
kept = synth({"G": [(True, 10, {"s": 1.0}), (True, 12, {"s": 0.5}), (True, 14, {"s": 0.2})]})
rep_old = orank.rank_report(kept, {}, np.ones(1), n_boot=10)
rep_new = orank.rank_report(kept, {}, np.ones(1), n_boot=10, status=st_all)
check(rep_old["groups_by_class"]["saturated"] == 1 and rep_new["groups_by_class"]["live"] == 1
      and rep_new["s"]["degenerate"]["saturated"]["groups"] == 0 and rep_new["records"][0]["status"] == "live",
      "a live group whose losers were not scored stays live, and is not read as a saturated one")

# --- 7. the rollout stores the document prompt text only when the check is on --
import agent_system.multi_turn_rollout.rollout_loop as _rl  # noqa: E402
from verl.utils.dataset.rl_dataset import collate_fn  # noqa: E402


def _collector(self_check):
    return _rl.TrajectoryCollector(
        config=OmegaConf.create({
            "data": {"max_prompt_length": 4096, "truncation": "left", "return_raw_chat": False,
                     "apply_chat_template_kwargs": {}},
            "env": {"rollout": {"n": 2}},
            "algorithm": {"oci_slots": {"enable": False},
                          "oci_rank": {"enable": True, "tasks": ["alfworld"],
                                       "self_check_rows": self_check}},
        }), tokenizer=tok, processor=None)


from verl.protocol import DataProto as _DP  # noqa: E402

gb2 = _DP.from_dict(tensors={"dummy": torch.zeros(2)},
                    non_tensors={"raw_prompt": np.array([[{"role": "user", "content": "x"}]] * 2, dtype=object),
                                 "data_source": np.array(["alfworld"] * 2, dtype=object),
                                 "task_name": np.array(["alfworld"] * 2, dtype=object)})
obs2 = {"text": [plain_content] * 2, ol.OCI_DOC_KEY: [doc_content, ""]}
col_on = _collector(8)
on_rows = [col_on.preprocess_single_sample(item=i, gen_batch=gb2, obs=obs2) for i in range(2)]
check(on_rows[0].get("oci_doc_prompt") == tok.apply_chat_template(
          [{"role": "user", "content": doc_content}], add_generation_prompt=True, tokenize=False)
      and on_rows[1].get("oci_doc_prompt") == "" and int(on_rows[0]["oci_doc_len"]) > 0,
      "self_check_rows > 0: a row carries the document prompt text its edit was taken against ('' without one)")
ph = col_on._placeholder_single_sample(item=1, gen_batch=gb2, obs=obs2, template=on_rows[0])
stacked2 = collate_fn([on_rows[0], ph])
check(ph.get("oci_doc_prompt") == "" and len(stacked2["oci_doc_prompt"]) == 2,
      "and a finished row carries an empty one, so collate sees one schema")
off_row = _collector(0).preprocess_single_sample(item=0, gen_batch=gb2, obs=obs2)
check("oci_doc_prompt" not in off_row and int(off_row["oci_doc_len"]) > 0,
      "self_check_rows = 0: no text column at all, and the edit is still recorded")

# --- 8. both self-checks on the real tokenizer ----------------------------------
CKPT = "/opt1/ohara/offline_ladder/probe_hf/klwctl_step300"
if not os.path.isdir(CKPT) or gf is None:
    print(f"  SKIP  no tokenizer at {CKPT}; the real-tokenizer self-check needs one")
else:
    from transformers import AutoTokenizer

    from verl.utils.torch_functional import tokenize_and_postprocess_data

    rtok = AutoTokenizer.from_pretrained(CKPT)
    real_plain = ("You are an expert agent operating in the ALFRED Embodied Environment.\n"
                  "Your current observation is: You are in the middle of a room. Looking quickly around you, "
                  "you see a cabinet 1, a countertop 1, and a microwave 1.\nYour admissible actions of the "
                  "current situation are: [\n 'go to cabinet 1'\n 'go to countertop 1'\n 'look'].\n"
                  "Now it's your turn to take an action.")
    real_doc = em._build_wrong_plan(gf, "walkthrough") + real_plain
    rmsgs = [{"role": "user", "content": real_plain}]
    redit = rl._oci_render_edit(rtok, rmsgs, real_doc, 4096, {}, prompt_window=4096)
    plain_r = rtok.apply_chat_template(rmsgs, add_generation_prompt=True, tokenize=False)
    doc_r = rtok.apply_chat_template([{"role": "user", "content": real_doc}], add_generation_prompt=True,
                                     tokenize=False)
    check(redit is not None, "real tokenizer: the document edit is recordable")
    if redit:
        PW, RR = len(rtok(plain_r, add_special_tokens=False)["input_ids"]) + 7, 4
        p_ids, p_am = tokenize_and_postprocess_data(plain_r, rtok, PW, 0, left_pad=True, truncation="error")
        resp_ids = torch.tensor([[rtok.eos_token_id or 1, 11, 12, 0]])
        resp_am = torch.tensor([[1, 1, 1, 0]])
        p_pos = torch.clamp(torch.cumsum(p_am[0], -1) - 1, min=0)
        rb = DataProto.from_dict(tensors={
            "input_ids": torch.cat([p_ids, resp_ids], 1), "attention_mask": torch.cat([p_am, resp_am], 1),
            "position_ids": torch.cat([p_pos, p_pos[-1] + torch.arange(1, RR + 1)]).unsqueeze(0),
            "responses": resp_ids,
            "oci_doc_off": torch.tensor([redit[0]]), "oci_doc_len": torch.tensor([redit[1]]),
            "oci_doc_repl": torch.tensor([list(redit[2]) + [0] * (4096 - len(redit[2]))], dtype=torch.int32),
            "oci_doc_repl_len": torch.tensor([len(redit[2])])})
        rdoc, rspl = orank.with_document(rb, 0)
        rW = int(rdoc.batch["input_ids"].shape[1]) - RR
        rdir, rtl = orank.direct_render(rdoc, [doc_r], rtok, 0, rW)
        mask_s, mask_d = rdoc.batch["attention_mask"][0].bool(), rdir.batch["attention_mask"][0].bool()
        check(bool(rspl[0]) and not rtl
              and rdir.batch["input_ids"][0][mask_d].tolist() == rdoc.batch["input_ids"][0][mask_s].tolist()
              and rdir.batch["position_ids"][0][mask_d].tolist() == rdoc.batch["position_ids"][0][mask_s].tolist()
              and orank.prompt_tokens(rdoc, [0])[0] == rtok(doc_r, add_special_tokens=False)["input_ids"],
              "real tokenizer: the splice equals the document prompt tokenized from text (prompt, response, positions)")
        rnull, _ = orank.with_document(orank.null_edit(rb), 0, min_grow=rW - PW)
        check(orank.prompt_tokens(rnull, [0]) == orank.prompt_tokens(rb, [0])
              and rnull.batch["position_ids"][0][rnull.batch["attention_mask"][0].bool()].tolist()
              == rb.batch["position_ids"][0][rb.batch["attention_mask"][0].bool()].tolist(),
              "real tokenizer: the null edit gives back the plain prompt and positions")


# --- 9. the driver's report end to end, on fake workers ---------------------------
# A fake model whose log-prob of each response token depends on every live token
# before it and on its live position -- so it notices a wrong splice, a wrong
# position or a padding leak, and is blind to left padding like a real one.
from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer  # noqa: E402


def _fake_lp(data):
    ids, am = data.batch["input_ids"], data.batch["attention_mask"]
    pos = data.batch["position_ids"]
    R = int(data.batch["responses"].shape[1])
    out = torch.zeros(ids.shape[0], R)
    for i in range(ids.shape[0]):
        live = am[i].bool()
        toks, p = ids[i][live].tolist(), pos[i][live].tolist()
        n_prompt = int(am[i, :ids.shape[1] - R].sum())
        acc = 0
        for t in range(len(toks)):
            if t >= n_prompt:
                out[i, t - n_prompt] = -((acc + 7 * p[t]) % 97) / 100.0
            acc = (acc * 31 + toks[t]) % 1000003
    return out


class _FakeWG:
    world_size = 2

    def __init__(self, key):
        self.key = key

    def compute_log_prob(self, data):
        return DataProto.from_dict(tensors={self.key: _fake_lp(data)})

    compute_ref_log_prob = compute_log_prob


def _rank_batch(corrupt_row=None):
    R, rows = 3, []
    layout = [("G1", "a", True), ("G1", "b", False), ("G2", "c", True), ("G2", "d", True)]
    for uid, tr, won in layout:
        for k in range(2):
            plain = f"obs {uid} {tr} turn {k}\nNow act."
            doc = f"DOC walkthrough for {uid}\nstep one\n\n" + plain
            msgs = [{"role": "user", "content": plain}]
            e = rl._oci_render_edit(tok, msgs, doc, 4096, {}, prompt_window=4096)
            rows.append(dict(uid=uid, tr=tr, won=won, k=k, ids=tok.encode(tok.apply_chat_template(msgs)),
                             edit=e, text=tok.apply_chat_template([{"role": "user", "content": doc}])))
    rows.append(dict(rows[0], copy=True))   # adjust_batch's copies: same traj_uid as their source
    rows.append(dict(rows[5], copy=True))
    P = max(len(r["ids"]) for r in rows) + 2
    n = len(rows)
    ids = torch.zeros(n, P + R, dtype=torch.long); am = torch.zeros_like(ids); pos = torch.zeros_like(ids)
    off, take, rl_ = (torch.zeros(n, dtype=torch.long) for _ in range(3))
    repl = torch.zeros(n, 4096, dtype=torch.int32)
    for i, r in enumerate(rows):
        L = len(r["ids"])
        ids[i, P - L:P] = torch.tensor(r["ids"]); am[i, P - L:P] = 1
        pos[i, P - L:P] = torch.arange(L)
        ids[i, P:] = torch.tensor([100 + i, 200 + r["k"], 300]); am[i, P:] = 1
        pos[i, P:] = L - 1 + torch.arange(1, R + 1)
        o, t_, rp = r["edit"]
        off[i], take[i], rl_[i] = o + (1 if i == corrupt_row else 0), t_, len(rp)
        repl[i, :len(rp)] = torch.tensor(list(rp), dtype=torch.int32)
    b = DataProto.from_dict(
        tensors={"input_ids": ids, "attention_mask": am, "position_ids": pos, "responses": ids[:, P:].clone(),
                 "response_mask": am[:, P:].clone(), "oci_doc_off": off, "oci_doc_len": take,
                 "oci_doc_repl": repl, "oci_doc_repl_len": rl_,
                 "is_padding_row": torch.tensor([bool(r.get("copy")) for r in rows])},
        non_tensors={"uid": np.array([r["uid"] for r in rows], dtype=object),
                     "traj_uid": np.array([r["tr"] for r in rows], dtype=object),
                     "turn_step": np.array([r["k"] for r in rows], dtype=object),
                     "task_name": np.array(["alfworld"] * n, dtype=object),
                     "gamefile": np.array([""] * n, dtype=object),
                     "episode_rewards": np.array([10.0 if r["won"] else 0.0 for r in rows], dtype=object),
                     "oci_doc_prompt": np.array([r["text"] for r in rows], dtype=object)})
    return b


tok.decode = lambda ids, skip_special_tokens=True: f"<think>x</think><action>act {ids[-1] % 2}</action>"
_fake_self = SimpleNamespace(
    config=OmegaConf.create({"algorithm": {"oci_rank": {"enable": True, "tasks": ["alfworld"],
                                                        "self_check_rows": 4}}}),
    tokenizer=tok, actor_rollout_wg=_FakeWG("old_log_probs"),
    teacher_wg={"alfworld": _FakeWG("ref_log_prob")}, _normalize_task_name=lambda t: t)
_fake_self._padded_log_prob = OPDRayTrainer._padded_log_prob.__get__(_fake_self)
_fake_self._oci_rank_self_check = OPDRayTrainer._oci_rank_self_check.__get__(_fake_self)
rep = OPDRayTrainer._oci_rank_report(_fake_self, _rank_batch(), {"rank_bootstrap": 20})
sc = rep.get("self_check", {})
check("error" not in rep and rep.get("rows_padding_copies") == 2 and rep.get("trajectories_dropped_incomplete") == 0
      and rep.get("rows_scored") == 8 and rep.get("trajectories") == 4,
      f"driver: two padding copies, all 4 trajectories scored on all 8 rows "
      f"({ {k: rep.get(k) for k in ('rows_padding_copies', 'rows_scored', 'trajectories_dropped_incomplete')} })")
check(rep.get("groups_by_class") == {"live": 1, "stuck": 0, "saturated": 1},
      f"driver: classes from all real trajectories ({rep.get('groups_by_class')})")
check("error" not in sc and sc.get("rows") == 4
      and sc["null_edit"]["max_abs_row_sum_diff"] == 0.0 and sc["null_edit"]["rows_prompt_identical"] == 4
      and sc["direct_render"]["max_abs_row_sum_diff"] == 0.0 and sc["direct_render"]["rows_prompt_identical"] == 4
      and sc["noise_floor"]["plain"]["max_abs_row_sum_diff"] == 0.0,
      f"driver self-checks pass on a correct splice ({sc})")
# Every scored row checked (self_check_rows >= 8), one of them spliced one token off.
_fake_self.config.algorithm.oci_rank.self_check_rows = 8
bad_rep = OPDRayTrainer._oci_rank_report(_fake_self, _rank_batch(corrupt_row=1), {"rank_bootstrap": 20})
bsc = bad_rep.get("self_check", {})
check("direct_render" in bsc and bsc["direct_render"]["rows"] == 8
      and bsc["direct_render"]["rows_prompt_identical"] == 7 and bsc["direct_render"]["max_abs_row_sum_diff"] > 0.0
      and bsc["null_edit"]["max_abs_row_sum_diff"] == 0.0,
      f"and the direct-render check catches an edit recorded one token off, which the null edit cannot "
      f"({bsc.get('direct_render')})")


def test_rank():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
