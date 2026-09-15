"""CPU test: ten rollouts per group, eight trained, and which eight.

WHAT THIS FILE IS FOR. The arm is a LAYOUT, and every part of it is a place where
two numbers have to agree: the env manager decides what a slot is shown from its
own local slot index, the rollout loop records the slot from the row's position in
the generation batch, and the driver reads both back after the batch has been
regrouped by task, padded and reordered. The single-candidate arm got exactly
this wrong once -- a mark keyed by env slot, read by the global row index, marked
one alfworld group of fifteen and passed its own "at least one is marked" assert
-- so the checks below run the real manager and hold the mark against the
position rather than trusting either.

The rest is the rule itself: a group trains its eight ordinary rollouts unless it
is degenerate, in which case the eighth is replaced by the special rollout that
disagrees -- and never by one that cannot be re-scored on the plain prompt, since
that is what makes the shaped term meaningful.
"""
import glob, json, math, os, re, sys, zlib
from types import SimpleNamespace

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import agent_system.environments.env_manager as em  # noqa: E402
from agent_system.environments import oci_layout as ol  # noqa: E402
from verl.trainer.ppo import oci_slots as osl  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def cfg(group_n_note="", **over):
    slots = {"enable": True, "tasks": ["alfworld"], "doc_mode": "walkthrough_stepwise",
             "foreign_task": "webshop", "gamma": 0.1, "opd_on_special": False}
    slots.update(over)
    return SimpleNamespace(env=SimpleNamespace(history_length=2),
                           algorithm={"oci_slots": slots})


def envs(group_n=10, is_train=True):
    return SimpleNamespace(group_n=group_n, is_train=is_train)


# --- 1. which slot is which --------------------------------------------------
roles = [ol.role_for_slot(i, 10) for i in range(10)]
check(roles == [ol.ROLE_PLAIN] * 7 + [ol.ROLE_RESERVE, ol.ROLE_DOC, ol.ROLE_FOREIGN],
      f"group_n=10: seven plain, then reserve, document, foreign ({roles})")
check(ol.used_per_group(10) == 8, "eight trajectories per group are trained, as control trains eight")
check([ol.role_for_slot(i, 4) for i in range(4)]
      == [ol.ROLE_PLAIN, ol.ROLE_RESERVE, ol.ROLE_DOC, ol.ROLE_FOREIGN],
      "the layout still holds at its smallest size (group_n=4)")
check(all(ol.role_for_slot(i, 3) == ol.ROLE_NONE for i in range(3)),
      "below four there is no layout: a verdict needs two ordinary rollouts and the slots need two more")
check([ol.slot_role(i, envs(), cfg()) for i in range(10)] == roles,
      "a training manager marks its slots")
check(all(ol.slot_role(i, envs(group_n=1, is_train=False), cfg()) == ol.ROLE_NONE for i in range(4)),
      "the VALIDATION manager marks nothing -- it is what the arm is measured with")
check(all(ol.slot_role(i, envs(), None) == ol.ROLE_NONE for i in range(10))
      and all(ol.slot_role(i, envs(), cfg(enable=False)) == ol.ROLE_NONE for i in range(10)),
      "switch off, or no config at all: no slot is special")
# The arithmetic the rollout loop's slot column rests on. Under
# TASK_BALANCE_INTERLEAVE the batch is laid out at the PROMPT level (alf0,
# search0, webshop0, alf1, ...) and each prompt is then repeated group_n times
# CONTIGUOUSLY -- so every group starts at a multiple of group_n and a row's
# global position modulo group_n is its slot within its own group. That is the
# same property tests/oci/test_interleaved_layout.py pins for the mark itself;
# if it ever stopped holding, select_rollouts refuses the batch rather than
# training it, because the role and the position would disagree.
_global = [3 * 10 * (i // 10) + i % 10 for i in range(15 * 10)]
check(all(g % 10 == i % 10 for i, g in enumerate(_global)),
      "in the task-interleaved batch a row's global position modulo the group size is its slot")

# --- 2. the foreign prompt is another task's OWN prompt -----------------------
key = "/data/alfworld/json_2.1.1/train/pick_and_place-Mug-None-Desk-301/trial_1/game.tw-pddl"
instr = ol.WEBSHOP_TRAIN_INSTRUCTIONS[zlib.crc32(ol.game_key(key).encode()) % len(ol.WEBSHOP_TRAIN_INSTRUCTIONS)]
check(ol.game_key(key) == "train/pick_and_place-Mug-None-Desk-301/trial_1/game.tw-pddl"
      and ol.foreign_prompt("webshop", key) == ol.foreign_prompt("webshop", "/opt1/other/root" + key),
      "the prompt is chosen by the path below the split directory, so two hosts with the data "
      "under different roots show a game the same one")
wm = em.WebshopEnvironmentManager.__new__(em.WebshopEnvironmentManager)
wm.config = SimpleNamespace(env=SimpleNamespace(history_length=2))
wm.tasks = [instr]
# What WebShop's own manager builds on its landing page, from the observation and
# available actions a live WebShop env returns at reset.
webshop_own = wm.build_text_obs(
    ["'Search'"],
    [{"available_actions": {"has_search_bar": True, "clickables": ["search"]}}],
    init=True)[0]
check(ol.foreign_prompt("webshop", key) == webshop_own,
      "the foreign slot is shown WebShop's own turn-0 prompt, byte for byte")
check(ol.foreign_prompt("webshop", key) == ol.foreign_prompt("webshop", key)
      and len({ol.foreign_prompt("webshop", f"game{i}") for i in range(40)}) > 1,
      "the same game is shown the same prompt every time, and different games spread over the pool")
check("<action>" in webshop_own and "<think>" in webshop_own and "<search>" not in webshop_own,
      "it asks for the <think>/<action> shape alfworld's projection checks, so the rows take no "
      "invalid-action penalty")

# --- 3. the manager, end to end, on a real game ------------------------------
games = sorted(glob.glob(os.path.expanduser(
    "~/data/alfworld/json_2.1.1/train/pick_heat_then_place_in_recep*/*/game.tw-pddl")))
gf = next((g for g in games if len(json.load(open(g)).get("walkthrough") or []) >= 5), None)
if gf is None:
    print("SKIP: no long alfworld walkthrough on this host")
else:
    walk = json.load(open(gf))["walkthrough"]

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
    c = cfg()
    m = em.AlfWorldEnvironmentManager(FakeEnvs(), lambda acts, adm: (list(acts), [1] * len(acts)), c)
    obs, _ = m.reset(kwargs=None)
    text, role, plain = obs["text"], obs[ol.OCI_ROLE_KEY], obs[ol.OCI_PLAIN_KEY]
    block = em._build_wrong_plan(gf, "walkthrough")

    check(role == [ol.ROLE_PLAIN, ol.ROLE_RESERVE, ol.ROLE_DOC, ol.ROLE_FOREIGN],
          f"the manager marks its own slots ({role})")
    check(all("[Privileged Solution Path]" not in t and "WebShop" not in t for t in text[:2]),
          "the two ordinary slots are shown nothing: they are what the group is judged on")
    check(text[2].startswith(block) and f"step 1 of {len(walk)}: {walk[0]}" in text[2],
          "the document slot carries the walkthrough and the line naming the step it owes")
    check(text[3] == ol.foreign_prompt("webshop", gf),
          "the foreign slot carries another task's prompt and nothing of this one")
    check(plain[0] == "" and plain[1] == "",
          "an ordinary slot has nothing to strip")
    check(plain[2] and plain[3] and plain[2] == plain[3] == text[0],
          "both special slots carry the render an ordinary slot got this turn")
    check(text[2] != plain[2] and text[3] != plain[3],
          "and it differs from what they were actually shown")

    def step(actions):
        o, *_ = m.step(actions)
        return o

    o1 = step(["look", "look", walk[0], "<action>search[shorts]</action>"])
    check(f"Your next action is step 2 of {len(walk)}: {walk[1]}" in o1["text"][2],
          "taking the owed step moves the document slot's pointer")
    check(m._guide_ptr == [0, 0, 1, 0], f"and moves nobody else's ({m._guide_ptr})")
    check(o1["text"][3] == ol.foreign_prompt("webshop", gf),
          "the foreign slot keeps being asked the other task's turn-0 question")
    _fp = o1[ol.OCI_PLAIN_KEY][3]
    check(_fp.lstrip().startswith("You are an expert agent operating in the ALFRED")
          and "search[shorts]" in _fp and "WebShop" not in _fp,
          "its plain render is the alfworld prompt this trajectory would have had at this "
          "turn -- its own actions in the history -- so rho is taken on THIS turn's prompt")

    # training success counts the ordinary slots only
    batch_list = [[{"active_masks": True}] for _ in range(4)]
    infos = [[{"won": False, "extra.gamefile": gf}], [{"won": True, "extra.gamefile": gf}],
             [{"won": True, "extra.gamefile": gf}], [{"won": False, "extra.gamefile": gf}]]
    succ = m.success_evaluator(total_infos=infos, total_batch_list=batch_list,
                               episode_rewards=None, episode_lengths=None)
    check(succ["success_rate"].tolist() == [0.0, 1.0],
          f"success_rate is the ordinary slots' ({succ['success_rate'].tolist()})")
    check(succ["oci_doc_success_rate"].tolist() == [1.0]
          and succ["oci_foreign_success_rate"].tolist() == [0.0],
          "the document and foreign slots are reported on their own keys")

    # both privileged arms at once is refused, loudly
    c2 = cfg()
    c2.algorithm["oci_sat"] = {"enable": True, "plan_corruption": "walkthrough"}
    m2 = em.AlfWorldEnvironmentManager(FakeEnvs(), lambda acts, adm: (list(acts), [1] * len(acts)), c2)
    try:
        m2.reset(kwargs=None)
        check(False, "two privileged arms at once should have been refused")
    except ValueError as exc:
        check("oci_sat" in str(exc) and "oci_slots" in str(exc),
              "oci_sat and oci_slots together are refused at reset")

# --- 3b. the columns the rollout loop writes ---------------------------------
# A toy tokenizer, as tests/ray_cpu/test_privileged_notice_rollout.py uses: the
# same chat rendering with whitespace tokens. What is checked here is the
# plumbing -- which row is marked, how wide the replacement column is, and that
# collate still sees one schema -- not the text, which section 4 checks with the
# real tokenizer.
from omegaconf import OmegaConf  # noqa: E402

import agent_system.multi_turn_rollout.rollout_loop as rl  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.utils.dataset.rl_dataset import collate_fn  # noqa: E402


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


def collector(enable=True, n=10, width=4096):
    return rl.TrajectoryCollector(
        config=OmegaConf.create({
            "data": {"max_prompt_length": width, "truncation": "left",
                     "return_raw_chat": False, "apply_chat_template_kwargs": {}},
            "env": {"rollout": {"n": n}},
            "algorithm": {"oci_slots": {"enable": enable, "tasks": ["alfworld"],
                                        "doc_mode": "walkthrough_stepwise",
                                        "foreign_task": "webshop"}},
        }),
        tokenizer=_Tok(), processor=None)


plain_text = "\nYou are an expert agent. Your admissible actions are: ['go to cabinet 1'].\n"
shown = {8: "[Privileged Solution Path]\n1. go to cabinet 1\n\n" + plain_text + "\n[progress] step 1\n",
         9: ol.foreign_prompt("webshop", "g")}
rows_obs = {"text": [shown.get(i, plain_text) for i in range(10)],
            ol.OCI_ROLE_KEY: [ol.role_for_slot(i, 10) for i in range(10)],
            ol.OCI_PLAIN_KEY: ["" if i < 8 else plain_text for i in range(10)]}
gb = DataProto.from_dict(
    tensors={"dummy": torch.zeros(10)},
    non_tensors={"raw_prompt": np.array([[{"role": "user", "content": "x"}]] * 10, dtype=object),
                 "data_source": np.array(["alfworld"] * 10, dtype=object),
                 "task_name": np.array(["alfworld"] * 10, dtype=object)})
col = collector()
built = [col.preprocess_single_sample(item=i, gen_batch=gb, obs=rows_obs) for i in range(10)]
check([int(r["oci_slot"]) for r in built] == list(range(10))
      and [int(r["oci_role"]) for r in built] == [ol.role_for_slot(i, 10) for i in range(10)],
      "every row records which slot of its group it is and what that slot is for")
check(all(int(r["oci_candidate"]) == 0 and int(r["oci_plan_len"]) == 0 for r in built[:8]),
      "the eight ordinary rows have no span to strip")
check(all(int(r["oci_candidate"]) == 1 and int(r["oci_plan_len"]) > 0
          and int(r["oci_plan_repl_len"]) > 0 for r in built[8:]),
      "both special rows do -- the two-place edit and the whole-prompt swap alike")
stacked = collate_fn(built)
check(tuple(stacked["oci_plan_repl"].shape) == (10, 4096),
      f"collate sees one schema, at the prompt's own width {tuple(stacked['oci_plan_repl'].shape)}")
off_row = collector(enable=False).preprocess_single_sample(
    item=3, gen_batch=gb, obs={"text": [plain_text] * 10})
check(off_row["oci_plan_repl"].shape[0] == rl.OCI_REPL_WIDTH and int(off_row["oci_slot"]) == 0
      and int(off_row["oci_role"]) == 0,
      "with the layout off the column stays narrow and the marks stay zero")

# --- 4. the strip: one replacement that reproduces the plain render -----------
CKPT = "/opt1/ohara/offline_ladder/probe_hf/klwctl_step300"
if not os.path.isdir(CKPT) or gf is None:
    print(f"  SKIP  no tokenizer at {CKPT}; the render-edit check needs one")
else:
    from transformers import AutoTokenizer

    from agent_system.multi_turn_rollout.rollout_loop import _oci_render_edit
    from verl.trainer.ppo.oci_reachability import splice_span

    tok = AutoTokenizer.from_pretrained(CKPT)
    PAD = tok.pad_token_id or 0

    def round_trip(shown, plain, width, window=4096):
        msgs = [{"role": "user", "content": shown}]
        edit = _oci_render_edit(tok, msgs, plain, width, {}, prompt_window=window)
        if edit is None:
            return None
        off, take, repl = edit
        ids_w = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                           add_special_tokens=False)
        ids_o = tok.encode(tok.apply_chat_template([{"role": "user", "content": plain}],
                                                   add_generation_prompt=True, tokenize=False),
                           add_special_tokens=False)
        # The row's prompt region, as production has it: data.max_prompt_length,
        # which the edit is checked against. A window too small for the PLAIN
        # render is what the guard below is about.
        P, R = max(len(ids_w), len(ids_o)) + 3, 4
        ids = torch.full((1, P + R), PAD, dtype=torch.long)
        mask = torch.zeros((1, P + R), dtype=torch.long)
        ids[0, P - len(ids_w):P] = torch.tensor(ids_w)
        mask[0, P - len(ids_w):P] = 1
        ids[0, P:P + 2] = torch.tensor([9001, 9002])
        mask[0, P:P + 2] = 1
        padded = torch.tensor([repl + [0] * (width - len(repl))])
        s_ids, s_mask, _ = splice_span(ids, mask, torch.tensor([off]), torch.tensor([take]),
                                       padded, torch.tensor([len(repl)]), PAD, response_length=R)
        got = s_ids[0, :P][s_mask[0, :P].bool()].tolist()
        return got == ids_o, len(repl), torch.equal(s_ids[0, P:], ids[0, P:])

    plain_obs = em.ALFWORLD_TEMPLATE.format(
        task_description="heat some mug and put it somewhere", step_count=2, history_length=2,
        action_history="[Observation 1: 'x', Action 1: 'go to fridge 1']",
        current_step=3, current_observation="You see a mug 1.",
        admissible_actions="'go to fridge 1'\n 'take mug 1 from countertop 1'")
    doc_shown = em._insert_guide(em._build_wrong_plan(gf, "walkthrough") + plain_obs,
                                 em._guide_line(walk, 2))
    got = round_trip(doc_shown, plain_obs, 4096)
    check(got is not None and got[0] and got[2],
          f"the document slot's TWO edits (block and progress line) come out as one "
          f"replacement that reproduces the plain render{'' if got is None else f' ({got[1]} tokens)'}, "
          "with the response region untouched")
    foreign_shown = ol.foreign_prompt("webshop", gf)
    got_f = round_trip(foreign_shown, plain_obs, 4096)
    check(got_f is not None and got_f[0] and got_f[2],
          "the foreign slot's whole-prompt swap does too")
    check(round_trip(doc_shown, plain_obs, 16) is None,
          "a replacement that does not fit the column is reported as not strippable, "
          "not spliced approximately")
    # The foreign slot's prompt is SHORTER than the alfworld one it stands in for,
    # so its splice grows the prompt. Past the window splice_span leaves the row
    # untouched -- and an untouched row is re-scored on the prompt it was
    # generated with, i.e. rho = 1 and the full advantage on exactly the
    # conditional distribution the shaping exists to keep it off.
    check(round_trip(foreign_shown, plain_obs, 4096, window=32) is None,
          "a plain render that does not fit the prompt window is not strippable either")

# --- 5. the rule: which eight a group trains ---------------------------------
TURNS = 2
GROUP_N = 10


def group(uid, task, plain_rets, reserve_ret, doc_ret, foreign_ret,
          doc_strippable=True, foreign_strippable=True, layout=True):
    """One group's rows: seven plain, a reserve, a document and a foreign slot."""
    rows = []
    specs = [(s, r, ol.role_for_slot(s, GROUP_N) if layout else ol.ROLE_NONE)
             for s, r in enumerate(list(plain_rets) + [reserve_ret, doc_ret, foreign_ret])]
    for slot, ret, role in specs:
        strip = 1
        if role == ol.ROLE_DOC and not doc_strippable:
            strip = 0
        if role == ol.ROLE_FOREIGN and not foreign_strippable:
            strip = 0
        for _ in range(TURNS):
            rows.append({"uid": uid, "traj_uid": f"{uid}:{slot}", "task_name": task,
                         "episode_rewards": float(ret), "oci_role": role,
                         "oci_slot": slot, "oci_plan_len": (3 if role in (ol.ROLE_DOC, ol.ROLE_FOREIGN) else 0) * strip,
                         "oci_plan_truncated": 0})
    return rows


def batch_of(rows):
    tens = {k: torch.tensor([r[k] for r in rows], dtype=torch.long)
            for k in ("oci_role", "oci_slot", "oci_plan_len", "oci_plan_truncated")}
    nont = {k: np.array([r[k] for r in rows], dtype=object)
            for k in ("uid", "traj_uid", "task_name", "episode_rewards")}
    return SimpleNamespace(batch=tens, non_tensor_batch=nont, meta_info={})


rows = []
rows += group("live", "alfworld", [10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0], 0.0, 10.0, 0.0)
rows += group("stuck_rescued", "alfworld", [0.0] * 7, 0.0, 10.0, 0.0)
rows += group("stuck_failed", "alfworld", [0.0] * 7, 0.0, 0.0, 0.0)
rows += group("stuck_unstrippable", "alfworld", [0.0] * 7, 0.0, 10.0, 0.0, doc_strippable=False)
rows += group("sat_failed", "alfworld", [10.0] * 7, 10.0, 10.0, 0.0)
rows += group("sat_survived", "alfworld", [10.0] * 7, 10.0, 10.0, 10.0)
rows += group("shop", "webshop", [1.0] * 7, 1.0, 1.0, 1.0, layout=False)
b = batch_of(rows)
keep, injected, metrics = osl.select_rollouts(b, tasks=["alfworld"], group_n=GROUP_N)

role_col = b.batch["oci_role"].numpy()
uid_col = np.array([r["uid"] for r in rows])


def kept_roles(uid):
    sel = (uid_col == uid) & keep
    return sorted({int(x) for x in role_col[sel]})


def n_traj(uid, mask):
    sel = (uid_col == uid) & mask
    return len({rows[i]["traj_uid"] for i in np.nonzero(sel)[0]})


check(all(n_traj(u, keep) == 8 for u in set(uid_col)),
      "every group trains exactly eight trajectories, the number control trains")
check(kept_roles("live") == [ol.ROLE_PLAIN, ol.ROLE_RESERVE],
      "a live group is the eight ordinary rollouts -- control's group exactly")
check(kept_roles("stuck_rescued") == [ol.ROLE_PLAIN, ol.ROLE_DOC],
      "a stuck group whose document rollout solved the game trades its reserve for it")
check(kept_roles("stuck_failed") == [ol.ROLE_PLAIN, ol.ROLE_RESERVE],
      "a stuck group whose document rollout also failed keeps its reserve (nothing to contrast)")
check(kept_roles("stuck_unstrippable") == [ol.ROLE_PLAIN, ol.ROLE_RESERVE],
      "a rescue that cannot be re-scored on the plain prompt is NOT used: rho would compare "
      "two prompts differing in more than the conditioning")
check(kept_roles("sat_failed") == [ol.ROLE_PLAIN, ol.ROLE_FOREIGN],
      "a saturated group takes the foreign rollout that failed")
check(kept_roles("sat_survived") == [ol.ROLE_PLAIN, ol.ROLE_RESERVE],
      "a foreign rollout that somehow did not fail leaves the group as control's")
check(kept_roles("shop") == [ol.ROLE_NONE] and n_traj("shop", keep) == 8
      and sorted(b.batch["oci_slot"].numpy()[(uid_col == "shop") & keep].tolist())[-1] == 7,
      "a task without the layout keeps its first eight rollouts, chosen before any return is read")

inj_uids = {rows[i]["uid"] for i in np.nonzero(injected)[0]}
check(inj_uids == {"stuck_rescued", "sat_failed"},
      f"only the special rollouts a group actually took are marked for the shaped term ({sorted(inj_uids)})")
check(all(bool(keep[i]) for i in np.nonzero(injected)[0]),
      "and every marked row is one of the eight")
check(metrics["oci_slots/groups_stuck/alfworld"] == 3
      and metrics["oci_slots/groups_saturated/alfworld"] == 2
      and metrics["oci_slots/groups_live/alfworld"] == 1,
      "the classes are counted per task")
check(metrics["oci_slots/doc_solved/alfworld"] == 2
      and metrics["oci_slots/document_used/alfworld"] == 1
      and metrics["oci_slots/doc_unstrippable/alfworld"] == 1,
      "a rescue that is not usable is counted separately from one that is")
check(metrics["oci_slots/foreign_failed/alfworld"] == 1
      and metrics["oci_slots/foreign_used/alfworld"] == 1,
      "so is the saturated side")
check(abs(metrics["oci_slots/doc_rescue_rate/alfworld"] - 2 / 3) < 1e-9
      and abs(metrics["oci_slots/foreign_fail_rate/alfworld"] - 0.5) < 1e-9,
      "the rates are taken over the class each slot serves")
check(metrics["oci_slots/rows_trained"] == int(keep.sum())
      and metrics["oci_slots/rows_dropped"] == int((~keep).sum())
      and metrics["oci_slots/rows_dropped"] == 2 * TURNS * 7,
      "two rollouts per group are dropped")

# a batch whose marks and positions disagree is refused rather than trained on
bad = batch_of(group("x", "alfworld", [0.0] * 7, 0.0, 10.0, 0.0))
bad.batch["oci_role"][0] = ol.ROLE_DOC
try:
    osl.select_rollouts(bad, tasks=["alfworld"], group_n=GROUP_N)
    check(False, "a role that its slot does not predict should be refused")
except AssertionError as exc:
    check("slot" in str(exc), "a role that its slot does not predict is refused")

# --- 6. what leaves the batch, and the mark that travels with it -------------
from verl import DataProto  # noqa: E402

proto = DataProto.from_dict(
    tensors={k: v for k, v in b.batch.items()},
    non_tensors={k: v for k, v in b.non_tensor_batch.items()})
out = osl.apply_selection(proto, keep, injected)
check(len(out) == int(keep.sum()), f"the dropped rows leave the batch ({len(proto)} -> {len(out)})")
check(int(out.batch[osl.INJECTED_KEY].sum()) == int(injected.sum()),
      "and the shaped-row mark travels with the rows that stay")

# --- 7. the actor's side: the special row trains the gradient, not the teacher -
src = open(os.path.join(REPO, "verl/workers/actor/dp_actor.py")).read()
check(re.search(r"_kld_for_loss = teacher_kld if _pb_w is None else teacher_kld \* _pb_w\n"
                r"\s+if oci_opd_skip:\n(\s+#[^\n]*\n)*\s+_kld_for_loss = _kld_for_loss \* _oci_opd_keep\(",
                src) is not None,
      "the distillation term is zeroed on the special rows at _kld_for_loss, which BOTH loss "
      "branches read (zeroing response_mask instead would take the policy gradient with it)")
check("oci_slots_on = bool(_oci_slots_cfg.get(\"enable\", False))" in src
      and "_oci_shape_wanted = bool(_oci_shape_cfg.get(\"enable\", False)) or oci_slots_on" in src,
      "the layout turns the shaping on by itself: there is no unshaped version of this arm")

# --- 7b. the shaped rows, driven through the real _oci_shaped_rows -------------
# The first two runs of this path on a GPU died inside it, each on something no
# CPU test had exercised: a tensor built on the wrong device, diagnostics handed
# to _defer as floats, and the plain log-prob taken from the wrong slot of the
# forward's 3-tuple. This drives the real function with a stand-in actor whose
# forward returns what the real one does -- (entropy, log_probs, topk_out) -- and
# holds the row replacement, the sub-batch it forwards, and the diagnostics'
# types against it. The return order itself is pinned by source, below.
try:
    from verl.workers.actor.dp_actor import _oci_opd_keep, _oci_shaped_rows
    _HAVE_ACTOR = True
except Exception as _exc:  # pragma: no cover - environment without the actor's deps
    print(f"  SKIP  dp_actor not importable here ({type(_exc).__name__}); shaped-row drive skipped")
    _HAVE_ACTOR = False
if _HAVE_ACTOR:
    PL, RL, BS = 8, 4, 3
    ids = torch.arange(100, 100 + BS * (PL + RL)).reshape(BS, PL + RL)
    mask = torch.ones(BS, PL + RL, dtype=torch.long)
    pos = torch.arange(PL + RL).repeat(BS, 1)
    mb = {"input_ids": ids, "attention_mask": mask, "position_ids": pos, "responses": ids[:, PL:].clone(),
          "oci_plan_off": torch.tensor([0, 1, 0]), "oci_plan_len": torch.tensor([0, 2, 0]),
          "oci_plan_repl": torch.zeros(BS, 4, dtype=torch.int32), "oci_plan_repl_len": torch.tensor([0, 0, 0]),
          "oci_plan_truncated": torch.tensor([0, 0, 0]), "oci_injected": torch.tensor([0, 1, 1])}
    seen = {}

    class _Actor:
        def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, **kw):
            seen["sub"] = micro_batch
            n = micro_batch["input_ids"].shape[0]
            return None, torch.full((n, RL), -1.0, requires_grad=True), None

    pg = torch.full((BS, RL), 7.0)
    adv = torch.tensor([[1.0] * RL, [2.0] * RL, [3.0] * RL])
    old = torch.full((BS, RL), -1.5)
    out, diag = _oci_shaped_rows(_Actor(), mb, pg, torch.tensor([False, True, True]),
                                 response_mask=torch.ones(BS, RL), advantages=adv, old_log_prob=old,
                                 temperature=1.0, gamma=0.1)
    rho = math.exp(-1.0 - (-1.5))
    check(torch.allclose(out[1], torch.full((RL,), -2.0 * rho / (rho + 0.1)))
          and torch.equal(out[0], pg[0]) and torch.equal(out[2], pg[2]),
          "the strippable injected row's loss is -A*f(rho); the other rows keep their clipped term "
          "(the unstrippable one too)")
    check(seen["sub"]["input_ids"].shape[0] == 1 and torch.equal(seen["sub"]["responses"][0], mb["responses"][1])
          and seen["sub"]["input_ids"][0, :PL][seen["sub"]["attention_mask"][0, :PL].bool()].tolist()
          == [ids[1, 0].item()] + ids[1, 3:PL].tolist(),
          "the forward receives only the strippable row, with its span taken out and its response intact")
    check(float(diag["oci/shaping/rows_injected"]) == 2 and float(diag["oci/shaping/rows_unstrippable"]) == 1
          and all(torch.is_tensor(v) and v.dim() == 0 and v.detach() is not None for v in diag.values())
          and "oci/shaping/frac_in_band" in diag,
          "the diagnostics are 0-d tensors (what _defer detaches) and count the unstrippable row")
    keep = _oci_opd_keep(mb, torch.zeros(BS, RL))
    check(keep.tolist() == [[1.0], [0.0], [0.0]], "the distillation keep factor is 0 exactly on the injected rows")
    src = open(os.path.join(REPO, "verl/workers/actor/dp_actor.py")).read()
    fm = src[src.index("def _forward_micro_batch("):]
    fm = fm[:fm.index("\n    def ", 10)]
    rets = [l.strip() for l in fm.splitlines() if l.strip().startswith("return ")]
    check(rets and all(r == "return entropy, log_probs, topk_out" for r in rets)
          and "_, lp_plain, _ = actor._forward_micro_batch(" in src,
          f"the real forward returns (entropy, log_probs, topk_out) at every exit ({len(rets)} of them) "
          "and the shaped path reads the second slot")

# --- 8. the launch is refused when it cannot be the arm ----------------------
def full_cfg(**over):
    alg = {"oci_slots": {"enable": True, "tasks": ["alfworld"]},
           "oci_sat": {"enable": False}, "oci_floor": {"enable": False},
           "filter_groups": {"enable": False}}
    alg["oci_slots"].update(over.pop("slots", {}))
    alg.update(over.pop("algorithm", {}))
    return SimpleNamespace(
        algorithm=alg,
        env=SimpleNamespace(rollout=SimpleNamespace(n=over.pop("n", 10))),
        actor_rollout_ref=SimpleNamespace(actor={"pg_loss_coef": 1.0,
                                                 "normalize_loss_by_task": True,
                                                 **over.pop("actor", {})}))


osl.check_config(full_cfg())
print("  OK   a complete launch passes check_config")
for name, kw in (("oci_sat also on", {"algorithm": {"oci_sat": {"enable": True}}}),
                 ("oci_floor also on", {"algorithm": {"oci_floor": {"enable": True}}}),
                 ("group of 3", {"n": 3}),
                 ("no policy gradient", {"actor": {"pg_loss_coef": 0.0}}),
                 ("no per-task weights", {"actor": {"normalize_loss_by_task": False}}),
                 ("filter_groups on", {"algorithm": {"filter_groups": {"enable": True}}}),
                 ("a task with no document", {"slots": {"tasks": ["alfworld", "webshop"]}})):
    try:
        osl.check_config(full_cfg(**kw))
        check(False, f"{name}: should have been refused")
    except AssertionError:
        check(True, f"refused before the first rollout: {name}")


# --- 9. the foreign slot, second design: this game with another game's goal ---
# The WebShop prompt's rows sat at rho = e^-40 and trained nothing. This one
# keeps the observation and swaps the goal sentence, so what is checked is the
# swap itself: a different object, the same choice on every host, every mention
# replaced, and the manager's plain render still the ordinary slot's.
lk = "/data/alfworld/json_2.1.1/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_2/game.tw-pddl"
check(ol.alfworld_goal_from_gamefile(lk) == ("look_at_obj_in_light", "AlarmClock", "DeskLamp"),
      "the target object is read off the game directory's name")
real = "examine the alarmclock with the desklamp."
alt = ol.alfworld_foreign_task(lk, real)
check(alt != real and "alarmclock" not in alt and alt in {s for s, _ in ol.ALFWORLD_TRAIN_TASKS},
      f"the foreign goal is another train game's sentence naming a different object ({alt!r})")
check(alt == ol.alfworld_foreign_task("/opt1/other/root" + lk, real),
      "the same game gets the same wrong goal under a different data root")
check(len({ol.alfworld_foreign_task(f"/d/train/pick_and_place-Mug-None-Desk-{i}/t/game.tw-pddl",
                                    "put a mug in desk.") for i in range(60)}) > 5,
      "different games spread over the pool")
check(all("mug" not in ol.alfworld_foreign_task(f"/d/train/pick_and_place-Mug-None-Desk-{i}/t/game.tw-pddl",
                                                 "put a mug in desk.") for i in range(60)),
      "and no game is ever shown a goal about its own object")
twice = f"Your task is to: {real}\nhistory: ... Your task is to: {real}\nnow"
swapped = ol.alfworld_foreign_obs(twice, real, lk)
check(real not in swapped and swapped.count(alt) == 2,
      "every mention of the real goal is replaced, the history's included")
try:
    ol.alfworld_foreign_obs("no goal here", real, lk)
    check(False, "an observation without the goal should be refused")
except ValueError:
    check(True, "an observation that does not carry the goal sentence is refused, not passed through")
try:
    ol.foreign_prompt("alfworld", lk)
    check(False, "foreign_prompt('alfworld') should point at alfworld_foreign_obs")
except ValueError:
    check(True, "foreign_prompt refuses 'alfworld': that design needs the observation")

if gf is not None:
    class FakeEnvs9:
        group_n, is_train = 4, True
        get_admissible_commands = [["look", "inventory"]] * 4

        def reset(self):
            obs = ["You are in the middle of a room.\n\nYour task is to: heat some mug and put it somewhere."] * 4
            return obs, None, [{"extra.gamefile": gf, "won": False}] * 4

        def step(self, actions):
            return ([f"You did: {a}." for a in actions], None, [0.0] * 4, [False] * 4,
                    [{"extra.gamefile": gf, "won": False}] * 4)

    em._WRONG_PLAN_CACHE.clear()
    m9 = em.AlfWorldEnvironmentManager(FakeEnvs9(), lambda acts, adm: (list(acts), [1] * len(acts)),
                                       cfg(foreign_task="alfworld"))
    o9, _ = m9.reset(kwargs=None)
    t9, p9 = o9["text"], o9[ol.OCI_PLAIN_KEY]
    goal9 = "heat some mug and put it somewhere."
    alt9 = ol.alfworld_foreign_task(gf, goal9)
    check(t9[3] == t9[0].replace(goal9, alt9) and goal9 not in t9[3] and alt9 in t9[3],
          "the foreign slot is shown the ordinary slot's render with only the goal swapped")
    check(p9[3] == t9[0] and t9[3] != p9[3],
          "its plain render is the ordinary slot's, so rho compares the two goals and nothing else")
    o9b = m9.step(["look", "look", "look", "go to fridge 1"])[0]
    check(goal9 not in o9b["text"][3] and alt9 in o9b["text"][3]
          and o9b[ol.OCI_PLAIN_KEY][3].count(goal9) == o9b["text"][3].count(alt9),
          "on later turns the swap covers the template's goal line and the history's copy alike")


def test_slots():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
