# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Driver half of the task-gradient probe: cut the batch into six groups.

The statistics are in ``task_gradient_conflict.py`` and the backward pass is in
the worker; what lives here is the partition, which is where the measurement can
quietly stop being the one it claims to be.

THREE PROPERTIES THE PARTITION HAS TO HAVE.

The halves must be INDEPENDENT DRAWS of the same task. The within-task cosine is
the reliability the cross-task numbers are divided by, so if the two halves
shared trajectories -- or shared prompts, since ``env.rollout.n`` puts eight
rollouts of one prompt in the batch -- the reliability would be inflated by the
overlap and every corrected cosine would be pulled toward zero by a factor
nobody measured. The split is therefore BY PROMPT GROUP, not by row: all eight
rollouts of a prompt land in the same half.

The groups must be EQUALLY SIZED and divisible by ``world_size *
micro_batch_size``. Ranks run the same number of micro-batches or FSDP's
collectives stop lining up, and a group that is one row short of a micro-batch
changes the token-mean denominator relative to its sibling. Rows past the
largest usable multiple are dropped, and the count that was dropped is reported
rather than absorbed.

The groups must carry the SAME per-row loss weights the training step used.
``normalize_loss_by_task`` weights each row so its task owns a third of the
loss; those weights are computed over the whole step's token totals and ride in
the batch. Recomputing them per group would rescale each task by its own group's
tokens, which is a different objective from the one the checkpoint was trained
under -- and since the probe compares tasks, a per-task rescaling is exactly the
confound to avoid. The weights are carried through untouched.
"""

import json
import os
import zlib
from collections import defaultdict

import numpy as np

from verl.trainer.ppo.task_gradient_conflict import attribution_shares

__all__ = [
    "partition_by_task_and_half",
    "teacher_endorsement_report",
    "accumulate_tau_probe",
    "run_attribution",
    "new_probe_state",
    "accumulate_grad_probe",
    "finish_grad_probe",
    "write_payload",
    "run_grad_probe",
]

HALVES = ("A", "B")


def _prompt_group_key(batch, i):
    """What identifies the prompt a row is a rollout of.

    ``uid`` is the GRPO group id -- the thing ``rollout.n`` replicates -- so it is
    the unit that must not straddle the two halves. Falling back to the row index
    makes every row its own group, which is correct (just less protective) for a
    batch that carries no uid.
    """
    for key in ("uid", "index", "traj_uid"):
        arr = batch.non_tensor_batch.get(key, None)
        if arr is not None:
            return str(arr[i])
    return str(i)


def partition_by_task_and_half(batch, task_id_names, *, block: int, seed: int = 0):
    """``{(task, half): [row indices]}``, equal sized, split by prompt group.

    Args:
        block: rows each group must be a multiple of, i.e.
            ``world_size * ppo_micro_batch_size_per_gpu``.

    Returns ``(groups, report)``; ``report`` names what was dropped and why, so a
    silently shrunken group cannot pass for a full one.
    """
    task_ids = batch.batch["task_ids"].reshape(-1).tolist()
    by_task = defaultdict(lambda: defaultdict(list))
    for i, t in enumerate(task_ids):
        if int(t) < 0:
            continue
        by_task[int(t)][_prompt_group_key(batch, i)].append(i)

    rng = np.random.default_rng(seed)
    groups, report = {}, {}
    for tid, prompt_groups in sorted(by_task.items()):
        name = task_id_names[tid] if tid < len(task_id_names) else str(tid)
        keys = sorted(prompt_groups)
        rng.shuffle(keys)
        # Deal whole prompt groups alternately, so the halves get independent
        # prompts and stay close in size.
        halves = {"A": [], "B": []}
        for n, k in enumerate(keys):
            halves[HALVES[n % 2]].extend(prompt_groups[k])
        usable = min(len(halves["A"]), len(halves["B"]))
        usable = (usable // block) * block if block > 0 else usable
        report[name] = {
            "rows_total": sum(len(v) for v in prompt_groups.values()),
            "prompt_groups": len(keys),
            "rows_per_half": usable,
            "rows_dropped": len(halves["A"]) + len(halves["B"]) - 2 * usable,
        }
        if usable == 0:
            continue
        for h in HALVES:
            groups[(name, h)] = sorted(halves[h][:usable])
    return groups, report


def partition_by_task_padded(batch, task_id_names, *, block: int):
    """``{task: {idx, n_real, n_pad, groups}}`` -- every prompt group kept.

    WHY NOT partition_by_task_and_half HERE. That one balances two halves and
    floors each to a multiple of ``block``, then truncates with ``[:usable]``,
    which cuts through prompt groups. On search that dropped 32 of 272 rows, and
    the probe's policy gradient came out exactly zero while the batch had one of
    fifteen groups carrying nonzero advantage -- the inference being that the one
    live group went with the dropped rows. Whether or not that inference was
    right, a measurement whose selection can remove the only informative group is
    measuring its own selection.

    So nothing is dropped. The rows are padded up to a multiple of ``block`` by
    repeating the first row, and the caller zeroes those rows' ``loss_mask`` and
    task weight, which removes them from both terms of the loss exactly.

    PADDING IS ONLY SAFE ON THE WEIGHTED AGGREGATION. ``agg_loss`` in token-mean
    divides by ``loss_mask.sum()``, so a micro-batch that is entirely padding
    would be 0/0. ``normalize_loss_by_task`` routes both terms through
    ``agg_loss_by_task_weights`` instead, which sums and does not divide, so a
    fully masked micro-batch contributes exactly zero. The caller asserts that
    setting rather than assuming it.
    """
    task_ids = batch.batch["task_ids"].reshape(-1).tolist()
    by_task = defaultdict(lambda: defaultdict(list))
    for i, t in enumerate(task_ids):
        if int(t) < 0:
            continue
        by_task[int(t)][_prompt_group_key(batch, i)].append(i)

    out = {}
    for tid, prompt_groups in sorted(by_task.items()):
        name = task_id_names[tid] if tid < len(task_id_names) else str(tid)
        idx = [i for k in sorted(prompt_groups) for i in prompt_groups[k]]
        if not idx:
            continue
        n_real = len(idx)
        n_pad = (-n_real) % block if block > 0 else 0
        out[name] = {
            "idx": idx + [idx[0]] * n_pad,
            "n_real": n_real,
            "n_pad": n_pad,
            "prompt_groups": len(prompt_groups),
            "rows_per_group": {str(k): len(v) for k, v in sorted(prompt_groups.items())},
        }
    return out


def mask_padded_rows_(sub, n_real: int) -> int:
    """Zero the loss contribution of the rows appended purely for divisibility.

    Both terms read the mask: the policy gradient and the teacher KL are each
    aggregated over ``loss_mask`` (update_policy takes its response mask from
    ``loss_mask`` when multi_turn is on, not from a response_mask column), and
    the per-task weight multiplies the row. Zeroing both is redundant on
    purpose -- one of them silently not applying is the failure that would look
    like a real measurement.
    """
    from verl.trainer.ppo.task_loss_weights import TASK_LOSS_WEIGHT_KEY

    n = len(sub.batch["responses"]) if "responses" in sub.batch else None
    if n is None or n_real >= n:
        return 0
    if "loss_mask" in sub.batch:
        sub.batch["loss_mask"][n_real:] = 0
    if TASK_LOSS_WEIGHT_KEY in sub.batch:
        sub.batch[TASK_LOSS_WEIGHT_KEY][n_real:] = 0
    if "attention_mask" in sub.batch:
        # Not the loss path, but it keeps any length statistic honest.
        pass
    return n - n_real


def new_probe_state() -> dict:
    return {"batches": 0, "rows": defaultdict(int), "partitions": []}


def accumulate_grad_probe(trainer, batch, state: dict, *, seed: int = 0) -> dict:
    """Fold ONE batch into the six running gradients. No step, no report.

    Called once per rollout. The pilot at one batch measured a within-task
    reliability of 0.126 (alfworld), 0.084 (webshop) and -0.002 (search) -- the
    denominator every corrected cosine is divided by -- which is far below the
    0.3-0.9 range the correction was verified over, and search's own gradient
    did not reproduce across its two halves at all. Accumulating over several
    batches is the way to raise that without touching the data pipeline: the
    per-task batch size and the parquet it is generated from stay exactly as the
    training run had them, and only the number of rollouts consumed changes.

    Gradients are SUMMED across batches. Each batch's loss is already a token
    mean, so the sum is N times the pooled-mean gradient; cosines are unaffected
    by that common factor and the norms scale together. The one asymmetry left is
    that a batch whose groups came out slightly smaller (rows dropped to reach
    micro-batch divisibility) contributes on equal terms with a full one; the
    per-batch row counts are recorded so the size of that is visible.

    The seed is offset per batch so successive batches deal their prompt groups
    differently. Reusing one seed would not repeat the same split -- the batches
    hold different prompts -- but varying it removes any chance that a fixed
    dealing order interacts with the order the sampler emits prompts in.
    """
    cfg = trainer.config
    task_id_names = list(batch.meta_info.get("task_id_names", []) or [])
    world = int(cfg.trainer.n_gpus_per_node) * int(cfg.trainer.nnodes)
    block = world * int(cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)

    groups, part_report = partition_by_task_and_half(
        batch, task_id_names, block=block, seed=seed + state["batches"]
    )
    assert groups, "the probe batch produced no usable task groups"
    n = state["batches"] + 1
    print(f"[grad_probe] batch {n}: partition {json.dumps(part_report)}", flush=True)

    for task in sorted({t for (t, _) in groups}):
        for half in HALVES:
            idx = groups.get((task, half), None)
            if idx is None:
                continue
            sub = batch.select_idxs(idx) if hasattr(batch, "select_idxs") else batch[idx]
            sub.meta_info = dict(batch.meta_info)
            sub.meta_info["probe_tag"] = f"{task}:{half}"
            # The same two the training loop sets before update_actor; the
            # forward rescales logits by the rollout temperature and the loss
            # mask depends on multi_turn, so a probe that omitted them would
            # differentiate a slightly different objective.
            sub.meta_info["temperature"] = cfg.actor_rollout_ref.rollout.temperature
            sub.meta_info["multi_turn"] = cfg.actor_rollout_ref.rollout.multi_turn.enable
            print(f"[grad_probe] batch {n} backward {task}:{half}  rows={len(idx)}", flush=True)
            trainer.actor_rollout_wg.grad_probe_accumulate(sub)
            state["rows"][f"{task}:{half}"] += len(idx)

    state["batches"] = n
    state["partitions"].append(part_report)
    return state


def normalize_eps(value) -> list:
    """One eps or several, whatever shape the config layer handed over.

    THREE CALL SITES COERCED THIS WITH float() AND ALL THREE HAD TO CHANGE, which
    is the reason it lives here now. Hydra delivers the bracket form as an
    omegaconf ListConfig -- not a list, not a tuple -- and float(ListConfig)
    raises, so a sweep that parsed correctly in the worker still died in the
    driver on the way there. Fixing the worker alone cost a full rollout.

    Accepts a scalar, a comma-separated string, or any iterable, and always
    returns an ascending list of floats.
    """
    if isinstance(value, str):
        out = [float(x) for x in value.replace(" ", "").split(",") if x]
    else:
        try:
            out = [float(x) for x in list(value)]
        except TypeError:
            out = [float(value)]
    if not out:
        raise ValueError("grad_probe eps is empty")
    return sorted(out)


def _row_selection_key(idx, n_real) -> str:
    """Exact identity of the rows a pass was handed, and how many are real.

    crc32 over the index list, not a reduction over the tensors: two passes that
    were given the same rows in the same order produce the same key by
    construction, and any difference in selection or padding changes it. The
    tensor fingerprint in the worker can only make a statistical argument;
    this one cannot be fooled by two rows with equal sums.
    """
    blob = ",".join(str(int(i)) for i in idx).encode()
    return f"{zlib.crc32(blob):08x}:{len(idx)}:{int(n_real)}"


def actor_loss_mask(batch, *, multi_turn: bool):
    """The mask update_policy will actually aggregate over.

    NOT the ``response_mask`` column. That column is
    ``attention_mask[:, -response_length:]`` (ray_trainer.compute_response_mask),
    which under multi-turn includes every environment-observation token; the
    actor takes its response mask from ``loss_mask`` instead
    (dp_actor: ``response_mask = data["loss_mask"][:, -response_length:]``).
    Advantages are built as ``A_row * response_mask`` (core_algos), so a
    diagnostic that counts nonzero advantages over the column counts tokens the
    loss never sees -- which is how the N=4 probe reported live groups for three
    cells whose policy gradient was identically zero.
    """
    n = batch.batch["responses"].shape[1]
    key = "loss_mask" if (multi_turn and "loss_mask" in batch.batch) else "attention_mask"
    return batch.batch[key][:, -n:].bool()


def effective_input_report(sub, *, multi_turn: bool) -> dict:
    """Stages 1-2 of the funnel, on the rows the probe is about to differentiate.

    Called AFTER select_idxs and after mask_padded_rows_, so the padding is
    already masked out and the group selection has already happened -- the
    previous report was taken on the parent batch and therefore described a
    different set of rows from the one the gradient came from.
    """
    m = actor_loss_mask(sub, multi_turn=multi_turn)
    adv = sub.batch.get("advantages", None)
    from verl.trainer.ppo.task_loss_weights import TASK_LOSS_WEIGHT_KEY

    w = sub.batch.get(TASK_LOSS_WEIGHT_KEY, None)
    live_row = m.any(-1)
    if w is not None:
        live_row = live_row & (w.reshape(-1) != 0)
    eff = m & live_row.reshape(-1, 1)
    rec = {
        "rows": int(len(live_row)),
        "eff_rows": int(live_row.sum()),
        "eff_tokens": int(eff.sum()),
        "rows_no_loss_token": int((~m.any(-1)).sum()),
    }
    if adv is None:
        return rec
    anz = eff & (adv != 0)
    rec["adv_nonzero_tokens"] = int(anz.sum())
    rec["adv_nonzero_rows"] = int(anz.any(-1).sum())
    rec["adv_abs_sum"] = float((adv.abs() * eff).sum())
    rec["adv_absmax"] = float((adv.abs() * eff).max()) if eff.any() else 0.0
    uids = (sub.non_tensor_batch or {}).get("uid", None)
    if uids is not None:
        rows_of = defaultdict(list)
        for i, u in enumerate(uids):
            rows_of[str(u)].append(i)
        rec["groups"] = len(rows_of)
        # A group counts as carrying signal only if a token the LOSS sees has a
        # nonzero advantage. That is the condition the gradient depends on;
        # "returns differ within the group" is upstream of it and not sufficient.
        rec["groups_with_eff_adv"] = int(sum(
            1 for _, rws in rows_of.items() if bool(anz[rws].any())
        ))
        rec["groups_with_eff_tokens"] = int(sum(
            1 for _, rws in rows_of.items() if bool(eff[rws].any())
        ))
        # THE IDS, not only the counts. Which groups carried signal is what lets
        # a later pass go back to the same groups, and what a count cannot say.
        rec["group_ids_with_eff_adv"] = sorted(
            g for g, rws in rows_of.items() if bool(anz[rws].any())
        )
        rec["group_ids_with_eff_tokens"] = sorted(
            g for g, rws in rows_of.items() if bool(eff[rws].any())
        )
        rew = sub.batch.get("token_level_rewards", None)
        if rew is not None:
            # Non-degenerate = the returns inside the group differ, which is the
            # upstream condition for a nonzero GRPO advantage. Unmasked row sums:
            # token_level_rewards is zero except where the reward was placed.
            r = rew.sum(-1)
            rec["group_ids_nondegenerate"] = sorted(
                g for g, rws in rows_of.items()
                if float(r[rws].max() - r[rws].min()) != 0.0
            )
    return rec


def advantage_report(batch, task_id_names, partition=None, *, multi_turn: bool = True) -> dict:
    """Per task: is there a policy gradient here at all, and if not, why not.

    THE FIRST RUN OF THE TERM PROBE RETURNED grad(L_rl) IDENTICALLY ZERO for
    search and webshop -- not small, zero nonzero coordinates out of 860M, on
    both ranks, while the OPD gradient on the same rows was normal. A
    group-relative advantage is exactly zero when every rollout in a prompt
    group earns the same return, so a task whose groups are all uniform
    contributes no policy gradient at all, and a probe that does not record
    this reports "no interference" for a term that was never there.

    Everything here is arithmetic on columns the batch already carries, so it
    costs no forward and no backward -- it only has to run while a batch
    exists, which the first run did not do.
    """
    import numpy as np

    out = {}
    adv = batch.batch.get("advantages", None)
    if adv is None:
        return {"error": "batch carries no advantages column"}
    # The mask the LOSS uses, not the response_mask column -- see actor_loss_mask.
    # The two differ by every environment-observation token, and the first version
    # of this report used the wrong one.
    mask = actor_loss_mask(batch, multi_turn=multi_turn)
    rew = batch.batch.get("token_level_rewards", None)
    task_ids = batch.batch["task_ids"].reshape(-1).tolist()
    uids = batch.non_tensor_batch.get("uid", None)

    for tid in sorted(set(int(t) for t in task_ids)):
        name = task_id_names[tid] if tid < len(task_id_names) else str(tid)
        rows = [i for i, t in enumerate(task_ids) if int(t) == tid]
        a = adv[rows]
        m = mask[rows]
        live = a[m]
        rec = {
            "rows": len(rows),
            "nonzero_advantage_frac": float((live != 0).to(float).mean()) if live.numel() else float("nan"),
            "advantage_absmax": float(live.abs().max()) if live.numel() else float("nan"),
        }
        if rew is not None:
            # Summed over the whole row, deliberately unmasked: token_level_rewards
            # is zero except where the reward was placed, so the row sum IS the
            # return. Masking it with the loss mask would turn "the reward token is
            # not model-generated" into "the group is degenerate", which is a
            # statement about the data the mask has no business making.
            r = rew[rows].sum(-1)
            rec["return_mean"] = float(r.mean())
            rec["return_std"] = float(r.std())
            if uids is not None:
                # COLLAPSE ROWS INTO TRAJECTORIES FIRST. A GRPO advantage is the
                # TRAJECTORY's return minus its group's mean, so a group is
                # degenerate when the n trajectory returns are equal -- not when
                # the ROWS are. Measured on the control step-300 batches, 2-7% of
                # trajectories carry the return on some of their rows and not on
                # others; at n=8 that is a ~28% chance per group of containing at
                # least one, and each one made a genuinely degenerate group read
                # as live. Over rows this table reported 22/60 live on alfworld
                # and 21/60 on webshop where the trajectory-level counts are 16
                # and 11. Over-reporting live groups is the exact failure this
                # function exists to prevent: it says a policy gradient is there
                # when it is not.
                tuids = batch.non_tensor_batch.get("traj_uid", None)
                by_group = {}
                if tuids is None:
                    rec["group_unit"] = "row"
                    for j, i in enumerate(rows):
                        by_group.setdefault(str(uids[i]), []).append(float(r[j]))
                else:
                    rec["group_unit"] = "trajectory"
                    by_traj = {}
                    for j, i in enumerate(rows):
                        by_traj.setdefault(str(tuids[i]), []).append((str(uids[i]), float(r[j])))
                    for entries in by_traj.values():
                        by_group.setdefault(entries[0][0], []).append(max(v for _, v in entries))
                spreads = {g: max(v) - min(v) for g, v in by_group.items()}
                rec["prompt_groups"] = len(by_group)
                # A degenerate group is one where the policy gradient cancels:
                # every rollout in it earned the same return.
                live = [g for g, x in spreads.items() if x != 0.0]
                rec["degenerate_groups"] = len(spreads) - len(live)
                rec["group_spread_max"] = float(max(spreads.values())) if spreads else float("nan")
                # WHICH LIVE GROUPS SURVIVED THE SELECTION. The previous run could
                # only infer that its one informative search group had been
                # dropped; with the padded partition nothing is dropped and this
                # records that rather than asserting it.
                if partition and name in partition:
                    kept = set(partition[name]["idx"][: partition[name]["n_real"]])
                    rows_of = {}
                    for j, i in enumerate(rows):
                        rows_of.setdefault(str(uids[i]), []).append(i)
                    rec["live_groups"] = len(live)
                    rec["live_groups_kept"] = int(sum(
                        1 for g in live if any(i in kept for i in rows_of.get(g, []))
                    ))
        out[name] = rec
    return out


def accumulate_term_probe(trainer, batch, state: dict, *, seed: int = 0) -> dict:
    """Split the loss into its RL and OPD terms and keep both gradients.

    WHY THIS IS A SEPARATE PASS AND NOT A SETTING. Everything measured so far --
    the cosines, the strata, the attribution -- came off ``verl_agent_opd_multitask``,
    which runs at ``pg_loss_coef=0``. It has no policy-gradient term at all, so
    "the tasks do not conflict" is a statement about OPD against OPD and says
    nothing about the arm the work is aimed at. This measures, at ONE parameter
    point of an OPD+GRPO checkpoint, three things the OPD-only probe cannot:
    whether the tasks' reward gradients agree with each other, whether the
    distillation gradients agree on this student's own state distribution, and
    whether approaching the teacher helps or fights the reward -- within a task
    as well as across tasks.

    SIX GRADIENTS, NOT TWELVE. Halves are what the reliability is built from, and
    the reliability is what magnitudes need. Signs need no correction -- they
    replicated at N=1 and N=2 in the OPD-only probe -- so the first pass drops
    the halves and spends its rollout on the term split instead. That keeps the
    host footprint at the six sets the existing probe already fits in.

    THE DECOMPOSITION IS CHECKED. A third pass per task differentiates the
    unmasked loss and the worker reports the relative residual of
    ``grad(L_rl) + grad(L_opd) - grad(L_total)`` per unit, then drops it. The
    split is two scalars on assembled terms, so that residual must be at
    rounding level; anything else means a branch went differently between the
    passes and the two gradients are not a decomposition of anything.
    """
    cfg = trainer.config
    task_id_names = list(batch.meta_info.get("task_id_names", []) or [])
    world = int(cfg.trainer.n_gpus_per_node) * int(cfg.trainer.nnodes)
    block = world * int(cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)
    # The padding below is only exact on the weighted aggregation; see
    # partition_by_task_padded. Checked against the CONDITION rather than a
    # config key: the column's presence is what routes both terms through
    # agg_loss_by_task_weights, and the two spellings of the setting
    # (algorithm.opd.normalize_loss_by_task and the actor key main_opd derives
    # from it) are easy to read the wrong one of -- the composed config shows
    # the actor one still false because the injection happens after Hydra.
    from verl.trainer.ppo.task_loss_weights import TASK_LOSS_WEIGHT_KEY

    assert TASK_LOSS_WEIGHT_KEY in batch.batch, (
        f"grad_probe.mode=terms pads to divisibility and masks the padding, which is "
        f"only exact when every row carries {TASK_LOSS_WEIGHT_KEY} and both terms go "
        f"through agg_loss_by_task_weights (which sums and does not divide by the mask). "
        f"Without it the loss uses agg_loss token-mean, and a fully padded micro-batch "
        f"is 0/0. Set algorithm.opd.normalize_loss_by_task=True."
    )

    part = partition_by_task_padded(batch, task_id_names, block=block)
    assert part, "the probe batch produced no usable task groups"
    n = state["batches"] + 1
    print(f"[grad_probe] term probe batch {n}: partition "
          f"{json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('idx', 'rows_per_group')} for k, v in part.items()})}",
          flush=True)

    # Recorded BEFORE any backward, because it explains a zero gradient and a
    # zero gradient is otherwise indistinguishable from "no interference".
    # Per group as well as per task: the earlier run could only infer which
    # groups its selection had removed, and inference is what this replaces.
    multi_turn = bool(cfg.actor_rollout_ref.rollout.multi_turn.enable)
    adv_report = advantage_report(batch, task_id_names, partition=part, multi_turn=multi_turn)
    print(f"[grad_probe] advantages: {json.dumps(adv_report)}", flush=True)

    checks, repeats, stages, eff = {}, {}, {}, {}
    for task in sorted(part):
        idx = part[task]["idx"]
        n_real = part[task]["n_real"]
        # rl, opd, the unmasked total to check they add up, then rl AND opd
        # again -- the floor each component's residual has to be read against.
        for term in ("rl", "opd", "check", "repeat_rl", "repeat_opd"):
            sub = batch.select_idxs(idx) if hasattr(batch, "select_idxs") else batch[idx]
            sub.meta_info = dict(batch.meta_info)
            n_masked = mask_padded_rows_(sub, n_real)
            if term == "rl" and n_masked:
                print(f"[grad_probe] {task}: padded {n_masked} rows to reach a "
                      f"multiple of {block}, masked out of both terms", flush=True)
            if term == "rl":
                # On the rows the gradient is about to be taken from, with the
                # padding already masked and the mask the actor will use.
                eff[task] = effective_input_report(sub, multi_turn=multi_turn)
                print(f"[grad_probe] {task} effective input: {json.dumps(eff[task])}", flush=True)
            sub.meta_info["probe_tag"] = f"{task}:{term}"
            sub.meta_info["probe_term"] = term
            # Empty unless asked for. When set, the columns that decide whether a
            # policy gradient exists are written to disk so the next round of this
            # investigation starts from a file instead of another rollout.
            probe_cfg = cfg.trainer.get("grad_probe", {}) or {}
            sub.meta_info["probe_dump_dir"] = str(probe_cfg.get("dump_dir", "") or "")
            sub.meta_info["probe_dump_terms"] = str(probe_cfg.get("dump_terms", "rl") or "rl")
            sub.meta_info["probe_dump_batches"] = int(probe_cfg.get("dump_batches", 1) or 1)
            # Which batch this pass belongs to. The worker's input-identity check
            # is per (batch, task): across batches the data is SUPPOSED to differ.
            sub.meta_info["probe_batch"] = n
            # The selection itself, exactly. The tensor reductions below it are a
            # corroboration; this is the thing that can actually differ between
            # passes, and comparing it is exact rather than statistical.
            sub.meta_info["probe_row_key"] = _row_selection_key(part[task]["idx"], n_real)
            sub.meta_info["temperature"] = cfg.actor_rollout_ref.rollout.temperature
            sub.meta_info["multi_turn"] = cfg.actor_rollout_ref.rollout.multi_turn.enable
            out = trainer.actor_rollout_wg.grad_probe_accumulate(sub)
            outs = out if isinstance(out, list) else [out]
            for o in outs:
                got = (o.meta_info or {}).get("probe_check", None)
                if got:
                    checks.update(got)
                rep = (o.meta_info or {}).get("probe_repeat", None)
                if rep:
                    repeats.update(rep)
                st = (o.meta_info or {}).get("probe_pg_stages", None)
                if st:
                    stages.update({k: v for k, v in st.items() if v})
            if term in ("rl", "opd"):
                state["rows"][f"{task}:{term}"] += n_real

    # Collapse this batch to its inner products and drop the gradients, so the
    # next batch is an independent replication rather than another summand.
    mom = trainer.actor_rollout_wg.grad_probe_term_moments()
    mom = mom if isinstance(mom, list) else [mom]
    got = next((m.meta_info.get("term_moments") for m in mom if (m.meta_info or {}).get("term_moments")), {})
    state.setdefault("batch_moments", []).append(got.get("whole_model", {}))
    state.setdefault("batch_per_unit", []).append(got.get("per_unit", {}))

    state["batches"] = n
    state["partitions"].append(
        {k: {kk: vv for kk, vv in v.items() if kk != "idx"} for k, v in part.items()}
    )
    state.setdefault("decomposition", []).append(checks)
    state.setdefault("advantages", []).append(adv_report)
    state.setdefault("determinism", []).append(repeats)
    state.setdefault("pg_stages", []).append(stages)
    state.setdefault("effective_input", []).append(eff)
    return state


def run_attribution(trainer, batch, *, directions, eps) -> dict:
    """Trace the accumulated directions back to token positions.

    Runs on the LAST batch, after all N have been folded in, because the
    directions are not known until then -- ``shared`` is the mean over tasks of
    gradients accumulated across every batch. Attribution is a distributional
    question (which roles carry the direction) and one batch already holds
    hundreds of thousands of token positions, so the direction gets all N
    batches' worth of precision while the attribution samples one. What it does
    NOT support is a claim about a particular batch's tokens.
    """
    cfg = trainer.config
    sub = batch
    sub.meta_info = dict(batch.meta_info)
    sub.meta_info["probe_directions"] = list(directions)
    eps = normalize_eps(eps)
    sub.meta_info["probe_eps"] = eps
    sub.meta_info["temperature"] = cfg.actor_rollout_ref.rollout.temperature
    sub.meta_info["multi_turn"] = cfg.actor_rollout_ref.rollout.multi_turn.enable
    print(f"[grad_probe] attribution: {list(directions)} at eps={eps}", flush=True)
    outs = trainer.actor_rollout_wg.grad_probe_attribution(sub)
    outs = outs if isinstance(outs, list) else [outs]
    merged = {}
    for o in outs:
        got = (o.meta_info or {}).get("attribution", {}) or {}
        refused = (o.meta_info or {}).get("refused", None)
        if refused:
            print(f"[grad_probe] attribution REFUSED: {refused}", flush=True)
            return {"refused": refused}
        merged = got or merged
    return merged


def write_payload(payload: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"[grad_probe] wrote {out_path}", flush=True)


def finish_grad_probe(trainer, state: dict, *, out_path: str, attribution: dict = None,
                      keep_snapshots: bool = False) -> dict:
    """Reduce the accumulated gradients to a report and write it.

    WRITTEN BEFORE ANYTHING OPTIONAL RUNS. The first N=8 attempt did attribution
    first and was killed inside it by the host memory monitor, which threw away
    eight batches of accumulated gradient because nothing had reached disk yet.
    The conflict report is the primary result and costs one file write; anything
    that can fail belongs after it.

    ``keep_snapshots`` leaves the six gradients on the worker so a later
    attribution pass can still build directions from them.
    """
    tasks = sorted({k.split(":")[0] for k in state["rows"]})
    reports = trainer.actor_rollout_wg.grad_probe_report(tasks, keep=keep_snapshots)
    report = reports[0] if isinstance(reports, list) else reports

    payload = {
        "tasks": tasks,
        "n_batches": state["batches"],
        "partitions": state["partitions"],
        "n_rows_per_group": dict(state["rows"]),
        "report": report,
        "attribution": attribution or {},
    }
    write_payload(payload, out_path)
    print(f"[grad_probe] ({state['batches']} batches accumulated)", flush=True)

    for geom in ("raw", "preconditioned"):
        if geom not in report:
            continue
        # The reliabilities are printed beside the summary, not buried in the
        # file, because they are what says whether the summary may be quoted at
        # all: the pilot's headline cosine of 0.914 sat on a within-task
        # reliability of 0.126, and two layers came back above 1.0, which no
        # true cosine can be.
        wmin = {}
        for entry in report[geom]["per_unit"].values():
            for t, v in entry.get("within", {}).items():
                if v == v:
                    wmin[t] = min(wmin.get(t, 1e9), v)
        med = _within_medians(report[geom]["per_unit"], tasks)
        print(f"[grad_probe] {geom} summary (corrected cosine): "
              f"{json.dumps(report[geom]['summary'], default=lambda v: round(float(v), 4))}", flush=True)
        print(f"[grad_probe] {geom} within-task reliability (median per task, "
              f"READ THIS FIRST -- under ~0.3 the corrected values are not usable): "
              f"{json.dumps(med, default=lambda v: round(float(v), 4))}", flush=True)

    for name, entry in (attribution or {}).items():
        if name.startswith("_") or not isinstance(entry, dict) or "aggregate" not in entry:
            continue
        shares = attribution_shares(entry["aggregate"])
        print(f"[grad_probe] attribution[{name}] role shares of |contribution|:", flush=True)
        for task, sh in sorted(shares.items()):
            row = {k: round(float(v), 3) for k, v in sh.items() if not k.startswith("_")}
            print(f"    {task:10s} {json.dumps(row)}", flush=True)
        if "expected_dot" in entry:
            got, want = entry["attributed_sum"], entry["expected_dot"]
            rel = abs(got - want) / max(abs(want), 1e-12)
            # The self-check: the per-position numbers must add up to the inner
            # product the gradients already gave. A large residual means eps is
            # wrong or the wiring is, and either way the shares above are not to
            # be read.
            print(f"    CHECK sum={got:.6g} vs <g,u>={want:.6g}  relative residual {rel:.3%}"
                  f"  {'OK' if rel < 0.05 else '<-- DO NOT READ THE SHARES'}", flush=True)
    return payload


def _within_medians(per_unit: dict, tasks) -> dict:
    out = {}
    for t in tasks:
        vals = sorted(
            e["within"][t] for e in per_unit.values()
            if t in e.get("within", {}) and e["within"][t] == e["within"][t]
        )
        out[t] = vals[len(vals) // 2] if vals else float("nan")
    return out


def run_grad_probe(trainer, batch, *, out_path: str, seed: int = 0) -> dict:
    """One batch, straight through. The N=1 case of the two calls above."""
    state = accumulate_grad_probe(trainer, batch, new_probe_state(), seed=seed)
    return finish_grad_probe(trainer, state, out_path=out_path)


# ---------------------------------------------------------------------------
# tau: does the teacher still tell this task's wins from its losses?
# ---------------------------------------------------------------------------
#
# WHAT THIS REPLACES. The retirement line for a task's teacher was going to be
# the teacher's own success rate, measured once before the run. That comparison
# is between two different distributions: the teacher scored on validation
# prompts at the validation temperature, the student scored on training prompts
# at T=1.0. tau is measured on one distribution -- the student's own rollouts,
# at the point in training where the decision is taken.
#
# THE STATISTIC. Per token the teacher's endorsement of what the student
# actually emitted is
#
#     delta_t = log pi_teacher(a_t | x) - log pi_student(a_t | x)
#
# which is the same delta the theory document's section 5.3(b) gate is built
# from. Aggregate it over a trajectory, rank the trajectories of one prompt
# group by it, rank them by return, and correlate the two rankings. tau > 0 is
# a teacher that endorses what worked and objects to what failed; tau ~ 0 is a
# teacher whose corrections are unrelated to the outcome; tau < 0 is a teacher
# pushing hardest against the trajectories that succeeded.
#
# WHY RANKS. The per-task teacher KL differs by a factor of five between search
# and alfworld, so no threshold on a KL-scaled quantity transfers across tasks.
# A rank correlation is bounded in [-1, 1] on every task, which is what lets one
# threshold govern three teachers.
#
# WHY GAMMA AND NOT KENDALL'S TAU-B. Returns tie heavily and the TIE STRUCTURE
# DIFFERS BY TASK, which is exactly what a cross-task threshold must not depend
# on. With n=8 binary returns and k successes, a teacher that separates the two
# classes perfectly scores tau-b = sqrt(k(8-k)/28): 0.76 at k=4, 0.50 at k=1. A
# task at 0.75 success and a task at 0.39 success therefore have different
# CEILINGS, so "tau below 0.3" would mean different things on alfworld and
# search. Goodman-Kruskal gamma, (concordant - discordant) / (concordant +
# discordant), drops tied pairs instead of penalising them and reaches +-1 at
# any success rate. On binary returns it is 2*AUC - 1, where AUC is the chance
# that the teacher endorses a randomly drawn winner over a randomly drawn
# loser -- so gamma = 0 is literally "the teacher's endorsement is a coin flip
# for telling this task's wins from its losses". That is the retirement line.
# tau-b is kept beside it because it is the standard statistic and a large
# disagreement between the two means the tie structure is doing the work.
#
# WHY THE GROUP IS THE UNIT. Ranking within a prompt group holds prompt
# difficulty fixed, which is the same control the GRPO advantage applies. A
# group whose trajectories all earned the same return carries no ranking
# information at all and is excluded here for the same reason it contributes no
# policy gradient there.
#
# COST. Arithmetic on columns the batch already carries: no forward, no
# backward. The only requirement is that the teacher's log-prob at the sampled
# token reached the driver, which it does under kl_loss_type=low_var_kl and
# under topk_kl WITHOUT student_indexed_topk. With student_indexed_topk=True
# the top-k values are resolved inside the actor from cached hidden states and
# never travel, so this reports its absence rather than guessing.


def _kendall_tau_b(x, y) -> float:
    """Kendall's tau-b over two equal-length sequences, ties handled.

    tau-b and not Spearman because returns tie heavily here -- a group where
    six of eight trajectories scored zero is the common case, not the
    exception -- and Spearman's mid-ranks turn those ties into agreement.
    n is rollout.n = 8, so the O(n^2) loop is 28 pairs.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2.0
    denom = ((n0 - tx) * (n0 - ty)) ** 0.5
    return float((conc - disc) / denom) if denom > 0 else float("nan")


def _goodman_kruskal_gamma(x, y) -> float:
    """(concordant - discordant) / (concordant + discordant); tied pairs dropped.

    Unlike tau-b this reaches +-1 whatever the tie structure, so one threshold
    is comparable across tasks with different success rates. On binary x it is
    ``2*AUC - 1`` for y as a ranker of x.
    """
    n = len(x)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    return float((conc - disc) / (conc + disc)) if (conc + disc) else float("nan")


def _partial_gamma(t_rd, t_rl, t_dl) -> float:
    """Rank correlation of (return, delta) with LENGTH held fixed.

    WHY IT IS NEEDED HERE. delta is a per-token mean and a successful alfworld
    trajectory is a SHORT one: measured at control step 300, return-vs-length is
    -0.79 and delta-vs-length is +0.64 on that task, so return and endorsement
    are tied together through length before the teacher says anything. The
    standard partial-tau form removes that path; what is left is the teacher's
    own ranking. On webshop, where both confounds are near zero, it changes
    nothing -- which is the check that it is not just shrinking everything.
    """
    den = ((1.0 - t_rl * t_rl) * (1.0 - t_dl * t_dl)) ** 0.5
    if den <= 1e-9 or t_rd != t_rd or t_rl != t_rl or t_dl != t_dl:
        return float("nan")
    return float((t_rd - t_rl * t_dl) / den)


def _mean_se(values) -> dict:
    """Mean over groups and the standard error of that mean."""
    v = [float(x) for x in values if x == x]  # drop nan
    if not v:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    m = sum(v) / len(v)
    if len(v) < 2:
        return {"mean": m, "se": float("nan"), "n": len(v)}
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return {"mean": m, "se": (var / len(v)) ** 0.5, "n": len(v)}


def teacher_logprob_at_sampled(batch):
    """``(rows, resp)`` teacher log-prob of the token the STUDENT emitted.

    Returns ``(tensor_or_None, source_string)``. The tensor carries nan where
    the sampled token fell outside the teacher's stored support; the caller
    counts those rather than treating them as zero endorsement.
    """
    import torch

    b = batch.batch
    if "teacher_log_probs" in b.keys():
        return b["teacher_log_probs"], "teacher_log_probs (low_var_kl, exact)"
    if "teacher_topk_logprobs" in b.keys() and "teacher_topk_ids" in b.keys():
        resp = b["responses"]
        ids = b["teacher_topk_ids"]
        lp = b["teacher_topk_logprobs"]
        hit = ids == resp.unsqueeze(-1)
        found = hit.any(-1)
        pos = hit.to(torch.int64).argmax(-1, keepdim=True)
        val = lp.gather(-1, pos).squeeze(-1).to(torch.float32)
        val = torch.where(found, val, torch.full_like(val, float("nan")))
        return val, "teacher_topk_logprobs (resolved at the sampled token)"
    return None, None


def teacher_endorsement_report(batch, task_id_names, *, multi_turn: bool = True) -> dict:
    """Per task: tau, the teacher's rank agreement with the reward.

    No forward and no backward -- every column read here is already in the
    batch. Run it before any gradient work, for the same reason
    ``advantage_report`` is: it explains the number the gradients will produce.
    """
    import torch

    out = {"unit": "trajectory"}
    t_lp, source = teacher_logprob_at_sampled(batch)
    if t_lp is None:
        return {
            "error": (
                "no teacher log-prob at the sampled token reached the driver. "
                "Under student_indexed_topk=True the top-k is resolved in the actor "
                "from cached hidden states and never travels. Re-run the probe with "
                "algorithm.opd.kl_loss_type=low_var_kl (and the matching "
                "actor_rollout_ref.actor.teacher_kl_loss_type), which writes "
                "teacher_log_probs (bs, resp) into the batch."
            )
        }
    out["source"] = source

    s_lp = None
    for key in ("rollout_log_probs", "old_log_probs"):
        if key in batch.batch.keys():
            s_lp = batch.batch[key]
            out["student_source"] = key
            break
    if s_lp is None:
        return {"error": "batch carries neither rollout_log_probs nor old_log_probs"}

    mask = actor_loss_mask(batch, multi_turn=multi_turn)
    rew = batch.batch.get("token_level_rewards", None)
    if rew is None:
        return {"error": "batch carries no token_level_rewards column"}
    task_ids = batch.batch["task_ids"].reshape(-1).tolist()
    uids = batch.non_tensor_batch.get("uid", None)
    tuids = batch.non_tensor_batch.get("traj_uid", None)
    if uids is None or tuids is None:
        return {"error": "tau needs both uid (prompt group) and traj_uid (trajectory)"}

    delta = (t_lp.to(torch.float32) - s_lp.to(torch.float32))
    live_tok = mask & torch.isfinite(delta)
    d_sum = torch.where(live_tok, delta, torch.zeros_like(delta)).sum(-1)
    d_cnt = live_tok.sum(-1)
    miss = (mask & ~torch.isfinite(delta)).sum().item()
    out["support_miss_frac"] = float(miss / max(int(mask.sum().item()), 1))
    row_return = rew.sum(-1)

    nonuniform = 0
    total_traj = 0
    for tid in sorted(set(int(t) for t in task_ids)):
        name = task_id_names[tid] if tid < len(task_id_names) else str(tid)
        rows = [i for i, t in enumerate(task_ids) if int(t) == tid]

        # rows -> trajectories
        by_traj = defaultdict(list)
        for i in rows:
            by_traj[str(tuids[i])].append(i)
        traj = {}
        for tk, idx in by_traj.items():
            rets = [float(row_return[i]) for i in idx]
            uniform = max(rets) == min(rets)
            nonuniform += 0 if uniform else 1
            total_traj += 1
            n_tok = int(sum(int(d_cnt[i]) for i in idx))
            if n_tok == 0:
                continue
            ssum = float(sum(float(d_sum[i]) for i in idx))
            traj[tk] = {
                "uid": str(uids[idx[0]]),
                "ret": rets[0] if uniform else float(sum(rets)),
                "sum": ssum,
                "mean": ssum / n_tok,
                "ntok": n_tok,
            }

        # trajectories -> prompt groups
        by_group = defaultdict(list)
        for rec in traj.values():
            by_group[rec["uid"]].append(rec)

        taus_mean, taus_sum, taus_b, taus_partial = [], [], [], []
        c_ret_len, c_del_len, top_bot, raw = [], [], [], []
        live = 0
        for g, recs in by_group.items():
            if len(recs) < 2:
                continue
            r = [x["ret"] for x in recs]
            if max(r) == min(r):
                continue  # degenerate: no ranking information, no policy gradient
            live += 1
            m = [x["mean"] for x in recs]
            s = [x["sum"] for x in recs]
            L = [float(x["ntok"]) for x in recs]
            g_rd = _goodman_kruskal_gamma(r, m)
            g_rl = _goodman_kruskal_gamma(r, L)
            g_dl = _goodman_kruskal_gamma(m, L)
            taus_mean.append(g_rd)
            taus_sum.append(_goodman_kruskal_gamma(r, s))
            taus_b.append(_kendall_tau_b(r, m))
            c_ret_len.append(g_rl)
            c_del_len.append(g_dl)
            taus_partial.append(_partial_gamma(g_rd, g_rl, g_dl))
            # The raw triples, so any other statistic can be recomputed from the
            # payload instead of from another rollout. 8 trajectories a group.
            raw.append({"ret": r, "delta": m, "ntok": L})
            hi = max(recs, key=lambda x: x["ret"])["mean"]
            lo = min(recs, key=lambda x: x["ret"])["mean"]
            top_bot.append(hi - lo)

        live_means = [x["mean"] for x in traj.values()]
        out[name] = {
            "rows": len(rows),
            "trajectories": len(traj),
            "groups": len(by_group),
            "live_groups": live,
            # THE DECISION VARIABLE: gamma, comparable across success rates.
            "tau": _mean_se(taus_mean),
            # Same ranking on the un-normalised trajectory log-likelihood ratio.
            # They should agree; if they do not, length is doing the work.
            "tau_sum": _mean_se(taus_sum),
            # Kendall's tau-b on the same pairs. Its ceiling moves with the
            # task's success rate, so it is a cross-check and not the line.
            "tau_b": _mean_se(taus_b),
            # tau with trajectory length held fixed. Read this one whenever the
            # two confounds below are large, and read it AS WELL as tau even
            # when they are not.
            "tau_partial_len": _mean_se(taus_partial),
            # Confounds, reported because delta is a per-token mean and a
            # successful alfworld trajectory is a SHORT one.
            "confound_return_vs_len": _mean_se(c_ret_len),
            "confound_delta_vs_len": _mean_se(c_del_len),
            # tau with a scale: nats per token of extra endorsement the teacher
            # gives the best-returning trajectory over the worst in its group.
            "endorse_top_minus_bottom": _mean_se(top_bot),
            "delta_per_token_mean": float(np.mean(live_means)) if live_means else float("nan"),
            "delta_per_token_std": float(np.std(live_means)) if live_means else float("nan"),
            "groups_raw": raw,
        }

    out["return_nonuniform_within_traj_frac"] = float(nonuniform / max(total_traj, 1))
    return out


def prompt_length_report(batch, task_id_names) -> dict:
    """Per task: how much of the rollout window a turn actually uses.

    THE NUMBER A CONTEXT CHANGE HAS TO BE ARGUED AGAINST. The rollout engine
    admits ``max_model_len`` tokens and ``max_response_length`` of them are
    reserved for the answer, so a turn's prompt has ``max_model_len -
    max_response_length`` to live in. Whether that bound BINDS is an empirical
    question nobody here has answered: ``data.max_prompt_length`` only ever
    touched the parquet row, and those are 8 tokens on alfworld and webshop --
    every real token comes from the environment template and the history, which
    no code truncates. If the p99 turn sits far below the bound, a skill prefix
    fits with the window unchanged and no baseline has to be re-run; if it sits
    near it, the window is already shaping the existing arms and that is a
    finding on its own.
    """
    import numpy as np

    am = batch.batch.get("attention_mask", None)
    resp = batch.batch.get("responses", None)
    if am is None or resp is None:
        return {}
    n = resp.shape[1]
    plen = am[:, :-n].sum(-1).to("cpu").numpy() if am.shape[1] > n else am.sum(-1).to("cpu").numpy()
    task_ids = batch.batch["task_ids"].reshape(-1).tolist()
    out = {}
    for tid in sorted(set(int(t) for t in task_ids)):
        name = task_id_names[tid] if tid < len(task_id_names) else str(tid)
        v = np.array([plen[i] for i, t in enumerate(task_ids) if int(t) == tid], dtype=float)
        if not v.size:
            continue
        out[name] = {
            "rows": int(v.size),
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "max": float(v.max()),
        }
    return out


def accumulate_tau_probe(trainer, batch, state: dict, *, seed: int = 0) -> dict:
    """One rollout batch, no gradients at all.

    The terms probe takes five backward passes per task per batch. tau needs
    none of them, so it is a separate mode rather than a field on that payload:
    the cost of the answer should be the cost of the rollout and nothing more.
    """
    cfg = trainer.config
    task_id_names = list(batch.meta_info.get("task_id_names", []) or [])
    multi_turn = bool(cfg.actor_rollout_ref.rollout.multi_turn.enable)
    n = state["batches"] + 1

    adv = advantage_report(batch, task_id_names, multi_turn=multi_turn)
    tau = teacher_endorsement_report(batch, task_id_names, multi_turn=multi_turn)
    # The same two allocation signals the training loop now logs every step, so a
    # probe batch and a training batch are read the same way.
    from verl.trainer.ppo.metric_utils import compute_group_metrics

    grp = compute_group_metrics(batch, with_records=True)
    plen = prompt_length_report(batch, task_id_names)
    print(f"[grad_probe] tau batch {n} advantages: {json.dumps(adv)}", flush=True)
    print(f"[grad_probe] tau batch {n}: {json.dumps(tau)}", flush=True)

    print(f"[grad_probe] tau batch {n} groups: {json.dumps(grp)}", flush=True)
    state.setdefault("advantages", []).append(adv)
    state.setdefault("tau", []).append(tau)
    state.setdefault("groups", []).append(grp)
    state.setdefault("prompt_len", []).append(plen)
    print(f"[grad_probe] tau batch {n} prompt_len: {json.dumps(plen)}", flush=True)
    state["batches"] = n
    return state


# ---------------------------------------------------------------------------
# rho: is an injected failure one the UNCONDITIONED student could also produce?
# ---------------------------------------------------------------------------
#
# THE NUMBER THAT DECIDES WHETHER INJECTION CAN WORK. A row injected into a
# saturated group carries advantage -sqrt(7); what reaches the weights is
#
#     dJ/dlog pi = A * gamma * rho / (rho + gamma)^2,
#     rho = pi_theta(a | x) / pi_theta(a | x, z)
#
# with the behaviour probability held fixed. That coefficient is ZERO at rho = 0
# and maximal at rho = gamma. So a failure the student would never produce
# without the privileged context z carries no gradient no matter how large its
# advantage is, and shaping does not rescue it -- shaping is what makes the
# coefficient vanish there. This is the reason base's failures are suspect and
# the reason a wrong-plan failure from the SAME weights is worth measuring: only
# its rho distribution says whether it is reachable.
#
# NOT A DERIVED QUANTITY. f(rho) near 1 is NOT evidence of reachability: at
# rho = gamma = 0.1 the shaped ratio is 0.5 and the coefficient is at its
# maximum. Report the coefficient itself, split by the sign of the advantage.
#
# COST. One forward pass of the actor's own weights over tokens that already
# exist. No generation, no second engine.


def reachability_report(trainer, batch, task_id_names, *, strip_fn, gamma: float = 0.1) -> dict:
    """Per task: the distribution of rho and of the shaping coefficient.

    ``strip_fn(batch) -> DataProto`` must return the same rows with the
    privileged block removed from the prompt; everything else identical. The
    caller owns that, because only it knows how z was inserted.
    """
    import numpy as np
    import torch

    lp_cond = batch.batch.get("rollout_log_probs", batch.batch.get("old_log_probs", None))
    if lp_cond is None:
        return {"error": "batch carries no rollout_log_probs/old_log_probs"}
    try:
        stripped = strip_fn(batch)
        out = trainer.actor_rollout_wg.compute_log_prob(stripped)
        lp_plain = out.batch["old_log_probs"]
    except Exception as exc:
        return {"error": f"re-scoring failed: {exc!r}"}
    if lp_plain.shape != lp_cond.shape:
        return {"error": f"shape mismatch {tuple(lp_plain.shape)} vs {tuple(lp_cond.shape)}"}

    mask = actor_loss_mask(batch, multi_turn=True)
    log_rho = (lp_plain.to(torch.float32) - lp_cond.to(torch.float32))
    rho = log_rho.clamp(min=-30.0, max=30.0).exp()
    coef = gamma * rho / (rho + gamma) ** 2

    task_ids = batch.batch["task_ids"].reshape(-1).tolist()
    res = {"gamma": gamma}
    for tid in sorted(set(int(t) for t in task_ids)):
        name = task_id_names[tid] if tid < len(task_id_names) else str(tid)
        rows = [i for i, t in enumerate(task_ids) if int(t) == tid]
        m = mask[rows]
        if not int(m.sum()):
            continue
        lr = log_rho[rows][m].detach().cpu().numpy()
        cf = coef[rows][m].detach().cpu().numpy()
        res[name] = {
            "tokens": int(lr.size),
            "log_rho": {q: float(np.percentile(lr, q)) for q in (5, 25, 50, 75, 95)},
            "rho_median": float(np.exp(np.percentile(lr, 50))),
            "coef": {q: float(np.percentile(cf, q)) for q in (5, 50, 95)},
            # the coefficient's ceiling is 1/4 at rho = gamma; report the share of
            # tokens that get at least a tenth of it, i.e. that carry any gradient
            "frac_coef_above_0.025": float((cf > 0.025).mean()),
        }
    return res
