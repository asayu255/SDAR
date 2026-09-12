"""
OPD + GRPO Trainer — multitask.

This is the OPD (On-Policy Distillation) multitask trainer with GRPO added back
on top. The student is trained jointly by:

  policy_loss = pg_loss * pg_loss_coef + teacher_kl_loss * teacher_kl_coef

i.e. the GRPO policy-gradient (group-relative advantages from the env reward)
*plus* the per-task teacher-KL distillation that pure OPD uses. Everything else —
per-task teacher routing, the 3-task (alfworld/search/webshop) data, batch sizes,
env settings, the student-top-k support and the cross-teacher sign weighting —
matches the pure-OPD multitask run.

Everything except the objective is inherited from :class:`OPDRayTrainer`, and
deliberately so. The teacher routing, the hidden-state cache and its witness, the
sign-weight pass, the env-reset prefetch and ``stop_after_steps`` are the parts
that have to stay identical for an A/B against pure OPD to mean anything; a
second copy of that loop is exactly how "the arms differ only in the objective"
quietly stops being true. So this subclass overrides only the two hooks
``OPDRayTrainer.fit`` calls:

* :meth:`_reward_and_advantage` — pure OPD scores the batch for monitoring only;
  here the same reward becomes ``token_level_rewards`` and, via ``old_log_prob``
  and ``compute_advantage``, the group-relative advantages the policy gradient
  reads.
* :meth:`_data_metrics` — the advantage-bearing batch can use the full
  ``compute_data_metrics`` instead of the advantage-free OPD variant.
"""

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_data_metrics_by_task,
    compute_group_metrics,
    get_task_names,
)
from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer
from verl.trainer.ppo.ray_trainer import (
    GRPO_STAT_EXCLUDE_KEY,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.reward import compute_reward

from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY

from agent_system.multi_turn_rollout import compute_log_prob_with_prefetch


class OPDGRPORayTrainer(OPDRayTrainer):
    """Multitask trainer combining GRPO policy-gradient with per-task teacher-KL distillation."""

    progress_desc = "OPD+GRPO Training"

    def _reward_and_advantage(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Score the batch, then turn that score into GRPO advantages.

        Unlike the pure-OPD base, the reward is NOT monitoring-only here: it is
        the policy gradient's whole signal, so a reward-manager failure is a
        failed step rather than something to print and carry on from. Hence no
        try/except around ``compute_reward``.

        Order matches the standard PPO loop: reward, then ``old_log_prob`` (the
        ratio's denominator, so it must be the policy that generated the
        responses — i.e. before ``update_actor``), then advantages.
        """
        with _timer("reward", timing_raw):
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

        # ---- old_log_prob (required by the GRPO policy-gradient) ----
        with _timer("old_log_prob", timing_raw):
            # Reuse any per-row log probs prefetched during the rollout
            # (ROLLOUT_PREFETCH_LOGPROB); computes everything normally when
            # nothing was prefetched.
            old_log_prob = compute_log_prob_with_prefetch(
                self.actor_rollout_wg,
                batch,
                self.traj_collector.take_prefetched_log_probs(),
                temperature=self.config.actor_rollout_ref.rollout.temperature,
            )
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            metrics.update(self._entropy_loss_metrics(batch, entropys, response_masks, loss_agg_mode))
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        # ---- advantages (GRPO) ----
        with _timer("adv", timing_raw):
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
                batch, invalid_metrics = apply_invalid_action_penalty(
                    batch,
                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                    invalid_action_penalty_coef_by_task=self.config.actor_rollout_ref.actor.get(
                        "invalid_action_penalty_coef_by_task", None
                    ),
                )
                metrics.update(invalid_metrics)

            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # ---- OCI-sat: swap one row of a saturated group for a failure ----
            # Before compute_advantage, because the point is to move the group
            # baseline. A saturated group's zero advantage is GRPO behaving
            # correctly -- the baseline equals the return -- so this manufactures
            # a signal rather than recovering one, and the arm exists to find out
            # whether the manufactured signal is worth anything. The verdict uses
            # only the rows NOT reserved for injection, and is never carried to
            # the next step: re-drawing the same dataset row changes its label 83%
            # of the time because the row is a placeholder and the environment
            # picks the game.
            oci_cfg = self.config.algorithm.get("oci_sat", None)
            oci_injected = None
            if oci_cfg is not None and bool(oci_cfg.get("enable", False)):
                import torch as _t

                from verl.trainer.ppo.oci_saturated import (
                    classify_groups, injection_metrics, select_saturated_injections,
                    token_mass_by_class)

                # NO FALLBACK. The mark is made where the row order means
                # something -- the env slot, in the rollout loop -- and travels as
                # a column. Deriving it here from the group layout is not a
                # degraded version of that, it is wrong: rows are regrouped by
                # task, padded and reordered by _balance_batch before this runs,
                # so any rule applied to this row order is applied to a permuted
                # one and marks trajectories that were never shown a plan. A
                # missing column means the rollout was not built with the switch
                # on, which is a launch error, not something to paper over.
                cand = batch.batch.get("oci_candidate", None)
                assert cand is not None, (
                    "algorithm.oci_sat.enable=True but the batch carries no "
                    "oci_candidate column. The column is emitted by the rollout "
                    "loop, which emits it when algorithm.oci_sat.enable reaches the "
                    "environment manager that builds the observations."
                )
                cand_np = cand.reshape(-1).detach().cpu().numpy().astype(bool)
                # PER GROUP, NOT "AT LEAST ONE". The previous version asserted
                # only that SOME row was marked, and that is not a check: when
                # the prefix travelled by env-slot index and was read by the
                # global row index, exactly one alfworld group of fifteen matched
                # and the assert passed. The other fourteen candidate
                # trajectories went unmarked, were judged as part of their own
                # group, and were trained as plain rows with the corrupted plan
                # still in the prompt. Every group on an oci_sat task gets
                # exactly one candidate trajectory or the batch is not what the
                # arm says it is.
                _tn_all = get_task_names(batch)
                _tasks_cfg = set(oci_cfg.get("tasks", ["alfworld"]) or ["alfworld"])
                _tuids = batch.non_tensor_batch.get("traj_uid", None)
                _uids = batch.non_tensor_batch.get("uid", None)
                if _tn_all is not None and _uids is not None and _tuids is not None:
                    _per_group = {}
                    for _i in range(len(batch)):
                        if str(_tn_all[_i]) not in _tasks_cfg:
                            continue
                        _per_group.setdefault(str(_uids[_i]), set())
                        if cand_np[_i]:
                            _per_group[str(_uids[_i])].add(str(_tuids[_i]))
                    _bad = {g: len(v) for g, v in _per_group.items() if len(v) != 1}
                    assert _per_group and not _bad, (
                        f"{len(_bad)} of {len(_per_group)} groups on "
                        f"{sorted(_tasks_cfg)} do not carry exactly one candidate "
                        f"trajectory (counts: {sorted(set(_bad.values()))}). "
                        "Zero everywhere means algorithm.oci_sat.enable did not reach "
                        "the alfworld environment manager, or the run is not a "
                        "training rollout. Zero on SOME groups means the mark and the row it "
                        "was read for are indexed differently; the prefix must "
                        "travel on the observation dict, which the multitask "
                        "merge reorders, not in a side channel keyed by env slot."
                    )
                else:
                    metrics["oci/error_cannot_verify_marks"] = 1
                    assert cand_np.any(), (
                        "algorithm.oci_sat.enable=True but not one row is marked "
                        "as a candidate, and the batch lacks the columns needed to "
                        "check this per group."
                    )
                # And nothing marked OFF the configured tasks. Separate from the
                # per-group check above, which only walks the on-task rows and so
                # cannot see a candidate that appeared on webshop or search. The
                # switch is gated on the alfworld manager, so this should be
                # unreachable; it is here because the two ends of that gate are in
                # different files.
                if _tn_all is not None and cand_np.any():
                    _off = sorted({str(t) for t, c in zip(_tn_all, cand_np)
                                   if c and str(t) not in _tasks_cfg})
                    assert not _off, (
                        f"oci_candidate marked rows on {_off}, which is not in "
                        f"algorithm.oci_sat.tasks={sorted(_tasks_cfg)}"
                    )

                grp = classify_groups(batch, judged_rows=~cand_np)
                oci_injected = select_saturated_injections(
                    batch, grp, candidate_rows=cand_np)

                # A candidate the group did not want must be inert, and that
                # takes BOTH of the following. Zeroing response_mask alone leaves
                # its return in the group's mean and std (nothing in
                # compute_grpo_outcome_advantage reads response_mask before the
                # statistic is formed), and under the turn-weighted statistic a
                # long wrong-plan failure moves that baseline once per turn -- so
                # live and stuck groups, which the design says it does not touch,
                # were having their yardstick moved by a rollout that was
                # supposed to have been discarded.
                drop = cand_np & ~oci_injected
                batch.batch[GRPO_STAT_EXCLUDE_KEY] = _t.as_tensor(
                    drop, device=batch.batch["response_mask"].device)
                if drop.any():
                    keep = _t.as_tensor(~drop, device=batch.batch["response_mask"].device)
                    batch.batch["response_mask"] = (
                        batch.batch["response_mask"] * keep.unsqueeze(-1).to(
                            batch.batch["response_mask"].dtype))
                    # The same rows must not train the distillation term either:
                    # dp_actor aggregates the teacher-KL over response_mask, so
                    # the line above already removes them from OPD. Recorded
                    # because the design document claims OPD is untouched, and it
                    # is not -- an arm trains OPD on 7 rollouts where control
                    # trains it on 8.
                    metrics["oci/rows_dropped"] = int(drop.sum())
                    metrics["oci/trajectories_dropped"] = len({
                        str(t) for t, d in zip(
                            batch.non_tensor_batch.get("traj_uid", []), drop) if d})
                # Shipped to the actor so arm A's shaping knows which rows to
                # take on the plan-stripped prompt. A column rather than a
                # recomputation: the selection depends on the group verdict,
                # which only the driver has.
                batch.batch["oci_injected"] = _t.as_tensor(
                    oci_injected, device=batch.batch["response_mask"].device).long()
                metrics.update(injection_metrics(
                    grp, oci_injected,
                    # Not a literal: widening oci_sat.tasks would leave the
                    # metric filed under a task the arm no longer only touches.
                    task="+".join(sorted(_tasks_cfg))))
                # tokens, not groups: a saturated group finishes early and a
                # stuck one runs to the turn cap, so the group count and the
                # token count disagree by about 4x and only the token count
                # bounds what injection can buy. Dropped rows are reported
                # separately rather than folded into a class.
                metrics.update(token_mass_by_class(
                    batch, grp,
                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                    exclude_rows=drop))

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                gigpo_mode=self.config.algorithm.gigpo.mode,
                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                # See ray_trainer: pinned rather than defaulted, because an
                # injected row changes its group's baseline through its length
                # under the turn-weighted statistic.
                compute_mean_std_cross_steps=self.config.algorithm.get(
                    "compute_mean_std_cross_steps", True),
            )

            # Arm B takes the injected row's gradient away AFTER the baseline
            # has already moved, so the seven successes keep their +1/sqrt(7) and
            # only the -sqrt(7) goes. A beating B is the only thing that shows
            # suppressing the failure is worth more than that re-reinforcement.
            if oci_injected is not None and not bool(oci_cfg.get("gradient_on_injected", True)):
                import torch as _t

                from verl.trainer.ppo.oci_saturated import zero_injected_advantage

                metrics["oci/rows_zeroed"] = zero_injected_advantage(batch, oci_injected)
                # AND THE DISTILLATION TERM. Zeroing the advantage removes the
                # policy gradient and nothing else: dp_actor aggregates the
                # teacher-KL over response_mask, so without this the injected row
                # still trains OPD -- on a plan-conditioned prompt, with the
                # teacher also reading the plan. B is supposed to be the arm that
                # makes NO claim about the injected row, so it must not learn from
                # it at all. Safe here and only here: the baseline has already
                # moved inside compute_advantage, so the seven successes keep the
                # advantage the injection bought them.
                if oci_injected.any():
                    _keep = _t.as_tensor(~oci_injected, device=batch.batch["response_mask"].device)
                    batch.batch["response_mask"] = (
                        batch.batch["response_mask"] * _keep.unsqueeze(-1).to(
                            batch.batch["response_mask"].dtype))

            batch = self._attach_advantage_reliability_columns(batch)

        return batch, reward_extra_infos_dict

    def _attach_advantage_reliability_columns(self, batch: DataProto) -> DataProto:
        """Per row: its advantage, and whether its prompt group carried any signal.

        The parameter-free cross-teacher arm calibrates each source teacher by
        correlating its residual support for the tokens the student emitted
        against the advantage of the trajectory it emitted them in, so it needs
        both a per-ROW advantage and a marker for which rows can inform that
        correlation. Both are driver-side facts -- the actor sees micro-batches
        and cannot see a prompt group at all -- and both are cheap here.

        ``adv_group_informative`` is false wherever the group has no spread of
        advantage. GRPO is group-relative, so a prompt whose rollouts all scored
        the same gives every one of its rows an advantage of zero; folding those
        into the correlation adds variance to the support score against none in
        the advantage and drags every pair's estimate toward zero for a reason
        that has nothing to do with the teachers. The comparison is against the
        advantages already computed -- no new threshold is introduced.

        Padding rows are excluded outright: ``adjust_batch`` appends them as
        copies carrying their original's uid, so leaving them in would count one
        trajectory twice.
        """
        adv = batch.batch["advantages"]
        mask = batch.batch["response_mask"].to(adv.dtype)
        denom = mask.sum(dim=-1).clamp(min=1)
        row_adv = (adv * mask).sum(dim=-1) / denom

        # Outcome GRPO broadcasts one score across the row, and the reliability
        # correlation is a statement about trajectories. Checked rather than
        # assumed: a step-level estimator would make the row mean an average of
        # different things and the correlation would quietly change meaning.
        spread = ((adv - row_adv.unsqueeze(-1)).abs() * mask).max()
        if float(spread) > 1e-4:
            print(
                f"[cross_teacher] advantages vary within a row (max deviation {float(spread):.3g}); "
                "the reliability correlation uses the masked row mean",
                flush=True,
            )

        uids = batch.non_tensor_batch.get("uid", batch.non_tensor_batch.get("traj_uid", None))
        real = torch.ones_like(row_adv, dtype=torch.bool)
        padding = batch.batch.get(PADDING_ROW_KEY, None)
        if padding is not None:
            real &= ~padding.reshape(-1).to(torch.bool)
        informative = torch.zeros_like(real)
        if uids is not None:
            by_uid = {}
            for i, u in enumerate(np.asarray(uids).reshape(-1).tolist()):
                if bool(real[i]):
                    by_uid.setdefault(u, []).append(i)
            for rows in by_uid.values():
                vals = [float(row_adv[i]) for i in rows]
                if len(vals) > 1 and max(vals) - min(vals) > 0:
                    for i in rows:
                        informative[i] = True

        # A DENSE group index, because the reliability statistic centres the
        # support score within the prompt group and a group's rollouts land in
        # different micro-batches and on different ranks: the accumulator pools
        # them by this id and all-reduces, which is exact where centring a local
        # fragment would not be. Dense and batch-local -- it names a prompt
        # within this step and nothing beyond it.
        group_id = torch.full_like(row_adv, -1, dtype=torch.long)
        if uids is not None:
            order = {}
            for i, u in enumerate(np.asarray(uids).reshape(-1).tolist()):
                if not bool(real[i]):
                    continue
                group_id[i] = order.setdefault(u, len(order))

        batch.batch["adv_row_value"] = row_adv
        batch.batch["adv_group_informative"] = informative
        batch.batch["adv_group_id"] = group_id
        return batch

    def _data_metrics(self, batch: DataProto) -> dict:
        """The full advantage-bearing statistics, which this arm's batch carries.

        Pure OPD reports :func:`compute_opd_data_metrics`, an advantage-free
        variant, precisely because it never computes advantages. Here they exist,
        so there is no reason to report less than the standard PPO loop does.
        """
        metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
        metrics.update(compute_data_metrics_by_task(batch=batch, use_critic=self.use_critic))
        # The allocation signals, from columns the batch already carries: whether
        # a rollout on this task buys a policy gradient at all, and if not
        # whether the task is stuck or solved. Costs one pass over the returns.
        metrics.update(compute_group_metrics(batch))
        return metrics
