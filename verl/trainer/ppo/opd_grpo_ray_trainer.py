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

import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_data_metrics_by_task,
    compute_group_metrics,
    get_task_names,
)
from verl.trainer.ppo.opd_ray_trainer import (
    OPDRayTrainer,
    _write_json_atomic,
    progress_value_config,
    progress_value_turn_caps,
)
from verl.trainer.ppo.ray_trainer import (
    GRPO_STAT_EXCLUDE_KEY,
    OCI_FLOOR_KEY,
    _check_progress_value_caps,
    _get_invalid_action_penalty_coef,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.reward import compute_reward

from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY

from agent_system.multi_turn_rollout import compute_log_prob_with_prefetch

PROGRESS_VALUE_ESTIMATOR = "progress_value_gae"
# The arms that move a group's statistic or its rows. progress_value_gae has no group statistic
# for them to move, so on its batch they would do nothing and still be reported as on.
_OCI_ARMS = ("oci_sat", "oci_floor", "oci_slots", "oci_rank")


def check_progress_value_config(config) -> bool:
    """True when the run's advantages come from progress_value_gae; refuses what would not mean what it says.

    ``algorithm.progress_value.enable`` is the switch for the RECORDS (the env managers count and the
    rollout loop writes the pv_* columns); the estimator is ``algorithm.adv_estimator``. So:

      * adv_estimator=progress_value_gae needs progress_value.enable -- without it no pv_* column is
        recorded and the first step would find nothing to read -- and it runs ALONE: progress_rank
        adds to outcome-GRPO advantages and asserts grpo itself, every OCI arm moves a GRPO group's
        statistic this estimator does not have, and a KL-in-reward penalty would sit in
        token_level_rewards, which this estimator never reads (R is episode_rewards > 0).
      * progress_value.enable beside another estimator is records-only: the columns ride along and
        nothing in the trainer reads them. False is returned and nothing is checked.

    Called by inject_opd_grpo_config, so a bad combination fails in the first seconds of a launch
    (the progress_value keys are validated there too), and again by the trainer on every step.
    The analysis records' own keys are checked here as well, beside any estimator
    (check_rollout_records_config).
    """
    check_rollout_records_config(config)
    alg = config.algorithm
    if alg.get("adv_estimator", None) != PROGRESS_VALUE_ESTIMATOR:
        return False
    pv_cfg = alg.get("progress_value", None)
    assert pv_cfg is not None and bool(pv_cfg.get("enable", False)), (
        "adv_estimator=progress_value_gae needs algorithm.progress_value.enable=True: it is what makes "
        "the environment managers and the rollout loop record the pv_* columns the estimator reads")
    pr_cfg = alg.get("progress_rank", None)
    assert not (pr_cfg is not None and bool(pr_cfg.get("enable", False))), (
        "adv_estimator=progress_value_gae and algorithm.progress_rank.enable are both on; (a) adds to "
        "outcome-GRPO advantages, and the value arm is to be measured with nothing else changed. "
        "The progress counters progress_rank's keys choose (alfworld_k, search_k, webshop_k) are "
        "read with it off.")
    for other in _OCI_ARMS:
        ocfg = alg.get(other, None)
        assert not (ocfg is not None and bool(ocfg.get("enable", False))), (
            f"adv_estimator=progress_value_gae and algorithm.{other}.enable are both on; the OCI arms "
            "move a GRPO group's statistic, which this estimator does not have")
    assert not bool(alg.get("use_kl_in_reward", False)), (
        "adv_estimator=progress_value_gae with algorithm.use_kl_in_reward: the penalty would go into "
        "token_level_rewards, which this estimator never reads, and silently do nothing")
    # The block's own values (features, buckets, n0, gamma < 1 without the gamma^t prefix, ...) and
    # the turn caps and k definitions the table is built for, checked now rather than at step 1 --
    # through the builder the trainer's table is made by, so what is checked is what will run.
    progress_value_config(config)
    return True


def check_rollout_records_config(config) -> bool:
    """True when the analysis records are on (algorithm.progress_value.records.enable); refuses what they cannot do.

    The records (verl/trainer/ppo/rollout_records.py) read the pv_* columns, so they need
    progress_value.enable. Beside an estimator other than progress_value_gae they also build a
    records-only shadow value table (records.shadow_value), with the same builder as a training table
    -- so a configuration that builder refuses (gamma < 1 without the gamma^t prefix, an unknown
    feature, a counter the environment managers do not know) is refused here, at launch, rather than
    leaving a run whose shadow never appears. Called from check_progress_value_config, beside any
    estimator.
    """
    from verl.trainer.ppo.rollout_records import check_records_config

    rcfg = check_records_config(config)
    if not rcfg.enable:
        return False
    if config.algorithm.get("adv_estimator", None) != PROGRESS_VALUE_ESTIMATOR and rcfg.shadow_value:
        try:
            progress_value_config(config)
        except AssertionError as e:
            raise AssertionError(
                "algorithm.progress_value.records.shadow_value: the records-only shadow value table cannot be "
                f"built under this configuration ({e}); fix it, or set records.shadow_value=False") from e
    return True


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

        # ---- self-distillation teacher (algorithm.opsd) ----
        # The skill-conditioned SELF as the distillation teacher, exactly as the
        # SDAR runs build it: the same current weights, the student's own
        # responses, the prompt prefixed with the task's skill documents. Written
        # to teacher_log_probs, which the actor reads under use_sdar_loss. After
        # old_log_prob and before the advantages, so the rows are already in the
        # order (and padding) the actor will see.
        opsd_cfg = self.config.algorithm.get("opsd", None)
        if opsd_cfg is not None and bool(opsd_cfg.get("enable", False)):
            with _timer("opsd_teacher", timing_raw):
                lp, valid = self._compute_self_teacher_log_probs(batch, opsd_cfg, metrics)
                if bool(opsd_cfg.get("measure_only", False)):
                    # ITS OWN COLUMN. measure_only builds the term to report its
                    # geometry and adds nothing to the loss, so nothing that
                    # consumes teacher_log_probs may see it -- least of all the
                    # external teacher's own path, which writes that name too.
                    batch.batch["opsd_teacher_log_probs"] = lp
                    batch.batch["opsd_valid"] = valid
                else:
                    batch.batch["teacher_log_probs"] = lp

        # ---- tied-group self-distillation (algorithm.tied_opsd) ----
        # After old_log_prob, like the OPSD teacher above: the rows are in their final order and padding,
        # and the group outcomes are known. Writes tied_{w,side,tok,topk_ids,topk_lp}; the actor adds the
        # term (dp_actor, tied block). The advantages below are GRPO's, untouched.
        tied_cfg = self.config.algorithm.get("tied_opsd", None)
        if tied_cfg is not None and bool(tied_cfg.get("enable", False)):
            with _timer("tied_teacher", timing_raw):
                self._tied_opsd_columns(batch, tied_cfg, metrics)

        # ---- advantages (GRPO) ----
        with _timer("adv", timing_raw):
            # ---- progress_value_gae: the value table the batch is scored against ----
            # Empty unless it IS the estimator (progress_value.enable beside another one only
            # records the pv_* columns). First, so a combination it refuses fails before any other
            # arm touches the batch. The table is read inside compute_advantage and left with this
            # batch's update staged; it is committed right after, once the batch has been scored
            # (dropped instead on a grad_probe batch, which takes no step: _progress_value_commit).
            pv_kwargs = self._progress_value_kwargs(batch)

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

            # ---- OCI-sat arm B': a virtual sample at the failure return ------
            # The same place in the pipeline as the injection above and for the
            # same reason -- the point is to move the group baseline before the
            # advantage is computed -- but nothing is injected. Arm B removed the
            # injected row's gradient, leaving the group's mean and std as the
            # only channel by which it reached the weights, and a sample that
            # only has to move a mean and a std does not have to be generated.
            # See oci_floor for what that removes: the corrupted plan, the
            # candidate column, the span bookkeeping, and the bar the plan could
            # not clear (cand_fail_rate 0.222 against 0.5 over five designs). It
            # also takes no rollout slot, so all eight rollouts are real, judged
            # and trained, and the 7-vs-8 confound against control is gone.
            oci_floor_cfg = self.config.algorithm.get("oci_floor", None)
            oci_floored = None
            if oci_floor_cfg is not None and bool(oci_floor_cfg.get("enable", False)):
                import torch as _t

                from verl.trainer.ppo.oci_floor import (
                    floor_metrics, select_floor_groups)
                from verl.trainer.ppo.oci_saturated import classify_groups

                assert not (oci_cfg is not None and bool(oci_cfg.get("enable", False))), (
                    "algorithm.oci_floor.enable and algorithm.oci_sat.enable are "
                    "both on. They are two implementations of the same arm -- a "
                    "real injected failure and a virtual one -- and running both "
                    "gives a saturated group two failures, one of which also eats "
                    "a rollout slot. Pick one."
                )
                _floor_tasks = list(oci_floor_cfg.get("tasks", ["alfworld"])
                                    or ["alfworld"])
                # EVERY TRAJECTORY IS JUDGED. The injection arm had to withhold
                # the candidate from its own group's verdict; here there is no
                # candidate, so the class is read off all eight.
                _grp = classify_groups(batch)
                oci_floored = select_floor_groups(
                    batch, _grp, tasks=_floor_tasks,
                    padding_mask=batch.batch.get(PADDING_ROW_KEY, None))
                batch.batch[OCI_FLOOR_KEY] = _t.as_tensor(
                    oci_floored, device=batch.batch["response_mask"].device)
                metrics.update(floor_metrics(
                    batch, _grp, oci_floored, tasks=_floor_tasks))

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
                # The return the virtual sample carries. 0.0 is ALFWorld's
                # failure return; it is read from the config rather than defaulted
                # because the reward scales are not comparable across tasks.
                oci_floor_value=float((oci_floor_cfg or {}).get("value", 0.0)
                                      if oci_floor_cfg is not None else 0.0),
                **pv_kwargs,
            )
            if pv_kwargs:
                metrics.update(self._progress_value_commit(pv_kwargs["progress_value_out"]))

            # WHAT THE FLOOR ACTUALLY BOUGHT, read off the advantage column after
            # the fact. The arm's whole claim is that these rows go from exactly
            # zero to a fixed positive number, and nothing else in the run checks
            # that the term reached the statistic.
            if oci_floored is not None and oci_floored.any():
                from verl.trainer.ppo.oci_floor import realized_advantage

                metrics.update(realized_advantage(batch, oci_floored))

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

            # ---- (a): rank stuck groups by progress (algorithm.progress_rank) ----
            # AFTER compute_advantage, and adding to it rather than touching the
            # reward: the ordinary advantage -- format penalty included -- is left
            # exactly as control computes it, and the ranking is a term on top. Mixing
            # progress into the reward would renormalise it together with the -0.1
            # penalty, which only acts inside degenerate groups and is what holds
            # verbosity down. See verl/trainer/ppo/progress_rank.py.
            pr_cfg = self.config.algorithm.get("progress_rank", None)
            if pr_cfg is not None and bool(pr_cfg.get("enable", False)):
                metrics.update(self._apply_progress_rank(batch, pr_cfg))

            batch = self._attach_advantage_reliability_columns(batch)

        # ---- analysis records (algorithm.progress_value.records) ----
        # LAST, once the advantages are what the actor will read (the value table already committed,
        # progress_rank already added): one record per group and the records/* metrics of this step's
        # advantages beside what GRPO and GiGPO would have given the same rollouts
        # (verl/trainer/ppo/rollout_records.py). It reads the batch and writes nothing to it, and never
        # touches the training value table. A no-op with records off.
        self._rollout_records(batch, metrics, timing_raw, pv_kwargs)

        return batch, reward_extra_infos_dict

    # --- algorithm.tied_opsd ---------------------------------------------------- #

    def _tied_controller(self, cfg):
        ctl = getattr(self, "_tied_ctl", None)
        if ctl is None:
            from verl.trainer.ppo.tied_opsd import TIED_TASKS, TiedController

            tasks = [t for t in TIED_TASKS if t in list(cfg.get("tasks", TIED_TASKS) or TIED_TASKS)]
            ctl = TiedController(tasks, retention=float(cfg.get("retention", 0.8)),
                                 guard=bool(cfg.get("guard", True)))
            pending = getattr(self, "_tied_pending_state", None)
            if pending:
                ctl.load_state_dict(pending["controller"])
                print(f"[tied_opsd] shares and M restored: {ctl.state_dict()}", flush=True)
            self._tied_ctl = ctl
        return ctl

    def _tied_refresh_teacher(self, cfg, metrics: dict) -> None:
        """The self-teacher holds the actor as it was after step k* = every*floor((step-1)/every).

        Steps 1..every read the initial model (the teacher was built from the same path). At step every+1
        the actor has finished step every, so it is copied in-process. On a resume the copy is read back
        from that step's actor checkpoint; if it is gone, the current actor is copied and the metric
        tied/teacher_restore_fallback says so.
        """
        every = int(cfg.get("refresh_every", 50))
        step = int(getattr(self, "global_steps", 1) or 1)
        want = every * ((step - 1) // every) if every > 0 else 0
        have = getattr(self, "_tied_teacher_from", None)
        refreshed, fallback = 0.0, 0.0
        if have is None:
            pending = getattr(self, "_tied_pending_state", None) or {}
            have = 0
            if want > 0:
                if step - 1 == want:
                    # Resumed exactly at a refresh boundary: the actor IS step `want`.
                    self._tied_copy_actor()
                else:
                    folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{want}", "actor")
                    ok = all(bool(x) for x in self.self_teacher_wg.load_ref_from_actor_checkpoint(folder))
                    if not ok:
                        self._tied_copy_actor()
                        fallback = 1.0
                        print(f"[tied_opsd] WARNING: no actor checkpoint at {folder}; the self-teacher copies the "
                              f"CURRENT actor (step {step - 1}) instead of step {want}", flush=True)
                    else:
                        print(f"[tied_opsd] self-teacher restored from {folder} (pending state said "
                              f"{pending.get('teacher_from')})", flush=True)
                have, refreshed = want, 1.0
        elif want != have:
            self._tied_copy_actor()
            have, refreshed = want, 1.0
            if step - 1 != want:
                fallback = 1.0
        self._tied_teacher_from = have
        metrics["tied/teacher_from_step"] = float(have)
        metrics["tied/teacher_refreshed"] = refreshed
        metrics["tied/teacher_restore_fallback"] = fallback

    def _tied_copy_actor(self) -> None:
        outs = self.self_teacher_wg.copy_weights_from_actor()
        for n, a, b in outs:
            assert abs(a - b) <= 1e-3 * max(1.0, abs(a)), f"self-teacher copy mismatch: {a} vs {b}"
        print(f"[tied_opsd] self-teacher copied from the actor ({outs[0][0]} tensors per rank)", flush=True)

    def _tied_opsd_columns(self, batch: DataProto, cfg, metrics: dict) -> None:
        from collections import defaultdict

        from verl.protocol import DataProtoConfig, pad_dataproto_to_divisor, unpad_dataproto
        from verl.trainer.ppo.tied_opsd import (SIDE_SAT, SIDE_STUCK, UNSCORED_LP, TokenMasker, apply_edit,
                                                classify_groups, live_abs_z, row_sides_and_weights)

        nt = batch.non_tensor_batch
        n = len(batch)
        k = int(cfg.get("topk", 20))
        real = np.ones(n, dtype=bool)
        pad = batch.batch.get(PADDING_ROW_KEY, None)
        if pad is not None:
            real &= ~pad.reshape(-1).to(torch.bool).cpu().numpy()
        tasks = get_task_names(batch)
        assert tasks is not None, "algorithm.tied_opsd needs per-row task names"
        tasks = np.asarray([str(t) for t in tasks], dtype=object)
        uids = np.asarray([str(u) for u in nt["uid"]], dtype=object)
        tuids = np.asarray([str(t) for t in nt["traj_uid"]], dtype=object)
        status, task_of, won = classify_groups(uids, tuids, tasks, nt["episode_rewards"], real)

        rows_of = defaultdict(list)
        for i in range(n):
            if real[i]:
                rows_of[uids[i]].append(i)
        live_z = defaultdict(list)
        for u, rows in rows_of.items():
            if status.get(u) == "live":
                live_z[task_of[u]] += live_abs_z([won[tuids[i]] for i in rows], [tuids[i] for i in rows])
        ctl = self._tied_controller(cfg)
        ctl.update(status, task_of, live_z)
        # The amount guard reads THIS batch's group counts per task (n_sat w_s <= n_live M).
        n_live = {t: sum(1 for u, s_ in status.items() if s_ == "live" and task_of.get(u) == t) for t in ctl.tasks}
        n_sat = {t: sum(1 for u, s_ in status.items() if s_ == "saturated" and task_of.get(u) == t) for t in ctl.tasks}
        wts = {t: ctl.weights(t, n_live=n_live[t], n_sat=n_sat[t]) for t in ctl.tasks}

        match = batch.batch["tied_match"].reshape(-1).cpu().numpy()
        doc_ok = batch.batch["oci_doc_len"].reshape(-1).cpu().numpy() > 0
        s_ok = batch.batch["tied_s_len"].reshape(-1).cpu().numpy() > 0
        on_task = np.asarray([t in ctl.tasks for t in tasks], dtype=bool)
        side, _ = row_sides_and_weights(uids, tasks, status, wts, eligible=np.ones(n, dtype=bool))
        elig = real & on_task & (match == 1) & doc_ok & ((side != SIDE_SAT) | s_ok)
        side, w = row_sides_and_weights(uids, tasks, status, wts, eligible=elig)

        resp = batch.batch["responses"]
        L = int(resp.shape[1])
        resp_mask = batch.batch["attention_mask"][:, -L:].cpu()
        evid = batch.batch["tied_evid"].reshape(-1).cpu().numpy()
        if getattr(self, "_tied_masker", None) is None:
            self._tied_masker = TokenMasker(self.tokenizer)
        tok = torch.zeros((n, L), dtype=torch.float32)
        resp_cpu = resp.cpu()
        for i in np.nonzero(w > 0)[0]:
            ln = int(resp_mask[i].sum())
            tok[i, :ln] = torch.from_numpy(self._tied_masker.row(resp_cpu[i, :ln].tolist(), tasks[i], bool(evid[i])))
        w[(tok.sum(-1) <= 0).numpy()] = 0.0

        self._tied_refresh_teacher(cfg, metrics)

        ids_full = torch.zeros((n, L, k), dtype=torch.long)
        lp_full = torch.full((n, L, k), UNSCORED_LP, dtype=torch.float32)
        rows = np.nonzero(w > 0)[0]
        spliced_ok = np.zeros(n, dtype=bool)
        if len(rows):
            sub = batch.select_idxs(rows)
            sb = sub.batch
            pad_id = int(self.tokenizer.pad_token_id)
            ids1, am1, pos1, ok1 = apply_edit(sb["input_ids"], sb["attention_mask"], sb.get("position_ids", None), L,
                                              sb["oci_doc_off"], sb["oci_doc_len"], sb["oci_doc_repl"],
                                              sb["oci_doc_repl_len"], pad_id)
            sat_rows = side[rows] == SIDE_SAT
            ids2, am2, pos2, ok2 = apply_edit(ids1, am1, pos1, L, sb["tied_s_off"], sb["tied_s_len"],
                                              sb["tied_s_repl"], sb["tied_s_repl_len"], pad_id, rows=sat_rows)
            ok = ok1 & (~sat_rows | ok2)
            spliced_ok[rows] = ok
            # THE SUPPORT, PER SIDE (the user, 2026-10-01): stuck rows on the TEACHER's top-k -- the forward half
            # must reach the rare correct action, which the student's own top-k can leave in the residual --
            # saturated rows on the STUDENT's top-k, the tight support for their reverse KL. The student's ids
            # come from the actor as it is before this step's update; the teacher is then read at them.
            budget = int(getattr(self, "_post_rollout_token_budget", 0) or 0)

            def _teacher_batch(sel, extra=None):
                t = torch.as_tensor(sel)
                tensors = {"input_ids": ids2[t], "attention_mask": am2[t], "responses": sb["responses"][t]}
                if pos2 is not None:
                    tensors["position_ids"] = pos2[t]
                tensors.update(extra or {})
                tb = DataProto.from_dict(tensors=tensors)
                tb.meta_info = {"topk_k": k, DataProtoConfig.auto_padding_key: True}
                if budget > 0:
                    tb.meta_info.update({"use_dynamic_bsz": True, "max_token_len": budget})
                return tb

            stuck_sel = ~sat_rows
            if stuck_sel.any():
                out = self.self_teacher_wg.compute_ref_topk_log_prob(_teacher_batch(stuck_sel))
                ids_full[rows[stuck_sel]] = out.batch["teacher_topk_ids"].to(torch.long).cpu()
                lp_full[rows[stuck_sel]] = out.batch["teacher_topk_logprobs"].to(torch.float32).cpu()
            if sat_rows.any():
                t_sat = torch.as_tensor(sat_rows)
                plain = {"input_ids": sb["input_ids"][t_sat], "attention_mask": sb["attention_mask"][t_sat],
                         "responses": sb["responses"][t_sat]}
                if sb.get("position_ids", None) is not None:
                    plain["position_ids"] = sb["position_ids"][t_sat]
                pb = DataProto.from_dict(tensors=plain)
                pb.meta_info = {"topk_k": k, DataProtoConfig.auto_padding_key: True}
                sids = self.actor_rollout_wg.compute_actor_topk_ids(pb).batch["student_topk_ids"].to(torch.long).cpu()
                out = self.self_teacher_wg.compute_ref_logprob_at_ids(_teacher_batch(sat_rows, {"gather_ids": sids}))
                ids_full[rows[sat_rows]] = sids
                lp_full[rows[sat_rows]] = out.batch["teacher_lp_at_ids"].to(torch.float32).cpu()
            w[rows[~ok]] = 0.0

        batch.batch["tied_w"] = torch.as_tensor(w, dtype=torch.float32)
        batch.batch["tied_side"] = torch.as_tensor(side, dtype=torch.long)
        batch.batch["tied_tok"] = tok
        batch.batch["tied_topk_ids"] = ids_full
        batch.batch["tied_topk_lp"] = lp_full

        for t in ctl.tasks:
            for key, val in wts[t].items():
                metrics[f"tied/{t}/{key}"] = float(val)
            sel = real & (tasks == t)
            stuck_rows = sel & (side == SIDE_STUCK)
            sat_rows_all = sel & (side == SIDE_SAT)
            tied_rows = stuck_rows | sat_rows_all
            metrics[f"tied/{t}/rows_stuck"] = float(stuck_rows.sum())
            metrics[f"tied/{t}/rows_saturated"] = float(sat_rows_all.sum())
            metrics[f"tied/{t}/rows_distilled"] = float((sel & (w > 0)).sum())
            metrics[f"tied/{t}/tokens_distilled"] = float(tok[torch.as_tensor(sel & (w > 0))].sum())
            if tied_rows.any():
                metrics[f"tied/{t}/mismatch_share"] = float((match[tied_rows] == 0).mean())
                metrics[f"tied/{t}/no_document_share"] = float((match[tied_rows] < 0).mean())
                metrics[f"tied/{t}/splice_fail_share"] = float(
                    ((match[tied_rows] == 1) & ~(doc_ok[tied_rows] & ((side[tied_rows] != SIDE_SAT) | s_ok[tied_rows]))).mean())
            groups = [u for u, tt in task_of.items() if tt == t]
            for st in ("stuck", "saturated", "live"):
                metrics[f"tied/{t}/groups_{st}"] = float(sum(1 for u in groups if status.get(u) == st))
        metrics["tied/rows_scored"] = float(len(rows))

    # --- (a) ------------------------------------------------------------------ #

    def _compute_self_teacher_log_probs(self, batch: DataProto, cfg, metrics: dict):
        """``(log pi_theta(y_t | privileged + x, y_<t), valid_rows)`` on every row.

        Two conditionings, ``algorithm.opsd.source``:

        ``skill``     the task's skill documents in front of the observation --
                      the SDAR baseline's own teacher, built from its own pieces
                      (``build_teacher_batch``, ``SkillProvider``) rather than a
                      copy. Every row is valid.
        ``document``  THIS instance's correct document (alfworld's walkthrough,
                      webshop's goal record, search's answer) spliced in by
                      :func:`oci_rank.with_document`, which is the same edit the
                      rank scorer is measured on. A row whose document did not
                      fit its prompt window is marked invalid rather than scored
                      on the plain prompt, which would read as a gap of zero.

        The external teachers keep running (their coefficient is the run's
        choice); they write ``teacher_cache_ids`` under student-indexed top-k,
        never ``teacher_log_probs`` -- checked at the first call, because the
        single-token OPD estimator writes that same column and would silently
        replace this one.
        """
        from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
        from verl.trainer.ppo.rlsd_utils import SkillProvider

        source = str(cfg.get("source", "skill") or "skill")
        assert source in ("skill", "document"), (
            f"algorithm.opsd.source={source!r}; expected 'skill' or 'document'"
        )
        if source == "document":
            return self._self_teacher_on_document(batch, metrics)

        if getattr(self, "_opsd_skill_provider", None) is None:
            assert self.config.algorithm.opd.get("kl_loss_type", None) == "topk_kl", (
                "algorithm.opsd needs the external OPD path in top-k mode: the "
                "single-token OPD estimator writes batch['teacher_log_probs'] too and "
                "would overwrite the self-teacher's column"
            )
            skills_dirs = cfg.get("skills_dirs", None)
            self._opsd_skill_provider = SkillProvider(
                skills_dir=cfg.get("skills_dir", "skills/alfworld"),
                skill_all=bool(cfg.get("skill_all", False)),
                skills_dirs=dict(skills_dirs) if skills_dirs is not None else None,
            )
        max_prompt_length = self.config.data.max_prompt_length
        teacher_batch = build_teacher_batch(
            batch=batch,
            skill_provider=self._opsd_skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )
        out = self.actor_rollout_wg.compute_log_prob(teacher_batch)

        # How much the skill prefix added, and how often the teacher prompt hit
        # the cap. build_teacher_batch keeps the END of an over-long prompt, so a
        # capped row lost the start of its skill text -- the one thing that makes
        # the teacher a teacher.
        response_length = batch.batch["responses"].size(1)
        student_len = batch.batch["attention_mask"][:, :-response_length].sum(-1).float()
        teacher_len = teacher_batch.batch["attention_mask"][:, :-response_length].sum(-1).float()
        metrics["opsd/prompt_tokens_added/mean"] = float((teacher_len - student_len).mean())
        metrics["opsd/prompt_capped_ratio"] = float((teacher_len >= max_prompt_length).float().mean())
        lp = out.batch["old_log_probs"]
        return lp, torch.ones(lp.shape[0], dtype=torch.float32, device=lp.device)

    def _self_teacher_on_document(self, batch: DataProto, metrics: dict):
        """The same weights with this instance's correct document in the prompt.

        The rollout records the edit that turns a row's prompt into its
        document-conditioned one (``oci_doc_off`` and friends), and it does so
        only when ``algorithm.oci_rank.enable`` is on -- by itself that switch
        costs no GPU work in the training step, it is what makes the columns
        exist. A run that asks for this teacher without them would otherwise be
        scored on the plain prompt, i.e. against itself, which reads as a gate of
        exactly one half everywhere; it is refused instead.
        """
        from verl.trainer.ppo.oci_rank import document_rows, with_document

        rows = document_rows(batch)
        assert rows.any(), (
            "algorithm.opsd.source=document, but no row carries a document edit. The rollout "
            "records it only under algorithm.oci_rank.enable=True (and for the tasks in "
            "algorithm.oci_rank.tasks); without it the 'privileged' teacher would be the "
            "student's own prompt"
        )
        doc_batch, spliced = with_document(batch, self.tokenizer.pad_token_id)
        out = self.actor_rollout_wg.compute_log_prob(doc_batch)
        lp = out.batch["old_log_probs"]
        valid = torch.as_tensor(spliced, dtype=torch.float32, device=lp.device)

        response_length = batch.batch["responses"].size(1)
        student_len = batch.batch["attention_mask"][:, :-response_length].sum(-1).float()
        doc_len = doc_batch.batch["attention_mask"][:, :-response_length].sum(-1).float()
        metrics["opsd/doc_rows_share"] = float(valid.mean())
        metrics["opsd/prompt_tokens_added/mean"] = float((doc_len - student_len)[valid > 0].mean())
        return lp, valid

    def _progress_rank_controller(self, cfg):
        ctl = getattr(self, "_progress_rank", None)
        if ctl is None:
            from verl.trainer.ppo.progress_rank import ProgressRankController

            ctl = ProgressRankController(
                rho=float(cfg.get("rho", 0.0)),
                ema_alpha=float(cfg.get("ema_alpha", 0.2)),
                ema_floor=float(cfg.get("ema_floor", 0.01)),
                cap_kappa=float(cfg.get("cap_kappa", 0.5)),
                min_top_k=dict(cfg.get("min_top_k", {}) or {}),
                tasks=list(cfg.get("tasks", ["alfworld", "webshop", "search"])),
                # The mean k is weighted the way the GRPO statistic weights samples,
                # so the ranking sums to zero over what the advantage sums to zero over.
                cross_steps=bool(self.config.algorithm.get("compute_mean_std_cross_steps", True)),
                # Saturated groups: winners ranked by turn count (0 = off).
                sat_rho=float(cfg.get("sat_rho", 0.0)),
                sat_tasks=list(cfg.get("sat_tasks", ["alfworld", "webshop"])),
                sat_min_spread=dict(cfg.get("sat_min_spread", {}) or {}),
                sat_turn_scale=dict(cfg.get("sat_turn_scale", {}) or {}),
                # The gate: a task's winners are ranked only while its saturated groups
                # outnumber its stuck ones (EMA): its reliably solved games outnumber
                # its reliably failed ones. No new constant: the same EMA alpha.
                sat_gate=bool(cfg.get("sat_gate", False)),
                # What a saturated group's turn difference is divided by: the task's
                # turn cap (task_constant, the default), the group's own document
                # length (document) or the group's own mean turns (group_mean). The
                # order inside a group never changes; only the weight between groups.
                sat_turn_scale_mode=str(cfg.get("sat_turn_scale_mode", "task_constant") or "task_constant"),
                # The null control: same firing and mass, the winners' scores permuted.
                sat_placebo=str(cfg.get("sat_placebo", "none") or "none"),
                # Which mean the winners' turns are centred on: turn (weighted by rows, the default)
                # or trajectory (the plain mean: no uniform bonus).
                sat_centring=str(cfg.get("sat_centring", "turn") or "turn"),
                # How the two sides' strength is set: budget_cap (rho E budget, kappa S cap, the gate if
                # on) or beta_mirror (a per-task Beta from the tied-group shares; see progress_rank).
                scale_mode=str(cfg.get("scale_mode", "budget_cap") or "budget_cap"),
                beta_pseudo_count=float(cfg.get("beta_pseudo_count", 1.0)),
                beta_group_size=int(cfg.get("beta_group_size", 8)),
                # Keep groups on goals the environment cannot pay out of the tied-group shares.
                beta_exclude_capped=bool(cfg.get("beta_exclude_capped", False)),
                # The Beta's input shares: the gate's EMAs (ema) or discounted group counts.
                beta_share_estimator=str(cfg.get("beta_share_estimator", "ema") or "ema"),
                # What each side's scores are divided by, and the per-group outlier guard.
                beta_denominator=str(cfg.get("beta_denominator", "step") or "step"),
                beta_cap=str(cfg.get("beta_cap", "none") or "none"),
                # Mixed groups: the failures ranked among themselves (0 = off).
                mixed_rho=float(cfg.get("mixed_rho", 0.0) or 0.0),
                mixed_tasks=(list(cfg.get("mixed_tasks")) if cfg.get("mixed_tasks", None) else None),
            )
            pending = getattr(self, "_progress_rank_pending_state", None)
            if pending:
                ctl.load_state_dict(pending)
            self._progress_rank = ctl
        return ctl

    def _apply_progress_rank(self, batch: DataProto, cfg) -> dict:
        """Add (a)'s term to ``batch.batch["advantages"]`` in place; return its metrics.

        Also writes the step's group records (see progress_rank.py, GROUP RECORDS)
        unless ``record_groups`` is off: to ``<grad_probe.out_path>.groups/b<n>.jsonl``
        in a probe, else ``<record_dir or default_local_dir/progress_rank_groups>/
        step<N>.jsonl`` -- one file per step, so a step re-run after a resume
        overwrites its own file instead of appending a second copy.
        """
        from verl.trainer.ppo.progress_rank import COVERAGE_D_KEY, PROGRESS_K_KEY, PROGRESS_TOTAL_KEY

        # ALONE, ON PURPOSE. The OCI arms move a group's statistic or its rows;
        # (a) is to be measured against control with nothing else changed.
        for other in ("oci_sat", "oci_floor", "oci_slots", "oci_rank"):
            ocfg = self.config.algorithm.get(other, None)
            assert not (ocfg is not None and bool(ocfg.get("enable", False))), (
                f"algorithm.progress_rank.enable and algorithm.{other}.enable are both on; "
                "(a) is measured against control with nothing else changed"
            )
        assert self.config.algorithm.adv_estimator == "grpo", (
            "progress_rank adds to outcome-GRPO advantages (one value per row); "
            f"adv_estimator={self.config.algorithm.adv_estimator!r} is not that"
        )
        nt = batch.non_tensor_batch
        missing = [k for k in ("uid", "traj_uid", "episode_rewards", PROGRESS_K_KEY, PROGRESS_TOTAL_KEY)
                   if k not in nt]
        assert not missing, (
            f"algorithm.progress_rank.enable=True but the batch has no {missing}. The counts "
            "are written by the environment managers and recorded by the rollout loop, both of "
            "which read algorithm.progress_rank.enable off their own copy of the config."
        )
        n = len(batch)
        real = np.ones(n, dtype=bool)
        pad = batch.batch.get(PADDING_ROW_KEY, None)
        if pad is not None:
            real &= ~pad.reshape(-1).to(torch.bool).cpu().numpy()
        stat = real.copy()
        exc = batch.batch.get(GRPO_STAT_EXCLUDE_KEY, None)
        if exc is not None:
            stat &= ~exc.reshape(-1).to(torch.bool).cpu().numpy()
        # The mask the actor's loss reads (dp_actor: loss_mask in multi-turn mode,
        # else the response part of attention_mask).
        resp_len = batch.batch["responses"].shape[1]
        key = ("loss_mask" if (self.config.actor_rollout_ref.rollout.multi_turn.enable
                               and "loss_mask" in batch.batch.keys()) else "attention_mask")
        mask = batch.batch[key][:, -resp_len:]

        task_names = get_task_names(batch)
        if task_names is None:
            # A single-task run carries no task_name column; its one task is the env's.
            from verl.trainer.ppo.metric_utils import normalize_task_name

            only = normalize_task_name(self.config.env.get("env_name", None))
            assert only is not None, "progress_rank needs per-row task names or env.env_name"
            task_names = np.array([only] * n, dtype=object)

        # The score GRPO normalised, per row: the format split and ProGPO's gate are
        # judged on it, never on |A| (float32 rounding makes |A| non-zero in a group
        # whose rows all score alike).
        tlr = batch.batch.get("token_level_rewards", None)
        row_scores = None if tlr is None else tlr.sum(-1).double().cpu().numpy()

        # The OTHER progress counts, for comparison on the same stuck groups: ALFWorld's
        # milestones with and without "arrived" and the walkthrough as a set, whichever
        # of them (a) is not ranking by.
        kdef = str(cfg.get("alfworld_k", "milestone") or "milestone")
        alt_counts = {}
        for name, kcol, tcol, active in (("milestone", "progress_k_milestone", "progress_total_milestone", "milestone"),
                                         ("arrive", "progress_k_arrive", "progress_total_arrive", "milestone_arrive"),
                                         ("walkset", "progress_k_walkset", "progress_total_walkset", "walkthrough_set")):
            if kdef != active and kcol in nt and tcol in nt:
                alt_counts[name] = (nt[kcol], nt[tcol])
        # ...and Search's two counts: the evidence alone (K 1) and evidence then answer
        # (K 2), whichever of them (a) is not ranking by.
        sdef = str(cfg.get("search_k", "evidence") or "evidence")
        for name, kcol, tcol, active in (("search_evidence", "progress_k_search_evidence",
                                          "progress_total_search_evidence", "evidence"),
                                         ("search_answered", "progress_k_search_answered",
                                          "progress_total_search_answered", "evidence_answered")):
            if sdef != active and kcol in nt and tcol in nt:
                alt_counts[name] = (nt[kcol], nt[tcol])

        ctl = self._progress_rank_controller(cfg)
        new_adv, out = ctl.apply(
            advantages=batch.batch["advantages"], mask=mask,
            uids=nt["uid"], tuids=nt["traj_uid"], task_names=task_names,
            episode_rewards=nt["episode_rewards"],
            k_rows=nt[PROGRESS_K_KEY], total_rows=nt[PROGRESS_TOTAL_KEY],
            real_rows=real, stat_rows=stat,
            row_scores=row_scores,
            valid_rows=nt["is_action_valid"] if "is_action_valid" in nt else None,
            coverage_rows=nt[COVERAGE_D_KEY] if COVERAGE_D_KEY in nt else None,
            episode_lengths=nt["episode_lengths"] if "episode_lengths" in nt else None,
            turn_caps=self._turn_caps(ctl.tasks),
            alt_counts=alt_counts,
            task_score_rows=nt["task_score"] if "task_score" in nt else None,
            committed_rows=nt["committed"] if "committed" in nt else None,
            search_count_rows=nt["searches"] if "searches" in nt else None,
            capped_rows=nt["goal_capped"] if "goal_capped" in nt else None,
            goal_rows=nt["goal_id"] if "goal_id" in nt else None,
            goal_price_rows=nt["goal_price_upper"] if "goal_price_upper" in nt else None,
            env_seed_rows=nt["env_seed"] if "env_seed" in nt else None,
            goal_product_price_rows=nt["goal_product_price"] if "goal_product_price" in nt else None,
            revisit_rows=nt["revisits"] if "revisits" in nt else None,
            done_walkset_rows=nt["progress_done_walkset"] if "progress_done_walkset" in nt else None,
            doc_len_rows=self._document_lengths(nt, task_names),
            gamefile_rows=nt["gamefile"] if "gamefile" in nt else None,
        )
        batch.batch["advantages"] = new_adv
        # The format channel, counted where the responses are (the controller sees
        # trajectories, not tokens). A guard, not a term: nothing reads it back.
        from verl.trainer.ppo.progress_rank import think_block_metrics

        out.update(think_block_metrics(responses=batch.batch["responses"], mask=mask,
                                       task_names=task_names, real=real))
        out.update(self._write_progress_rank_groups(ctl.last_group_records, cfg))
        return out

    @staticmethod
    def _document_lengths(nt, task_names):
        """Per row, the length of its task's document for this game, or NaN.

        ALFWorld: the walkthrough as a set of lines (progress_total_walkset).
        WebShop: the goal record's steps, 3 + options (progress_total).
        Search: 2 (a search that returns the answer, then the answer).
        What sat_turn_scale_mode=document divides a saturated group's turn
        differences by; recorded in the group records either way.
        """
        names = np.asarray([str(t) for t in task_names])
        out = np.full(len(names), np.nan)
        if "progress_total_walkset" in nt:
            v = np.asarray(nt["progress_total_walkset"], dtype=float)
            out = np.where(names == "alfworld", v, out)
        if "progress_total" in nt:
            v = np.asarray(nt["progress_total"], dtype=float)
            out = np.where(names == "webshop", v, out)
        out = np.where(names == "search", 2.0, out)
        return out

    def _turn_caps(self, tasks) -> dict:
        """``{task: max turns}`` as the environment managers apply them; {} if unknown."""
        caps = getattr(self, "_turn_caps_cache", None)
        if caps is None:
            caps = {}
            try:
                if self.config.env.get("multitask", None):
                    from agent_system.environments.env_manager import _get_multitask_task_max_steps

                    caps = _get_multitask_task_max_steps(self.config, [str(t) for t in tasks])
                elif self.config.env.get("max_steps", None) is not None and len(tasks) == 1:
                    caps = {str(tasks[0]): int(self.config.env.max_steps)}
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                print(f"[progress_rank] turn caps unknown ({e!r}); traj/<task>/fail_at_cap not reported")
                caps = {}
            self._turn_caps_cache = caps
        return caps

    def _write_progress_rank_groups(self, records, cfg) -> dict:
        """One JSON line per group; returns a metric that is 1 when the write failed.

        A failed write is loud but not fatal: the records are a diagnostic, and a
        full disk must not cost the training step they describe.
        """
        import json
        import os

        if not bool(cfg.get("record_groups", True)):
            return {}
        probe = self.config.trainer.get("grad_probe", None)
        extra = {"step": int(getattr(self, "global_steps", 0))}
        if probe is not None and bool(probe.get("enable", False)):
            # The probe accumulates AFTER this hook, so its counter is one behind.
            n = int((getattr(self, "_grad_probe_state", None) or {}).get("batches", 0)) + 1
            path = os.path.join(f"{probe.get('out_path', 'grad_probe.json')}.groups", f"b{n}.jsonl")
            extra["batch"] = n
        else:
            root = cfg.get("record_dir", None) or os.path.join(
                str(self.config.trainer.default_local_dir), "progress_rank_groups")
            path = os.path.join(str(root), f"step{extra['step']}.jsonl")
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                for rec in records:
                    f.write(json.dumps({**extra, **rec}, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[progress_rank] WARNING: group records not written to {path}: {e!r}", flush=True)
            return {"progress_rank/record_write_failed": 1.0}
        return {"progress_rank/record_write_failed": 0.0}

    # --- progress_value_gae ------------------------------------------------------ #

    def _progress_value_table(self):
        """The value table, built on first use and restored from the checkpoint the run resumed from.

        The shared loop's _load_checkpoint reads global_step_N/progress_value_state.json into
        ``_progress_value_pending_state`` (opd_ray_trainer) and its _save_checkpoint writes this
        table there. A table saved under another configuration (features, buckets, n0, retention,
        gamma, the turn caps, the k definitions, the feature / reward schema) is refused by
        load_state_dict: its cells would be read as other states. _load_checkpoint already refused
        it before the first rollout, configured by the same builder (progress_value_config); the
        check here covers a pending state set any other way.
        """
        table = getattr(self, "_progress_value", None)
        if table is None:
            from verl.trainer.ppo.progress_value import ProgressValueTable

            table = ProgressValueTable(progress_value_config(self.config))
            pending = getattr(self, "_progress_value_pending_state", None)
            if pending:
                table.load_state_dict(pending)
                mass = {t: round(sn[1], 1) for t, sn in sorted(table.roots.items())}
                print(f"[progress_value] table restored: {table.updates} updates, "
                      f"{len(table.cells)} cells, mass {mass}", flush=True)
            self._progress_value = table
        return table

    def _progress_value_kwargs(self, batch: DataProto) -> dict:
        """compute_advantage's progress_value_* arguments, or {} when the estimator is another one."""
        if not check_progress_value_config(self.config):
            return {}
        table = self._progress_value_table()
        # The caps the environment managers run under (opd_ray_trainer.progress_value_turn_caps:
        # env.multitask.max_steps per task in a multitask run, env.max_steps in a single-task one),
        # as the table was built for them: every row's pv_cap is checked against exactly the
        # horizons the table's fingerprint names.
        caps = dict(table.cfg.horizons)
        task_names = get_task_names(batch)
        if task_names is None:
            # A single-task run carries no task_name column; its one task is the env's.
            assert len(caps) == 1, "progress_value needs per-row task names or a single-task env.env_name"
            task_names = np.array(list(caps) * len(batch), dtype=object)
        return {
            "progress_value_table": table,
            # Checked against every row's pv_cap: the "turns remaining" feature is keyed on it.
            "progress_value_turn_caps": caps,
            "progress_value_task_names": task_names,
            # compute_advantage leaves the full result here (metrics, per-row V and delta).
            "progress_value_out": {},
        }

    def _progress_value_commit(self, out: dict) -> dict:
        """Add the scored batch to the table and return the step's progress_value/* metrics.

        Right after the advantages, so the step was scored against the table as the PREVIOUS steps
        left it, and before the checkpoint that follows the actor update, which saves the table
        with this step in it -- a run resumed from step N scores step N + 1 against exactly what an
        uninterrupted run would have. The per-task metrics describe the table the batch was scored
        against (before this commit).

        EXCEPT UNDER trainer.grad_probe: a probe batch takes no optimizer step and the probe holds
        the policy still while it reads it, so the table -- a parameter of the advantage -- is held
        still too. The staged batch is dropped, and every probe batch is scored against the table
        the checkpoint restored, as training step N + 1 would be; committed, batch k would be
        scored against that table decayed k - 1 times plus the earlier probe batches.
        """
        table = self._progress_value_table()
        probe = self.config.trainer.get("grad_probe", None)
        if probe is not None and bool(probe.get("enable", False)):
            assert table.discard(), "progress_value: compute_advantage staged no update for this batch"
        else:
            assert table.commit(), "progress_value: compute_advantage staged no update for this batch"
        metrics = dict(out["result"].metrics)
        metrics["progress_value/table_updates"] = float(table.updates)
        # A WebShop goal flagged as one the environment cannot pay, WON. The flag is what V = 0 and
        # the exclusion from the table rest on, and it scores one purchase (the goal's own product
        # with the goal's options), not every purchase: a win shows it wrong for that goal. Should be
        # 0 on every step; loud when it is not.
        won = metrics.get("progress_value/webshop/capped_won", 0.0)
        if won > 0:
            print(f"[progress_value] WARNING: {int(won)} WebShop trajectories on goal_capped goals WON at step "
                  f"{int(getattr(self, 'global_steps', 0))}: goal_capped is not a proof the goal cannot pay "
                  "(they were scored at V = 0 and kept out of the table)", flush=True)
        return metrics

    # --- analysis records (algorithm.progress_value.records) --------------------------------- #

    def _rollout_records(self, batch: DataProto, metrics: dict, timing_raw: dict, pv_kwargs: dict) -> None:
        """This step's analysis records: one JSON line per group, and the records/* metrics.

        OBSERVATION ONLY (verl/trainer/ppo/rollout_records.py): the batch is read and nothing is
        written to it, no random number is drawn, and the training value table is never staged or
        committed -- in the progress_value_gae arm the result compute_advantage computed is read. The
        one state kept is the records-only shadow table's (_rollout_records_shadow_value), an object
        of its own.

        A FAILURE COSTS THE STEP'S RECORDS, NOT THE STEP. Nothing downstream reads them, so taking a
        run down for one would trade the run for a diagnostic (the pure-OPD reward hook's reasoning):
        the traceback is printed and records/failed is 1 (0 on every good step). Off, a no-op that
        is not even timed.
        """
        from verl.trainer.ppo.rollout_records import RecordsConfig

        rcfg = RecordsConfig.from_config(self.config)
        if not rcfg.enable:
            return
        with _timer("records", timing_raw):
            try:
                out = self._rollout_records_step(batch, rcfg, pv_kwargs)
            except Exception as e:  # a record is never worth the run
                import traceback

                traceback.print_exc()
                print(f"[records] WARNING: step {int(getattr(self, 'global_steps', 0))}: the analysis records "
                      f"failed ({e!r}); none are written for this step, and training is unaffected", flush=True)
                out = {"records/failed": 1.0}
        metrics.update(out)

    def _rollout_records_step(self, batch: DataProto, rcfg, pv_kwargs: dict) -> dict:
        """_rollout_records' work: the value columns' source, the shadows' arguments as this trainer
        uses them, the computation, the file."""
        from verl.trainer.ppo import rollout_records
        from verl.trainer.ppo.task_loss_weights import pg_loss_norm_kwargs

        step = int(getattr(self, "global_steps", 0))
        alg, actor = self.config.algorithm, self.config.actor_rollout_ref.actor
        out: dict = {}
        task_names = get_task_names(batch)
        if task_names is None:
            # A single-task run carries no task_name column; its one task is the env's.
            from verl.trainer.ppo.metric_utils import normalize_task_name

            only = normalize_task_name(self.config.env.get("env_name", None))
            assert only is not None, "the records need per-row task names or env.env_name"
            task_names = np.array([only] * len(batch), dtype=object)

        if pv_kwargs:
            # The value arm: the result the step was scored with and the table committed from --
            # read here, never staged or committed a second time.
            value, source = pv_kwargs["progress_value_out"]["result"], "training"
        elif rcfg.shadow_value:
            value, shadow_metrics = self._rollout_records_shadow_value(batch, rcfg, task_names)
            source = "shadow"
            out.update(shadow_metrics)
        else:
            value, source = None, None

        # GiGPO's step returns take the invalid-action penalty as this run's scores took it: per row,
        # at the trainer's own coefficient (None when the run applies no penalty -- then its scores
        # carry none either).
        coefs = None
        if actor.get("use_invalid_action_penalty", True):
            names_col = batch.non_tensor_batch.get("task_name", None)
            coef, by_task = actor.invalid_action_penalty_coef, actor.get("invalid_action_penalty_coef_by_task", None)
            coefs = np.array([_get_invalid_action_penalty_coef(
                SimpleNamespace(non_tensor_batch={} if names_col is None else {"task_name": names_col[i]}),
                coef, by_task) for i in range(len(batch))], dtype=np.float64)
        # compute_advantage's GRPO arguments, as _reward_and_advantage hands them over.
        floor_cfg = alg.get("oci_floor", None)
        grpo_kwargs = {
            "norm_adv_by_std_in_grpo": alg.get("norm_adv_by_std_in_grpo", True),
            "compute_mean_std_cross_steps": alg.get("compute_mean_std_cross_steps", True),
            "exclude_mask": batch.batch.get(GRPO_STAT_EXCLUDE_KEY, None),
            "floor_mask": batch.batch.get(OCI_FLOOR_KEY, None),
            "floor_value": float((floor_cfg or {}).get("value", 0.0) if floor_cfg is not None else 0.0),
        }
        pr_cfg = alg.get("progress_rank", None)
        res = rollout_records.compute_step_records(
            batch, step=step, cfg=rcfg, task_names=task_names,
            multi_turn=bool(self.config.actor_rollout_ref.rollout.multi_turn.enable),
            pg_loss_norm=pg_loss_norm_kwargs(actor)["pg_loss_norm"],
            value=value, value_source=source, grpo_kwargs=grpo_kwargs, penalty_coefs=coefs,
            turn_caps=progress_value_turn_caps(self.config),
            # progress_rank reports traj/* and the think-block share itself when it is on: one writer
            # per key.
            traj_metrics=not (pr_cfg is not None and bool(pr_cfg.get("enable", False))))
        out.update(res.metrics)
        # Why a shadow was not emitted, said once per reason rather than every step.
        warned = getattr(self, "_rollout_records_warned", None)
        if warned is None:
            warned = self._rollout_records_warned = set()
        for note in res.notes:
            if note not in warned:
                warned.add(note)
                print(f"[records] {note}", flush=True)

        # The file: one per step, so a step re-run after a resume replaces its own. Under
        # trainer.grad_probe (whose step counter stands still) one per probe batch beside the probe's
        # output, as progress_rank's group records do.
        extra = {}
        probe = self.config.trainer.get("grad_probe", None)
        if probe is not None and bool(probe.get("enable", False)):
            n = int((getattr(self, "_grad_probe_state", None) or {}).get("batches", 0)) + 1
            path = os.path.join(f"{probe.get('out_path', 'grad_probe.json')}.pv_groups", f"b{n}.jsonl")
            extra["batch"] = n
        else:
            path = os.path.join(self._rollout_records_dir(rcfg), f"step{step}.jsonl")
        try:
            rollout_records.write_jsonl_atomic(path, res.groups, extra)
            out["records/write_failed"] = 0.0
        except OSError as e:
            print(f"[records] WARNING: group records not written to {path}: {e!r}", flush=True)
            out["records/write_failed"] = 1.0
        out["records/failed"] = 0.0
        return out

    def _rollout_records_dir(self, rcfg) -> str:
        """records.dir, or <trainer.default_local_dir>/progress_value_groups."""
        return rcfg.dir or os.path.join(str(self.config.trainer.default_local_dir), "progress_value_groups")

    def _rollout_records_shadow_value(self, batch: DataProto, rcfg, task_names):
        """``(result, metrics)``: the value estimator on this batch, from the records' OWN table.

        Records-only arms (records.shadow_value beside an estimator other than progress_value_gae), so
        the new estimator's advantage can be read on the control's rollouts. Everything the training
        path does, on another object: the table is built by the one builder (progress_value_config),
        the rows' pv_cap checked against its horizons, the batch scored against the table as the
        previous steps left it and committed right after -- dropped instead on a grad_probe batch, as
        _progress_value_commit drops it. Its metrics are the estimator's, under records/shadow_value/.

        NOT self._progress_value, and never in the checkpoint: the shared loop saves that attribute as
        the arm's value table, and a progress_value_gae arm started from this run's checkpoint would
        then resume on a table it did not build, without saying so (a warm start is
        allow_missing_table's decision). The state goes to <records dir>/shadow_value_state/
        step<N>.json after each commit, and a run resumed at global_step_N restores step<N>.json.
        """
        from verl.trainer.ppo.progress_value import compute_progress_value_advantage

        table = self._rollout_records_value_table(rcfg)
        columns = dict(batch.non_tensor_batch)
        pad = batch.batch.get(PADDING_ROW_KEY, None)
        if pad is not None:
            columns[PADDING_ROW_KEY] = pad.reshape(-1).to(torch.bool).cpu().numpy()
        columns["task_name"] = np.asarray(task_names, dtype=object)
        _check_progress_value_caps(columns, table.cfg, dict(table.cfg.horizons))
        res = compute_progress_value_advantage(columns, table)
        table.stage(res.records)
        metrics = {"records/shadow_value/" + k[len("progress_value/"):]: v for k, v in res.metrics.items()}
        probe = self.config.trainer.get("grad_probe", None)
        if probe is not None and bool(probe.get("enable", False)):
            table.discard()
        else:
            table.commit()
            path = os.path.join(self._rollout_records_dir(rcfg), "shadow_value_state",
                                f"step{int(getattr(self, 'global_steps', 0))}.json")
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                _write_json_atomic(path, table.state_dict())
                metrics["records/shadow_value/save_failed"] = 0.0
            except OSError as e:
                print(f"[records] WARNING: shadow value table not saved to {path}: {e!r}", flush=True)
                metrics["records/shadow_value/save_failed"] = 1.0
        metrics["records/shadow_value/table_updates"] = float(table.updates)
        return res, metrics

    def _rollout_records_value_table(self, rcfg):
        """The records-only shadow table, built on first use; after a resume, restored from the records'
        directory (global_steps is N + 1 on the first step after global_step_N, whose table is
        shadow_value_state/step<N>.json). Missing or of another configuration, it starts empty, loudly:
        a records-only shadow is not worth refusing a resume over, and its metrics show the restart
        (records/shadow_value/table_updates, fallback_share)."""
        table = getattr(self, "_rollout_records_value", None)
        if table is None:
            from verl.trainer.ppo.progress_value import ProgressValueTable

            table = ProgressValueTable(progress_value_config(self.config))
            step = int(getattr(self, "global_steps", 0))
            if step > 1:
                path = os.path.join(self._rollout_records_dir(rcfg), "shadow_value_state", f"step{step - 1}.json")
                if not os.path.exists(path):
                    print(f"[records] WARNING: no shadow value table for step {step - 1} at {path}; the "
                          f"records-only shadow starts empty at step {step}", flush=True)
                else:
                    try:
                        with open(path) as f:
                            table.load_state_dict(json.load(f))
                        print(f"[records] shadow value table restored from {path}: {table.updates} updates",
                              flush=True)
                    except (OSError, ValueError) as e:
                        table = ProgressValueTable(progress_value_config(self.config))
                        print(f"[records] WARNING: shadow value table {path} not restored ({e!r}); the "
                              f"records-only shadow starts empty at step {step}", flush=True)
            self._rollout_records_value = table
        return table

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
