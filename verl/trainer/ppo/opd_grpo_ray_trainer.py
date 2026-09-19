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
    OCI_FLOOR_KEY,
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
                batch.batch["teacher_log_probs"] = self._compute_self_teacher_log_probs(batch, opsd_cfg, metrics)

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
            )

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

        return batch, reward_extra_infos_dict

    # --- (a) ------------------------------------------------------------------ #

    def _compute_self_teacher_log_probs(self, batch: DataProto, cfg, metrics: dict) -> torch.Tensor:
        """log pi_theta(y_t | skills + x, y_<t) on every row, for the SDAR loss.

        Reuses the SDAR trainers' own pieces (``build_teacher_batch``,
        ``SkillProvider``) rather than a copy, so the teacher here is the one the
        SDAR baseline trained against. The external teachers keep running (their
        coefficient is the run's choice); they write ``teacher_cache_ids`` under
        student-indexed top-k, never ``teacher_log_probs`` -- checked at the first
        call, because the single-token OPD estimator writes that same column and
        would silently replace this one.
        """
        from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
        from verl.trainer.ppo.rlsd_utils import SkillProvider

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
        return out.batch["old_log_probs"]

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
        )
        batch.batch["advantages"] = new_adv
        out.update(self._write_progress_rank_groups(ctl.last_group_records, cfg))
        return out

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
