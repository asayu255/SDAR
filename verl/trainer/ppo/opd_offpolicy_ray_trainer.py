"""
Off-policy multitask distillation (offline KD) — Stage 1 generator + Stage 2 trainer.

This is the *off-policy* sibling of the on-policy OPD trainer
(``verl/trainer/ppo/opd_ray_trainer.py``). The standard form of off-policy
distillation trains the student on a *fixed dataset of sequences sampled from the
teacher* (sequence-level / GKD off-policy branch), rather than on the student's
own on-policy rollouts.

Two stages, sharing all data / env / config / teacher / loss machinery with OPD:

* **Stage 1 — ``TeacherTrajectoryGenerator``**: for a single task, a frozen
  per-task teacher (loaded as the ``actor_rollout`` model) rolls out multi-turn
  trajectories in that task's environment, then scores its own top-k
  (``teacher_topk_logprobs`` / ``teacher_topk_ids``) over the generated
  responses. The result is dumped to ``<out_dir>/<task>.pt`` as a ``DataProto``.

* **Stage 2 — ``OffPolicyOPDRayTrainer``**: loads the per-task ``.pt`` files,
  concatenates them into one fixed off-policy dataset, and trains the student in
  3-task-balanced batches with the *same* top-k teacher-KL used by OPD
  (``teacher_kl_loss_type=topk_kl``, ``teacher_kl_topk=20``). No generation and
  no teacher forward pass happen during training — the teacher top-k is already
  baked into the dataset. Only validation rolls the student out (identical to
  OPD via the inherited ``_validate``).
"""

import glob
import os
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import DataProtoConfig
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.opd_ray_trainer import compute_opd_data_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics

from agent_system.multi_turn_rollout import adjust_batch

# Tensor fields persisted per trajectory (Stage 1 -> Stage 2). These are exactly
# the keys update_policy's top-k distillation path consumes, plus prompts/
# response_mask for monitoring and balance_batch.
_SAVE_TENSOR_KEYS = [
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "prompts",
    "response_mask",
    "teacher_topk_logprobs",
    "teacher_topk_ids",
]
_SAVE_NON_TENSOR_KEYS = ["task_name", "traj_uid"]

# Keys popped from the prompt batch to form the generation input (mirrors OPD).
_GEN_BATCH_KEYS = ["input_ids", "attention_mask", "position_ids"]
_GEN_NON_TENSOR_KEYS = ["raw_prompt_ids", "data_source"]


def _build_gen_batch(batch: DataProto) -> DataProto:
    """Pop the model-input fields out of a prompt batch (same set as OPD/_validate)."""
    non_tensor_keys = list(_GEN_NON_TENSOR_KEYS)
    for opt in ("multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs", "task_name"):
        if opt in batch.non_tensor_batch:
            non_tensor_keys.append(opt)
    return batch.pop(batch_keys=list(_GEN_BATCH_KEYS), non_tensor_batch_keys=non_tensor_keys)


class TeacherTrajectoryGenerator(RayPPOTrainer):
    """Stage 1: roll out one task's frozen teacher and dump trajectories + top-k.

    The teacher checkpoint is loaded as the ``actor_rollout`` model (set
    ``config.actor_rollout_ref.model.path`` to the teacher path before
    construction), so the existing rollout engine generates the teacher's
    trajectories and ``compute_actor_topk_log_prob`` scores its top-k.
    """

    def generate(self):
        gen_cfg = self.config.gen
        task = gen_cfg.task
        out_dir = gen_cfg.out_dir
        topk = int(gen_cfg.get("topk", self.config.actor_rollout_ref.actor.get("teacher_kl_topk", 20)))
        # Optional cap on collected trajectories; default: one pass over the loader.
        target = gen_cfg.get("num_trajectories", None)
        target = int(target) if target else None

        self.global_steps = 0
        collected = []
        n_collected = 0
        os.makedirs(out_dir, exist_ok=True)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = _build_gen_batch(batch)
                del batch

                # Teacher generates the trajectories (is_train=True -> repeat by env.rollout.n).
                gen_out = self.traj_collector.multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.envs,
                    is_train=True,
                )
                gen_out = adjust_batch(self.config, gen_out)
                gen_out.batch["response_mask"] = compute_response_mask(gen_out)

                # Teacher scores its own top-k over the generated responses.
                topk_in = gen_out.select(
                    batch_keys=["responses", "input_ids", "attention_mask", "position_ids"]
                )
                topk_in.meta_info = dict(topk_in.meta_info)
                topk_in.meta_info["topk_k"] = topk
                # Per-call batch is not generally divisible by the DP world size.
                topk_in.meta_info[DataProtoConfig.auto_padding_key] = True
                topk_out = self.actor_rollout_wg.compute_actor_topk_log_prob(topk_in)
                gen_out.batch["teacher_topk_logprobs"] = topk_out.batch["teacher_topk_logprobs"]
                gen_out.batch["teacher_topk_ids"] = topk_out.batch["teacher_topk_ids"]

                # Backfill task_name if the rollout dropped it (single-task generation).
                if "task_name" not in gen_out.non_tensor_batch:
                    gen_out.non_tensor_batch["task_name"] = np.array(
                        [task] * len(gen_out), dtype=object
                    )

                tensor_keys = [k for k in _SAVE_TENSOR_KEYS if k in gen_out.batch]
                non_tensor_keys = [k for k in _SAVE_NON_TENSOR_KEYS if k in gen_out.non_tensor_batch]
                keep = gen_out.select(batch_keys=tensor_keys, non_tensor_batch_keys=non_tensor_keys)
                keep = keep.to("cpu")
                collected.append(keep)
                n_collected += len(keep)
                self.global_steps += 1
                print(f"[OPD-offpolicy gen][{task}] collected {n_collected} trajectories")

                if target is not None and n_collected >= target:
                    break
            else:
                continue
            break

        data = DataProto.concat(collected)
        out_path = os.path.join(out_dir, f"{task}.pt")
        data.save_to_disk(out_path)
        print(f"[OPD-offpolicy gen][{task}] saved {len(data)} trajectories -> {out_path}")
        return out_path


class OffPolicyOPDRayTrainer(RayPPOTrainer):
    """Stage 2: off-policy distillation on the fixed teacher-trajectory dataset.

    Reuses the base worker setup (student ``actor_rollout`` only; no reference
    policy, no teacher workers) and the inherited ``_validate`` /
    ``_save_checkpoint``. Only the training loop differs from OPD: it iterates a
    fixed off-policy dataset in 3-task-balanced batches and applies the *same*
    top-k teacher-KL via ``update_actor`` (no rollout, no teacher forward).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.use_reference_policy, "off-policy OPD must not create a reference-policy worker"
        opd_cfg = self.config.algorithm.get("opd", {})
        self.teacher_data_dir = opd_cfg.get("teacher_data_dir", None)
        assert self.teacher_data_dir is not None, (
            "off-policy OPD requires algorithm.opd.teacher_data_dir "
            "(directory of Stage-1 <task>.pt files)"
        )
        # Per-step, per-task trajectory count == OPD's per-task prompts * group size,
        # so the update batch matches OPD's update dynamics.
        per_task_prompts = int(self.config.data.task_balance.per_task_batch_size)
        group_size = int(self.config.env.rollout.n)
        self.per_task_traj_per_step = per_task_prompts * group_size
        self._load_offpolicy_data()

    def _load_offpolicy_data(self):
        files = sorted(glob.glob(os.path.join(self.teacher_data_dir, "*.pt")))
        assert files, f"no Stage-1 <task>.pt files found in {self.teacher_data_dir}"
        parts = [DataProto.load_from_disk(f) for f in files]
        self.offpolicy_data = DataProto.concat(parts)
        task_names = self.offpolicy_data.non_tensor_batch["task_name"]
        normalized = np.array([self._normalize_task_name(t) for t in task_names])
        self._task_to_indices = {
            task: np.where(normalized == task)[0] for task in sorted(set(normalized.tolist()))
        }
        sizes = {t: len(idx) for t, idx in self._task_to_indices.items()}
        print(f"[OPD-offpolicy] loaded {len(self.offpolicy_data)} trajectories from {len(files)} files; per-task sizes: {sizes}")
        for task, idx in self._task_to_indices.items():
            assert len(idx) > 0, f"no trajectories for task {task}"

    def _offpolicy_batch_iter(self):
        """Yield one task-balanced DataProto per training step (with replacement
        across epochs: pools are reshuffled and recycled when exhausted)."""
        seed = int(self.config.data.get("seed", 1))
        rng = np.random.default_rng(seed)
        pools = {t: rng.permutation(idx) for t, idx in self._task_to_indices.items()}
        cursors = {t: 0 for t in pools}

        def _draw(task, n):
            out = []
            while len(out) < n:
                pool = pools[task]
                c = cursors[task]
                take = min(n - len(out), len(pool) - c)
                out.extend(pool[c : c + take].tolist())
                cursors[task] = c + take
                if cursors[task] >= len(pool):
                    pools[task] = rng.permutation(self._task_to_indices[task])
                    cursors[task] = 0
            return out

        for _ in range(self.total_training_steps):
            sel = []
            for task in pools:
                sel.extend(_draw(task, self.per_task_traj_per_step))
            yield self.offpolicy_data.select_idxs(sel)

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPD-offpolicy Training")
        self.global_steps += 1
        last_val_metrics = None

        for batch in self._offpolicy_batch_iter():
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with _timer("step", timing_raw):
                # Pad the per-step batch to a DP/micro-divisible size (same as OPD's
                # post-rollout adjust_batch), then recompute response_mask / token counts.
                batch = adjust_batch(self.config, batch)
                batch.batch["response_mask"] = compute_response_mask(batch)
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                with _timer("update_actor", timing_raw):
                    # update_policy scales student logits by this temperature (same value
                    # compute_log_prob would set); OPD sets it here for the thin loop too.
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                test_start_step = self.config.trainer.get("test_start_step", 0)
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            metrics.update({
                "training/global_step": self.global_steps,
            })
            metrics.update(compute_opd_data_metrics(batch=batch))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1
            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
