"""
Multitask SFT (behaviour cloning on teacher trajectories) — Stage 2 trainer.

The hard-target sibling of off-policy distillation: the student is trained on the
same *fixed teacher-generated trajectory dataset* (Stage 1), in 3-task-balanced
batches, but with a plain cross-entropy / NLL loss on the teacher tokens instead
of the top-k teacher-KL. Everything else — the fixed-data loop, task balancing,
checkpointing and validation — is inherited unchanged from
``OffPolicyOPDRayTrainer``.

The SFT loss itself lives in ``DataParallelPPOActor.update_policy`` behind
``actor.use_sft_loss`` (``main_sft_multitask`` injects it). This trainer only
differs in *where the fixed dataset is read from* (``algorithm.sft.data_dir``)
and in *how many epochs* the fixed dataset is reused (``algorithm.sft.num_epochs``).

Unlike off-policy distillation (which reuses the fixed pool across a fixed number
of training steps), standard SFT reuses the dataset for a small number of epochs
(default 5, matching the SFT baseline in SakanaAI/TAID). The total number of
training steps is therefore derived from the pool size and epoch count rather
than taken from ``trainer.total_training_steps``.
"""

import math

from omegaconf import open_dict

from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer


class MultiTaskSFTTrainer(OffPolicyOPDRayTrainer):
    """Off-policy multitask SFT on a fixed teacher-trajectory dataset, reused for
    a small number of epochs (TAID-style SFT baseline)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reuse the fixed teacher-trajectory pool for ``num_epochs`` epochs, so
        # each trajectory is seen ~num_epochs times (standard SFT), instead of the
        # off-policy distillation regime (fixed step count, ~15x reuse).
        num_epochs = int(self.config.algorithm.get("sft", {}).get("num_epochs", 5))
        # One epoch == one pass over the largest task's trajectory pool; smaller
        # task pools wrap (task-balanced draws keep every step balanced).
        max_pool = max(len(trajs) for trajs in self._task_to_trajs.values())
        steps_per_epoch = self._steps_per_epoch(max_pool, self.per_task_traj_per_step)
        self.total_training_steps = num_epochs * steps_per_epoch
        # Keep the LR scheduler horizon consistent with the actual step count.
        with open_dict(self.config):
            if self.config.actor_rollout_ref.actor.get("optim", None) is not None:
                self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
        print(
            f"[SFT] num_epochs={num_epochs}, per_task_traj_per_step={self.per_task_traj_per_step}, "
            f"max_pool={max_pool}, steps_per_epoch={steps_per_epoch}, total_steps={self.total_training_steps}"
        )

    @staticmethod
    def _steps_per_epoch(max_pool: int, per_task_traj_per_step: int) -> int:
        """Number of steps to cover the largest task's pool once (>= 1)."""
        return max(1, math.ceil(max_pool / per_task_traj_per_step))

    def _resolve_data_dir(self):
        data_dir = self.config.algorithm.get("sft", {}).get("data_dir", None)
        assert data_dir is not None, (
            "multitask SFT requires algorithm.sft.data_dir "
            "(directory of Stage-1 <task>.pt teacher-trajectory files; "
            "pass via +algorithm.sft.data_dir=/path)"
        )
        return data_dir
