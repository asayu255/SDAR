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
differs in *where the fixed dataset is read from* (``algorithm.sft.data_dir``).
The training horizon remains the configured ``trainer.total_training_steps``;
the Stage-1 pool size is chosen so that fixed step count corresponds to the
intended number of SFT epochs.
"""

from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer


class MultiTaskSFTTrainer(OffPolicyOPDRayTrainer):
    """Off-policy multitask SFT on a fixed teacher-trajectory dataset."""

    def _resolve_data_dir(self):
        data_dir = self.config.algorithm.get("sft", {}).get("data_dir", None)
        assert data_dir is not None, (
            "multitask SFT requires algorithm.sft.data_dir "
            "(directory of Stage-1 <task>.pt teacher-trajectory files; "
            "pass via +algorithm.sft.data_dir=/path)"
        )
        return data_dir
