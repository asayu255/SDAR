"""
Multitask SFT (behaviour cloning on teacher trajectories) — Stage 2 trainer.

The hard-target sibling of off-policy distillation: the student is trained on the
same *fixed teacher-generated trajectory dataset* (Stage 1), in 3-task-balanced
batches, but with a plain cross-entropy / NLL loss on the teacher tokens instead
of the top-k teacher-KL. Everything else — the fixed-data loop, task balancing,
checkpointing and validation — is inherited unchanged from
``OffPolicyOPDRayTrainer``.

The SFT loss itself lives in ``DataParallelPPOActor.update_policy`` behind
``actor.use_sft_loss`` (``main_sft_multitask`` injects it). This trainer differs
from its parent in exactly two places: *where the fixed dataset is read from*
(``algorithm.sft.data_dir``) and *which columns of it are kept* (the teacher's
top-k distribution is KD's target, not SFT's -- see ``_drop_tensor_keys``).
The training horizon remains the configured ``trainer.total_training_steps``;
the Stage-1 pool size is chosen so that fixed step count corresponds to the
intended number of SFT epochs.
"""

from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer


class MultiTaskSFTTrainer(OffPolicyOPDRayTrainer):
    """Off-policy multitask SFT on a fixed teacher-trajectory dataset."""

    # The SFT arm shares the KD arm's Stage-1 pool -- same trajectories, same
    # teacher, so the two arms differ only in the loss and not in the data. That
    # pool is written with collect_topk=True, but the SFT loss is a plain NLL on
    # the teacher's sampled tokens (``responses``): update_policy's SFT branch
    # selects only responses/input_ids/attention_mask/position_ids, and
    # main_sft_multitask pins use_teacher_kl_loss=False, so the top-k columns are
    # never read here. They are also by far the largest thing in a row -- k
    # logprobs plus k ids per response token, ~120 KiB of the ~268 KiB -- so
    # carrying them for the whole run would cost roughly 100 GiB of host RAM for
    # nothing. ``_resolve_data_dir`` asserts the loss they belong to really is off.
    _drop_tensor_keys = OffPolicyOPDRayTrainer._drop_tensor_keys + (
        "teacher_topk_logprobs",
        "teacher_topk_ids",
    )

    def _resolve_data_dir(self):
        # Runs inside __init__ *before* the pool is loaded, so these fail in
        # seconds rather than after a multi-minute load of a pool whose top-k
        # columns are about to be thrown away.
        actor_cfg = self.config.actor_rollout_ref.actor
        assert actor_cfg.get("use_sft_loss", False), (
            "MultiTaskSFTTrainer requires actor.use_sft_loss=True "
            "(main_sft_multitask injects it)"
        )
        assert not actor_cfg.get("use_teacher_kl_loss", False), (
            "multitask SFT drops the teacher top-k columns at load, so "
            "actor.use_teacher_kl_loss must stay False; use the off-policy KD "
            "entry point (main_opd_offpolicy) for the distillation arm"
        )

        data_dir = self.config.algorithm.get("sft", {}).get("data_dir", None)
        assert data_dir is not None, (
            "multitask SFT requires algorithm.sft.data_dir "
            "(directory of Stage-1 <task>.pt teacher-trajectory files; "
            "pass via +algorithm.sft.data_dir=/path)"
        )
        return data_dir
