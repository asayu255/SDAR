"""
Main entry point for multitask OPD + GRPO training.

This is the OPD (On-Policy Distillation) multitask entrypoint with GRPO added
back on top. Each sample is still distilled from a separate, single-task
RL-trained teacher selected by its ``task_name`` (per-task teacher KL on the
student's own on-policy responses), but the GRPO policy-gradient is no longer
disabled: the student is trained jointly by

    policy_loss = pg_loss * pg_loss_coef + teacher_kl_loss * teacher_kl_coef

so the env-reward GRPO signal and the teacher distillation both shape the loss.
Everything else -- data, env, batch sizes, the teacher-KL support, the
cross-teacher sign weighting -- matches the pure-OPD multitask run, and does so
by sharing its code rather than by resembling it: the composition is
``main_opd.build_and_fit`` and the distillation settings are
``main_opd.inject_distillation_config``, both called unchanged. What this module
adds is the one difference the arm is named for.
"""

import hydra
import ray
from omegaconf import open_dict

from verl.trainer.main_opd import (
    build_and_fit,
    init_ray_for_opd,
    inject_distillation_config,
)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_opd_grpo(config)


def inject_opd_grpo_config(config) -> None:
    """Turn ``algorithm.opd.*`` into the actor settings the trainer actually reads.

    The distillation half is :func:`inject_distillation_config`, shared verbatim
    with pure OPD. What is left is this arm's whole definition, and it is
    entirely a matter of what is NOT done here: unlike ``inject_opd_config`` we
    do not zero ``pg_loss_coef`` or ``entropy_coeff``, and we do not force
    ``algorithm.use_kl_in_reward`` off, so the GRPO advantages can reach the loss.

    ``pg_loss_coef`` is taken directly from ``actor_rollout_ref.actor.pg_loss_coef``
    (the run script sets it there). It is deliberately NOT re-injected from
    ``algorithm.opd.pg_loss_coef``, a key the scripts never set: doing so silently
    overrode the script's value with the default.

    A module-level function, like its pure-OPD counterpart, because the intent
    lock validates the config AFTER it: a test that wants to know whether an arm
    will start has to apply this exact function rather than a copy that can drift.

    Modifies ``config`` in place.
    """
    inject_distillation_config(config)
    with open_dict(config):
        # Asserted rather than defaulted: on this arm pg_loss_coef is the ratio
        # between the two terms, i.e. the number the whole experiment is about.
        # Falling back to a default here would produce a run that trains fine and
        # answers a different question than the one asked.
        assert config.actor_rollout_ref.actor.get("pg_loss_coef", None) is not None, (
            "OPD+GRPO requires actor_rollout_ref.actor.pg_loss_coef -- it is the "
            "weight of the policy gradient against the teacher KL"
        )


def run_opd_grpo(config) -> None:
    init_ray_for_opd(config)
    runner = OPDGRPOTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class OPDGRPOTaskRunner:
    def run(self, config):
        from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer

        build_and_fit(
            config,
            inject_fn=inject_opd_grpo_config,
            trainer_cls=OPDGRPORayTrainer,
            tag="opd-grpo",
            label="OPD+GRPO",
            example_dir="opd_grpo_trainer",
        )


if __name__ == "__main__":
    main()
