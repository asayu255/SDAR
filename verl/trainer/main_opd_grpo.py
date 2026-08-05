"""
Main entry point for multitask OPD + GRPO training.

This is the OPD (On-Policy Distillation) multitask entrypoint with GRPO added
back on top. Each sample is still distilled from a separate, single-task
RL-trained teacher selected by its ``task_name`` (per-task teacher KL on the
student's own on-policy responses), but the GRPO policy-gradient is no longer
disabled: the student is trained jointly by

    policy_loss = pg_loss * pg_loss_coef + teacher_kl_loss * teacher_kl_coef

so the env-reward GRPO signal and the teacher distillation both shape the loss.
Everything else (data, env, batch sizes) matches the pure-OPD multitask run.
"""

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_opd_grpo(config)


def run_opd_grpo(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = OPDGRPOTaskRunner.remote()
    ray.get(runner.run.remote(config))


def inject_opd_grpo_config(config):
    """Turn ``algorithm.opd.*`` into the actor settings the trainer actually reads.

    Separated from ``run`` so the EFFECTIVE config — the thing the expectations
    file is checked against — can be produced without standing up Ray. A test that
    composes the run script's arguments and applies this is the only way to catch
    a lock/script disagreement before a run does, and the lock's whole purpose is
    to fail in seconds instead of hours.

    Modifies ``config`` in place and returns it.
    """
    from omegaconf import open_dict

    opd_cfg = config.algorithm.get("opd", {})
    teacher_paths = opd_cfg.get("teacher_paths", None)
    assert teacher_paths is not None, (
        "OPD+GRPO requires algorithm.opd.teacher_paths.{alfworld,search,webshop} "
        "(pass via +algorithm.opd.teacher_paths.<task>=/path)"
    )

    # Enable the per-task teacher KL distillation AND keep the GRPO policy
    # gradient active (pg_loss_coef != 0). Unlike pure OPD (main_opd.py), we
    # do NOT zero pg_loss_coef and we do NOT force use_kl_in_reward; those are
    # left to config so GRPO advantages can flow into the loss.
    with open_dict(config):
        config.actor_rollout_ref.actor.use_teacher_kl_loss = True
        config.actor_rollout_ref.actor.teacher_kl_loss_coef = opd_cfg.get("kl_loss_coef", 1.0)
        config.actor_rollout_ref.actor.teacher_kl_loss_type = opd_cfg.get("kl_loss_type", "low_var_kl")
        # top-k (+tail) dense KL support size; only used when kl_loss_type=topk_kl.
        config.actor_rollout_ref.actor.teacher_kl_topk = opd_cfg.get("topk", 20)
        # Equal per-task share of the loss, instead of the token-count share the
        # plain token-mean gives. On this arm it weights the policy gradient and
        # the teacher KL alike, so pg_loss_coef still means the ratio between
        # them. Scientific knob, so it is surfaced under algorithm.opd like the
        # other loss settings and pinned in the expectations file rather than
        # left to the actor's default.
        config.actor_rollout_ref.actor.normalize_loss_by_task = opd_cfg.get(
            "normalize_loss_by_task", False
        )
        # pg_loss_coef is taken directly from actor_rollout_ref.actor.pg_loss_coef
        # (the run script sets it there). It used to be re-injected from
        # algorithm.opd.pg_loss_coef — which the script never sets — silently
        # overriding the script's value with 1.0; that injection is removed.
        # No reference-KL term and no SDL/SDAR distillation paths.
        config.actor_rollout_ref.actor.use_kl_loss = False
        config.actor_rollout_ref.actor.use_sdl_loss = False
        config.actor_rollout_ref.actor.use_sdar_loss = False
    return config


@ray.remote(num_cpus=1)
class OPDGRPOTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        inject_opd_grpo_config(config)

        # Fail-fast intent check: validate the EFFECTIVE config (after the
        # injection above) against the version-controlled expectations file.
        # Required — a run without a pinned intent is exactly how the
        # low_var_kl-instead-of-topk_kl mishap happened.
        from verl.utils.expected_config import enforce_expected_config

        expect_file = config.trainer.get("expected_config", None)
        assert expect_file is not None, (
            "OPD+GRPO requires +trainer.expected_config=<expectations yaml> "
            "(see examples/opd_grpo_trainer/expected_multitask_config.yaml)"
        )
        enforce_expected_config(config, expect_file, tag="opd-grpo expected-config")

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # NOTE: like pure OPD, no Role.RefPolicy. Teachers are created inside
        # OPDGRPORayTrainer.init_workers as additional role="ref" worker groups.

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Re-read from the config rather than carrying a local across:
        # inject_opd_grpo_config owns reading algorithm.opd, and it has already run
        # (and asserted this key is present), so this is the same value it saw.
        teacher_paths = config.algorithm.opd.teacher_paths
        teacher_paths_plain = (
            OmegaConf.to_container(teacher_paths, resolve=True)
            if OmegaConf.is_config(teacher_paths)
            else dict(teacher_paths)
        )
        print(f"[OPD+GRPO] teacher_paths: {teacher_paths_plain}")
        print(f"[OPD+GRPO] teacher_kl_loss_coef: {config.actor_rollout_ref.actor.teacher_kl_loss_coef}")
        print(f"[OPD+GRPO] teacher_kl_loss_type: {config.actor_rollout_ref.actor.teacher_kl_loss_type}")
        print(f"[OPD+GRPO] pg_loss_coef: {config.actor_rollout_ref.actor.pg_loss_coef}")
        print(f"[OPD+GRPO] adv_estimator: {config.algorithm.adv_estimator}")

        from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer

        trainer = OPDGRPORayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
