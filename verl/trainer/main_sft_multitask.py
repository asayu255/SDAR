"""
Stage 2 entry point for multitask SFT (behaviour cloning on teacher trajectories).

The student is trained on the fixed teacher-trajectory dataset produced by
``main_opd_offpolicy_gen`` -- specifically, on the *same pool the off-policy KD
arm uses*, so the two arms differ in the loss and in nothing else. Batches are
3-task-balanced, and the loss is a cross-entropy / NLL on the teacher tokens.
GRPO policy-gradient, entropy, reference-KL, teacher-KL and reward are all
disabled so the only training signal is the SFT cross-entropy. The pool's top-k
teacher columns are KD's target and are dropped at load here (see
``MultiTaskSFTTrainer._drop_tensor_keys``). Only validation rolls the student
out (identical to OPD / off-policy distillation).
"""

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_sft_multitask(config)


def run_sft_multitask(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = SFTMultiTaskTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class SFTMultiTaskTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        sft_cfg = config.algorithm.get("sft", {})
        data_dir = sft_cfg.get("data_dir", None)
        assert data_dir is not None, (
            "multitask SFT requires algorithm.sft.data_dir "
            "(directory of Stage-1 <task>.pt files; pass via +algorithm.sft.data_dir=/path)"
        )

        # Inject the pure-SFT invariants: cross-entropy on teacher tokens is the
        # only training signal.
        with open_dict(config):
            config.actor_rollout_ref.actor.use_sft_loss = True
            config.actor_rollout_ref.actor.sft_loss_coef = sft_cfg.get("loss_coef", 1.0)
            config.actor_rollout_ref.actor.use_teacher_kl_loss = False
            config.actor_rollout_ref.actor.pg_loss_coef = 0          # no GRPO policy gradient
            config.actor_rollout_ref.actor.entropy_coeff = 0         # no entropy bonus
            config.actor_rollout_ref.actor.use_kl_loss = False       # no reference-KL term
            config.actor_rollout_ref.actor.use_sdl_loss = False
            config.actor_rollout_ref.actor.use_sdar_loss = False
            config.algorithm.use_kl_in_reward = False                # reward never shapes the loss

        # Fail-fast intent check: validate the EFFECTIVE config (after the
        # injection above) against the version-controlled expectations file.
        from verl.utils.expected_config import enforce_expected_config

        expect_file = config.trainer.get("expected_config", None)
        assert expect_file is not None, (
            "multitask SFT requires +trainer.expected_config=<expectations yaml> "
            "(see examples/sft_trainer/expected_multitask_sft_config.yaml)"
        )
        enforce_expected_config(config, expect_file, tag="sft expected-config")

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        # Envs are used only for validation (student rollout), identical to OPD.
        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

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

        # NOTE: multitask SFT registers neither Role.RefPolicy nor teacher workers;
        # the target tokens are precomputed in the Stage-1 dataset.

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
            "keep actor_rollout_ref.rollout.n=1; GRPO group size is env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        # train_dataset is unused by the SFT loop (it iterates the fixed teacher
        # dataset) but is built to satisfy the base dataloader / validation plumbing;
        # val_dataset drives _validate exactly like OPD.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        print(f"[SFT-multitask] data_dir: {data_dir}")
        print(f"[SFT-multitask] sft_loss_coef: {config.actor_rollout_ref.actor.sft_loss_coef}")

        from verl.trainer.ppo.sft_multitask_ray_trainer import MultiTaskSFTTrainer

        trainer = MultiTaskSFTTrainer(
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
