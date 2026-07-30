"""
Stage 1 entry point for off-policy multitask distillation (offline KD).

Generates a *fixed* dataset of teacher trajectories for ONE task: a frozen
per-task teacher checkpoint is loaded as the ``actor_rollout`` model, rolls out
multi-turn trajectories in that task's environment, scores its own top-k over
the generated responses, and dumps everything to ``<out_dir>/<task>.pt``.

Run once per task (see ``examples/opd_trainer/run_multitask_offpolicy_qwen3.sh``):

    python3 -m verl.trainer.main_opd_offpolicy_gen \
        +gen.task=alfworld +gen.teacher_path=/path/to/alfworld_teacher \
        +gen.out_dir=$HOME/data/.../teacher_traj +gen.num_trajectories=2400 \
        actor_rollout_ref.model.path=Qwen/Qwen3-1.7B ... (same rollout/env knobs as OPD)
"""

import os

import hydra
import ray
from omegaconf import OmegaConf, open_dict


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_gen(config)


def _filter_parquet_to_task(src_path: str, task: str, out_dir: str) -> str:
    """Write a per-task copy of the prompt parquet so the single-task env sees
    only its own samples."""
    import pandas as pd

    df = pd.read_parquet(src_path)
    if "task_name" not in df.columns:
        raise ValueError(f"{src_path} has no 'task_name' column; cannot filter to task={task}")
    df_task = df[df["task_name"] == task].reset_index(drop=True)
    assert len(df_task) > 0, f"no rows for task={task} in {src_path}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"_gen_{task}_{os.path.basename(src_path)}")
    df_task.to_parquet(out_path, index=False)
    return out_path


def run_gen(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = OPDGenTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class OPDGenTaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)

        gen_cfg = config.get("gen", None)
        assert gen_cfg is not None and gen_cfg.get("task", None) and gen_cfg.get("teacher_path", None), (
            "Stage 1 requires +gen.task=<task> +gen.teacher_path=<ckpt> +gen.out_dir=<dir>"
        )
        task = gen_cfg.task
        out_dir = gen_cfg.out_dir

        # Fail-fast intent check: validate the composed config BEFORE the
        # single-task restriction below (which rewrites tasks / batch sizes /
        # model.path per invocation and filters the parquet on disk). The
        # expectations file pins only task-agnostic knobs, which the
        # restriction does not touch.
        #
        # Optional, not required. It was briefly mandatory, for a Stage 1 the SFT
        # arm no longer has -- that arm now reads the KD arm's pool directly, so
        # the only caller left is the KD generation script, which has never
        # shipped an expectations file and whose pool is already on disk. Writing
        # one now would pin knobs reconstructed after the fact, which is a worse
        # guarantee than none. The warning keeps the omission visible.
        expect_file = config.trainer.get("expected_config", None)
        if expect_file is None:
            print(
                "[OPD-offpolicy gen] WARNING: no +trainer.expected_config given; "
                "this run's knobs are not checked against a version-controlled "
                "intent file. The generated pool is only as trustworthy as the "
                "command line that produced it."
            )
        else:
            from verl.utils.expected_config import enforce_expected_config

            enforce_expected_config(config, expect_file, tag="offpolicy-gen expected-config")

        # ---- Restrict everything to this single task; teacher drives the rollout. ----
        per_task_batch_size = int(config.data.task_balance.per_task_batch_size)
        with open_dict(config):
            config.actor_rollout_ref.model.path = gen_cfg.teacher_path
            config.env.env_name = "multitask"
            config.env.multitask.tasks = [task]
            config.env.multitask.val_per_task_batch_size = per_task_batch_size
            config.data.task_balance.enable = True
            config.data.task_balance.tasks = [task]
            # Single-task batch sizes must match the env's per-task batch.
            config.data.train_batch_size = per_task_batch_size
            config.data.val_batch_size = per_task_batch_size
            # Per-task prompt parquet (env sees only this task's samples).
            task_train = _filter_parquet_to_task(config.data.train_files, task, out_dir)
            config.data.train_files = task_train
            config.data.val_files = task_train

        pprint(OmegaConf.to_container(config, resolve=True))

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
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id}
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        assert reward_manager_name == "episode"
        from agent_system.reward_manager import EpisodeRewardManager

        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "keep actor_rollout_ref.rollout.n=1; GRPO group size is env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.opd_offpolicy_ray_trainer import TeacherTrajectoryGenerator

        generator = TeacherTrajectoryGenerator(
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
        generator.init_workers()
        generator.generate()


if __name__ == "__main__":
    main()
