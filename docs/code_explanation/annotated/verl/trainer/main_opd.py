"""
Main entry point for multitask OPD (On-Policy Distillation) training.

Each sample is distilled from a separate, single-task RL-trained teacher selected
by its ``task_name``. The student is trained purely by the KL to its per-task
teacher on the student's own on-policy responses; GRPO policy-gradient, entropy,
reference-KL and reward signals are all disabled so no other signal enters the loss.
"""

import hydra
import ray
from omegaconf import OmegaConf


# `@hydra.main`が`ppo_trainer.yaml`へCLI overrideをcomposeし、未解決のDictConfigをPure OPD専用`run_opd`へ渡すentrypointである。
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_opd(config)


# 既存Ray runtimeがなければ、VERL既定runtime_envと`config.ray_init.runtime_env`をmergeしてdriverからclusterへ接続する。
# `ray.init`はactorを作る処理ではなく、後続のremote task/worker groupが使うruntimeを初期化する境界である。
def run_opd(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # `OPDTaskRunner.remote()`で1 CPUのRay actorを作り、`runner.run.remote(config)`を非同期投入する。
    # 外側の`ray.get`がdriverの明示的blocking pointであり、worker初期化・training・validation・checkpointが終了するまで`run_opd`は戻らない。
    runner = OPDTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class OPDTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        # `OmegaConf.resolve`後の値を基準にteacher pathの存在を確認するため、Hydra interpolationやCLI overrideを反映した実効設定を検査する。
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        opd_cfg = config.algorithm.get("opd", {})
        teacher_paths = opd_cfg.get("teacher_paths", None)
        assert teacher_paths is not None, (
            "OPD requires algorithm.opd.teacher_paths.{alfworld,search,webshop} "
            "(pass via +algorithm.opd.teacher_paths.<task>=/path)"
        )

        # Inject the pure-distillation invariants so no signal other than the
        # per-task teacher KL can flow into the actor loss.
        # `open_dict(config)`はOmegaConfのstruct制約をこのblockだけ一時解除し、compose済み設定へPure OPD固有keyと実効値を書き込む。
        # resource、autocast、no-gradの寿命を管理するcontext managerではない。
        with open_dict(config):
            config.actor_rollout_ref.actor.use_teacher_kl_loss = True
            config.actor_rollout_ref.actor.teacher_kl_loss_coef = opd_cfg.get("kl_loss_coef", 1.0)
            config.actor_rollout_ref.actor.teacher_kl_loss_type = opd_cfg.get("kl_loss_type", "low_var_kl")
            # top-k (+tail) dense KL support size; only used when kl_loss_type=topk_kl.
            config.actor_rollout_ref.actor.teacher_kl_topk = opd_cfg.get("topk", 20)
            config.actor_rollout_ref.actor.pg_loss_coef = 0          # no GRPO policy gradient
            config.actor_rollout_ref.actor.entropy_coeff = 0         # no entropy bonus
            config.actor_rollout_ref.actor.use_kl_loss = False       # no reference-KL term
            config.actor_rollout_ref.actor.use_sdl_loss = False
            config.actor_rollout_ref.actor.use_sdar_loss = False
            config.algorithm.use_kl_in_reward = False                # reward never shapes the loss

        # Fail-fast intent check: validate the EFFECTIVE config (after the
        # injection above) against the version-controlled expectations file.
        # Required — a run without a pinned intent is exactly how the
        # low_var_kl-instead-of-topk_kl mishap happened.
        # 検証順はHydra compose → Pure OPD invariant注入 → expected-config比較である。
        # `enforce_expected_config`はraw defaultではなくworkerが読む直前のeffective configをdot pathごとに照合する。
        from verl.utils.expected_config import enforce_expected_config

        expect_file = config.trainer.get("expected_config", None)
        assert expect_file is not None, (
            "OPD requires +trainer.expected_config=<expectations yaml> "
            "(see examples/opd_trainer/expected_multitask_config.yaml)"
        )
        enforce_expected_config(config, expect_file, tag="opd expected-config")

        # `actor_rollout_ref.model.path`だけをlocalizeしてstudent tokenizer/processorを構築する。
        # task別teacher pathはここでstudent pathへ混ぜず、後の`OPDRayTrainer.init_workers`で各ref-role copyへ設定する。
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

        # actor strategyに応じてFSDPまたはMegatron worker group実装を選ぶ。現在runはFSDPで、同じ`ActorRolloutRefWorker` classをactor_rollout roleとtask別ref roleに使い分ける。
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

        # global resource poolへActorRolloutとCriticのclassを登録するが、`Role.RefPolicy`は登録しない。
        # task別teacherはshared ref policyではなく、OPDRayTrainerが同じpool上へ追加する独立worker groupである。
        # reward managerはmonitoring用scoreを作るが、Pure OPD thin loopのscalar lossには接続されない。
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

        # NOTE: OPD does NOT register Role.RefPolicy. Teachers are created inside
        # OPDRayTrainer.init_workers as additional role="ref" worker groups, one per task.

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # engine側`actor_rollout_ref.rollout.n`を1に固定し、GRPO groupの8 trajectoryは環境側`env.rollout.n`で複製する。
        # この分離によりmulti-turn collectorがtrajectory UIDとturn-rowを一貫して管理する。
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

        teacher_paths_plain = (
            OmegaConf.to_container(teacher_paths, resolve=True)
            if OmegaConf.is_config(teacher_paths)
            else dict(teacher_paths)
        )
        print(f"[OPD] teacher_paths: {teacher_paths_plain}")
        print(f"[OPD] teacher_kl_loss_coef: {config.actor_rollout_ref.actor.teacher_kl_loss_coef}")
        print(f"[OPD] teacher_kl_loss_type: {config.actor_rollout_ref.actor.teacher_kl_loss_type}")

        from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

        # `OPDRayTrainer`はcompose済みconfigからteacher pathを読み、`init_workers`でactorの前に3 teacherを初期化する。
        # `fit`は通常PPO loopではなく、rollout → monitoring reward → teacher forward → actor updateのthin loopを実行する。
        trainer = OPDRayTrainer(
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
