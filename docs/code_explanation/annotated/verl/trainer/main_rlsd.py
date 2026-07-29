# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Main entry point for RLSD (Reinforcement Learning with Self-Distillation) training.
Based on main_ppo.py with RLSD-specific extensions.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import hydra
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
# [EXPLAIN] `main` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main(config):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    run_rlsd(config)


# [EXPLAIN] `run_rlsd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_rlsd(config) -> None:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not ray.is_initialized():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        default_runtime_env = get_ppo_ray_runtime_env()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ray_init_kwargs = config.get("ray_init", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"ray init kwargs: {ray_init_kwargs}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    runner = RLSDTaskRunner.remote()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.get(runner.run.remote(config))


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(num_cpus=1)
# [EXPLAIN] `RLSDTaskRunner` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RLSDTaskRunner:
    # [EXPLAIN] `run` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def run(self, config):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from pprint import pprint

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.fs import copy_to_local

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        pprint(OmegaConf.to_container(config, resolve=True))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        OmegaConf.resolve(config)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments import make_envs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        envs, val_envs = make_envs(config)

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils import hf_processor, hf_tokenizer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        trust_remote_code = config.data.get("trust_remote_code", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.utils.vllm_utils import is_version_ge

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.single_controller.ray import RayWorkerGroup
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ray_worker_group_cls = RayWorkerGroup

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_rollout_cls = ActorRolloutRefWorker
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_pool_id = "global_pool"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.reward_model.enable:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from verl.workers.fsdp_workers import RewardModelWorker
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif config.reward_model.strategy == "megatron":
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from verl.workers.megatron_workers import RewardModelWorker
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise NotImplementedError
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mapping[Role.RewardModel] = global_pool_id

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mapping[Role.RefPolicy] = global_pool_id

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if reward_manager_name == "episode":
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from agent_system.reward_manager import EpisodeRewardManager

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward_manager_cls = EpisodeRewardManager
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.multi_turn_rollout import TrajectoryCollector

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.dataset.rl_dataset import collate_fn
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Load skill provider for RLSD
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.rlsd_utils import SkillProvider

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skills_dir = config.algorithm.rlsd.get("skills_dir", "skills/alfworld")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skill_all = config.algorithm.rlsd.get("skill_all", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[RLSD] Loaded skills from {skills_dir}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[RLSD] Available skills: {list(skill_provider.skill_contents.keys())}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[RLSD] Task-to-skill mapping: {skill_provider.task_to_skill}")

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        trainer = RLSDRayTrainer(
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
            skill_provider=skill_provider,
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        trainer.init_workers()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        trainer.fit()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
