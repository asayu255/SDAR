# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Main entry point for Skill-GRPO training.

GRPO training with skill information injected into the model's rollout prompt.
Unlike SDAR where skills are only seen by the teacher, here the student directly
sees skills during training. Evaluation is done both with and without skills.
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
    run_skill_grpo(config)


# [EXPLAIN] `run_skill_grpo` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_skill_grpo(config) -> None:
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
    runner = SkillGRPOTaskRunner.remote()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.get(runner.run.remote(config))


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(num_cpus=1)
# [EXPLAIN] `SkillGRPOTaskRunner` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SkillGRPOTaskRunner:
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

        # Load skill provider
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.rlsd_utils import SkillProvider

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skill_grpo_cfg = config.algorithm.get("skill_grpo", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skills_dir = skill_grpo_cfg.get("skills_dir", "skills/alfworld")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skill_all = skill_grpo_cfg.get("skill_all", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[Skill-GRPO] Loaded skills from {skills_dir}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[Skill-GRPO] Available skills: {list(skill_provider.skill_contents.keys())}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[Skill-GRPO] Task-to-skill mapping: {skill_provider.task_to_skill}")

        # Create skill-aware trajectory collector
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_collector = SkillTrajectoryCollector(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            skill_provider=skill_provider,
        )

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

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        trainer = SkillGRPORayTrainer(
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
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        trainer.init_workers()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        trainer.fit()


# [EXPLAIN] `SkillTrajectoryCollector` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SkillTrajectoryCollector:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    TrajectoryCollector wrapper that injects skill information into observation text.
    Delegates all actual work to the underlying TrajectoryCollector instance.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config, tokenizer, processor=None, skill_provider=None):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.multi_turn_rollout import TrajectoryCollector

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.skill_provider = skill_provider
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.inject_skill = True
        # Expose config for external access
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.processor = processor

    # [EXPLAIN] `_inject_skills_into_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _inject_skills_into_obs(self, obs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Prepend skill text to each observation's text field."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.inject_skill or self.skill_provider is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return obs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_texts = obs.get("text", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if obs_texts is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return obs
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(obs_texts)):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if obs_texts[i] is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                skill_text = self.skill_provider.get_privileged_info_from_prompt(obs_texts[i])
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if skill_text:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    obs_texts[i] = f"[Privileged Skill Information]\n{skill_text}\n\n{obs_texts[i]}"
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs

    # [EXPLAIN] `preprocess_single_sample` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def preprocess_single_sample(self, item, gen_batch, obs):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._collector.preprocess_single_sample(item, gen_batch, obs)

    # [EXPLAIN] `preprocess_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def preprocess_batch(self, gen_batch, obs, active_mask=None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs = self._inject_skills_into_obs(obs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._collector.preprocess_batch(gen_batch, obs, active_mask=active_mask)

    # [EXPLAIN] `vanilla_multi_turn_loop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._collector.vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)

    # [EXPLAIN] `dynamic_multi_turn_loop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def dynamic_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._collector.dynamic_multi_turn_loop(gen_batch, actor_rollout_wg, envs)

    # [EXPLAIN] `gather_rollout_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def gather_rollout_data(self, *args, **kwargs):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._collector.gather_rollout_data(*args, **kwargs)

    # [EXPLAIN] `multi_turn_loop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Override multi_turn_loop to inject skills into observations.
        We monkey-patch the inner collector's preprocess_batch to inject skills.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_train:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.inject_skill = True
        # For validation, inject_skill is controlled externally by the trainer

        # Temporarily replace the inner collector's preprocess_batch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        original_preprocess_batch = self._collector.preprocess_batch

        # [EXPLAIN] `skill_preprocess_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def skill_preprocess_batch(gen_batch, obs, active_mask=None):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs = self._inject_skills_into_obs(obs)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return original_preprocess_batch(gen_batch, obs, active_mask=active_mask)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._collector.preprocess_batch = skill_preprocess_batch
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result = self._collector.multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train)
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._collector.preprocess_batch = original_preprocess_batch

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return result


# [EXPLAIN] `SkillGRPORayTrainer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SkillGRPORayTrainer:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Wrapper around RayPPOTrainer that performs dual validation
    (with skill and without skill) during evaluation.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, **kwargs):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._trainer = RayPPOTrainer(**kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._traj_collector = kwargs["traj_collector"]

    # [EXPLAIN] `init_workers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_workers(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._trainer.init_workers()

    # [EXPLAIN] rollout 生成から teacher forward、actor 更新、検証、checkpoint までの学習 phase を順序付ける trainer loop である。
    def fit(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Override fit to intercept _validate calls for dual evaluation.
        We monkey-patch _validate on the trainer instance.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        original_validate = self._trainer._validate
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_collector = self._traj_collector

        # [EXPLAIN] `dual_validate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def dual_validate():
            # First validation: with skill
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            traj_collector.inject_skill = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metrics_with_skill = original_validate()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefixed_with = {f"val_with_skill/{k.replace('val/', '')}": v for k, v in metrics_with_skill.items()}

            # Second validation: without skill
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            traj_collector.inject_skill = False
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metrics_no_skill = original_validate()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefixed_no = {f"val_no_skill/{k.replace('val/', '')}": v for k, v in metrics_no_skill.items()}

            # Combine both
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            combined = {}
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            combined.update(prefixed_with)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            combined.update(prefixed_no)
            # Also keep original val/ keys from with_skill run for backward compatibility
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            combined.update(metrics_with_skill)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return combined

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._trainer._validate = dual_validate
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._trainer.fit()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
