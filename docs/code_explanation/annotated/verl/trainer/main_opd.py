# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Main entry point for multitask OPD (On-Policy Distillation) training.

Each sample is distilled from a separate, single-task RL-trained teacher selected
by its ``task_name``. The student is trained purely by the KL to its per-task
teacher on the student's own on-policy responses; GRPO policy-gradient, entropy,
reference-KL and reward signals are all disabled so no other signal enters the loss.
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
    run_opd(config)


# [EXPLAIN] `run_opd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_opd(config) -> None:
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
    runner = OPDTaskRunner.remote()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.get(runner.run.remote(config))


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(num_cpus=1)
# [EXPLAIN] `OPDTaskRunner` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OPDTaskRunner:
    # [EXPLAIN] `run` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def run(self, config):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from pprint import pprint

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf, open_dict

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.fs import copy_to_local

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        pprint(OmegaConf.to_container(config, resolve=True))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        OmegaConf.resolve(config)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        opd_cfg = config.algorithm.get("opd", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_paths = opd_cfg.get("teacher_paths", None)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert teacher_paths is not None, (
            "OPD requires algorithm.opd.teacher_paths.{alfworld,search,webshop} "
            "(pass via +algorithm.opd.teacher_paths.<task>=/path)"
        )

        # Inject the pure-distillation invariants so no signal other than the
        # per-task teacher KL can flow into the actor loss.
        # [EXPLAIN] ここは raw Hydra config を worker が実際に読む effective config へ変換する境界である。
        # [EXPLAIN] `pg_loss_coef=0` と `entropy_coeff=0` により GRPO/entropy の scalar 寄与をゼロ化し、
        # [EXPLAIN] reference-KL・SDL・SDAR・reward-KL も無効化して、teacher KL だけを gradient 経路に残す。
        # [EXPLAIN] `adv_estimator=grpo` が raw config に残っていても、thin loop が advantage を作らず、
        # [EXPLAIN] actor も `pg_loss_coef==0` なら `old_log_probs`/`advantages` を要求しない点が重要である。
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open_dict(config):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.actor_rollout_ref.actor.use_teacher_kl_loss = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.actor_rollout_ref.actor.teacher_kl_loss_coef = opd_cfg.get("kl_loss_coef", 1.0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.actor_rollout_ref.actor.teacher_kl_loss_type = opd_cfg.get("kl_loss_type", "low_var_kl")
            # top-k (+tail) dense KL support size; only used when kl_loss_type=topk_kl.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.actor_rollout_ref.actor.teacher_kl_topk = opd_cfg.get("topk", 20)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.actor_rollout_ref.actor.pg_loss_coef = 0          # no GRPO policy gradient
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.actor_rollout_ref.actor.entropy_coeff = 0         # no entropy bonus
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.actor_rollout_ref.actor.use_kl_loss = False       # no reference-KL term
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.actor_rollout_ref.actor.use_sdl_loss = False
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.actor_rollout_ref.actor.use_sdar_loss = False
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.algorithm.use_kl_in_reward = False                # reward never shapes the loss

        # Fail-fast intent check: validate the EFFECTIVE config (after the
        # injection above) against the version-controlled expectations file.
        # Required — a run without a pinned intent is exactly how the
        # low_var_kl-instead-of-topk_kl mishap happened.
        # [EXPLAIN] 検証順は Hydra compose → この entry-point injection → expected-config 比較である。
        # [EXPLAIN] したがって expectations file は raw default ではなく、実験開始直前の実効値を固定する。
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.expected_config import enforce_expected_config

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expect_file = config.trainer.get("expected_config", None)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert expect_file is not None, (
            "OPD requires +trainer.expected_config=<expectations yaml> "
            "(see examples/opd_trainer/expected_multitask_config.yaml)"
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        enforce_expected_config(config, expect_file, tag="opd expected-config")

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

        # NOTE: OPD does NOT register Role.RefPolicy. Teachers are created inside
        # OPDRayTrainer.init_workers as additional role="ref" worker groups, one per task.

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

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_paths_plain = (
            OmegaConf.to_container(teacher_paths, resolve=True)
            if OmegaConf.is_config(teacher_paths)
            else dict(teacher_paths)
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[OPD] teacher_paths: {teacher_paths_plain}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[OPD] teacher_kl_loss_coef: {config.actor_rollout_ref.actor.teacher_kl_loss_coef}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[OPD] teacher_kl_loss_type: {config.actor_rollout_ref.actor.teacher_kl_loss_type}")

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
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
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        trainer.init_workers()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        trainer.fit()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
