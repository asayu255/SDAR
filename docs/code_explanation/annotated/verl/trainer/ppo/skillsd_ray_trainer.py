# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
SkillSD (Skill-based Self-Distillation) Trainer.

Extends RLSDRayTrainer. Unlike RLSD which replaces advantages with token-level
teacher-weighted advantages, SkillSD keeps the original GRPO advantages and
adds an auxiliary SDL loss computed inside the actor's update_policy().
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pprint import pprint

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.core_algos import agg_loss
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.rlsd_utils import SkillProvider
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer, build_teacher_batch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.metric import reduce_metrics
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import masked_mean
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_data_metrics_by_task,
    compute_throughout_metrics,
    compute_timing_metrics,
    normalize_task_name,
    task_row_indices,
)

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.multi_turn_rollout import adjust_batch, compute_log_prob_with_prefetch

# Overlap envs.reset() for the next rollout with this step's GPU training phases.
# The reset is pure CPU / subprocess / HTTP work and the env managers are idle
# between rollouts; the reset still runs exactly once per rollout and in the same
# order, so stateful env schedules (alfworld's game-file iterator) are unchanged.
# Opt-in; see TrajectoryCollector.prefetch_env_reset.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ENV_RESET_PREFETCH = os.environ.get("ENV_RESET_PREFETCH", "0").strip().lower() in ("1", "true", "yes", "on")


# [EXPLAIN] `SkillSDRayTrainer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SkillSDRayTrainer(RLSDRayTrainer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    SkillSD trainer that extends RLSDRayTrainer.

    Keeps the original GRPO advantages unchanged and passes teacher_log_probs
    to the actor, where the SDL loss is computed inside update_policy().
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(*args, skill_provider=skill_provider, **kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skillsd_cfg = self.config.algorithm.get("skillsd", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sdl_lambda = skillsd_cfg.get("sdl_lambda", 0.1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sdl_warmdown_steps = skillsd_cfg.get("warmdown_steps", -1)

    # [EXPLAIN] `_get_sdl_lambda` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_sdl_lambda(self, step: int) -> float:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.sdl_warmdown_steps <= 0:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.sdl_lambda
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if step >= self.sdl_warmdown_steps:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 0.0
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.sdl_lambda * (1.0 - step / self.sdl_warmdown_steps)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `_normalize_task_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _normalize_task_name(task_name):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return normalize_task_name(task_name)

    # [EXPLAIN] `_task_kl_loss_coef_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _task_kl_loss_coef_tensor(self, batch: DataProto):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        coef_by_task = self.config.actor_rollout_ref.actor.get("kl_loss_coef_by_task", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not coef_by_task:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if OmegaConf.is_config(coef_by_task):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            coef_by_task = OmegaConf.to_container(coef_by_task, resolve=True)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fallback = float(self.config.actor_rollout_ref.actor.kl_loss_coef)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_names = batch.non_tensor_batch.get("task_name", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_names is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            env_kwargs = batch.non_tensor_batch.get("env_kwargs", None)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if env_kwargs is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                task_names = [
                    item.get("task_name") if isinstance(item, dict) else None
                    for item in env_kwargs
                ]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_names is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.full((len(batch),), fallback, dtype=torch.float32)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        coefs = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task_name in task_names:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task = self._normalize_task_name(task_name)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            coefs.append(float(coef_by_task.get(task, fallback)))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.tensor(coefs, dtype=torch.float32)

    # [EXPLAIN] rollout 生成から teacher forward、actor 更新、検証、checkpoint までの学習 phase を順序付ける trainer loop である。
    def fit(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        The training loop of SkillSD. Identical to RLSD except:
        - Advantages are NOT replaced with token-level advantages
        - teacher_log_probs are passed through to the actor for SDL loss
        """
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.tracking import Tracking

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.global_steps = 0
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._load_checkpoint()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._fast_forward_env_schedules()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            val_metrics = self._validate()
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert val_metrics, f"{val_metrics=}"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            pprint(f"Initial validation metrics: {val_metrics}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.log(data=val_metrics, step=self.global_steps)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.trainer.get("val_only", False):
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.global_steps += 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        last_val_metrics = None

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for epoch in range(self.config.trainer.total_epochs):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_iter = iter(self.train_dataloader)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            peeked_batch_dict = None
            # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
            while True:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if peeked_batch_dict is not None:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    batch_dict = peeked_batch_dict
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    peeked_batch_dict = None
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    batch_dict = next(batch_iter, None)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if batch_dict is None:
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        break
                # Reset the pre-peek dataloader snapshot each step; it is set
                # again below if this step peeks ahead (see _save_checkpoint).
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self._pre_peek_dataloader_state = None
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metrics = {}
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                timing_raw = {}
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "multi_modal_data" in batch.non_tensor_batch:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "raw_prompt" in batch.non_tensor_batch:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "tools_kwargs" in batch.non_tensor_batch:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "env_kwargs" in batch.non_tensor_batch:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "task_name" in batch.non_tensor_batch:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    non_tensor_batch_keys_to_pop.append("task_name")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                is_last_step = self.global_steps >= self.total_training_steps

                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with _timer("step", timing_raw):
                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("gen", timing_raw):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    # The train envs are idle from here until the next rollout;
                    # kick off their reset for the next step in a background
                    # thread so it overlaps the GPU training phases below.
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if (
                        _ENV_RESET_PREFETCH
                        and not is_last_step
                        and not self.config.algorithm.filter_groups.enable
                    ):
                        # Snapshot the dataloader state before peeking: the peeked
                        # batch is trained on the NEXT step, so a checkpoint saved
                        # this step must record the pre-peek position or a resumed
                        # run would skip that batch (see _save_checkpoint).
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if hasattr(self.train_dataloader, "state_dict"):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            self._pre_peek_dataloader_state = self.train_dataloader.state_dict()
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        peeked_batch_dict = next(batch_iter, None)
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if peeked_batch_dict is not None and "env_kwargs" in peeked_batch_dict:
                            # Same repeat the next multi_turn_loop applies to its
                            # gen_batch (repeat(n, interleave=True) on non-tensors
                            # is an element-wise np.repeat).
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            next_env_kwargs = np.repeat(
                                peeked_batch_dict["env_kwargs"], self.config.env.rollout.n
                            )
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            self.traj_collector.prefetch_env_reset(self.envs, next_env_kwargs)

                    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
                    del batch
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    batch = gen_batch_output

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    batch = adjust_batch(self.config, batch)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.trainer.balance_batch:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        self._balance_batch(batch, metrics=metrics)

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("reward", timing_raw):
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.use_rm:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            batch = batch.union(reward_tensor)

                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.config.reward_model.launch_reward_fn_async:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("old_log_prob", timing_raw):
                        # Reuse any per-row log probs prefetched during the rollout
                        # (ROLLOUT_PREFETCH_LOGPROB); computes everything normally
                        # when nothing was prefetched.
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        old_log_prob = compute_log_prob_with_prefetch(
                            self.actor_rollout_wg,
                            batch,
                            self.traj_collector.take_prefetched_log_probs(),
                            temperature=self.config.actor_rollout_ref.rollout.temperature,
                        )
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        entropys = old_log_prob.batch["entropys"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        response_masks = batch.batch["response_mask"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        old_log_prob_metrics = self._entropy_loss_metrics(batch, entropys, response_masks, loss_agg_mode)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics.update(old_log_prob_metrics)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        old_log_prob.batch.pop("entropys")
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch = batch.union(old_log_prob)

                    # ---- SkillSD: Teacher forward pass (same as RLSD) ----
                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("teacher_forward", timing_raw):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.use_reference_policy:
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("ref", timing_raw):
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if not self.ref_in_actor:
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                            else:
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            batch = batch.union(ref_log_prob)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.use_critic:
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("values", timing_raw):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            values = self.critic_wg.compute_values(batch)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            batch = batch.union(values)

                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("adv", timing_raw):
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        reward_extra_infos_dict: dict[str, list]
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.config.reward_model.launch_reward_fn_async:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        batch.batch["token_level_scores"] = reward_tensor

                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if reward_extra_infos_dict:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                invalid_action_penalty_coef_by_task=self.config.actor_rollout_ref.actor.get(
                                    "invalid_action_penalty_coef_by_task", None
                                ),
                            )
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            metrics.update(invalid_metrics)

                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.config.algorithm.use_kl_in_reward:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            metrics.update(kl_metrics)
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        # ---- SkillSD: Do NOT replace advantages ----
                        # Advantages stay as standard GRPO sequence-level advantages.
                        # The SDL loss is computed inside dp_actor.update_policy().

                        # Log teacher-student gap metrics
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        response_mask = batch.batch["response_mask"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        student_log_probs = batch.batch["old_log_probs"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        teacher_lp = batch.batch["teacher_log_probs"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        delta_t = (teacher_lp - student_log_probs) * response_mask
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        current_sdl_lambda = self._get_sdl_lambda(self.global_steps)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics["skillsd/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics["skillsd/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metrics["skillsd/sdl_lambda"] = current_sdl_lambda
                        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                        for task, rows in task_row_indices(batch).items():
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            task_rows = torch.from_numpy(rows)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            task_delta_t, task_mask = delta_t[task_rows], response_mask[task_rows]
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            metrics[f"skillsd/teacher_student_gap_mean/{task}"] = masked_mean(task_delta_t, task_mask).item()
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            metrics[f"skillsd/teacher_student_gap_std/{task}"] = masked_mean(task_delta_t ** 2, task_mask).sqrt().item()

                        # Save per-token gap data if SAVE_SDAR_DEBUG=1, at test_freq interval
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if os.environ.get("SAVE_SDAR_DEBUG", "0") == "1" and \
                                self.config.trainer.test_freq > 0 and \
                                self.global_steps % self.config.trainer.test_freq == 0:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            save_dir = os.environ.get(
                                "SAVE_SDAR_DEBUG_DIR",
                                "outputs/sdar_debug"
                            )
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            os.makedirs(save_dir, exist_ok=True)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            save_path = os.path.join(save_dir, f"step_{self.global_steps}.jsonl")

                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            bs = response_mask.shape[0]
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            turn_steps = batch.non_tensor_batch.get("turn_step", np.zeros(bs, dtype=object))
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            traj_uids = batch.non_tensor_batch.get("traj_uid", np.array([""] * bs, dtype=object))
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            episode_rewards_arr = batch.non_tensor_batch.get("episode_rewards", np.zeros(bs, dtype=object))
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            episode_lengths_arr = batch.non_tensor_batch.get("episode_lengths", np.zeros(bs, dtype=object))
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            response_ids = batch.batch["responses"]

                            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                            with open(save_path, "w") as f:
                                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                                for i in range(bs):
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    mask_i = response_mask[i].bool()
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    valid_count = mask_i.sum().item()
                                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                    if valid_count == 0:
                                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                                        continue
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    token_ids_i = response_ids[i][mask_i].cpu().tolist()
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    tokens_i = [self.tokenizer.decode([tid]) for tid in token_ids_i]
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    gaps_i = delta_t[i][mask_i].cpu().tolist()
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    teacher_lps_i = teacher_lp[i][mask_i].cpu().tolist()
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    student_lps_i = student_log_probs[i][mask_i].cpu().tolist()

                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    record = {
                                        "global_step": self.global_steps,
                                        "sample_idx": i,
                                        "turn_step": int(turn_steps[i]),
                                        "traj_uid": str(traj_uids[i]),
                                        "episode_reward": float(episode_rewards_arr[i]),
                                        "episode_length": float(episode_lengths_arr[i]),
                                        "tokens": tokens_i,
                                        "token_ids": token_ids_i,
                                        "gaps": gaps_i,
                                        "teacher_log_probs": teacher_lps_i,
                                        "student_log_probs": student_lps_i,
                                    }
                                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"[SDAR Debug] Saved per-token gap data to {save_path} ({bs} samples)")

                    # tag rows with their task so the actor can split its metrics
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    self._attach_task_ids(batch)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.use_critic:
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("update_critic", timing_raw):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            critic_output = self.critic_wg.update_critic(batch)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics.update(critic_output_metrics)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("update_actor", timing_raw):
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            kl_loss_coef = self._task_kl_loss_coef_tensor(batch)
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if kl_loss_coef is not None:
                                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                                batch.batch["kl_loss_coef"] = kl_loss_coef
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics.update(actor_output_metrics)

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if rollout_data_dir:
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("dump_rollout_generations", timing_raw):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("testing", timing_raw):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            val_metrics: dict = self._validate()
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if is_last_step:
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                last_val_metrics = val_metrics
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics.update(val_metrics)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("save_checkpoint", timing_raw):
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            self._save_checkpoint()

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics.update(compute_data_metrics_by_task(batch=batch, use_critic=self.use_critic))
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                n_gpus = self.resource_pool_manager.get_n_gpus()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.log(data=metrics, step=self.global_steps)

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                progress_bar.update(1)
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                self.global_steps += 1
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if is_last_step:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    progress_bar.close()
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return
