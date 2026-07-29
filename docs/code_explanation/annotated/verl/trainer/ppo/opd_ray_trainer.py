# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
OPD (On-Policy Distillation) Trainer — multitask.

Unlike OPSD/SDAR (which use the *same* policy with a privileged skill prepend as
the teacher), OPD routes each sample to a *separate, single-task RL-trained*
teacher checkpoint based on its ``task_name`` (e.g. an alfworld sample is
distilled from the alfworld teacher). The student is trained purely by the KL
to its per-task teacher, evaluated on the student's own on-policy responses —
no GRPO policy-gradient, entropy, reference-KL, or reward signal enters the loss.

Teachers are created as ``role="ref"`` worker groups (one per task), each loading
a distinct ``model.path``. ``role="ref"`` forces FSDP ``CPUOffload`` at build time,
so the teachers live on CPU and only ride to GPU during log-prob computation.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import copy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pprint import pprint

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import open_dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProtoConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray import RayClassWithInitArgs
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.base import create_colocated_worker_cls
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.metric_utils import (
    _compute_response_info,
    compute_metrics_by_task,
    compute_trajectory_response_tokens,
    compute_throughout_metrics,
    compute_timing_metrics,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    _timer,
    compute_response_mask,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.reward import compute_reward
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.metric import reduce_metrics

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.multi_turn_rollout import adjust_batch

# Overlap envs.reset() for the next rollout with this step's GPU training phases.
# The reset is pure CPU / subprocess / HTTP work and the env managers are idle
# between rollouts; the reset still runs exactly once per rollout and in the same
# order, so stateful env schedules (alfworld's game-file iterator) are unchanged.
# Opt-in; see TrajectoryCollector.prefetch_env_reset.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ENV_RESET_PREFETCH = os.environ.get("ENV_RESET_PREFETCH", "0").strip().lower() in ("1", "true", "yes", "on")


# [EXPLAIN] `compute_opd_data_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_opd_data_metrics(batch: DataProto) -> dict:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Lightweight, advantage-free replacement for ``compute_data_metrics``.

    ``compute_data_metrics`` unconditionally reads ``advantages``/``returns``/
    ``token_level_rewards``; OPD never computes advantages, so we report only
    sequence-length stats plus (defensively) any reward/episode signals that
    happen to be present for monitoring.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_info = _compute_response_info(batch)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompt_length = response_info["prompt_length"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = response_info["response_length"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_response_length = batch.batch["responses"].shape[-1]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_prompt_length = batch.batch["attention_mask"].shape[-1] - max_response_length

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metrics = {
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # Monitoring-only: env reward / success rate, never fed to the loss.
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "token_level_scores" in batch.batch:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq_score = batch.batch["token_level_scores"].sum(-1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        metrics["opd/score/mean"] = torch.mean(seq_score).detach().item()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        metrics["opd/score/max"] = torch.max(seq_score).detach().item()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        metrics["opd/score/min"] = torch.min(seq_score).detach().item()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "traj_uid" in batch.non_tensor_batch:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, unique_idx = np.unique(batch.non_tensor_batch["traj_uid"], return_index=True)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k, v in batch.non_tensor_batch.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "success_rate" in k:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics[f"episode/{k}"] = float(v[0])
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "episode_rewards" in batch.non_tensor_batch:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics["episode/reward/mean"] = float(batch.non_tensor_batch["episode_rewards"][unique_idx].mean())
        # tokens a whole trajectory generated (response_length above is per turn)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        trajectory_response_tokens = compute_trajectory_response_tokens(batch)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if trajectory_response_tokens is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics["episode/response_tokens/mean"] = float(trajectory_response_tokens.mean())
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics["episode/response_tokens/max"] = float(trajectory_response_tokens.max())
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics["episode/response_tokens/min"] = float(trajectory_response_tokens.min())

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return metrics


# [EXPLAIN] `compute_opd_data_metrics_by_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_opd_data_metrics_by_task(batch: DataProto) -> dict:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Per-task breakdown of :func:`compute_opd_data_metrics`.

    Success rates are dropped from the per-task slices: they are batch-wide
    constants broadcast onto every row, and the multitask env manager already
    reports them per task as ``episode/{task}_success_rate``.
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return compute_metrics_by_task(
        batch,
        lambda task_batch: {
            name: value for name, value in compute_opd_data_metrics(task_batch).items() if "success_rate" not in name
        },
    )


# [EXPLAIN] `OPDRayTrainer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OPDRayTrainer(RayPPOTrainer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Multitask on-policy distillation trainer with per-task teacher routing."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, *args, **kwargs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(*args, **kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        opd_cfg = self.config.algorithm.get("opd", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_paths = opd_cfg.get("teacher_paths", None)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert teacher_paths is not None, (
            "OPD requires algorithm.opd.teacher_paths.{alfworld,search,webshop}"
        )
        # Normalize to a plain {task_name: checkpoint_path} dict.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.teacher_paths = {
            self._normalize_task_name(task): path for task, path in dict(teacher_paths).items()
        }
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert None not in self.teacher_paths, "teacher_paths contains an unknown task name"
        # Pure-distillation invariants (also enforced by main_opd config injection).
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert not self.use_reference_policy, "OPD must not create a reference-policy worker"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.teacher_wg = {}
        # Distillation KL mode: "topk_kl" uses dense top-k (+tail) KL; otherwise a
        # single-sampled-token estimator (low_var_kl / kl / ...).
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_cfg = self.config.actor_rollout_ref.actor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.teacher_topk_kl = actor_cfg.get("teacher_kl_loss_type", "low_var_kl") == "topk_kl"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.teacher_kl_topk = int(actor_cfg.get("teacher_kl_topk", 20))

    # ------------------------------------------------------------------ #
    # Worker setup: actor_rollout (+ optional critic/rm) + N teachers.
    # ------------------------------------------------------------------ #
    # [EXPLAIN] `init_workers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_workers(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.resource_pool_manager.create_resource_pool()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # actor + rollout (hybrid engine)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.hybrid_engine:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_critic:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # One teacher worker group per task, each with its own checkpoint.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._teacher_keys = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task, path in self.teacher_paths.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            teacher_cfg = copy.deepcopy(self.config.actor_rollout_ref)
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with open_dict(teacher_cfg):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                teacher_cfg.model.path = path
                # Avoid the LoRA branch in compute_ref_log_prob; teachers are full models.
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                teacher_cfg.model.lora_rank = 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            key = f"teacher_{task}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._teacher_keys[task] = key
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            teacher_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=teacher_cfg,
                role="ref",
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.resource_pool_to_cls[teacher_pool][key] = teacher_cls

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_rm:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_wg = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        wg_kwargs = {}
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            all_wg.update(spawn_wg)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_critic:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.critic_wg = all_wg["critic"]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.critic_wg.init_model()

        # Initialize teachers before the rollout engine (matches actor-last ordering).
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task, key in self._teacher_keys.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            wg = all_wg[key]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            wg.init_model()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.teacher_wg[task] = wg

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_rm:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.rm_wg = all_wg["rm"]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.rm_wg.init_model()

        # rollout is created last so vLLM gets a better kv-cache estimate.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.actor_rollout_wg = all_wg["actor_rollout"]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_rollout_wg.init_model()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.async_rollout_mode = False
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.actor_rollout_ref.rollout.mode == "async":
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.async_rollout_mode = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    # [EXPLAIN] `_save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _save_checkpoint(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Same as the base trainer, except when ENV_RESET_PREFETCH has peeked
        one dataloader batch ahead this step. The peeked batch has only had its
        env_kwargs used for the background env reset — it is trained on the
        *next* step — so the checkpoint must record the pre-peek dataloader
        position; saving the live (post-peek) state would make a resumed run
        skip that batch entirely.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pre_peek_state = getattr(self, "_pre_peek_dataloader_state", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pre_peek_state is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return super()._save_checkpoint()
        # Shadow the bound state_dict with the pre-peek snapshot for the
        # duration of the base save (which calls train_dataloader.state_dict()).
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_dataloader.state_dict = lambda: pre_peek_state
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return super()._save_checkpoint()
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            del self.train_dataloader.state_dict

    # ------------------------------------------------------------------ #
    # Per-task teacher routing.
    # ------------------------------------------------------------------ #
    # [EXPLAIN] task_name で分割した row を各 teacher worker group へ送り、得られた teacher signal を元の global row 順で入力 batch に書き戻す。
    def compute_teacher_log_probs(self, batch: DataProto) -> None:
        # [EXPLAIN] 入力 `responses` は (batch, response_length) で、student 自身が on-policy 生成した token 列である。
        # [EXPLAIN] teacher 用 prompt/response を新規生成せず、同じ row を task 別 checkpoint へ forward する。
        # [EXPLAIN] 戻り値は使わず、単一 token 方式なら (batch,response_length)、top-k 方式なら
        # [EXPLAIN] (batch,response_length,k) の CPU Tensor を `batch.batch` に mutation する。
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Route each sample to its task's teacher and write the distillation
        signal into ``batch.batch`` in original order.

        The student's exact (prompt, response) is fed to the teacher — no skill
        prepend. The teacher call is ``DP_COMPUTE_PROTO``-dispatched; per-task
        slices are auto-padded to the DP world size and unpadded on return.

        - default (single-token estimator): sets ``teacher_log_probs`` (bs, resp).
        - top-k mode: sets ``teacher_topk_logprobs`` and ``teacher_topk_ids``
          (bs, resp, k), the teacher's top-k log-softmax and ids per token.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_names = batch.non_tensor_batch.get("task_name", None)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert task_names is not None, "OPD requires task_name on every sample for teacher routing"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        normalized = [self._normalize_task_name(t) for t in task_names]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bs = batch.batch["responses"].size(0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        resp_len = batch.batch["responses"].size(1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seen = [False] * bs

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.teacher_topk_kl:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            k = self.teacher_kl_topk
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            teacher_topk_logprobs = torch.zeros((bs, resp_len, k), dtype=torch.float32)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            teacher_topk_ids = torch.zeros((bs, resp_len, k), dtype=torch.long)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            teacher_log_probs = torch.zeros((bs, resp_len), dtype=torch.float32)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task, wg in self.teacher_wg.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idxs = [i for i, t in enumerate(normalized) if t == task]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not idxs:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] `idxs` は task-local row 番号ではなく元 batch の global row 番号である。
            # [EXPLAIN] `select_idxs` は Tensor/non-Tensor を同じ index 集合で抽出する一方、meta_info は共有し得るため、
            # [EXPLAIN] 下で shallow copy して auto-padding 指定が親 batch へ漏れないようにする。
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sub = batch.select_idxs(idxs)
            # Task slices are not generally divisible by the teacher group's world
            # size, so enable auto-padding: the DP dispatch pads the input to a
            # multiple of world_size and unpads the output back to len(idxs). Copy
            # meta_info first so we don't mutate the parent batch (select_idxs
            # shares the parent's meta_info dict by reference).
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sub.meta_info = dict(sub.meta_info)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sub.meta_info[DataProtoConfig.auto_padding_key] = True
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.teacher_topk_kl:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                sub.meta_info["topk_k"] = k
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                out = wg.compute_ref_topk_log_prob(sub)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tlp = out.batch["teacher_topk_logprobs"]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tid = out.batch["teacher_topk_ids"]
                # [EXPLAIN] teacher worker の戻り順 `j` を元の global row `i` へ scatter する。
                # [EXPLAIN] task ごとに分割・DP padding されても actor 入力の row 順は変わらない。
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for j, i in enumerate(idxs):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    teacher_topk_logprobs[i] = tlp[j]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    teacher_topk_ids[i] = tid[j]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    seen[i] = True
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                out = wg.compute_ref_log_prob(sub)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                lp = out.batch["ref_log_prob"]
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for j, i in enumerate(idxs):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    teacher_log_probs[i] = lp[j]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    seen[i] = True

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not all(seen):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            missing = sorted({normalized[i] for i in range(bs) if not seen[i]})
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(
                f"No teacher configured for task_name(s) {missing}; "
                f"available teachers: {sorted(self.teacher_wg.keys())}"
            )

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.teacher_topk_kl:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            batch.batch["teacher_topk_logprobs"] = teacher_topk_logprobs
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            batch.batch["teacher_topk_ids"] = teacher_topk_ids
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            batch.batch["teacher_log_probs"] = teacher_log_probs

    # ------------------------------------------------------------------ #
    # Thin training loop: rollout -> teacher_log_probs -> update_actor.
    # No old_log_prob / ref / values / advantage / reward-in-loss.
    # ------------------------------------------------------------------ #
    # [EXPLAIN] rollout 生成から teacher forward、actor 更新、検証、checkpoint までの学習 phase を順序付ける trainer loop である。
    def fit(self):
        # [EXPLAIN] この thin loop の gradient 経路は `gen → teacher_forward → update_actor` で閉じる。
        # [EXPLAIN] reward は episode/per-task metric 用に計算されるが、critic value、advantage、returns、
        # [EXPLAIN] old-log-prob、shared reference policy forward、reward-KL の phase は呼ばれない。
        # [EXPLAIN] optional env-reset prefetch は次 step の環境初期化を GPU 学習と重ねる性能処理であり、
        # [EXPLAIN] teacher KL の目的関数や現在 step の trajectory 内容を変更するものではない。
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
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPD Training")
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

                    # Reward is computed for monitoring only (task success); it is never
                    # turned into advantages and never enters the loss.
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    reward_extra_infos_dict = {}
                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("reward", timing_raw):
                        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                        try:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            batch.batch["token_level_scores"] = reward_tensor
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if reward_extra_infos_dict:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                        except Exception as e:  # monitoring must never break training
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"[OPD] reward computation skipped: {e}")

                    # ---- Per-task teacher forward pass (the only training signal) ----
                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("teacher_forward", timing_raw):
                        # writes teacher_log_probs OR teacher_topk_{logprobs,ids} into batch
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        self.compute_teacher_log_probs(batch)

                    # tag rows with their task so the actor can split its metrics
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    self._attach_task_ids(batch)

                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("update_actor", timing_raw):
                        # update_policy scales the student logits by this temperature to
                        # match the rollout sampling distribution. The standard loop gets
                        # it from compute_log_prob; the thin OPD loop skips that step, so
                        # set it explicitly (same value compute_log_prob would set).
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    metrics.update(actor_output_metrics)

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if rollout_data_dir and "token_level_scores" in batch.batch:
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
                metrics.update(compute_opd_data_metrics(batch=batch))
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics.update(compute_opd_data_metrics_by_task(batch=batch))
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
