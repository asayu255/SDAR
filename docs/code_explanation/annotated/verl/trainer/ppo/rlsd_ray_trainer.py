# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
RLSD (Reinforcement Learning with Self-Distillation) Trainer.

Extends the standard RayPPOTrainer with a teacher forward pass that uses
privileged information (skills) to construct token-level advantages.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pprint import pprint

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
from verl.trainer.ppo.rlsd_utils import SkillProvider, compute_rlsd_token_advantage
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
    task_row_indices,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.multi_turn_rollout import adjust_batch, compute_log_prob_with_prefetch

# Overlap envs.reset() for the next rollout with this step's GPU training phases.
# See SkillSDRayTrainer / TrajectoryCollector.prefetch_env_reset for details.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ENV_RESET_PREFETCH = os.environ.get("ENV_RESET_PREFETCH", "0").strip().lower() in ("1", "true", "yes", "on")


# [EXPLAIN] `build_teacher_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def build_teacher_batch(
    batch: DataProto,
    skill_provider: SkillProvider,
    tokenizer,
    max_prompt_length: int,
    truncation: str = "error",
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Build a teacher batch by prepending privileged skill info to each sample's prompt.

    The teacher sees (x, r) where r is the skill/privileged information.
    We prepend the skill text as a system message before the user prompt,
    then re-tokenize to get teacher input_ids/attention_mask/position_ids.
    The responses remain unchanged.

    Args:
        batch: The original student batch with prompts and responses.
        skill_provider: SkillProvider instance for loading skills.
        tokenizer: The tokenizer.
        max_prompt_length: Maximum prompt length.
        truncation: Truncation mode.

    Returns:
        teacher_batch: A DataProto with modified input_ids/attention_mask/position_ids
            but the same responses, suitable for computing teacher log probs.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bs = batch.batch["input_ids"].size(0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = batch.batch["responses"].size(1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_input_ids_list = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_attention_mask_list = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_position_ids_list = []

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(bs):
        # Decode the original prompt (student input minus response)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        original_input_ids = batch.batch["input_ids"][i]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        original_attention_mask = batch.batch["attention_mask"][i]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_length = original_input_ids.size(0) - response_length

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_ids = original_input_ids[:prompt_length]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_mask = original_attention_mask[:prompt_length]

        # Find the first non-padding token in prompt
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(valid_start) > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_start = valid_start[0].item()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_start = 0

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_prompt_ids = prompt_ids[valid_start:]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        # Get privileged skill info based on gamefile, data_source, or prompt text
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gamefile = batch.non_tensor_batch.get("gamefile", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_source = batch.non_tensor_batch.get("data_source", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_name = batch.non_tensor_batch.get("task_name", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gf = gamefile[i] if gamefile is not None else None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ds = data_source[i] if data_source is not None else None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task = task_name[i] if task_name is not None else None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skill_text = skill_provider.get_privileged_info_for_sample(
            task_name=task,
            gamefile=gf,
            data_source=ds,
            prompt_text=prompt_text,
        )

        # Construct teacher prompt: prepend skill as a system message
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skill_prefix = f"[Privileged Skill Information]\n{skill_text}\n\n"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_prompt_text = skill_prefix + prompt_text

        # Tokenize the teacher prompt
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)

        # Truncate if needed (left truncation to keep the end of prompt)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(teacher_prompt_ids) > max_prompt_length:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actual_prompt_len = len(teacher_prompt_ids)

        # Pad to max_prompt_length (left padding)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_length = max_prompt_length - actual_prompt_len
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pad_length > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        # Combine with response
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_ids = batch.batch["responses"][i]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_mask = original_attention_mask[-response_length:]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        teacher_input_ids_list.append(teacher_full_ids)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        teacher_attention_mask_list.append(teacher_full_mask)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        teacher_position_ids_list.append(teacher_position_ids)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_input_ids = torch.stack(teacher_input_ids_list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_attention_mask = torch.stack(teacher_attention_mask_list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_position_ids = torch.stack(teacher_position_ids_list)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
            "responses": batch.batch["responses"],
        },
    )

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return teacher_batch


# [EXPLAIN] `RLSDRayTrainer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RLSDRayTrainer(RayPPOTrainer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    RLSD trainer that extends RayPPOTrainer with self-distillation
    using privileged skill information as teacher signal.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(*args, **kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.skill_provider = skill_provider
        # RLSD hyperparams from config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rlsd_cfg = self.config.algorithm.get("rlsd", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rlsd_lambda_init = rlsd_cfg.get("rlsd_lambda", 0.5)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rlsd_lambda_warmdown_steps = rlsd_cfg.get("warmdown_steps", 50)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rlsd_clip_eps = rlsd_cfg.get("clip_eps", 0.2)

    # [EXPLAIN] `_get_rlsd_lambda` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_rlsd_lambda(self, step: int) -> float:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Linearly decay λ from rlsd_lambda_init to 0 over warmdown_steps."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if step >= self.rlsd_lambda_warmdown_steps:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 0.0
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.rlsd_lambda_init * (1.0 - step / self.rlsd_lambda_warmdown_steps)

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

    # [EXPLAIN] rollout 生成から teacher forward、actor 更新、検証、checkpoint までの学習 phase を順序付ける trainer loop である。
    def fit(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        The training loop of RLSD, extending the standard PPO/GRPO loop
        with teacher forward pass and token-level advantage computation.
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
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="RLSD Training")
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

                    # Compute student log probs (old_log_probs)
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

                    # ---- RLSD: Teacher forward pass ----
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

                        # Compute standard GRPO sequence-level advantages
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

                        # ---- RLSD: Replace sequence-level advantage with token-level advantage ----
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        seq_advantages = batch.batch["advantages"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        student_log_probs = batch.batch["old_log_probs"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        teacher_log_probs = batch.batch["teacher_log_probs"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        response_mask = batch.batch["response_mask"]

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        current_lambda = self._get_rlsd_lambda(self.global_steps)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        token_advantages = compute_rlsd_token_advantage(
                            seq_advantages=seq_advantages,
                            student_log_probs=student_log_probs,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            rlsd_lambda=current_lambda,
                            rlsd_clip_eps=self.rlsd_clip_eps,
                        )

                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        batch.batch["advantages"] = token_advantages

                        # Log RLSD-specific metrics
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        delta_t = (teacher_log_probs - student_log_probs) * response_mask
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics["rlsd/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics["rlsd/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metrics["rlsd/lambda"] = current_lambda
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metrics["rlsd/clip_eps"] = self.rlsd_clip_eps
                        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                        for task, rows in task_row_indices(batch).items():
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            task_rows = torch.from_numpy(rows)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            task_delta_t, task_mask = delta_t[task_rows], response_mask[task_rows]
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            metrics[f"rlsd/teacher_student_gap_mean/{task}"] = masked_mean(task_delta_t, task_mask).item()
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            metrics[f"rlsd/teacher_student_gap_std/{task}"] = masked_mean(task_delta_t ** 2, task_mask).sqrt().item()

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

    # [EXPLAIN] `_compute_teacher_log_probs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Compute teacher log probs by running forward pass with privileged skill info
        prepended to the prompt. Uses the same model π_θ but conditioned on (x, r).
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_batch = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )

        # Use the same actor to compute teacher log probs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_log_probs = teacher_output.batch["old_log_probs"]

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return teacher_log_probs
