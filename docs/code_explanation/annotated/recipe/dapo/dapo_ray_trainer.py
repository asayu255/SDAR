# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import uuid
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from copy import deepcopy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pprint import pprint

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.core_algos import agg_loss
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask


# [EXPLAIN] `RayDAPOTrainer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RayDAPOTrainer(RayPPOTrainer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # [EXPLAIN] rollout 生成から teacher forward、actor 更新、検証、checkpoint までの学習 phase を順序付ける trainer loop である。
    def fit(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
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

        # load checkpoint before doing anything
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
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

        # add tqdm
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.global_steps += 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        last_val_metrics = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        timing_raw = defaultdict(float)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_prompt_in_batch = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_gen_batches = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for epoch in range(self.config.trainer.total_epochs):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for batch_dict in self.train_dataloader:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metrics = {}

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                num_gen_batches += 1
                # pop those keys for generation
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                is_last_step = self.global_steps >= self.total_training_steps

                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with _timer("step", timing_raw):
                    # generate a batch
                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("gen", timing_raw):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("gen_max", timing_raw):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            gen_baseline_batch = deepcopy(gen_batch)
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            gen_baseline_batch.meta_info["do_sample"] = False
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            new_batch = new_batch.union(gen_baseline_output)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
                            del gen_baseline_batch, gen_baseline_output

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    new_batch = new_batch.union(gen_batch_output)

                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.use_rm:
                            # we first compute reward model score
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        reward_extra_infos_dict: dict[str, list]
                        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                        try:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_tensor = reward_result["reward_tensor"]
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                        except Exception as e:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"Error in reward_fn: {e}")
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_tensor = self.reward_fn(new_batch)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_extra_infos_dict = {}

                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        new_batch.batch["token_level_scores"] = reward_tensor

                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if reward_extra_infos_dict:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.config.algorithm.use_kl_in_reward:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if not self.config.algorithm.filter_groups.enable:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch = new_batch
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        metric_name = self.config.algorithm.filter_groups.metric
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        elif metric_name == "seq_reward":
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        prompt_uid2metric_vals = defaultdict(list)
                        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            prompt_uid2metric_vals[uid].append(metric_val)

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        prompt_uid2metric_std = {}
                        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        num_prompt_in_batch += len(kept_prompt_uids)

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        kept_traj_idxs = []
                        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if traj_from_prompt_uid in kept_prompt_uids:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                kept_traj_idxs.append(idx)

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        new_batch = new_batch[kept_traj_idxs]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        prompt_bsz = self.config.data.train_batch_size
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if num_prompt_in_batch < prompt_bsz:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                print(f"{num_gen_batches=}. Keep generating...")
                                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                                continue
                            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                            else:
                                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # Align the batch
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.trainer.balance_batch:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer("old_log_prob", timing_raw):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        entropys = old_log_prob.batch["entropys"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        response_masks = batch.batch["response_mask"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics.update(old_log_prob_metrics)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        old_log_prob.batch.pop("entropys")
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch = batch.union(old_log_prob)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.use_reference_policy:
                        # compute reference log_prob
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("ref", timing_raw):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            batch = batch.union(ref_log_prob)

                    # compute values
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
                        # compute advantages, executed on the driver process
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
                        )

                    # update critic
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

                    # implement critic warmup
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer("update_actor", timing_raw):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics.update(actor_output_metrics)

                    # validate
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
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

                # collect metrics
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                n_gpus = self.resource_pool_manager.get_n_gpus()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                timing_raw = defaultdict(float)  # clear timing

                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                metrics["train/num_gen_batches"] = num_gen_batches
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                batch = None
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_prompt_in_batch = 0
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.log(data=metrics, step=self.global_steps)

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if is_last_step:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    progress_bar.close()
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                progress_bar.update(1)
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                self.global_steps += 1
