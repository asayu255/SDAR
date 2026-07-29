# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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


# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import itertools
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import math
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from recipe.spin.core_algos import compute_online_dpo_loss, get_batch_logps
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.actor import DataParallelPPOActor

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ['DataParallelPPOActor']

# [EXPLAIN] `SPINDataParallelPPOActor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SPINDataParallelPPOActor(DataParallelPPOActor):
    
    # [EXPLAIN] `compute_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_module.eval()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = data.meta_info['micro_batch_size']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = data.select(batch_keys=select_keys).batch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if has_multi_modal_inputs:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_select_keys = ['multi_modal_inputs']
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif use_dynamic_bsz:
            # split using dynamic bsz
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches = batch.split(micro_batch_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs_lst = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for micro_batch in micro_batches:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(micro_batch, DataProto):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with torch.no_grad():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_probs_lst.append(log_probs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs = torch.concat(log_probs_lst, dim=0)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            indices = list(itertools.chain.from_iterable(indices))
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            log_probs = log_probs[revert_indices]

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return log_probs

    # [EXPLAIN] `update_policy_dpo_with_ref` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_policy_dpo_with_ref(self, data: DataProto):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Performs the DPO update step using pre-calculated reference log probs
        from an external, periodically updated reference model.
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_module.train() # Ensure training mode

        # --- Retrieve necessary data ---
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # Expects batch prepared by fit_dpo loop, including reference log probs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_td = data.batch
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            chosen_labels = batch_td['chosen_labels']
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rejected_labels = batch_td['rejected_labels']
            # ... other needed tensors like chosen/rejected input_ids, attention_mask, position_ids ...

            # === Get PRE-CALCULATED reference log probs from input data ===
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reference_chosen_logps = batch_td['reference_chosen_logps'] # Should be sequence-level logps
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reference_rejected_logps = batch_td['reference_rejected_logps'] # Should be sequence-level logps
            # ============================================================

            # Get DPO params from meta_info
            # beta = data.meta_info.get('dpo_beta', 0.1) # Default beta
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            beta = self.config.get('dpo_beta', 0.1) # Default beta
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            loss_type = data.meta_info.get('dpo_loss_type', 'sigmoid')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            label_smoothing = data.meta_info.get('dpo_label_smoothing', 0.0)
            # reference_free should now be False as we provide ref logps
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reference_free = data.meta_info.get('reference_free', False) # Default False

        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except KeyError as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"ERROR: Missing required key for DPO update (in update_policy_dpo): {e}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Available keys in data.batch: {list(batch_td.keys())}") # Debug print
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {} # Return empty metrics on error
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e_data:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"ERROR accessing data for DPO update (in update_policy_dpo): {e_data}")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {}

        # --- Micro-batching Setup ---
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = self.config.get('ppo_micro_batch_size_per_gpu')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if micro_batch_size is None:
            # Fallback or default if not set, or raise error
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batch_size = 1 # Example fallback, adjust as needed
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Warning: 'ppo_micro_batch_size_per_gpu' not set, defaulting to {micro_batch_size}")
            # raise ValueError("Config 'ppo_micro_batch_size_per_gpu' must be set.")

        # Ensure chosen_input_ids exists before getting shape
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'chosen_input_ids' not in batch_td:
             # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
             print("ERROR: 'chosen_input_ids' not found in batch_td for DPO update.")
             # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
             return {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz = batch_td['chosen_input_ids'].shape[0]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if bsz == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Warning: DPO batch size is 0 in update_policy_dpo. Skipping update.")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {'actor/dpo_loss': 0.0, 'actor/grad_norm': 0.0} # Return zero metrics if batch is empty

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_micro_batches = math.ceil(bsz / micro_batch_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gradient_accumulation_steps = num_micro_batches

        # --- Metrics Accumulation ---
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_loss = 0.0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        accumulated_metrics = defaultdict(list)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = {} # Final metrics dict

        # --- Zero Gradients ---
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_optimizer.zero_grad(set_to_none=True)

        # --- Micro-batch Loop ---
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(num_micro_batches):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            start_idx = i * micro_batch_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            end_idx = min(start_idx + micro_batch_size, bsz)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if start_idx >= end_idx: continue

            # Slice the full DPO batch into micro-batches
            # Important: Slice ALL required tensors, including labels and inputs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batch_chosen_labels = chosen_labels[start_idx:end_idx]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batch_rejected_labels = rejected_labels[start_idx:end_idx]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batch_chosen_inputs = {
                'input_ids': batch_td['chosen_input_ids'][start_idx:end_idx],
                'attention_mask': batch_td['chosen_attention_mask'][start_idx:end_idx]
            }
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 'chosen_position_ids' in batch_td:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                micro_batch_chosen_inputs['position_ids'] = batch_td['chosen_position_ids'][start_idx:end_idx]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batch_rejected_inputs = {
                'input_ids': batch_td['rejected_input_ids'][start_idx:end_idx],
                'attention_mask': batch_td['rejected_attention_mask'][start_idx:end_idx]
            }
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 'rejected_position_ids' in batch_td:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                micro_batch_rejected_inputs['position_ids'] = batch_td['rejected_position_ids'][start_idx:end_idx]


            # Determine autocast dtype
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            autocast_dtype = torch.bfloat16 # Or get dynamically from config/FSDP settings
            # --- Autocast Forward Pass ---
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                # --- Step 1: Forward pass for CURRENT policy log probs (with grad) ---
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                policy_chosen_outputs = self.actor_module(**micro_batch_chosen_inputs, use_cache=False)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                policy_rejected_outputs = self.actor_module(**micro_batch_rejected_inputs, use_cache=False)

                # --- Step 2: Calculate CURRENT policy log probs using get_batch_logps ---
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                policy_chosen_logps = get_batch_logps(
                    policy_chosen_outputs.logits, micro_batch_chosen_labels, average_log_prob=False
                )
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                policy_rejected_logps = get_batch_logps(
                    policy_rejected_outputs.logits, micro_batch_rejected_labels, average_log_prob=False
                )

                # --- Step 3: Retrieve PRE-CALCULATED reference log probs (NO grad needed) ---
                # Slice the full batch reference logps for the current micro-batch
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_ref_chosen_logps = reference_chosen_logps[start_idx:end_idx]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_ref_rejected_logps = reference_rejected_logps[start_idx:end_idx]
                # --- The ActorAsRef calculation block is REMOVED ---

                # --- Step 4: Calculate DPO Logits and Loss ---
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                pi_logratios = policy_chosen_logps - policy_rejected_logps
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ref_logratios = micro_ref_chosen_logps - micro_ref_rejected_logps # Uses pre-calculated values
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                logits = pi_logratios - ref_logratios # DPO logits

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss = compute_online_dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,         # Has grad
                    policy_rejected_logps=policy_rejected_logps,       # Has grad
                    reference_chosen_logps=micro_ref_chosen_logps,   # No grad (from input)
                    reference_rejected_logps=micro_ref_rejected_logps, # No grad (from input)
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    reference_free=reference_free # Should be False now
                )

                # --- Scale loss for gradient accumulation ---
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                scaled_loss = loss / gradient_accumulation_steps

                # --- Accumulate Metrics ---
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                total_loss += loss.item() # Unscaled loss
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                accumulated_metrics['actor/dpo_loss_batch'].append(loss.item())
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                accumulated_metrics['actor/dpo_logits_batch'].append(logits.mean().item())
                # Accumulate policy and reference log probs/ratios if needed for debugging
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                accumulated_metrics['actor/policy_chosen_logps_batch'].append(policy_chosen_logps.mean().item())
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                accumulated_metrics['actor/policy_rejected_logps_batch'].append(policy_rejected_logps.mean().item())
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                accumulated_metrics['actor/reference_chosen_logps_batch'].append(micro_ref_chosen_logps.mean().item())
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                accumulated_metrics['actor/reference_rejected_logps_batch'].append(micro_ref_rejected_logps.mean().item())

            # --- Backward Pass (outside autocast) ---
            # Check if loss requires grad before backward
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if scaled_loss.requires_grad:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                scaled_loss.backward()
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"Warning: Scaled loss at micro-batch {i} does not require grad. Skipping backward.")


        # --- End Micro-batch Loop ---

        # --- Optimizer Step (after accumulating gradients for all micro-batches) ---
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        grad_norm = self._optimizer_step()

        # --- Populate Final Metrics ---
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if num_micro_batches > 0 and bsz > 0: # Check if any processing happened
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metrics['actor/dpo_loss'] = total_loss / num_micro_batches
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics['actor/grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) and torch.isfinite(grad_norm) else float('inf')
            # Average other accumulated metrics
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, val_list in accumulated_metrics.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if val_list: metrics[key.replace('_batch','')] = np.mean(val_list)

            # Calculate accuracy / rewards / margins based on averaged logprobs if desired
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 'actor/policy_chosen_logps' in metrics and 'actor/policy_rejected_logps' in metrics and \
               'actor/reference_chosen_logps' in metrics and 'actor/reference_rejected_logps' in metrics:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                policy_ratio_mean = metrics['actor/policy_chosen_logps'] - metrics['actor/policy_rejected_logps']
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ref_ratio_mean = metrics['actor/reference_chosen_logps'] - metrics['actor/reference_rejected_logps']
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                logits_mean = policy_ratio_mean - ref_ratio_mean
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics['actor/rewards_chosen'] = beta * (metrics['actor/policy_chosen_logps'] - metrics['actor/reference_chosen_logps'])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics['actor/rewards_rejected'] = beta * (metrics['actor/policy_rejected_logps'] - metrics['actor/reference_rejected_logps'])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metrics['actor/rewards_accuracies'] = float(logits_mean > 0) # Mean accuracy proxy
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                metrics['actor/rewards_margins'] = metrics['actor/rewards_chosen'] - metrics['actor/rewards_rejected']

        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else: # Handle case where no micro-batches were run (e.g., bsz=0)
             # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
             metrics['actor/dpo_loss'] = 0.0
             # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
             metrics['actor/grad_norm'] = 0.0
             # Initialize other metrics to 0 or NaN as appropriate
             # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
             for key in accumulated_metrics.keys():
                 # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                 metrics[key.replace('_batch','')] = 0.0
             # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
             metrics['actor/rewards_chosen'] = 0.0
             # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
             metrics['actor/rewards_rejected'] = 0.0
             # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
             metrics['actor/rewards_accuracies'] = 0.0
             # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
             metrics['actor/rewards_margins'] = 0.0


        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return metrics # Return aggregated metrics
