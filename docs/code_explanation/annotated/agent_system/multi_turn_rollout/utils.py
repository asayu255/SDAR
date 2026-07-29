# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List, Tuple, Dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import math
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from PIL import Image
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# [EXPLAIN] `to_list_of_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def to_list_of_dict(batch: DataProto) -> list[dict]:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensors = batch.batch
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensor = batch.non_tensor_batch
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = len(tensors['input_ids'])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    save_list = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for bs in range(batch_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        save_dict = dict()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in tensors.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            save_dict[key] = val[bs]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in non_tensor.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            save_dict[key] = val[bs]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        save_list.append(save_dict)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return save_list


# [EXPLAIN] `torch_to_numpy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def torch_to_numpy(tensor, is_object=False):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(tensor, torch.Tensor):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor = tensor.detach().cpu().numpy()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(tensor, np.ndarray):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Unsupported type: {type(tensor)})")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_object:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor = tensor.astype(object)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tensor

# [EXPLAIN] `numpy_to_torch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def numpy_to_torch(array, device):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(array, np.ndarray):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        array = torch.from_numpy(array).to(device)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(array, torch.Tensor):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        array = array.to(device)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Unsupported type: {type(array)})")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return array


# [EXPLAIN] `process_image` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(image, torch.Tensor):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image = torch_to_numpy(image)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if image.max() < 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image = image * 255.0
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if image.dtype != np.uint8:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image = image.astype(np.uint8)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image = Image.fromarray(image)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if (image.width * image.height) > max_pixels:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image = image.resize((width, height))

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if (image.width * image.height) < min_pixels:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image = image.resize((width, height))

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if image.mode != 'RGB':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image = image.convert('RGB')

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return image


# [EXPLAIN] `compute_log_prob_with_prefetch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_log_prob_with_prefetch(actor_rollout_wg, batch: DataProto, prefetched: Dict, temperature=None) -> DataProto:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """old_log_prob phase that reuses per-row results prefetched during rollout.

    `prefetched` maps (traj_uid, turn_step) -> (old_log_probs_row, entropys_row),
    produced by TrajectoryCollector._prefetch_pending_log_probs with the same
    frozen actor weights this phase would use. Rows found in the map are filled
    from it; the remaining rows are computed by the normal
    actor_rollout_wg.compute_log_prob on a padded sub-batch. Duplicated rows
    (adjust_batch mode="copy") share a key and receive the same value, exactly as
    recomputation would produce. Falls back to the full computation whenever the
    prefetched rows cannot be used verbatim.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not prefetched:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return actor_rollout_wg.compute_log_prob(batch)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensor = batch.non_tensor_batch
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "traj_uid" not in non_tensor or "turn_step" not in non_tensor:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return actor_rollout_wg.compute_log_prob(batch)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = batch.batch["responses"].shape[1]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sample_log_probs, sample_entropys = next(iter(prefetched.values()))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if sample_log_probs.shape[-1] != response_length:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(
            "[prefetch-logprob] response length mismatch "
            f"({sample_log_probs.shape[-1]} vs {response_length}); recomputing all rows."
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return actor_rollout_wg.compute_log_prob(batch)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    keys = [
        (non_tensor["traj_uid"][i], int(non_tensor["turn_step"][i]))
        for i in range(len(batch))
    ]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    missing_idx = [i for i, key in enumerate(keys) if key not in prefetched]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(missing_idx) == len(keys):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return actor_rollout_wg.compute_log_prob(batch)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if missing_idx:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sub_batch = batch.select_idxs(missing_idx)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sub_padded, pad_size = pad_dataproto_to_divisor(sub_batch, actor_rollout_wg.world_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        computed = actor_rollout_wg.compute_log_prob(sub_padded)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        computed = unpad_dataproto(computed, pad_size=pad_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs_dtype = computed.batch["old_log_probs"].dtype
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        entropys_dtype = computed.batch["entropys"].dtype
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        meta_info = computed.meta_info
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        computed = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs_dtype = sample_log_probs.dtype
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        entropys_dtype = sample_entropys.dtype
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        meta_info = {} if temperature is None else {"temperature": temperature}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_log_probs = torch.zeros((len(batch), response_length), dtype=log_probs_dtype)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_entropys = torch.zeros((len(batch), response_length), dtype=entropys_dtype)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i, key in enumerate(keys):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if key in prefetched:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            row_log_probs, row_entropys = prefetched[key]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_log_probs[i] = row_log_probs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_entropys[i] = row_entropys
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if computed is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        missing_t = torch.as_tensor(missing_idx, dtype=torch.long)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_log_probs[missing_t] = computed.batch["old_log_probs"].to(log_probs_dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_entropys[missing_t] = computed.batch["entropys"].to(entropys_dtype)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prefetched_count = len(keys) - len(missing_idx)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"[prefetch-logprob] reused {prefetched_count}/{len(keys)} rows from rollout prefetch")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return DataProto.from_dict(
        tensors={"old_log_probs": full_log_probs, "entropys": full_entropys},
        meta_info=meta_info,
    )


# [EXPLAIN] `adjust_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def adjust_batch(config, data: DataProto, mode="copy") -> DataProto:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * world_size
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        size_divisor_ref = config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu * world_size
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        size_divisor_ref = size_divisor_rollout
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "multi_modal_inputs" in data.non_tensor_batch:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bs = len(data)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    remainder = bs % size_divisor
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if remainder == 0:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data
    
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if mode == "delete":
        # Generate indices to remove, rather than indices to keep
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        remove_indices = np.random.choice(bs, remainder, replace=False)
        # Sort remove_indices to maintain stability when deleting
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keep_mask = np.ones(bs, dtype=bool)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keep_mask[remove_indices] = False

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_data = data.batch[keep_mask_tensor]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del data
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif mode == "copy":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        to_add = size_divisor - remainder
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dup_indices = np.random.choice(bs, to_add, replace=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dup_proto = data.select_idxs(dup_indices)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        adjusted_batch = DataProto.concat([data, dup_proto])
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Unsupported mode: {mode}")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return adjusted_batch


# [EXPLAIN] `filter_group_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        tool_callings: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if last_try:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = config.data.train_batch_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group_n = config.env.rollout.n
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if group_n <= 1:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    keep_indices = np.array([], dtype=np.int64)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(batch_size):
        # Get the indices of the current group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for index in group_indices:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, success, and tool_callings based on the keep_indices
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_list = [batch_list[i] for i in keep_indices]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    episode_rewards = episode_rewards[keep_indices]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    traj_uid = traj_uid[keep_indices]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tool_callings = tool_callings[keep_indices]

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

