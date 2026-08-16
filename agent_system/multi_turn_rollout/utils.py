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

import torch
import numpy as np
import random
from typing import List, Tuple, Dict
import math
from PIL import Image
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

def to_list_of_dict(batch: DataProto, rows=None) -> list[dict]:
    """One dict per row, aligned to ``rows`` when given.

    ``rows`` exists because the caller usually keeps only the rows that are still
    active: a finished trajectory's row is recorded nowhere, so slicing every
    tensor of every column for it and then dropping the dict is the whole cost for
    none of the result. In a 50-turn episode most rows are finished for most
    turns.
    """
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    if rows is None:
        rows = range(len(tensors['input_ids']))
    save_list = []
    for bs in rows:
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def compute_log_prob_with_prefetch(actor_rollout_wg, batch: DataProto, prefetched: Dict, temperature=None) -> DataProto:
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
    if not prefetched:
        return actor_rollout_wg.compute_log_prob(batch)

    non_tensor = batch.non_tensor_batch
    if "traj_uid" not in non_tensor or "turn_step" not in non_tensor:
        return actor_rollout_wg.compute_log_prob(batch)

    response_length = batch.batch["responses"].shape[1]
    sample_log_probs, sample_entropys = next(iter(prefetched.values()))
    if sample_log_probs.shape[-1] != response_length:
        print(
            "[prefetch-logprob] response length mismatch "
            f"({sample_log_probs.shape[-1]} vs {response_length}); recomputing all rows."
        )
        return actor_rollout_wg.compute_log_prob(batch)

    keys = [
        (non_tensor["traj_uid"][i], int(non_tensor["turn_step"][i]))
        for i in range(len(batch))
    ]
    missing_idx = [i for i, key in enumerate(keys) if key not in prefetched]
    if len(missing_idx) == len(keys):
        return actor_rollout_wg.compute_log_prob(batch)

    if missing_idx:
        sub_batch = batch.select_idxs(missing_idx)
        sub_padded, pad_size = pad_dataproto_to_divisor(sub_batch, actor_rollout_wg.world_size)
        computed = actor_rollout_wg.compute_log_prob(sub_padded)
        computed = unpad_dataproto(computed, pad_size=pad_size)
        log_probs_dtype = computed.batch["old_log_probs"].dtype
        entropys_dtype = computed.batch["entropys"].dtype
        meta_info = computed.meta_info
    else:
        computed = None
        log_probs_dtype = sample_log_probs.dtype
        entropys_dtype = sample_entropys.dtype
        meta_info = {} if temperature is None else {"temperature": temperature}

    full_log_probs = torch.zeros((len(batch), response_length), dtype=log_probs_dtype)
    full_entropys = torch.zeros((len(batch), response_length), dtype=entropys_dtype)
    for i, key in enumerate(keys):
        if key in prefetched:
            row_log_probs, row_entropys = prefetched[key]
            full_log_probs[i] = row_log_probs
            full_entropys[i] = row_entropys
    if computed is not None:
        missing_t = torch.as_tensor(missing_idx, dtype=torch.long)
        full_log_probs[missing_t] = computed.batch["old_log_probs"].to(log_probs_dtype)
        full_entropys[missing_t] = computed.batch["entropys"].to(entropys_dtype)

    prefetched_count = len(keys) - len(missing_idx)
    print(f"[prefetch-logprob] reused {prefetched_count}/{len(keys)} rows from rollout prefetch")
    return DataProto.from_dict(
        tensors={"old_log_probs": full_log_probs, "entropys": full_entropys},
        meta_info=meta_info,
    )


def adjust_batch(config, data: DataProto, mode="copy") -> DataProto:
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * world_size
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        size_divisor_ref = config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu * world_size
    else:
        size_divisor_ref = size_divisor_rollout
    if "multi_modal_inputs" in data.non_tensor_batch:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    else:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    remainder = bs % size_divisor
    if remainder == 0:
        return data
    
    if mode == "delete":
        # Generate indices to remove, rather than indices to keep
        remove_indices = np.random.choice(bs, remainder, replace=False)
        # Sort remove_indices to maintain stability when deleting
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        del data
    elif mode == "copy":
        to_add = size_divisor - remainder
        dup_indices = np.random.choice(bs, to_add, replace=False)
        dup_proto = data.select_idxs(dup_indices)

        adjusted_batch = DataProto.concat([data, dup_proto])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return adjusted_batch


def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        tool_callings: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        # Get the indices of the current group
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        for index in group_indices:
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, success, and tool_callings based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]
    tool_callings = tool_callings[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

