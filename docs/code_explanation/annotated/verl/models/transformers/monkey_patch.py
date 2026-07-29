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
Apply monkey-patch function to models
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from types import SimpleNamespace
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.modeling_flash_attention_utils import _flash_attention_forward
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.modeling_utils import PreTrainedModel

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.import_utils import is_trl_available
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.transformers_compat import is_transformers_version_in_range
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)


# [EXPLAIN] `repeat_kv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if n_rep == 1:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return hidden_states
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


# [EXPLAIN] `_ulysses_flash_attention_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    For transformers>=4.55, the flash attention api has changed,
    we need to pass the query_length after doing ulysses all2all.
    See https://github.com/huggingface/transformers/issues/40399

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)

    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    ########## AlltoAll for Ulysses ##########
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ulysses_sp_size > 1:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"

        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = repeat_kv(key_states, repeats)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
        # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
        # https://github.com/huggingface/transformers/pull/33932

        # (bsz, seq_len/n) -> (bsz, seq_len)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = torch.concat(position_ids_list, dim=-1)

    # (bsz, seq_len, n_head/n, head_dim)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_length = query_states.size(1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = _flash_attention_forward(
        query_states, key_states, value_states, attention_mask, query_length, *args, position_ids=position_ids, **kwargs
    )

    ########## AlltoAll for Ulysses ##########
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return attn_output


# [EXPLAIN] `patch_vlm_for_ulysses_input_slicing` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def patch_vlm_for_ulysses_input_slicing(model_class: type):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.
    """

    # [EXPLAIN] `_create_ulysses_wrapped_decoder_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _create_ulysses_wrapped_decoder_forward(original_forward):
        # [EXPLAIN] `ulysses_wrapped_decoder_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            inputs_embeds = kwargs.get("inputs_embeds")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = kwargs.get("position_ids")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            visual_pos_masks = kwargs.get("visual_pos_masks")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            call_kwargs = kwargs.copy()

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if slice_now:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                call_kwargs["inputs_embeds"] = slice_input_tensor(inputs_embeds, dim=1, padding=False)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                call_kwargs["position_ids"] = slice_input_tensor(position_ids, dim=-1, padding=False)
                # Also slice visual_pos_masks and deepstack_visual_embeds for Qwen3 VL models
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if visual_pos_masks is not None:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    original_visual_mask = visual_pos_masks
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    sliced_visual_mask = slice_input_tensor(visual_pos_masks, dim=1, padding=False)
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    call_kwargs["visual_pos_masks"] = sliced_visual_mask

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if deepstack_visual_embeds is not None:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        sliced_embeds = []

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        num_visual_before = original_visual_mask.sum().item()
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        num_visual_in_shard = sliced_visual_mask.sum().item()

                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if num_visual_in_shard > 0 and num_visual_before > 0:
                            # Calculate which visual embeddings belong to this shard
                            # We need to find the offset of visual tokens in this shard
                            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                            from verl.utils.ulysses import get_ulysses_sequence_parallel_rank

                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            rank = get_ulysses_sequence_parallel_rank()
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            seq_len = original_visual_mask.shape[1]
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            local_seq_len = seq_len // current_ulysses_sp_size
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            start_idx = rank * local_seq_len
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            end_idx = start_idx + local_seq_len

                            # Get total visual tokens before and up to the end of the shard's sequence slice
                            # This correctly handles batches by summing across all samples
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            visual_start = original_visual_mask[:, :start_idx].sum().item() if start_idx > 0 else 0
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            visual_end = original_visual_mask[:, :end_idx].sum().item()

                            # Slice each tensor in deepstack_visual_embeds
                            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                            for embed in deepstack_visual_embeds:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                sliced_embeds.append(embed[visual_start:visual_end])
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # No visual tokens in this shard, create empty tensors to maintain gradient flow
                            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                            for embed in deepstack_visual_embeds:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                sliced_embeds.append(embed[:0])
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        call_kwargs["deepstack_visual_embeds"] = sliced_embeds

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self._needs_initial_slice = False
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return original_forward(self, *args, **call_kwargs)
            # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
            finally:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if slice_now:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self._needs_initial_slice = True

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ulysses_wrapped_decoder_forward

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    original_forward = model_class.forward
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_class.forward = wrapped_forward
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Monkey patch {model_class.__name__}.forward for Ulysses SP input slicing.")


# [EXPLAIN] `patch_forward_with_backends` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def patch_forward_with_backends(
    model: PreTrainedModel,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Choose the forward function based on the model and backend.
    Args:
        model (PreTrainedModel): The model to apply the monkey patch.
        use_fused_kernels (bool): Whether to use fused kernels.
        fused_kernels_backend (str): The backend to use for fused kernels.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not use_fused_kernels or fused_kernels_backend not in ["triton", "torch"]:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(
            f"Skipping monkey patch for {model.__class__.__name__} as use_fused_kernels is "
            f"{use_fused_kernels} or fused_kernels_backend is {fused_kernels_backend}"
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    forward_with_torch_backend_function = model.__class__.forward
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    forward_with_triton_backend_function = model.__class__.forward
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if model.config.model_type in ["qwen2_5_vl", "qwen2_vl"]:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.transformers.qwen2_vl import forward_with_torch_backend, forward_with_triton_backend

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_with_torch_backend_function = forward_with_torch_backend
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_with_triton_backend_function = forward_with_triton_backend
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif model.config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.transformers.qwen3_vl import forward_with_torch_backend, forward_with_triton_backend

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_with_torch_backend_function = forward_with_torch_backend
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_with_triton_backend_function = forward_with_triton_backend
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif model.config.model_type == "glm4v":
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.transformers.glm4v import forward_with_torch_backend, forward_with_triton_backend

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_with_torch_backend_function = forward_with_torch_backend
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_with_triton_backend_function = forward_with_triton_backend
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_with_torch_backend_function = forward_with_torch_backend
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_with_triton_backend_function = forward_with_triton_backend

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if fused_kernels_backend == "triton":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model.__class__.forward = forward_with_triton_backend_function
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Using Triton backend for fused kernels in {model.__class__.__name__}")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif fused_kernels_backend == "torch":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model.__class__.forward = forward_with_torch_backend_function
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Using Torch backend for fused kernels in {model.__class__.__name__}")
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Unsupported fused_kernels_backend: {fused_kernels_backend}. Choose 'triton' or 'torch'.")


# [EXPLAIN] `apply_monkey_patch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Apply monkey patch to the models for ulysses sequence parallel and fused kernel.

    In the end of this function forward function of the model is patched for fused kernel.
    If the model is not supported with fused kernel, please return after patch.
    """

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    module = sys.modules[model.__module__]

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert num_attention_heads % ulysses_sp_size == 0, (
        f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
    )
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size "
        f"{ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,"
        f"kv heads are repeated to ensure correctness."
    )

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_trl_available():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from trl import AutoModelForCausalLMWithValueHead  # type: ignore

        # [EXPLAIN] `state_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def state_dict(self, *args, **kwargs):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.nn.Module.state_dict(self, *args, **kwargs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        AutoModelForCausalLMWithValueHead.state_dict = state_dict
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")

    # TODO: VLM models only, unify monkey patch to LLM models.
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if model.config.model_type in ["qwen2_5_vl", "qwen2_vl"]:
        # Step 1: patch model to support image-text mixed data
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_transformers_version_in_range(min_version="4.52.0"):
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2_5_VLModel,
                Qwen2_5_VLTextModel,
            )
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLForConditionalGeneration,
                Qwen2VLModel,
                Qwen2VLTextModel,
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as Qwen2_5_VLTextModel
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as Qwen2VLTextModel

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            Qwen2_5_VLModel = SimpleNamespace(forward=None)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            Qwen2VLModel = SimpleNamespace(forward=None)

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.transformers.qwen2_vl import forward_with_normal_backend, qwen2_vl_base_forward

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Qwen2_5_VLModel.forward = qwen2_vl_base_forward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Qwen2VLModel.forward = qwen2_vl_base_forward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Qwen2_5_VLForConditionalGeneration.forward = forward_with_normal_backend
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Qwen2VLForConditionalGeneration.forward = forward_with_normal_backend
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Monkey patch {model.__class__.__name__} model forward")

        # Step 2: patch attention to support ulysses parallelism
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_transformers_version_in_range(min_version="4.54.0"):
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif is_transformers_version_in_range(min_version="4.53.0"):
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError("Transformers 4.53.* is bugged. Use transformers 4.54.0 or later.")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLFlashAttention2 as Qwen2_5_VLAttention,
            )
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2 as Qwen2VLAttention

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_remove_padding or ulysses_sp_size > 1:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.transformers.qwen2_vl import qwen2_vl_attn_forward

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            Qwen2_5_VLAttention.forward = qwen2_vl_attn_forward
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            Qwen2VLAttention.forward = qwen2_vl_attn_forward
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Monkey patch {model.__class__.__name__} attention layer")

        # Step 3: patch input for multimodal sequence parallelism
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ulysses_sp_size > 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            patch_vlm_for_ulysses_input_slicing(Qwen2VLTextModel)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif model.config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:
        # Step 1: patch model to support image-text mixed data
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLModel,
            Qwen3VLTextModel,
        )
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
            Qwen3VLMoeModel,
            Qwen3VLMoeTextModel,
        )

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.transformers.qwen3_vl import forward_with_normal_backend, qwen3_vl_base_forward

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Qwen3VLModel.forward = qwen3_vl_base_forward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Qwen3VLMoeModel.forward = qwen3_vl_base_forward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Qwen3VLForConditionalGeneration.forward = forward_with_normal_backend
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Qwen3VLMoeForConditionalGeneration.forward = forward_with_normal_backend
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Monkey patch {model.__class__.__name__} model forward")

        # Step 2: patch input for multimodal sequence parallelism
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ulysses_sp_size > 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            patch_vlm_for_ulysses_input_slicing(Qwen3VLTextModel)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            patch_vlm_for_ulysses_input_slicing(Qwen3VLMoeTextModel)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif model.config.model_type == "glm4v":
        # Step 1: patch model to support image-text mixed data

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from transformers.models.glm4v.modeling_glm4v import (
            Glm4vForConditionalGeneration,
            Glm4vModel,
            Glm4vTextAttention,
            Glm4vTextModel,
        )

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.transformers.glm4v import forward_with_normal_backend, glm4v_base_forward

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Glm4vModel.forward = glm4v_base_forward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Glm4vForConditionalGeneration.forward = forward_with_normal_backend
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Monkey patch {model.__class__.__name__} model forward")

        # Step 2: patch attention to support ulysses parallelism
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_remove_padding or ulysses_sp_size > 1:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.transformers.glm4v import glm4v_attn_forward

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            Glm4vTextAttention.forward = glm4v_attn_forward
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Monkey patch {model.__class__.__name__} attention layer")

        # Step 3: patch input for multimodal sequence parallelism
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ulysses_sp_size > 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            patch_vlm_for_ulysses_input_slicing(Glm4vTextModel)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif model.config.model_type == "kimi_vl":
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_remove_padding or ulysses_sp_size > 1:
            # TODO: Changes need to be made when transformers are adapted.
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.transformers.kimi_vl import _ulysses_flash_attn_forward

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            module.DeepseekV3FlashAttention2.forward = _ulysses_flash_attn_forward
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Monkey patch FlashAttention2.forward in KimiVL")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ulysses_sp_size > 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            patch_vlm_for_ulysses_input_slicing(module.DeepseekV3ForCausalLM)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_fused_kernels:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Not support fused kernels for KimiVL")

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if use_remove_padding or ulysses_sp_size > 1:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if hasattr(module, "_flash_attention_forward"):  # transformers <= 4.47.1 or legacy models
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            module._flash_attention_forward = _ulysses_flash_attention_forward
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Monkey patch _flash_attention_forward in {model.__module__}")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.integrations import flash_attention

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    patch_forward_with_backends(model, use_fused_kernels=use_fused_kernels, fused_kernels_backend=fused_kernels_backend)
