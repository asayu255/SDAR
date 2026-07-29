# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# there is some bug in mcore 0.12, so we need to patch it
# 1. `get_query_key_value_tensors` in `multi_latent_attention.py` works wrong when packed_seq_params is not None


# [EXPLAIN] `apply_patch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def apply_patch():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import torch
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention, apply_rotary_pos_emb, deprecate_inference_params, gather_from_sequence_parallel_region, gather_from_tensor_model_parallel_region, scatter_to_sequence_parallel_region
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import parallel_state, tensor_parallel

    # [EXPLAIN] `patch_get_query_key_value_tensors` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def patch_get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(inference_context, None, hidden_states, self.config, packed_seq_params)

        # rotary_pos_emb:[s, b, 1, 64]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mscale = 1.0
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.rope_type == "rope":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len)

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if q_compressed.size(-1) != self.config.q_lora_rank:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.config.sequence_parallel:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_compressed = self.q_layernorm(q_compressed)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_compressed = hidden_states

        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv_compressed, k_pos_emb = torch.split(kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv_compressed, k_pos_emb = torch.split(kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================
        # [EXPLAIN] `qkv_up_proj_and_rope_apply` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.q_lora_rank is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q, _ = self.linear_q_up_proj(q_compressed)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # hidden_states:[s, b, 2048], q: [s, b, n * 192]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q, _ = self.linear_q_proj(q_compressed)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_len, bsz, _ = q.size()

            # q: [s, b, n, 192]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

            # kv: [s, b, 2048]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            # kv: [s, b, n, 256]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv = kv.view(
                q_len,
                bsz,
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if inference_context is not None:
                # add offset to the sequence start for inference
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                sequence_start = inference_context.sequence_len_offset
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                sequence_end = sequence_start + q_len
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # Shorten rotary_pos_emb to the sequence length when inference_params
                # is not provided. This makes sure we can run forward directly with
                # any sequence length. During training, the sequence length is always
                # the full rotary_pos_emb length.
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rotary_pos_emb = rotary_pos_emb[0:q_len]

            # [s, b, 64] -> [s, b, 1, 64]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            k_pos_emb = torch.unsqueeze(k_pos_emb, 2)

            # q: [s, b, n, 128], q_pos_emb: [s, b, n, 64]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

            # k_no_pe: [s, b, n, 128], value: [s, b, n, 128]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if packed_seq_params is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q_pos_emb = q_pos_emb.squeeze(1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                k_pos_emb = k_pos_emb.squeeze(1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q_no_pe = q_no_pe.squeeze(1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                k_no_pe = k_no_pe.squeeze(1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                value = value.squeeze(1)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cu_seqlens_q = cu_seqlens_kv = None

            # q_pos_emb: [s, b, n, 64], k_pos_emb:[s, b, 1, 64]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=mscale,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
            )

            # query: [s, b, n, 192]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if packed_seq_params is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # key: [s, b, n, 192]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            query = query.contiguous()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            key = key.contiguous()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            value = value.contiguous()
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return query, key, value

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.recompute_up_proj:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            query, key, value = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return query, key, value

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    MLASelfAttention.get_query_key_value_tensors = patch_get_query_key_value_tensors
