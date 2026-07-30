# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
# 【このファイルの役割（日本語補足）】
# - 1つのdata-parallel rank上で動くactor/ref policyのforwardとactor更新を担当する。
# 上位のRay workerはDataProtoをこのクラスへ渡し、このクラスが実際のTransformer forward、
# log-prob計算、各種lossの合成、backward、gradient clipping、optimizer stepを行う。
# - actorとreference/teacherで同じクラスを再利用する。actor_optimizerがNoneのインスタンスは
# 更新されないreference/teacherとして使われ、compute_log_probまたはcompute_topk_log_probだけを実行する。
# - Pure OPDで特に重要な経路は次の3つ。
# (1) compute_topk_log_prob(): teacher自身のtop-k token IDとlog-probをno_gradで計算する。
# (2) _forward_micro_batch(..., topk_ids=...): studentがteacherと同じtoken ID位置のlog-probを計算し、
# student側だけにgradientを残す。
# (3) update_policy(): pg_loss_coef=0のときPPO/GRPO lossを無効化し、teacher KLだけを
# scalar lossへ加えてactorを更新する。
# - remove-padding、dynamic batch、Ulysses sequence parallel、FSDP1/FSDP2、multimodal入力など、
# 複数の実行形態を同じコードで扱うため、Tensorの並び順とshapeの復元処理が重要である。
"""
Single Process Actor
"""

# dynamic batchで並べ替えたrow indexをflattenするために使用する。
import itertools
# 既存実装との互換性のため残る時間計測用標準モジュール。
import time
# rankごとの警告やGPU memory loggerへ渡すloggerを構築する。
import logging
# VERL_LOGGING_LEVEL環境変数の読み取りに使用する。
import os
from typing import Tuple

import torch
from torch import nn
# FSDP1の型判定と、shardを考慮したgradient clippingに使用する。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# entropyやlog-probなど、verl共通のTensor演算を利用する。
import verl.utils.torch_functional as verl_F
# Tensor batch、non-tensor batch、meta_infoをまとめて運ぶverlのデータ構造。
# actorで利用し得るloss集約・PPO/GSPO・KL関連関数をまとめてimportする。
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, agg_loss_with_sample_weights, compute_policy_loss, compute_policy_loss_gspo, kl_penalty, topk_kl_per_token
# mixed-task batchをtaskごとのrow maskへ分割し、診断metricを再集約する。
from verl.trainer.ppo.metric_utils import iter_task_row_masks
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
# FSDP2はFSDP1とgradient clipping APIが異なるため、専用helperを利用する。
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
# metric辞書へ同名metricをlist形式で蓄積する。
from verl.utils.py_functional import append_to_dict
# token数基準のdynamic micro-batch作成と、元row順を復元するindex計算に使う。
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
# Ulysses sequence parallelのpad、rank slice、all-gather、unpadを担当する。
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

# remove-padding経路はdeviceごとのFlashAttention padding utilityに依存する。
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        # actor_moduleはFSDP/FSDP2でwrapされたTransformerまたは通常のnn.Module。
        # actor_optimizer=Noneならreference/teacher用途で、更新経路は呼ばれない。
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        # padding tokenをforward対象から除外し、FlashAttention varlen形式で計算するかを保持する。
        # 有効token数が系列長より大幅に少ない場合、計算量とmemory trafficを削減できる。
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        # fused kernel経路ではmodel側がlog-prob/entropyを直接返す。
        # 一方、top-k KLはfull logitsへアクセスする必要があるためfused kernelと両立しない。
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        # Ulyssesはsequence次元を複数rankへ分割するsequence parallel方式。
        # size=1なら通常のdata parallel forwardとして扱う。
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        # entropy計算はshapeが実行ごとに変わり得るためdynamic=Trueでcompileする。
        # compile無効時は通常のPython関数をそのまま呼ぶ。
        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        # autocastへ渡すdevice type（cuda/npu等）を実行環境から決定する。
        self.device_name = get_device_name()

    # 1 micro-batchのTransformer forwardとresponse token上のlog-probを計算する。
    # 通常経路:
    # remove-padding経路:
    # -> 元の(B,S)へpad_inputで復元 -> response位置(B,R)だけを切り出す
    # top-k teacher mode (`topk_k`指定):
    # teacher自身のtop-k IDと、その位置のfull-vocab log-softmaxを返す。
    # top-k student mode (`topk_ids`指定):
    # teacherが選んだID位置のstudent log-softmaxを返す。student側Tensorはgradientを保持する。
    # entropy: (B,R)。calculate_entropy=FalseならNone。
    # log_probs: (B,R)。各response tokenとして実際に選ばれたtokenのlog-prob。
    # topk_out: None、teacher用tuple(logprob, ids)、またはstudent logprob(B,R,K)。
    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, topk_k=None, topk_ids=None) -> Tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_out: optional top-k output for distillation KL. None unless
                topk_k or topk_ids is given. When ``topk_k`` is set (teacher mode):
                a tuple (topk_logprob, topk_ids) each (bs, response_len, k), where
                topk_logprob are full-vocab log-softmax values at the model's own
                top-k ids. When ``topk_ids`` is given (student mode): a tensor
                (bs, response_len, k) of this model's full-vocab log-softmax values
                gathered at the provided ids (carries gradient).
        """
        topk_out = None
        # fused kernelはfull logitsを外へ返さないため、任意token ID位置をgatherするtop-k KLを計算できない。
        if (topk_k is not None or topk_ids is not None) and self.use_fused_kernels:
            raise NotImplementedError("top-k KL forward is not supported with fused kernels")
        # responsesは右padding済み(B,R)。Rは後でfull sequenceからresponse位置を切り出すために使う。
        response_length = micro_batch["responses"].size(-1)
        # VLM入力はrowごとのdictとしてnon-tensor側から届くため、keyごとにbatch次元へ連結する。
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        # model forwardはbf16 autocastで実行する。KL集約など数値精度が必要な部分は後段でfloat32へ変換する。
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                # Qwen2-VLのmRoPE position_idsは通常の(B,S)ではなく(B,4,S)。
                # modelが期待する(4,B,S)へchannel次元を先頭に移す。
                # input tokenと同じindicesでposition_idsもunpadし、rotary embeddingの位置対応を保つ。
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                # Ulysses SPではnnzをSP sizeで割り切れる長さへpadし、rankごとにsequence sliceを持つ。
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        # VLMはembedding後にsequence分割するため、ここではpadだけ行う。
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        # text-only modelはpad後、そのrankが担当するsequence sliceだけをforwardする。
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        # attention_mask=1のtokenだけを取り出す。
                        # indicesはflattenした(B*S)空間における有効token位置で、後のpad_input復元にも再利用する。
                        )
                    # label列もinputと同じSP分割にしないとlogitと教師tokenの位置がずれる。
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    # causal LMのlogit位置tは次token t+1を予測する。
                    # token列を1つ左へrollし、各logitと教師token IDを対応させる。
                    )

                # logprobs_from_logitsはlabelを1次元で受けるためbatch次元1を除去する。
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                # remove-padding時はattention_mask=Noneを渡し、model側のFlashAttention varlen経路を有効にする。
                extra_args = {}
                if self.use_fused_kernels:
                    # fused implementationは内部でtemperature適用とlog-prob/entropy計算まで行う。
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    # training/evaluation forwardなのでgeneration用KV cacheを作らない
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    # fused kernel出力は既にsampled-token log-probとentropyになっている。
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    # temperature Tはsoftmax前logitをlogit/Tへ変換する。
                    # Tが小さいほど分布が鋭く、大きいほど平坦になる。
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    # 通常はmemory節約のためlogprobs_from_logits内でlogitsをin-place利用できる。
                    # entropyまたはtop-k KLはこの後もfull logitsを読むため、その場合は破壊を禁止する。
                    inplace_backward = True
                    if calculate_entropy or topk_k is not None or topk_ids is not None:
                        # top-k KL reads logits_rmpad after this call, so don't mutate it.
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # entropyは各token位置のfull vocabulary分布から計算する。
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    # 各SP rankが持つsequence sliceをgatherし、Ulysses用paddingを除去して元nnz順へ戻す。
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                # unpad時に保存したindicesを使い、nnz列を元の(B,S)へscatterする。
                # logit at position t predicts token t+1なので、response tokenのlog-probは
                # full sequence末尾R tokenそのものではなく[-R-1:-1]のlogit位置にある。
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                # top-k distillation log-probs (teacher: own top-k; student: gather at given ids)
                # top-k distillationはteacherとstudentで役割が異なる。
                # teacher: 自身のtop-k token IDとそのlog-probを作る。
                # student: teacherから渡された同じID位置のlog-probをgatherし、gradientを保持する。
                if topk_k is not None or topk_ids is not None:
                    # Ulyssesではfull vocabulary logitsとglobal sequence位置の対応をこの実装が未対応。
                    if self.use_fused_kernels or self.use_ulysses_sp:
                        raise NotImplementedError("top-k KL forward is not supported with fused kernels or ulysses SP")
                    # logsumexpを一度計算し、任意token logit - logsumexpでfull-vocab log-softmaxを得る。
                    lse = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)  # (total_nnz, 1)
                    if topk_k is not None:
                        # teacher mode: 各token位置でteacher logit上位K個を選ぶ。
                        tvals, tids = torch.topk(logits_rmpad, k=topk_k, dim=-1)  # (total_nnz, k)
                        # Use float32 for pad_input: bf16 cannot represent vocab ids
                        # (>256) exactly, and float32 keeps log-probs precise.
                        # bf16は大きなvocab IDを整数として正確に表現できない。
                        # pad_inputがfloating tensorを想定するため一時的にfloat32化し、復元後round+longへ戻す。
                        t_lp_rmpad = (tvals - lse).float()
                        full_t_lp = pad_input(t_lp_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                        full_t_id = pad_input(tids.float(), indices=indices, batch=batch_size, seqlen=seqlen)
                        topk_out = (
                            full_t_lp[:, -response_length - 1 : -1, :],
                            full_t_id[:, -response_length - 1 : -1, :].round().long(),
                        )
                    else:
                        # student mode: teacher_topk_ids(B,R,K)をfull sequence位置へ埋め戻し、
                        # unpadded logits_rmpadと同じnnz順へ変換してgatherする。
                        k = topk_ids.size(-1)
                        full_ids = torch.zeros((batch_size, seqlen, k), dtype=torch.long, device=logits_rmpad.device)
                        full_ids[:, -response_length - 1 : -1, :] = topk_ids
                        ids_rmpad = index_first_axis(rearrange(full_ids, "b s k -> (b s) k"), indices)  # (total_nnz, k)
                        s_lp_rmpad = (logits_rmpad.gather(-1, ids_rmpad) - lse).float()  # (total_nnz, k), keeps grad
                        full_s_lp = pad_input(s_lp_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                        topk_out = full_s_lp[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                # attention_maskをmodelへ渡し、paddingを含む(B,S)のままforwardする。
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                # gatherはstudent logitsへの微分可能なindex選択であり、student側にgradientが流れる。
                # 長さの近いrowを再配置し、各micro-batchの総token数が上限内になるよう分割する。
                # indicesは並べ替え後rowが元batchのどのrowかを表す。
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    # fused出力からresponseを予測するlogit位置[-R-1:-1]だけを切り出す。
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                # paddingを保持する通常経路。Ulysses SPはremove-padding経路でのみ扱う。
                else:
                    logits = output.logits

                    logits.div_(temperature)
                    # prompt位置のlogitsを捨て、response tokenに対応するR位置だけを保持する。
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                    if topk_k is not None or topk_ids is not None:
                        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # (bsz, response_length, 1)
                        if topk_k is not None:
                            # teacher mode: teacher自身のtop-k supportを作る。
                            tvals, tids = torch.topk(logits, k=topk_k, dim=-1)
                            topk_out = ((tvals - lse).float(), tids.long())
                        else:
                            # student mode: teacherが選んだ(B,R,K)のID位置をgatherする。
                            topk_out = (logits.gather(-1, topk_ids) - lse).float()

            return entropy, log_probs, topk_out

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        # shardされたparameter全体のnormを正しく求めるため、wrapper種別ごとの専用APIを使う。
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        # NaN/Inf gradientでparameterを破壊しないよう、このstepのoptimizer更新をskipする。
        # zero_gradにより異常gradientを次mini-batchへ持ち越さない。
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    # 固定されたactor/ref/teacherに対して、入力responseのtoken log-probをno_gradで計算する。
    # micro_batch_size: 固定micro-batch時のrow数
    # temperature: rolloutと同じ確率分布を再現するtemperature
    # use_dynamic_bsz: row数ではなく総token数でmicro-batchを構成するか
    # max_token_len: dynamic batch 1組あたりのtoken上限
    # entropys: calculate_entropy=Trueなら(B,R)、それ以外はNone
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
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
        # dropout等を無効化し、同じinputに対する確定的なlog-prob評価へ切り替える。
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # forwardに必要なTensorだけを選択し、不要なreward/metric等をGPUへ送らない。
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            # multimodal rowはnon-tensor入力とTensorを同じchunk境界で分割する必要がある。
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        # Ulysses SPでは各rankがsequence sliceを持つため、許容token数をSP size分拡張する。
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            # multimodal chunkはDataProtoで届くため、forwardが扱うdictへ展開する。
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            # 旧方策/ref/teacher評価ではparameter gradientは不要。
            # no_gradによりactivation保存を避け、GPU memoryを削減する。
            with torch.no_grad():
                entropy, log_probs, _ = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        # micro-batch順に連結する。dynamic bsz時点では元batch順とは限らない。
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            # nested indicesをflattenし、rearrange前のglobal row順へ戻すinverse permutationを作る。
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    # teacher/ref側で、各response tokenに対するteacher自身のtop-k supportを計算する。
    # topk_logprob: (B,R,K)。teacher full-vocab log-softmaxのtop-k位置。
    # topk_ids: (B,R,K), int64。teacherが選んだvocabulary ID。
    # この関数全体はno_gradであり、返されたteacher Tensorからstudent parameterへの
    # gradient edgeは作られない。studentは後で同じtopk_ids位置のlog-probを別forwardで計算する。
    def compute_topk_log_prob(self, data: DataProto, topk_k: int):
        """Teacher-side: per response token, the teacher's top-k token ids and the
        teacher's full-vocab log-softmax values at those ids.

        Returns:
            topk_logprob: (bs, response_length, k)
            topk_ids:     (bs, response_length, k) int64
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # 現実装はmultimodal top-k KLのID対応とpadding復元を実装していないため明示的に拒否する。
        assert not has_multi_modal_inputs, "top-k KL is not supported for multi-modal inputs"

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        topk_logprob_lst = []
        topk_ids_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                _, _, topk_out = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=False, topk_k=topk_k)
            tlp, tids = topk_out
            topk_logprob_lst.append(tlp)
            topk_ids_lst.append(tids)

        topk_logprob = torch.concat(topk_logprob_lst, dim=0)
        topk_ids = torch.concat(topk_ids_lst, dim=0)
        if use_dynamic_bsz:
            # teacher signalもstudent batchのglobal row順と一致させる必要があるため、両Tensorを同じinverse permutationで戻す。
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == topk_logprob.size(0), f"{len(indices)} vs. {topk_logprob.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            topk_logprob = topk_logprob[revert_indices]
            topk_ids = topk_ids[revert_indices]

        return topk_logprob, topk_ids

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        # pg_loss_coef=0はpolicy-gradient項を完全に無効化する。
        pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
        use_teacher_kl_loss = self.config.get("use_teacher_kl_loss", False)
        teacher_kl_loss_type = self.config.get("teacher_kl_loss_type", "low_var_kl")
        teacher_topk_kl = use_teacher_kl_loss and teacher_kl_loss_type == "topk_kl"
        # Exact token-mean under dynamic bsz: scale each micro-batch by its valid-token
        # share of the mini-batch (token-weighted) instead of by sample count, so the
        # objective is grouping-invariant and matches the true global token-mean.
        # dynamic micro-batchはrow数が不均一になる。
        # token-mean objectiveを正確に再現する場合、各micro-batch lossをrow数ではなく有効token数比で重み付けする。
        dynamic_bsz_token_scale = (
            self.config.use_dynamic_bsz
            and self.config.get("dynamic_bsz_token_scale", False)
            and self.config.loss_agg_mode == "token-mean"
        )
        # どのloss経路を有効にするかに応じて必要Tensorだけを選ぶ。
        # これによりPure OPDでは存在しないadvantages/old_log_probsを誤って要求しない。
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        # advantages / old_log_probs are only needed by the policy-gradient (and SDL) paths.
        # Pure teacher-KL distillation (pg_loss_coef==0) does not produce them, so don't require them.
        if pg_loss_coef != 0:
            # PPO/GRPO policy lossには旧方策log-probとadvantageが必要。
            select_keys += ["old_log_probs", "advantages"]
        elif self.config.get("use_sdl_loss", False):
            # SDLはpolicy-gradientを使わなくてもold_log_probsをweight計算へ利用する。
            select_keys.append("old_log_probs")
        if multi_turn:
            # multi-turnでは生成済みtoken全体ではなく、学習対象turnを示すloss_maskを使う。
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            # 通常のreference-policy KL regularization用signal。
            select_keys.append("ref_log_prob")
            if "kl_loss_coef" in data.batch:
                # multitask等でsampleごとのKL係数を使う場合のみ追加する。
                select_keys.append("kl_loss_coef")
        if self.config.get("use_sdl_loss", False) or self.config.get("use_sdar_loss", False) or (use_teacher_kl_loss and not teacher_topk_kl):
            # sampled-token型teacher lossはteacher_log_probs(B,R)を使う。
            select_keys.append("teacher_log_probs")
        if teacher_topk_kl:
            # dense top-k KLではteacherのsupport IDとそのlog-probを両方必要とする。
            select_keys += ["teacher_topk_logprobs", "teacher_topk_ids"]
        # Multitask runs tag every row with its task id (see RayPPOTrainer._attach_task_ids)
        # so the loss metrics below can also be reported per task. Absent in single-task runs.
        # task_idsはloss自体を変えず、同じTensorをtask rowごとに再集約した診断metric用。
        task_id_names = data.meta_info.get("task_id_names", None)
        if "task_ids" in data.batch.keys():
            select_keys.append("task_ids")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # PPO mini-batchは1 optimizer stepで処理する論理単位。
        # micro-batchはそのmini-batchをGPU memoryへ収めるために細分化した単位。
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        # 同じrollout batchをppo_epochs回再利用する。Pure OPDでも設定値に従って複数epoch更新し得る。
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                # Total valid response tokens in this mini-batch (for exact token-mean
                # scaling under dynamic bsz). Same mask rule as the per-micro loss below.
                # exact token-mean scaleの分母として、mini-batch全体の有効response token数を一度だけ数える。
                minibatch_valid_tokens = None
                if dynamic_bsz_token_scale:
                    _resp_len = mini_batch["responses"].size(1)
                    _mb_mask = mini_batch["loss_mask"][:, -_resp_len:] if multi_turn else mini_batch["attention_mask"][:, -_resp_len:]
                    minibatch_valid_tokens = float(_mb_mask.sum().clamp(min=1))
                # 全maskが0でも0除算しないよう最低1へclampする。
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                # row数ではなくtoken総数で分割するため、各micro-batchのBは異なり得る。
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                # mini-batch開始時にのみgradientを消去し、各micro-batchのbackwardを同じparameter.gradへ蓄積する。
                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    # DataProtoの場合はTensorを現在deviceへ移し、non-tensor項目と1 dictへ統合する。
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        # parameter offload時はactorがCPU上に戻っている場合があるため、forward直前に現在deviceへ移す。
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    task_ids = data.get("task_ids", None) if task_id_names else None
                    # loss計算対象tokenを(B,R) maskとして切り出す。
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    # PPO/GSPO clippingとentropy、loss集約方式をconfigから読む。
                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    # entropy係数が0ならfull-vocabulary entropyを計算せずforward costを削減する。
                    # top-k KLではteacher support上のstudent log-probを同じforwardから得る。
                    fwd_topk_ids = data["teacher_topk_ids"] if teacher_topk_kl else None
                    entropy, log_prob, student_topk_logprobs = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, topk_ids=fwd_topk_ids)
                    
                    # vanilla PPO/GRPO token lossまたはGSPO sequence lossを選択する。
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    if pg_loss_coef != 0:
                        old_log_prob = data["old_log_probs"]
                        advantages = data["advantages"]
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    else:
                        # Pure teacher-KL distillation: no policy-gradient signal.
                        # Pure OPDではpolicy-gradient signalを作らない。
                        # metric schemaを共通化するため、PPO関連値はdevice/dtypeを合わせたscalar zeroにする。
                        old_log_prob = data.get("old_log_probs", None)
                        zero = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
                        pg_loss = pg_clipfrac = ppo_kl = pg_clipfrac_lower = zero

                    # entropy bonusは最大化したいため、最小化するpolicy_lossには負号で加える。
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss * pg_loss_coef

                    if self.config.use_kl_loss:
                        # 通常の固定reference policyに対するKL正則化。
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        kl_loss_coef = data.get("kl_loss_coef", None)
                        if kl_loss_coef is None:
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        else:
                            # task別などsampleごとの係数(B,)をtoken lossへbroadcastして集約する。
                            weighted_kl_loss = agg_loss_with_sample_weights(
                                loss_mat=kld,
                                # paddingおよび学習対象外turnをmaskし、設定された方式でscalarへ集約する。
                                loss_mask=response_mask,
                                sample_weights=kl_loss_coef,
                                loss_agg_mode=loss_agg_mode,
                            # 全sample共通のscalar係数。
                            )
                            policy_loss = policy_loss + weighted_kl_loss
                            metrics["actor/kl_coef"] = kl_loss_coef.float().mean().detach().item()
                        metrics["actor/kl_loss"] = kl_loss.detach().item()

                    if self.config.get("use_sdl_loss", False):
                        # SkillSD: 元のGRPO advantageは保ちつつ、teacher signalを補助lossとして加える。
                        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdl_loss = compute_sdl_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            old_log_probs=old_log_prob,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdl_coef = self.config.get("sdl_loss_coef", 0.1)
                        policy_loss = policy_loss + sdl_loss * sdl_coef
                        metrics["actor/sdl_loss"] = sdl_loss.detach().item()
                        metrics["actor/sdl_coef"] = sdl_coef

                    if self.config.get("use_sdar_loss", False):
                        # SDAR: teacher-student gapをgateへ使う専用蒸留loss。
                        from verl.trainer.ppo.sdar_utils import compute_sdar_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdar_loss, sdar_metrics = compute_sdar_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            gate_beta=self.config.get("sdar_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdar_coef = self.config.get("sdar_loss_coef", 0.1)
                        policy_loss = policy_loss + sdar_loss * sdar_coef
                        metrics.update(sdar_metrics)
                        metrics["sdar/coef"] = sdar_coef

                    if use_teacher_kl_loss:
                        # On-policy distillation: KL between student and a (per-task) teacher,
                        # evaluated on the student's own on-policy responses. Only the student
                        # log-probs carry gradients; teacher values are detached upstream.
                        # OPDはstudent自身が生成したon-policy response上で、task対応teacherへ近づける。
                        # teacher signalは上流でdetach/no_grad済みで、student log-probだけがgradientを持つ。
                        if teacher_topk_kl:
                            # Dense reverse KL over the teacher's top-k support (+ tail bucket).
                            # student/teacherは同じteacher top-k support上の(B,R,K) log-prob。
                            # K個のtokenにtop-k外をまとめたtail bucketを加えたdense reverse KLを計算する。
                            teacher_kld = topk_kl_per_token(
                                student_topk_logprob=student_topk_logprobs,
                                teacher_topk_logprob=data["teacher_topk_logprobs"],
                            )
                        else:
                            # Single-sampled-token estimator (low_var_kl / kl / mse / abs).
                            # low_var_kl/kl/mse/abs等は実際にsampleされたtoken位置のscalar log-probを比較する。
                            teacher_kld = kl_penalty(
                                logprob=log_prob,
                                ref_logprob=data["teacher_log_probs"],
                                kl_penalty=teacher_kl_loss_type,
                            )
                        teacher_kl_loss = agg_loss(loss_mat=teacher_kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        teacher_kl_coef = self.config.get("teacher_kl_loss_coef", 1.0)
                        policy_loss = policy_loss + teacher_kl_loss * teacher_kl_coef
                        metrics["actor/teacher_kl_loss"] = teacher_kl_loss.detach().item()
                        metrics["actor/teacher_kl_coef"] = teacher_kl_coef

                    # micro-batch lossをmini-batch全体のobjectiveへ正規化する。
                    if self.config.use_dynamic_bsz:
                        if minibatch_valid_tokens is not None:
                            # Exact global token-mean: policy_loss is a token-mean
                            # (denominator = this micro-batch's valid tokens), so
                            # policy_loss * micro_valid_tokens = token-sum, and dividing
                            # by the mini-batch's total valid tokens makes the per-token
                            # weight independent of how tokens were grouped into
                            # micro-batches (unlike the sample-count factor below).
                            # policy_lossはmicro-batch内token mean。
                            # valid token数を掛ければtoken sumになり、mini-batch総token数で割れば
                            # micro-batchのグルーピングに依存しない正確なglobal token meanになる。
                            micro_valid_tokens = response_mask.sum()
                            loss = policy_loss * (micro_valid_tokens / minibatch_valid_tokens)
                        # 従来互換のsample数比。token長が不均一なら厳密なtoken meanとは一致しない。
                        else:
                            # relative to the dynamic bsz (sample-count reweighting)
                            loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        # 固定micro-batchでは各backwardを1/gradient_accumulationにし、
                        # 全micro-batchのgradient和をmini-batch mean相当にする。
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    if task_ids is not None:
                        # Same losses, re-aggregated over the rows of one task at a
                        # time. Diagnostics only: nothing here touches the graph the
                        # optimizer step above was built from.
                        # ここからは診断専用。既にbackwardしたgraphやoptimizer updateを変更しない。
                        # 同じloss Tensorをtask rowごとに再計算し、multitask runの偏りを可視化する。
                        with torch.no_grad():
                            for task, rows in iter_task_row_masks(task_ids, task_id_names):
                                task_response_mask = response_mask[rows]
                                task_metrics = {}

                                if pg_loss_coef != 0:
                                    task_pg_loss, task_pg_clipfrac, task_ppo_kl, task_pg_clipfrac_lower = policy_loss_fn(
                                        old_log_prob=old_log_prob[rows],
                                        log_prob=log_prob[rows],
                                        advantages=advantages[rows],
                                        response_mask=task_response_mask,
                                        cliprange=clip_ratio,
                                        cliprange_low=clip_ratio_low,
                                        cliprange_high=clip_ratio_high,
                                        clip_ratio_c=clip_ratio_c,
                                        loss_agg_mode=loss_agg_mode,
                                    )
                                else:
                                    task_pg_loss = task_pg_clipfrac = task_ppo_kl = task_pg_clipfrac_lower = zero
                                task_metrics[f"actor/pg_loss/{task}"] = task_pg_loss.detach().item()
                                task_metrics[f"actor/pg_clipfrac/{task}"] = task_pg_clipfrac.detach().item()
                                task_metrics[f"actor/ppo_kl/{task}"] = task_ppo_kl.detach().item()
                                task_metrics[f"actor/pg_clipfrac_lower/{task}"] = task_pg_clipfrac_lower.detach().item()

                                if entropy_coeff != 0:
                                    task_metrics[f"actor/entropy_loss/{task}"] = (
                                        agg_loss(loss_mat=entropy[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )

                                if self.config.use_kl_loss:
                                    task_metrics[f"actor/kl_loss/{task}"] = (
                                        agg_loss(loss_mat=kld[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )
                                    if kl_loss_coef is not None:
                                        task_metrics[f"actor/kl_coef/{task}"] = kl_loss_coef[rows].float().mean().detach().item()

                                if self.config.get("use_sdl_loss", False):
                                    from verl.trainer.ppo.skillsd_utils import compute_sdl_loss

                                    task_metrics[f"actor/sdl_loss/{task}"] = compute_sdl_loss(
                                        student_log_probs=log_prob[rows],
                                        teacher_log_probs=teacher_log_probs[rows],
                                        old_log_probs=old_log_prob[rows],
                                        response_mask=task_response_mask,
                                        loss_agg_mode=loss_agg_mode,
                                    ).detach().item()

                                if self.config.get("use_sdar_loss", False):
                                    from verl.trainer.ppo.sdar_utils import compute_sdar_loss

                                    _, task_sdar_metrics = compute_sdar_loss(
                                        student_log_probs=log_prob[rows],
                                        teacher_log_probs=teacher_log_probs[rows],
                                        response_mask=task_response_mask,
                                        gate_beta=self.config.get("sdar_gate_beta", 5.0),
                                        loss_agg_mode=loss_agg_mode,
                                    )
                                    task_metrics.update({f"{name}/{task}": value for name, value in task_sdar_metrics.items()})

                                if use_teacher_kl_loss:
                                    task_metrics[f"actor/teacher_kl_loss/{task}"] = (
                                        agg_loss(loss_mat=teacher_kld[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )

                                append_to_dict(metrics, task_metrics)

                    data = {
                        # 全体metricもmicro-batchごとにlistへ蓄積し、上位worker/trainer側でreduceする。
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                # 全micro-batchのgradientが蓄積された後に1回だけclipとoptimizer stepを行う。
                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        # 呼び出し終了時にgradient bufferを空にし、次のupdate_policy呼出しへ状態を残さない。
        self.actor_optimizer.zero_grad()
        return metrics
