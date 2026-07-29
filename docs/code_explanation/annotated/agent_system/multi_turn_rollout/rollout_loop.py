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
import copy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import threading
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from concurrent.futures import ThreadPoolExecutor

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils import gpu_profiler
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.dataset.rl_dataset import collate_fn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.torch_functional as verl_F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PreTrainedTokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import uuid
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.environments import EnvironmentManagerBase
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List, Dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# Opt-in per-turn rollout timing. When off (default) the loop is unchanged and
# adds only a few cheap perf_counter() reads. Set ROLLOUT_TURN_TIMING=1 to print,
# at the end of every rollout, a per-turn breakdown of where the gen phase spends
# wall time (preproc / generate / decode / env.step) plus the GPU SM utilization
# measured *during* generate_sequences (GEN-UTIL). Pairs with GPU_PROFILER=1.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ROLLOUT_TURN_TIMING = os.environ.get("ROLLOUT_TURN_TIMING", "0").strip().lower() in ("1", "true", "yes", "on")

# Accuracy-safe throughput optimization: skip the prompt tokenization for already
# finished trajectories. Finished rows are excluded from generation
# (batch_input[active_idx]) and dropped by gather_rollout_data() (active_masks=
# False), so their tokens are never consumed. Active rows go through the
# *unchanged* preprocess_single_sample(), so the generation input — and every
# training number — is byte-for-byte identical to the un-optimized path.
# NOTE: an earlier "measured neutral" conclusion was invalid — that run had not
# pulled this code, so the optimization was never actually exercised. Default ON;
# set ROLLOUT_SKIP_DONE_PREPROC=0 to restore full-batch preprocessing for A/B.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ROLLOUT_SKIP_DONE_PREPROC = os.environ.get("ROLLOUT_SKIP_DONE_PREPROC", "1").strip().lower() in ("1", "true", "yes", "on")

# Keep vLLM awake for the whole rollout instead of waking/sleeping (and re-syncing
# the frozen actor weights) every turn. See ActorRolloutRefWorker.begin_rollout_
# session(). Accuracy-safe: identical frozen weights are used on every turn, so the
# generated tokens are unchanged; only the per-turn weight-sync / wake / sleep
# overhead is removed (and vLLM's prefix cache can persist across turns). Opt-in;
# OFF reproduces the current per-turn behavior exactly.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ROLLOUT_KEEP_VLLM_AWAKE = os.environ.get("ROLLOUT_KEEP_VLLM_AWAKE", "0").strip().lower() in ("1", "true", "yes", "on")

# Parallelize the per-row prompt tokenization (apply_chat_template + tokenize) of
# preprocess_batch across a thread pool. HF fast tokenizers release the GIL in
# their Rust encode path, and every row is processed by an *identical, independent*
# call, so the outputs are byte-for-byte the same as the sequential loop — only
# wall time changes. Each worker thread uses its own deepcopy of the tokenizer
# because HF fast tokenizers mutate internal truncation/padding state per call and
# are not safe to share across threads. 0 (default) keeps the sequential loop;
# multimodal batches always fall back to sequential (processor thread-safety is
# not established).
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ROLLOUT_PREPROC_WORKERS = int(os.environ.get("ROLLOUT_PREPROC_WORKERS", "0"))

# Decode only the rows that were actually generated this turn. Finished rows'
# scattered filler is all pad tokens, which batch_decode(skip_special_tokens=True)
# already renders as '' — so filling '' directly gives the same env input while
# skipping the wasted decode of pad-only rows (2/3 of the batch during the
# alfworld tail). Default ON; set ROLLOUT_DECODE_ACTIVE_ONLY=0 for A/B.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ROLLOUT_DECODE_ACTIVE_ONLY = os.environ.get("ROLLOUT_DECODE_ACTIVE_ONLY", "1").strip().lower() in ("1", "true", "yes", "on")

# Do not record per-turn rows for already-finished trajectories. Those rows carry
# active_masks=False and are dropped by gather_rollout_data() anyway; because a
# trajectory's active rows always form a prefix of its turn list, the enumerate-
# based turn_step and the "last active entry" scans in success_evaluator /
# filter_group_data see identical data. Saves the per-turn dict materialization
# and memory for the inactive 2/3 of the batch during the multitask tail.
# Default ON; set ROLLOUT_COMPACT_RECORD=0 for A/B.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ROLLOUT_COMPACT_RECORD = os.environ.get("ROLLOUT_COMPACT_RECORD", "1").strip().lower() in ("1", "true", "yes", "on")

# Prefetch old_log_prob for trajectories that already finished, overlapped with
# env.step: each turn, envs.step() (CPU/HTTP/IPC — GPU idle) runs in a background
# thread while the driver issues actor_rollout_wg.compute_log_prob() on a bounded
# chunk of finished-trajectory rows. The actor weights are frozen for the whole
# rollout and are exactly the weights the trainer's old_log_prob phase would use
# after it, so the prefetched values are computed by the same function on the
# same weights and the same rows — only earlier. Accuracy class: same standard as
# vLLM batch-composition changes (the micro-batch grouping differs from the
# monolithic phase; per-row results are computed independently under rmpad).
# Rows not prefetched by rollout end are computed by the trainer as usual.
# Opt-in; OFF reproduces the current serial behavior exactly.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ROLLOUT_PREFETCH_LOGPROB = os.environ.get("ROLLOUT_PREFETCH_LOGPROB", "0").strip().lower() in ("1", "true", "yes", "on")

# Rows per prefetch call. Sized so one compute_log_prob chunk roughly fits inside
# one env.step window; leftovers roll over to later turns (the alfworld tail has
# ~35 such windows). Too large a chunk extends the turn past env.step — the work
# is still subtracted from the old_log_prob phase, but the overlap is lost.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ROLLOUT_PREFETCH_LOGPROB_CHUNK = int(os.environ.get("ROLLOUT_PREFETCH_LOGPROB_CHUNK", "64"))


# [EXPLAIN] `_env_kwargs_equal` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _env_kwargs_equal(a, b) -> bool:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Compare two env_kwargs sequences (None or sequence of per-env dicts)."""
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if a is None or b is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return a is None and b is None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    a = list(a) if not isinstance(a, list) else a
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    b = list(b) if not isinstance(b, list) else b
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(a) != len(b):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for item_a, item_b in zip(a, b):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if item_a is item_b:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(item_a, dict) or not isinstance(item_b, dict) or item_a.keys() != item_b.keys():
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return False
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in item_a:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            va, vb = item_a[key], item_b[key]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not np.array_equal(np.asarray(va), np.asarray(vb)):
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return False
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif va != vb:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return False
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return True


# [EXPLAIN] `_now` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _now():
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return time.perf_counter()


# [EXPLAIN] `_fmt_per_gpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _fmt_per_gpu(vals):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not vals:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "-"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return "/".join(f"{v:.0f}" if v is not None else "-" for v in vals)


# [EXPLAIN] `_print_turn_timing` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _print_turn_timing(records):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Pretty-print the per-turn breakdown collected during one rollout.

    The perGPU% column shows per-GPU SM util during the turn's generation; the
    spread between GPUs reveals data-parallel load imbalance from mixing tasks
    across the DP split. The DP-IMBALANCE summary line tracks it (the interleaved
    task layout, TASK_BALANCE_INTERLEAVE, shrinks it on the mixed-task turns).
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not records:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    header = (
        f"{'turn':>4}{'active':>8}{'preproc':>9}{'gen':>9}{'decode':>9}"
        f"{'envstep':>9}{'total':>9}{'genGPU%':>9}{'  perGPU%':>12}"
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lines = ["[rollout-turn-timing] per-turn breakdown (seconds); GPU busy only during 'gen'", header, "-" * len(header)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tot = {k: 0.0 for k in ("preproc", "gen", "decode", "envstep", "total")}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_util, shrunk_util = [], []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp_spreads = []  # per-turn max-min across GPUs during gen (DP imbalance)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    first_active = records[0]["active"]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for r in records:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total = r["preproc"] + r["gen"] + r["decode"] + r["envstep"]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k in ("preproc", "gen", "decode", "envstep"):
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            tot[k] += r[k]
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        tot["total"] += total
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gu = r["gen_util"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gu_s = f"{gu:.0f}" if gu is not None else "-"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pg = r.get("gen_util_per_gpu")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pg_s = _fmt_per_gpu(pg)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        lines.append(
            f"{r['turn']:>4}{r['active']:>8}{r['preproc']:>9.2f}{r['gen']:>9.2f}"
            f"{r['decode']:>9.2f}{r['envstep']:>9.2f}{total:>9.2f}{gu_s:>9}{pg_s:>12}"
        )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if gu is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (full_util if r["active"] >= first_active else shrunk_util).append(gu)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pg:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            present = [v for v in pg if v is not None]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(present) >= 2:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                dp_spreads.append(max(present) - min(present))
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    lines.append("-" * len(header))
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    lines.append(
        f"{'TOTAL':>4}{'':>8}{tot['preproc']:>9.1f}{tot['gen']:>9.1f}"
        f"{tot['decode']:>9.1f}{tot['envstep']:>9.1f}{tot['total']:>9.1f}"
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cpu_glue = tot["preproc"] + tot["decode"] + tot["envstep"]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if tot["total"] > 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        lines.append(
            f"SHARE  gen(GPU-busy)={100*tot['gen']/tot['total']:.1f}%  "
            f"cpu-glue(preproc+decode+envstep, GPU-idle)={100*cpu_glue/tot['total']:.1f}%"
        )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fa = sum(full_util) / len(full_util) if full_util else None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sa = sum(shrunk_util) / len(shrunk_util) if shrunk_util else None
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    lines.append(
        f"GEN-UTIL  full_batch(active>={first_active})="
        f"{f'{fa:.0f}%' if fa is not None else '-'}   "
        f"shrunk(active<{first_active})={f'{sa:.0f}%' if sa is not None else '-'}   "
        f"(full~shrunk => not batch-underfed; full>>shrunk => alfworld plateau underfeed)"
    )
    # data-parallel imbalance (lower is better; the interleaved task layout
    # shrinks it on mixed-task turns)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dp_spreads:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mean_spread = sum(dp_spreads) / len(dp_spreads)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        lines.append(
            f"DP-IMBALANCE  mean |maxGPU-minGPU| during gen = {mean_spread:.1f} pp "
            f"(lower=better; TASK_BALANCE_INTERLEAVE shrinks this on mixed turns)"
        )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("\n".join(lines), flush=True)


# [EXPLAIN] `TrajectoryCollector` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TrajectoryCollector:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.processor = processor
        # ROLLOUT_PREPROC_WORKERS: lazily-built thread pool + per-thread tokenizer
        # clones (HF fast tokenizers are not safe to share across threads).
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._preproc_executor = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._preproc_local = threading.local()
        # ROLLOUT_PREFETCH_LOGPROB state: rows of finished trajectories waiting for
        # a prefetched compute_log_prob, and the per-row results keyed by
        # (traj_uid, turn_step). Cleared at the start of every multi_turn_loop.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._logprob_prefetch_enabled = False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._logprob_pending = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._prefetched_log_probs = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._env_step_executor = None
        # ENV_RESET_PREFETCH state: one outstanding background envs.reset, launched
        # by the trainer between rollouts (see prefetch_env_reset).
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._env_reset_prefetch = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._env_reset_executor = None

    # [EXPLAIN] `_get_preproc_executor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_preproc_executor(self) -> ThreadPoolExecutor:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._preproc_executor is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._preproc_executor = ThreadPoolExecutor(max_workers=_ROLLOUT_PREPROC_WORKERS)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._preproc_executor

    # [EXPLAIN] `_thread_tokenizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _thread_tokenizer(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer = getattr(self._preproc_local, "tokenizer", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tokenizer is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tokenizer = copy.deepcopy(self.tokenizer)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._preproc_local.tokenizer = tokenizer
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tokenizer

    # [EXPLAIN] `_run_full_preprocess` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _run_full_preprocess(self, items: List[int], gen_batch: DataProto, obs: Dict) -> Dict[int, dict]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Run preprocess_single_sample over `items`, optionally in parallel.

        The parallel path calls the *same* function with per-thread tokenizer
        clones, so every row's output is identical to the sequential loop.
        Multimodal batches always use the sequential path.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_threads = (
            _ROLLOUT_PREPROC_WORKERS > 1
            and len(items) > 1
            and obs.get('image', None) is None
        )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not use_threads:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {
                item: self.preprocess_single_sample(item=item, gen_batch=gen_batch, obs=obs)
                for item in items
            }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        executor = self._get_preproc_executor()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = {
            item: executor.submit(
                self._preprocess_single_sample_threadsafe, item, gen_batch, obs
            )
            for item in items
        }
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {item: future.result() for item, future in futures.items()}

    # [EXPLAIN] `_preprocess_single_sample_threadsafe` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _preprocess_single_sample_threadsafe(self, item: int, gen_batch: DataProto, obs: Dict):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.preprocess_single_sample(
            item=item, gen_batch=gen_batch, obs=obs, tokenizer=self._thread_tokenizer()
        )

    # [EXPLAIN] `preprocess_single_sample` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        tokenizer=None,
    ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tokenizer is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tokenizer = self.tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_texts = obs.get('text', None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_images = obs.get('image', None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_anchors = obs.get('anchor', None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_text = obs_texts[item] if obs_texts is not None else None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_image = obs_images[item] if obs_images is not None else None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_multi_modal = obs_image is not None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_content = ''
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if obs_text is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            obs_content += obs_text
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Warning: No text observation found!")

        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_with_chat_template = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        row_dict = {}
        
        # Process multimodal data
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image_grid_thw = image_inputs['image_grid_thw']
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if image_grid_thw is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                merge_length = self.processor.image_processor.merge_size**2
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                index = 0
                # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
                while '<image>' in prompt_with_chat_template:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    index += 1

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            raw_prompt = prompt_with_chat_template
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_multi_modal:

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from verl.models.transformers.qwen3_vl import get_rope_index
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from verl.models.transformers.qwen2_vl import get_rope_index

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_mask = attention_mask[0].bool()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = compute_position_id_with_mask(attention_mask)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.data.truncation == "left":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.config.data.truncation == "right":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.config.data.truncation == "middle":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                left_half = self.config.data.max_prompt_length // 2
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                right_half = self.config.data.max_prompt_length - left_half
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.config.data.truncation == "error":
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'task_name' in gen_batch.non_tensor_batch:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            row_dict['task_name'] = gen_batch.non_tensor_batch['task_name'][item]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.data.get('return_raw_chat', False):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            row_dict['raw_prompt'] = chat.tolist()
        
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return row_dict

    # [EXPLAIN] `_placeholder_single_sample` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _placeholder_single_sample(self, item, gen_batch, obs, template):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Cheap stand-in for an already-finished trajectory's row.

        Reproduces only the *cheap* metadata that preprocess_single_sample sets
        (dict lookups, no apply_chat_template / tokenization) and fills the
        model-input tensors (input_ids/attention_mask/position_ids) with padding
        cloned from `template` — an already-processed active row — so the shapes
        and dtypes match exactly for collate. These rows are excluded from
        generation and dropped by gather_rollout_data(), so their token contents
        are never consumed; only shape-consistency matters here.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_texts = obs.get('text', None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_anchors = obs.get('anchor', None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_text = obs_texts[item] if obs_texts is not None else None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_token_id = self.tokenizer.pad_token_id
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pad_token_id is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pad_token_id = 0

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        row_dict = {
            'input_ids': torch.full_like(template['input_ids'], pad_token_id),
            'attention_mask': torch.zeros_like(template['attention_mask']),
            'position_ids': torch.zeros_like(template['position_ids']),
            'raw_prompt_ids': [pad_token_id],
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': gen_batch.non_tensor_batch['data_source'][item],
        }
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'task_name' in gen_batch.non_tensor_batch:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            row_dict['task_name'] = gen_batch.non_tensor_batch['task_name'][item]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.data.get('return_raw_chat', False):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            chat = np.array([{"content": obs_text if obs_text is not None else '', "role": "user"}])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            row_dict['raw_prompt'] = chat.tolist()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return row_dict

    # [EXPLAIN] `preprocess_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def preprocess_batch(
        self,
        gen_batch: DataProto,
        obs: Dict,
        active_mask: np.ndarray = None,
    ) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Process a batch of observation samples, converting environment observations into model-processable format.

        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            active_mask (np.ndarray or None): Boolean mask of trajectories still
                active. When provided, only active rows are fully tokenized; finished
                rows get a cheap placeholder (see _placeholder_single_sample). When
                None, every row is fully processed (original behavior). Active rows
                are processed identically either way, so generation is unaffected.

        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = len(gen_batch.batch['input_ids'])

        # Decide which rows need full tokenization. Falling back to "all active"
        # (None / all-True / multimodal) reproduces the original code path exactly.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skip_done = (
            active_mask is not None
            and not bool(active_mask.all())
            and obs.get('image', None) is None  # multimodal rows always fully processed
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        processed_samples = [None] * batch_size

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not skip_done:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_items = list(range(batch_size))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            processed = self._run_full_preprocess(full_items, gen_batch, obs)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for item in full_items:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                processed_samples[item] = processed[item]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # Full tokenization for active rows (unchanged path); the first such
            # row becomes the shape/dtype template for finished-row placeholders.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            active_items = [item for item in range(batch_size) if active_mask[item]]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            processed = self._run_full_preprocess(active_items, gen_batch, obs)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            template = processed[active_items[0]] if active_items else None
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for item in active_items:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                processed_samples[item] = processed[item]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for item in range(batch_size):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not active_mask[item]:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    processed_samples[item] = self._placeholder_single_sample(
                        item=item,
                        gen_batch=gen_batch,
                        obs=obs,
                        template=template,
                    )

        # Aggregate batch data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return new_batch


    # [EXPLAIN] `gather_rollout_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = len(total_batch_list)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        success_rate = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, value in success.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            success_rate[key] = np.mean(value)
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        effective_batch = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for step_idx, data in enumerate(total_batch_list[bs]):
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if data['active_masks']:
                    # episode_rewards
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    data['tool_callings'] = tool_callings[bs]
                    # turn_step: which step within the trajectory
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    data['turn_step'] = step_idx
                    # success_rate
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for key, value in success_rate.items():
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        data[key] = value

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return gen_batch_output

    # [EXPLAIN] `_scatter_active_to_full` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _scatter_active_to_full(
            self,
            active_output: DataProto,
            active_idx: np.ndarray,
            batch_size: int,
            ) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Expand a generation output that was produced for the active trajectories
        only back to the full batch size, so it can be unioned with the full-size
        batch and recorded per environment.

        Rows that were skipped (already-finished trajectories) are filled with
        padding tokens for token tensors and zeros otherwise, and with a copy of an
        arbitrary active value for non-tensor fields. These rows always carry
        active_masks=False and are dropped by gather_rollout_data(), so their
        contents never affect training; the fillers exist only to keep tensor
        shapes valid and decoding harmless.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        active_idx_t = torch.as_tensor(active_idx, dtype=torch.long)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_token_id = self.tokenizer.pad_token_id
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pad_token_id is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pad_token_id = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_like_keys = {"responses", "input_ids", "prompts"}

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_tensors = {}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if active_output.batch is not None:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, tensor in active_output.batch.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                fill_value = pad_token_id if key in token_like_keys else 0
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_tensor = tensor.new_full((batch_size,) + tuple(tensor.shape[1:]), fill_value)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_tensor[active_idx_t] = tensor
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_tensors[key] = full_tensor

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_non_tensors = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in active_output.non_tensor_batch.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_val = np.empty((batch_size,) + val.shape[1:], dtype=val.dtype)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(val) > 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_val[:] = val[0]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_val[active_idx] = val
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_non_tensors[key] = full_val

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto.from_dict(
            tensors=full_tensors,
            non_tensors=full_non_tensors if full_non_tensors else None,
            meta_info=active_output.meta_info,
        )

    # [EXPLAIN] `_prefetch_pending_log_probs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _prefetch_pending_log_probs(self, actor_rollout_wg):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Run compute_log_prob on a bounded chunk of finished-trajectory rows.

        Called while envs.step() runs in a background thread. Uses the same
        frozen actor weights and the same per-row tensors the trainer's
        old_log_prob phase would use, so each row's result is the value that
        phase would have produced; the trainer then computes only the rows that
        were not prefetched (see compute_log_prob_with_prefetch).
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk = self._logprob_pending[:_ROLLOUT_PREFETCH_LOGPROB_CHUNK]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not chunk:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del self._logprob_pending[:len(chunk)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keys = [key for key, _ in chunk]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rows = [row for _, row in chunk]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensors = {
            name: torch.stack([row[name] for row in rows])
            for name in ("input_ids", "attention_mask", "position_ids", "responses")
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sub = DataProto.from_dict(tensors=tensors)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sub_padded, pad_size = pad_dataproto_to_divisor(sub, actor_rollout_wg.world_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = actor_rollout_wg.compute_log_prob(sub_padded)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = unpad_dataproto(output, pad_size=pad_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs = output.batch["old_log_probs"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        entropys = output.batch["entropys"]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for j, key in enumerate(keys):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._prefetched_log_probs[key] = (log_probs[j], entropys[j])

    # [EXPLAIN] `take_prefetched_log_probs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def take_prefetched_log_probs(self) -> Dict:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Hand the prefetched (traj_uid, turn_step) -> (old_log_probs, entropys)
        rows to the trainer and clear them. Empty unless ROLLOUT_PREFETCH_LOGPROB
        was on during the last multi_turn_loop."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prefetched = self._prefetched_log_probs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._prefetched_log_probs = {}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return prefetched

    # [EXPLAIN] `prefetch_env_reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def prefetch_env_reset(self, envs: EnvironmentManagerBase, env_kwargs):
        # [EXPLAIN] 次 step の env reset を background future として開始し、現在 step の teacher/actor GPU 計算と重ねる。
        # [EXPLAIN] env identity と kwargs が一致する場合だけ1回消費し、resume 用 dataloader pre-peek state と分離する。
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Launch envs.reset() for the *next* rollout in a background thread.

        Called by the trainer right after a rollout's data has been collected
        (envs are idle from then until the next rollout), so the reset — pure
        CPU / subprocess work such as alfworld game loading — overlaps the GPU
        training phases. The next multi_turn_loop on the same `envs` object
        consumes the result instead of calling reset again; reset is still
        executed exactly once per rollout, so stateful env schedules (alfworld's
        game-file iterator) advance identically to the serial order.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._env_reset_prefetch is not None:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError("prefetch_env_reset called while a prefetched reset is still pending.")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._env_reset_executor is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._env_reset_executor = ThreadPoolExecutor(max_workers=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._env_reset_prefetch = {
            "envs_id": id(envs),
            "kwargs": env_kwargs,
            "future": self._env_reset_executor.submit(envs.reset, env_kwargs),
        }

    # [EXPLAIN] `_reset_envs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _reset_envs(self, envs: EnvironmentManagerBase, env_kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Consume a matching prefetched reset, or reset synchronously."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pending = self._env_reset_prefetch
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pending is not None and pending["envs_id"] == id(envs):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._env_reset_prefetch = None
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not _env_kwargs_equal(pending["kwargs"], env_kwargs):
                # The prefetched reset already advanced stateful env schedules;
                # resetting again would silently change the sampled episodes.
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise RuntimeError(
                    "Prefetched env reset consumed with mismatched env_kwargs; "
                    "disable ENV_RESET_PREFETCH or fix the trainer-side prefetch."
                )
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return pending["future"].result()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs.reset(kwargs=env_kwargs)

    # [EXPLAIN] `vanilla_multi_turn_loop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def vanilla_multi_turn_loop(
        # [EXPLAIN] 1 trajectory を複数 env turn row へ展開する同期 loop で、prompt batch 数と actor row 数は一致し得ない。
        # [EXPLAIN] 各 turn で active trajectory のみ生成し、done row は mask/placeholder で global trajectory 順を維持する。
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = len(gen_batch.batch)

        # Initial observations from the environment (uses a trainer-prefetched
        # reset when one is pending for this env manager, see prefetch_env_reset)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs, infos = self._reset_envs(envs, gen_batch.non_tensor_batch.pop('env_kwargs', None))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.env.rollout.n > 0: # env grouping
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            uid_batch = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(batch_size):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if i % self.config.env.rollout.n == 0:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    uid = str(uuid.uuid4())
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                uid_batch.append(uid)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            uid_batch = np.array(uid_batch, dtype=object)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else: # no env grouping, set all to the same uid
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            uid = str(uuid.uuid4())
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_done = np.zeros(batch_size, dtype=bool)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_batch_list = [[] for _ in range(batch_size)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_infos = [[] for _ in range(batch_size)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _turn_records = [] if _ROLLOUT_TURN_TIMING else None
        # Trajectory collection loop
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for _step in range(self.config.env.max_steps):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            active_masks = np.logical_not(is_done)

            # Phase-1 throughput optimization: only generate for trajectories that
            # are still active. Steps belonging to already-finished trajectories
            # carry active_masks=False and are discarded by gather_rollout_data(),
            # so generating them is pure wasted GPU compute. This waste is large in
            # the multitask setting where e.g. search episodes finish within a few
            # turns while the loop keeps running up to alfworld's max_steps. As
            # episodes finish, the vLLM generation batch shrinks accordingly.
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not active_masks.any():
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            active_idx = np.nonzero(active_masks)[0]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _m0 = _now()

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _pre_active_mask = active_masks if _ROLLOUT_SKIP_DONE_PREPROC else None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs, active_mask=_pre_active_mask)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
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
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_input.meta_info = gen_batch.meta_info
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _m_preproc = _now()  # end of CPU preprocess (tokenize/pop)

            # Restrict generation to active trajectories only; done rows are filled
            # back in afterwards (and dropped downstream via active_masks).
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            generate_all = len(active_idx) == batch_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            active_batch_input = batch_input if generate_all else batch_input[active_idx]

            # pad to be divisible by dp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_input_padded, pad_size = pad_dataproto_to_divisor(active_batch_input, actor_rollout_wg.world_size)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _gw0 = gpu_profiler.now()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _gw1 = gpu_profiler.now()
            # # unpad
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            active_batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _m_gen = _now()  # end of GPU generation window

            # Scatter active outputs back to the full batch size for union/recording.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_output = active_batch_output if generate_all else \
                self._scatter_active_to_full(active_batch_output, active_idx, batch_size)

            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            batch.non_tensor_batch['uid'] = uid_batch
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            batch.non_tensor_batch['traj_uid'] = traj_uid

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch = batch.union(batch_output)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if generate_all or not _ROLLOUT_DECODE_ACTIVE_ONLY:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # Decode only the generated rows; finished rows' scattered filler
                # is pad-only, which batch_decode(skip_special_tokens=True) would
                # render as '' anyway.
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                active_actions = self.tokenizer.batch_decode(
                    active_batch_output.batch['responses'], skip_special_tokens=True
                )
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                text_actions = [''] * batch_size
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for pos, idx in enumerate(active_idx):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    text_actions[idx] = active_actions[pos]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _m_decode = _now()  # end of CPU decode (+ scatter/union glue)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._logprob_prefetch_enabled and self._logprob_pending:
                # Overlap: envs.step (CPU/HTTP/IPC, GPU idle) runs in a background
                # thread while the GPU prefetches old_log_prob for finished
                # trajectories. The prefetch call returns before the next
                # generate_sequences is issued, so it never contends with
                # generation on the worker actors.
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self._env_step_executor is None:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self._env_step_executor = ThreadPoolExecutor(max_workers=1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                env_future = self._env_step_executor.submit(envs.step, text_actions)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self._prefetch_pending_log_probs(actor_rollout_wg)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                next_obs, rewards, dones, infos = env_future.result()
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                next_obs, rewards, dones, infos = envs.step(text_actions)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _m_env = _now()  # end of env.step (CPU / HTTP / IPC)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if _turn_records is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _turn_records.append({
                    "turn": _step,
                    "active": int(len(active_idx)),
                    "preproc": _m_preproc - _m0,
                    "gen": _m_gen - _m_preproc,
                    "decode": _m_decode - _m_gen,
                    "envstep": _m_env - _m_decode,
                    "gen_util": gpu_profiler.mean_util_between(_gw0, _gw1),
                    "gen_util_per_gpu": gpu_profiler.per_gpu_util_between(_gw0, _gw1),
                })


            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(rewards.shape) == 2:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rewards = rewards.squeeze(1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                dones = dones.squeeze(1)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 'is_action_valid' in infos[0]:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 'tool_calling' in infos[0]:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            episode_lengths[active_masks] += 1

            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_list: list[dict] = to_list_of_dict(batch)

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(batch_size):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if _ROLLOUT_COMPACT_RECORD and not active_masks[i]:
                    # Finished trajectories' rows carry active_masks=False and are
                    # dropped by gather_rollout_data(); skip materializing them.
                    # Active rows form a prefix of each trajectory's list, so the
                    # enumerate-based turn_step and the last-active-entry scans in
                    # success_evaluator / filter_group_data are unchanged.
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                total_batch_list[i].append(batch_list[i])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                total_infos[i].append(infos[i])

            # Update done states
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            newly_done = np.logical_and(active_masks, dones)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_done = np.logical_or(is_done, dones)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._logprob_prefetch_enabled:
                # Trajectories that finished this turn now have all their rows
                # final; queue them for prefetched old_log_prob on later turns.
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for i in np.nonzero(newly_done)[0]:
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for step_idx, row in enumerate(total_batch_list[i]):
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if row['active_masks']:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            self._logprob_pending.append(((traj_uid[i], step_idx), row))

            # Update observations for next step
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs = next_obs

            # Break if all environments are done
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if is_done.all():
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if _turn_records is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            _print_turn_timing(_turn_records)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    # [EXPLAIN] `dynamic_multi_turn_loop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_batch_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_episode_rewards = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_episode_lengths = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_success = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_traj_uid = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_tool_callings = []
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try_count: int = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(total_batch_list) > 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try_count += 1

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            total_batch_list += batch_list
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            total_episode_rewards.append(episode_rewards)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            total_episode_lengths.append(episode_lengths)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            total_success.append(success)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            total_traj_uid.append(traj_uid)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            total_tool_callings.append(tool_callings)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    # [EXPLAIN] `multi_turn_loop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def multi_turn_loop(
        # [EXPLAIN] production entry は config/env var に応じて vanilla/dynamic 経路を選び、
        # [EXPLAIN] session、active-only decode、compact record、prefetch などの性能機構を適用する。
        # [EXPLAIN] Pure OPD では old-log-prob 消費 phase がないため、ROLLOUT_PREFETCH_LOGPROB は実効的に不要である。
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_train:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        # Log-prob prefetch only makes sense for training rollouts (validation
        # never computes old_log_prob). Multimodal runs are excluded: the
        # prefetch sub-batch carries only the four token tensors, so it would
        # compute log probs without multi_modal_inputs and merge WRONG values —
        # the trainer's full-batch path (which passes multi_modal_inputs) must
        # handle those rows. State is cleared per rollout; anything still
        # pending at the end is simply computed by the trainer as usual.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._logprob_prefetch_enabled = (
            _ROLLOUT_PREFETCH_LOGPROB and is_train and self.processor is None
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._logprob_pending = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._prefetched_log_probs = {}

        # Initial observations from the environment
        # Open one vLLM session for the whole rollout (opt-in). end_rollout_session
        # runs in finally so the engine is always returned to its slept/offloaded
        # state before the post-rollout (gather/teacher/train) phases.
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if _ROLLOUT_KEEP_VLLM_AWAKE:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            actor_rollout_wg.begin_rollout_session()
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.algorithm.filter_groups.enable and is_train:
                # Dynamic Sampling (for DAPO and Dynamic GiGPO)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.dynamic_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # Vanilla Sampling
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if _ROLLOUT_KEEP_VLLM_AWAKE:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                actor_rollout_wg.end_rollout_session()
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(total_batch_list) == len(total_episode_rewards)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(total_batch_list) == len(total_episode_lengths)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(total_batch_list) == len(total_traj_uid)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return gen_batch_output
