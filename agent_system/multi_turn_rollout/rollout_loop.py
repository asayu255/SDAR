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

import os
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from verl import DataProto
from verl.utils import gpu_profiler
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# Opt-in per-turn rollout timing. When off (default) the loop is unchanged and
# adds only a few cheap perf_counter() reads. Set ROLLOUT_TURN_TIMING=1 to print,
# at the end of every rollout, a per-turn breakdown of where the gen phase spends
# wall time (preproc / generate / decode / env.step) plus the GPU SM utilization
# measured *during* generate_sequences (GEN-UTIL). Pairs with GPU_PROFILER=1.
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
_ROLLOUT_SKIP_DONE_PREPROC = os.environ.get("ROLLOUT_SKIP_DONE_PREPROC", "1").strip().lower() in ("1", "true", "yes", "on")

# Keep vLLM awake for the whole rollout instead of waking/sleeping (and re-syncing
# the frozen actor weights) every turn. See ActorRolloutRefWorker.begin_rollout_
# session(). Accuracy-safe: identical frozen weights are used on every turn, so the
# generated tokens are unchanged; only the per-turn weight-sync / wake / sleep
# overhead is removed (and vLLM's prefix cache can persist across turns). Opt-in;
# OFF reproduces the current per-turn behavior exactly.
_ROLLOUT_KEEP_VLLM_AWAKE = os.environ.get("ROLLOUT_KEEP_VLLM_AWAKE", "0").strip().lower() in ("1", "true", "yes", "on")

# Decode only the rows that were actually generated this turn. Finished rows'
# scattered filler is all pad tokens, which batch_decode(skip_special_tokens=True)
# already renders as '' — so filling '' directly gives the same env input while
# skipping the wasted decode of pad-only rows (2/3 of the batch during the
# alfworld tail). Default ON; set ROLLOUT_DECODE_ACTIVE_ONLY=0 for A/B.
_ROLLOUT_DECODE_ACTIVE_ONLY = os.environ.get("ROLLOUT_DECODE_ACTIVE_ONLY", "1").strip().lower() in ("1", "true", "yes", "on")

# Do not record per-turn rows for already-finished trajectories. Those rows carry
# active_masks=False and are dropped by gather_rollout_data() anyway; because a
# trajectory's active rows always form a prefix of its turn list, the enumerate-
# based turn_step and the "last active entry" scans in success_evaluator /
# filter_group_data see identical data. Saves the per-turn dict materialization
# and memory for the inactive 2/3 of the batch during the multitask tail.
# Default ON; set ROLLOUT_COMPACT_RECORD=0 for A/B.
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
_ROLLOUT_PREFETCH_LOGPROB = os.environ.get("ROLLOUT_PREFETCH_LOGPROB", "0").strip().lower() in ("1", "true", "yes", "on")

# Rows per prefetch call. Sized so one compute_log_prob chunk roughly fits inside
# one env.step window; leftovers roll over to later turns (the alfworld tail has
# ~35 such windows). Too large a chunk extends the turn past env.step — the work
# is still subtracted from the old_log_prob phase, but the overlap is lost.
_ROLLOUT_PREFETCH_LOGPROB_CHUNK = int(os.environ.get("ROLLOUT_PREFETCH_LOGPROB_CHUNK", "64"))


def _env_kwargs_equal(a, b) -> bool:
    """Compare two env_kwargs sequences (None or sequence of per-env dicts)."""
    if a is None or b is None:
        return a is None and b is None
    a = list(a) if not isinstance(a, list) else a
    b = list(b) if not isinstance(b, list) else b
    if len(a) != len(b):
        return False
    for item_a, item_b in zip(a, b):
        if item_a is item_b:
            continue
        if not isinstance(item_a, dict) or not isinstance(item_b, dict) or item_a.keys() != item_b.keys():
            return False
        for key in item_a:
            va, vb = item_a[key], item_b[key]
            if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                if not np.array_equal(np.asarray(va), np.asarray(vb)):
                    return False
            elif va != vb:
                return False
    return True


def _now():
    return time.perf_counter()


_SAID_ROLLOUT_ENV = False


def _say_rollout_env():
    """Print, once, the flags as THIS process resolved them.

    The flags are read at import time in whichever process runs the rollout loop
    -- the trainer Ray actor, not the launcher and not the rollout workers. On the
    eval arm that distinction cost 13% of wall clock and took a 2-second
    nvidia-smi trace to notice, because a driver that never calls
    begin_rollout_session() and one whose workers refuse to open a session look
    identical from outside: vLLM wakes and sleeps every turn either way. Reading
    the value off the process that actually branches on it is the only reading
    that answers the question.
    """
    global _SAID_ROLLOUT_ENV
    if _SAID_ROLLOUT_ENV:
        return
    _SAID_ROLLOUT_ENV = True
    print(
        f"[rollout-merge] driver: ROLLOUT_MERGE_GENERATES="
        f"{os.environ.get('ROLLOUT_MERGE_GENERATES', '<unset>')!r} -> queued generate calls "
        f"{'ARE merged into one' if _ROLLOUT_MERGE_GENERATES else 'run one after another'}",
        flush=True,
    )
    print(
        f"[rollout-session] driver: ROLLOUT_KEEP_VLLM_AWAKE="
        f"{os.environ.get('ROLLOUT_KEEP_VLLM_AWAKE', '<unset>')!r} -> session mode "
        f"{'ON' if _ROLLOUT_KEEP_VLLM_AWAKE else 'OFF (vLLM wakes and sleeps every turn)'}"
        f"; ROLLOUT_TURN_TIMING={'on' if _ROLLOUT_TURN_TIMING else 'off'}"
        f"; GPU_PROFILER={os.environ.get('GPU_PROFILER', '<unset>')!r} -> the turn table's "
        f"genGPU%/perGPU% columns will be {'filled' if gpu_profiler.enabled() else 'EMPTY (-)'}",
        flush=True,
    )


_SLOT_LABEL = threading.local()


@contextmanager
def slot_label(name):
    """Tag this thread's rollout with the pipeline slot that launched it.

    Thread-local rather than an attribute on the collector: the label belongs to
    the run, not to the object, and every stub a test hands the pipeline would
    otherwise have to accept being written to.
    """
    previous = getattr(_SLOT_LABEL, "name", None)
    _SLOT_LABEL.name = name
    try:
        yield
    finally:
        _SLOT_LABEL.name = previous


def _current_slot():
    return getattr(_SLOT_LABEL, "name", None) or "-"


# Wall-clock accounting across batches. The per-batch turn table cannot say what
# pipelining did, because what a batch costs is not what changes.
#
# The figure that carries between runs is SECONDS OF WALL PER BATCH. Everything
# else here is diagnosis, and one number in particular is a trap: the occupancy
# ratio -- the sum of the batches' spans over the wall clock -- is NOT a speedup.
# Under pipelining a batch's own span INFLATES, because the generate call it sits
# in is queued behind another batch's. Two slots each reporting a doubled span
# put the ratio at 2.00x with nothing whatsoever gained. It says how many slots
# were occupied, and that is all it says. Measured 1.82x on a run that moved
# s/batch by 1.5%.
#
# s/batch is reported over a trailing window as well as from the start, because
# the run is not homogeneous: alfworld and webshop are the first two batches and
# cost multiples of a search batch, so a figure cumulative over 413 batches
# carries a prefix the comparison does not want.
_WALL_LOCK = threading.Lock()
_WALL_WINDOW = 20
_WALL_STATE = {"batches": 0, "first_start": None, "serial": 0.0, "recent": [], "rows": 0}


def reset_batch_wall():
    """Start a fresh accounting period (called at the top of each validation)."""
    with _WALL_LOCK:
        _WALL_STATE.update(batches=0, first_start=None, serial=0.0, recent=[], rows=0)


def _record_batch_wall(start, end, slot, rows=None):
    """Fold one batch into the running totals and return the line(s) to print.

    ``rows`` makes the run comparable to one that batches differently. Seconds
    per batch only compares runs whose batches hold the same number of rows, and
    the batch NUMBER stops being a fixed point too -- widening search's batches
    turns 413 of them into 208, so batch #171 is no longer the same rows. Rows
    processed is the invariant either way.
    """
    with _WALL_LOCK:
        if _WALL_STATE["first_start"] is None:
            _WALL_STATE["first_start"] = start
        index = _WALL_STATE["batches"]
        _WALL_STATE["batches"] += 1
        span = end - start
        _WALL_STATE["serial"] += span
        serial = _WALL_STATE["serial"]
        wall = end - _WALL_STATE["first_start"]
        if rows:
            _WALL_STATE["rows"] += int(rows)
        rows_total = _WALL_STATE["rows"]
        recent = _WALL_STATE["recent"]
        recent.append((end, rows_total))
        # one more than the window: the rate over N completions needs the end
        # time of the batch before them, not just the N end times themselves.
        del recent[: -(_WALL_WINDOW + 1)]
        window = (recent[-1][0] - recent[0][0]) / (len(recent) - 1) if len(recent) > 1 else float("nan")
        window_rows = recent[-1][1] - recent[0][1]
        window_per_row = (recent[-1][0] - recent[0][0]) / window_rows * 1000 if window_rows else float("nan")
    occupancy = serial / wall if wall > 0 else float("nan")
    lines = []
    if index == 0:
        lines.append(
            "WALL   legend: s/batch is the figure to compare between runs. slots-busy is "
            "OCCUPANCY, not speedup -- a pipelined batch's span inflates while it waits on "
            "another batch's generate, so two slots read 2.00x whether or not anything was gained."
        )
    per_row = ""
    if rows_total:
        all_per_row = wall / rows_total * 1000
        per_row = f"  ms/row last{_WALL_WINDOW}={window_per_row:.0f} all={all_per_row:.0f}"
    lines.append(
        f"WALL   slot={slot}  batch#{index}  rows={rows or '-'}  span={span:.1f}s  "
        f"s/batch last{_WALL_WINDOW}={window:.1f}s all={wall / (index + 1):.1f}s{per_row}  "
        f"wall={wall:.1f}s  slots-busy={occupancy:.2f}x"
    )
    return "\n".join(lines)


# Merge generate calls that are already queued behind one another.
#
# A search batch's last turns decode for a handful of trajectories spread over
# three ranks, and the ranks do not finish together: twelve active trajectories
# measured 61/89/56 per-GPU, the call returning only once the rank holding the
# longest responses is done. The empty seats can only be filled from another
# batch, and the pipeline has one -- but runs it out of phase on purpose, which
# is where its 16.5% came from. So nothing waits: a call merges only what is
# ALREADY queued behind it, which the worker group would have serialised anyway.
#
# Off by default. It changes how rows are grouped inside a generate call, and
# although the rows and their sampling parameters are identical either way, the
# scoring path is not somewhere to enable a thing by assumption.
_ROLLOUT_MERGE_GENERATES = os.environ.get("ROLLOUT_MERGE_GENERATES", "0").strip().lower() in ("1", "true", "yes", "on")
_GENERATE_MERGER = None


def _merge_key(batch):
    """Calls sharing this may be merged; calls that do not, may never be.

    Everything that makes two calls differ apart from their rows: the sampling
    parameters, and the tensor widths a concatenation would have to agree on.
    """
    widths = tuple(sorted((name, tuple(tensor.shape[1:])) for name, tensor in batch.batch.items()))
    return (repr(sorted((str(k), repr(v)) for k, v in (batch.meta_info or {}).items())), widths)


def _split_by_rows(output, sizes):
    start, parts = 0, []
    for size in sizes:
        parts.append(output.slice(start, start + size))
        start += size
    return parts


def _generate_sequences(actor_rollout_wg, batch_input_padded):
    """One generate call, merged with whatever is queued when merging is on."""
    if not _ROLLOUT_MERGE_GENERATES:
        return actor_rollout_wg.generate_sequences(batch_input_padded)
    global _GENERATE_MERGER
    if _GENERATE_MERGER is None:
        from verl.utils.generate_merge import GenerateMerger

        _GENERATE_MERGER = GenerateMerger(concat=DataProto.concat, split=_split_by_rows)
    return _GENERATE_MERGER.call(
        _merge_key(batch_input_padded),
        batch_input_padded,
        len(batch_input_padded),
        actor_rollout_wg.generate_sequences,
    )


def _token_counts(batch_input_padded, batch_output_padded):
    """Prompt and generated token counts for one generate call, or (None, None).

    The split between them is what decides whether a turn is prefill work or
    decode work, and no other instrument here can see it: the driver times the
    call as one span, NVML says only whether a kernel was resident, and vLLM's
    own stats logger emits nothing on this path even with disable_log_stats
    false and its logging at INFO (verified: the KV-cache lines print, the
    throughput lines do not).

    Both counts come from attention_mask sums, which are exactly the tokens the
    engine was handed -- padding rows from pad_dataproto_to_divisor included,
    since the engine processes those too. The generated count is the output's
    total minus the prompt's, which needs no assumption about pad ids.
    """
    try:
        prompt = int(batch_input_padded.batch["attention_mask"].sum())
        total = int(batch_output_padded.batch["attention_mask"].sum())
    except (AttributeError, KeyError, TypeError):
        return None, None
    return prompt, max(0, total - prompt)


def _fmt_per_gpu(vals):
    if not vals:
        return "-"
    return "/".join(f"{v:.0f}" if v is not None else "-" for v in vals)


def _print_turn_timing(records, span=None, slot="-", rows=None):
    """Pretty-print the per-turn breakdown collected during one rollout.

    The perGPU% column shows per-GPU SM util during the turn's generation; the
    spread between GPUs reveals data-parallel load imbalance from mixing tasks
    across the DP split. The DP-IMBALANCE summary line tracks it (the interleaved
    task layout, TASK_BALANCE_INTERLEAVE, shrinks it on the mixed-task turns).
    """
    if not records:
        return
    header = (
        f"{'turn':>4}{'active':>8}{'preproc':>9}{'gen':>9}{'decode':>9}"
        f"{'envstep':>9}{'total':>9}{'genGPU%':>9}{'  perGPU%':>12}"
        f"{'promptTok':>12}{'genTok':>10}"
    )
    lines = ["[rollout-turn-timing] per-turn breakdown (seconds); GPU busy only during 'gen'", header, "-" * len(header)]
    tot = {k: 0.0 for k in ("preproc", "gen", "decode", "envstep", "total")}
    toks = {"prompt_tok": 0, "gen_tok": 0}
    full_util, shrunk_util = [], []
    dp_spreads = []  # per-turn max-min across GPUs during gen (DP imbalance)
    first_active = records[0]["active"]
    for r in records:
        total = r["preproc"] + r["gen"] + r["decode"] + r["envstep"]
        for k in ("preproc", "gen", "decode", "envstep"):
            tot[k] += r[k]
        tot["total"] += total
        gu = r["gen_util"]
        gu_s = f"{gu:.0f}" if gu is not None else "-"
        pg = r.get("gen_util_per_gpu")
        pg_s = _fmt_per_gpu(pg)
        pt, gt = r.get("prompt_tok"), r.get("gen_tok")
        for key, value in (("prompt_tok", pt), ("gen_tok", gt)):
            if value is not None:
                toks[key] += value
        lines.append(
            f"{r['turn']:>4}{r['active']:>8}{r['preproc']:>9.2f}{r['gen']:>9.2f}"
            f"{r['decode']:>9.2f}{r['envstep']:>9.2f}{total:>9.2f}{gu_s:>9}{pg_s:>12}"
            f"{(f'{pt:,}' if pt is not None else '-'):>12}"
            f"{(f'{gt:,}' if gt is not None else '-'):>10}"
        )
        if gu is not None:
            (full_util if r["active"] >= first_active else shrunk_util).append(gu)
        if pg:
            present = [v for v in pg if v is not None]
            if len(present) >= 2:
                dp_spreads.append(max(present) - min(present))
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':>4}{'':>8}{tot['preproc']:>9.1f}{tot['gen']:>9.1f}"
        f"{tot['decode']:>9.1f}{tot['envstep']:>9.1f}{tot['total']:>9.1f}"
        f"{'':>9}{'':>12}{toks['prompt_tok']:>12,}{toks['gen_tok']:>10,}"
    )
    if toks["prompt_tok"] or toks["gen_tok"]:
        # promptTok is what was HANDED to the engine, not what it recomputed:
        # with prefix caching on, a turn's shared prefix is served from cache.
        # So the prompt figure is an UPPER bound on prefill work, and the ratio
        # below is the most prefill-heavy reading the data allows. If generated
        # tokens dominate even here, the turn is decode-bound for certain.
        lines.append(
            f"TOKENS prompt={toks['prompt_tok']:,} (submitted; prefix-cache hits are not recomputed, "
            f"so this is an UPPER bound on prefill work)  generated={toks['gen_tok']:,} (decode steps)"
        )
    cpu_glue = tot["preproc"] + tot["decode"] + tot["envstep"]
    if tot["total"] > 0:
        lines.append(
            f"SHARE  gen(GPU-busy)={100*tot['gen']/tot['total']:.1f}%  "
            f"cpu-glue(preproc+decode+envstep, GPU-idle)={100*cpu_glue/tot['total']:.1f}%"
        )
    fa = sum(full_util) / len(full_util) if full_util else None
    sa = sum(shrunk_util) / len(shrunk_util) if shrunk_util else None
    lines.append(
        f"GEN-UTIL  full_batch(active>={first_active})="
        f"{f'{fa:.0f}%' if fa is not None else '-'}   "
        f"shrunk(active<{first_active})={f'{sa:.0f}%' if sa is not None else '-'}   "
        f"(full~shrunk => not batch-underfed; full>>shrunk => alfworld plateau underfeed)"
    )
    # data-parallel imbalance (lower is better; the interleaved task layout
    # shrinks it on mixed-task turns)
    if dp_spreads:
        mean_spread = sum(dp_spreads) / len(dp_spreads)
        lines.append(
            f"DP-IMBALANCE  mean |maxGPU-minGPU| during gen = {mean_spread:.1f} pp "
            f"(lower=better; TASK_BALANCE_INTERLEAVE shrinks this on mixed turns)"
        )
    if span is not None:
        lines.append(_record_batch_wall(span[0], span[1], slot, rows=rows))
    print("\n".join(lines), flush=True)


@contextmanager
def rollout_session(actor_rollout_wg):
    """Hold vLLM awake for everything inside this block.

    Without a session, generate_sequences re-enters the sharding manager on every
    call: it re-gathers the FSDP state dict, re-syncs the full model weights, and
    wakes then sleeps the engine. The weights are frozen for as long as nobody
    trains, so one sync at the top of the block produces the identical weights
    for every call inside it -- the generated tokens are unchanged.

    Nest it as widely as the frozen weights allow. multi_turn_loop opens one per
    rollout, and _validate opens one around the whole validation; on the SFT arm
    that is 413 rollouts, and paying a 21 GB unmap and remap between each of them
    measured 10.4% of the evaluation's wall clock. The worker counts scopes, so
    the inner ones are free.

    A no-op when ROLLOUT_KEEP_VLLM_AWAKE is off, and on any rollout that is not
    vLLM's -- the worker decides that, not this.
    """
    _say_rollout_env()
    if not _ROLLOUT_KEEP_VLLM_AWAKE:
        yield
        return
    actor_rollout_wg.begin_rollout_session()
    try:
        yield
    finally:
        actor_rollout_wg.end_rollout_session()


# Reuse of the prompt tokenisation for raw_prompt_ids.
#
# In the text-only path preprocess_single_sample tokenises the SAME string
# twice: once through tokenize_and_postprocess_data (which calls the tokenizer
# with add_special_tokens=False and pads), and once through tokenizer.encode
# with add_special_tokens=False to build raw_prompt_ids. The non-pad tokens of
# the first ARE the second, for the truncation modes this arm uses -- both
# sides cut "left" the same way, "right" the same way, and "error" raises
# before this point. That second pass is 252 encodes per turn on the calling
# thread the pipeline waits between batches for.
#
# The equality is load-bearing for scoring (raw_prompt_ids is what vLLM
# generates from), so it is not assumed: the first _RAW_IDS_VERIFY calls run
# both paths and compare, and any mismatch disables the reuse for the process
# and says so. "middle" truncation and multimodal prompts always take the old
# path -- middle is not a mode postprocess_data implements, and multimodal
# raw_prompt is a different string.
_RAW_IDS_REUSE = os.environ.get("ROLLOUT_RAW_IDS_REUSE", "1").strip().lower() not in ("0", "false", "no")
_RAW_IDS_VERIFY = 8
_RAW_IDS_STATE = {"enabled": _RAW_IDS_REUSE, "verified": 0}


def _prompt_ids_from_tensors(input_ids_row, attention_mask_row):
    """The prompt's token ids, as the already-run tokenisation produced them."""
    return input_ids_row[attention_mask_row.bool()].tolist()


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        # ROLLOUT_PREFETCH_LOGPROB state: rows of finished trajectories waiting for
        # a prefetched compute_log_prob, and the per-row results keyed by
        # (traj_uid, turn_step). Cleared at the start of every multi_turn_loop.
        self._logprob_prefetch_enabled = False
        self._logprob_pending = []
        self._prefetched_log_probs = {}
        self._env_step_executor = None
        # ENV_RESET_PREFETCH state: one outstanding background envs.reset, launched
        # by the trainer between rollouts (see prefetch_env_reset).
        self._env_reset_prefetch = None
        self._env_reset_executor = None

    def _run_full_preprocess(self, items: List[int], gen_batch: DataProto, obs: Dict) -> Dict[int, dict]:
        """Run preprocess_single_sample over `items`."""
        return {
            item: self.preprocess_single_sample(item=item, gen_batch=gen_batch, obs=obs)
            for item in items
        }

    def _raw_prompt_ids(self, tokenizer, raw_prompt, input_ids_row, attention_mask_row, is_multi_modal):
        """raw_prompt_ids for one row, reusing the tokenisation already done.

        See the _RAW_IDS_REUSE comment above for why the reuse is sound and how
        it is verified. The fallback is the original double-encode, kept intact
        for multimodal rows, "middle" truncation, and any process where the
        self-check ever failed.
        """
        truncation = self.config.data.truncation
        eligible = _RAW_IDS_STATE["enabled"] and not is_multi_modal and truncation in ("left", "right", "error")
        if eligible:
            extracted = _prompt_ids_from_tensors(input_ids_row, attention_mask_row)
            if _RAW_IDS_STATE["verified"] >= _RAW_IDS_VERIFY:
                return extracted
            encoded = self._encode_raw_prompt(tokenizer, raw_prompt)
            if encoded == extracted:
                _RAW_IDS_STATE["verified"] += 1
                return extracted
            _RAW_IDS_STATE["enabled"] = False
            print(
                "[raw-ids] reused prompt tokens differ from a fresh encode "
                f"({len(extracted)} vs {len(encoded)} ids); reuse disabled for this process, "
                "falling back to double tokenisation.",
                flush=True,
            )
            return encoded
        return self._encode_raw_prompt(tokenizer, raw_prompt)

    def _encode_raw_prompt(self, tokenizer, raw_prompt):
        raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")
        return raw_prompt_ids

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
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

        tokenizer = self.tokenizer
        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self._raw_prompt_ids(
            tokenizer, raw_prompt, input_ids[0], attention_mask[0], is_multi_modal
        )

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if 'task_name' in gen_batch.non_tensor_batch:
            row_dict['task_name'] = gen_batch.non_tensor_batch['task_name'][item]

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def _placeholder_single_sample(self, item, gen_batch, obs, template):
        """Cheap stand-in for an already-finished trajectory's row.

        Reproduces only the *cheap* metadata that preprocess_single_sample sets
        (dict lookups, no apply_chat_template / tokenization) and fills the
        model-input tensors (input_ids/attention_mask/position_ids) with padding
        cloned from `template` — an already-processed active row — so the shapes
        and dtypes match exactly for collate. These rows are excluded from
        generation and dropped by gather_rollout_data(), so their token contents
        are never consumed; only shape-consistency matters here.
        """
        obs_texts = obs.get('text', None)
        obs_anchors = obs.get('anchor', None)
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor
        obs_text = obs_texts[item] if obs_texts is not None else None

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        row_dict = {
            'input_ids': torch.full_like(template['input_ids'], pad_token_id),
            'attention_mask': torch.zeros_like(template['attention_mask']),
            'position_ids': torch.zeros_like(template['position_ids']),
            'raw_prompt_ids': [pad_token_id],
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': gen_batch.non_tensor_batch['data_source'][item],
        }
        if 'task_name' in gen_batch.non_tensor_batch:
            row_dict['task_name'] = gen_batch.non_tensor_batch['task_name'][item]
        if self.config.data.get('return_raw_chat', False):
            chat = np.array([{"content": obs_text if obs_text is not None else '', "role": "user"}])
            row_dict['raw_prompt'] = chat.tolist()
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto,
        obs: Dict,
        active_mask: np.ndarray = None,
    ) -> DataProto:
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
        batch_size = len(gen_batch.batch['input_ids'])

        # Decide which rows need full tokenization. Falling back to "all active"
        # (None / all-True / multimodal) reproduces the original code path exactly.
        skip_done = (
            active_mask is not None
            and not bool(active_mask.all())
            and obs.get('image', None) is None  # multimodal rows always fully processed
        )

        processed_samples = [None] * batch_size

        if not skip_done:
            full_items = list(range(batch_size))
            processed = self._run_full_preprocess(full_items, gen_batch, obs)
            for item in full_items:
                processed_samples[item] = processed[item]
        else:
            # Full tokenization for active rows (unchanged path); the first such
            # row becomes the shape/dtype template for finished-row placeholders.
            active_items = [item for item in range(batch_size) if active_mask[item]]
            processed = self._run_full_preprocess(active_items, gen_batch, obs)
            template = processed[active_items[0]] if active_items else None
            for item in active_items:
                processed_samples[item] = processed[item]
            for item in range(batch_size):
                if not active_mask[item]:
                    processed_samples[item] = self._placeholder_single_sample(
                        item=item,
                        gen_batch=gen_batch,
                        obs=obs,
                        template=template,
                    )

        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
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
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for step_idx, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # turn_step: which step within the trajectory
                    data['turn_step'] = step_idx
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def _scatter_active_to_full(
            self,
            active_output: DataProto,
            active_idx: np.ndarray,
            batch_size: int,
            ) -> DataProto:
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
        active_idx_t = torch.as_tensor(active_idx, dtype=torch.long)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        token_like_keys = {"responses", "input_ids", "prompts"}

        full_tensors = {}
        if active_output.batch is not None:
            for key, tensor in active_output.batch.items():
                fill_value = pad_token_id if key in token_like_keys else 0
                full_tensor = tensor.new_full((batch_size,) + tuple(tensor.shape[1:]), fill_value)
                full_tensor[active_idx_t] = tensor
                full_tensors[key] = full_tensor

        full_non_tensors = {}
        for key, val in active_output.non_tensor_batch.items():
            full_val = np.empty((batch_size,) + val.shape[1:], dtype=val.dtype)
            if len(val) > 0:
                full_val[:] = val[0]
            full_val[active_idx] = val
            full_non_tensors[key] = full_val

        return DataProto.from_dict(
            tensors=full_tensors,
            non_tensors=full_non_tensors if full_non_tensors else None,
            meta_info=active_output.meta_info,
        )

    def _prefetch_pending_log_probs(self, actor_rollout_wg):
        """Run compute_log_prob on a bounded chunk of finished-trajectory rows.

        Called while envs.step() runs in a background thread. Uses the same
        frozen actor weights and the same per-row tensors the trainer's
        old_log_prob phase would use, so each row's result is the value that
        phase would have produced; the trainer then computes only the rows that
        were not prefetched (see compute_log_prob_with_prefetch).
        """
        chunk = self._logprob_pending[:_ROLLOUT_PREFETCH_LOGPROB_CHUNK]
        if not chunk:
            return
        del self._logprob_pending[:len(chunk)]
        keys = [key for key, _ in chunk]
        rows = [row for _, row in chunk]
        tensors = {
            name: torch.stack([row[name] for row in rows])
            for name in ("input_ids", "attention_mask", "position_ids", "responses")
        }
        sub = DataProto.from_dict(tensors=tensors)
        sub_padded, pad_size = pad_dataproto_to_divisor(sub, actor_rollout_wg.world_size)
        output = actor_rollout_wg.compute_log_prob(sub_padded)
        output = unpad_dataproto(output, pad_size=pad_size)
        log_probs = output.batch["old_log_probs"]
        entropys = output.batch["entropys"]
        for j, key in enumerate(keys):
            self._prefetched_log_probs[key] = (log_probs[j], entropys[j])

    def take_prefetched_log_probs(self) -> Dict:
        """Hand the prefetched (traj_uid, turn_step) -> (old_log_probs, entropys)
        rows to the trainer and clear them. Empty unless ROLLOUT_PREFETCH_LOGPROB
        was on during the last multi_turn_loop."""
        prefetched = self._prefetched_log_probs
        self._prefetched_log_probs = {}
        return prefetched

    def prefetch_env_reset(self, envs: EnvironmentManagerBase, env_kwargs):
        """Launch envs.reset() for the *next* rollout in a background thread.

        Called by the trainer right after a rollout's data has been collected
        (envs are idle from then until the next rollout), so the reset — pure
        CPU / subprocess work such as alfworld game loading — overlaps the GPU
        training phases. The next multi_turn_loop on the same `envs` object
        consumes the result instead of calling reset again; reset is still
        executed exactly once per rollout, so stateful env schedules (alfworld's
        game-file iterator) advance identically to the serial order.
        """
        if self._env_reset_prefetch is not None:
            raise RuntimeError("prefetch_env_reset called while a prefetched reset is still pending.")
        if self._env_reset_executor is None:
            self._env_reset_executor = ThreadPoolExecutor(max_workers=1)
        self._env_reset_prefetch = {
            "envs_id": id(envs),
            "kwargs": env_kwargs,
            "future": self._env_reset_executor.submit(envs.reset, env_kwargs),
        }

    def _reset_envs(self, envs: EnvironmentManagerBase, env_kwargs):
        """Consume a matching prefetched reset, or reset synchronously."""
        pending = self._env_reset_prefetch
        if pending is not None and pending["envs_id"] == id(envs):
            self._env_reset_prefetch = None
            if not _env_kwargs_equal(pending["kwargs"], env_kwargs):
                # The prefetched reset already advanced stateful env schedules;
                # resetting again would silently change the sampled episodes.
                raise RuntimeError(
                    "Prefetched env reset consumed with mismatched env_kwargs; "
                    "disable ENV_RESET_PREFETCH or fix the trainer-side prefetch."
                )
            return pending["future"].result()
        return envs.reset(kwargs=env_kwargs)

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
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

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment (uses a trainer-prefetched
        # reset when one is pending for this env manager, see prefetch_env_reset)
        obs, infos = self._reset_envs(envs, gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        _turn_records = [] if _ROLLOUT_TURN_TIMING else None
        if _turn_records is not None:
            # Feeds the genGPU% column. Nothing on the validation path would
            # otherwise start the sampler -- push_phase is its only other caller
            # and that lives in the trainer's fit loop.
            gpu_profiler.ensure_started()
        _batch_started = _now()
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            # Phase-1 throughput optimization: only generate for trajectories that
            # are still active. Steps belonging to already-finished trajectories
            # carry active_masks=False and are discarded by gather_rollout_data(),
            # so generating them is pure wasted GPU compute. This waste is large in
            # the multitask setting where e.g. search episodes finish within a few
            # turns while the loop keeps running up to alfworld's max_steps. As
            # episodes finish, the vLLM generation batch shrinks accordingly.
            if not active_masks.any():
                break
            active_idx = np.nonzero(active_masks)[0]
            _m0 = _now()

            _pre_active_mask = active_masks if _ROLLOUT_SKIP_DONE_PREPROC else None
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs, active_mask=_pre_active_mask)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            _m_preproc = _now()  # end of CPU preprocess (tokenize/pop)

            # Restrict generation to active trajectories only; done rows are filled
            # back in afterwards (and dropped downstream via active_masks).
            generate_all = len(active_idx) == batch_size
            active_batch_input = batch_input if generate_all else batch_input[active_idx]

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(active_batch_input, actor_rollout_wg.world_size)
            _gw0 = gpu_profiler.now()
            batch_output_padded = _generate_sequences(actor_rollout_wg, batch_input_padded)
            _gw1 = gpu_profiler.now()
            if _turn_records is not None:
                _prompt_tok, _gen_tok = _token_counts(batch_input_padded, batch_output_padded)
            # # unpad
            active_batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            _m_gen = _now()  # end of GPU generation window

            # Scatter active outputs back to the full batch size for union/recording.
            batch_output = active_batch_output if generate_all else \
                self._scatter_active_to_full(active_batch_output, active_idx, batch_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)

            if generate_all or not _ROLLOUT_DECODE_ACTIVE_ONLY:
                text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            else:
                # Decode only the generated rows; finished rows' scattered filler
                # is pad-only, which batch_decode(skip_special_tokens=True) would
                # render as '' anyway.
                active_actions = self.tokenizer.batch_decode(
                    active_batch_output.batch['responses'], skip_special_tokens=True
                )
                text_actions = [''] * batch_size
                for pos, idx in enumerate(active_idx):
                    text_actions[idx] = active_actions[pos]
            _m_decode = _now()  # end of CPU decode (+ scatter/union glue)

            if self._logprob_prefetch_enabled and self._logprob_pending:
                # Overlap: envs.step (CPU/HTTP/IPC, GPU idle) runs in a background
                # thread while the GPU prefetches old_log_prob for finished
                # trajectories. The prefetch call returns before the next
                # generate_sequences is issued, so it never contends with
                # generation on the worker actors.
                if self._env_step_executor is None:
                    self._env_step_executor = ThreadPoolExecutor(max_workers=1)
                env_future = self._env_step_executor.submit(envs.step, text_actions)
                self._prefetch_pending_log_probs(actor_rollout_wg)
                next_obs, rewards, dones, infos = env_future.result()
            else:
                next_obs, rewards, dones, infos = envs.step(text_actions)
            _m_env = _now()  # end of env.step (CPU / HTTP / IPC)

            if _turn_records is not None:
                _turn_records.append({
                    "turn": _step,
                    "active": int(len(active_idx)),
                    "preproc": _m_preproc - _m0,
                    "gen": _m_gen - _m_preproc,
                    "decode": _m_decode - _m_gen,
                    "envstep": _m_env - _m_decode,
                    "gen_util": gpu_profiler.mean_util_between(_gw0, _gw1),
                    "gen_util_per_gpu": gpu_profiler.per_gpu_util_between(_gw0, _gw1),
                    "prompt_tok": _prompt_tok,
                    "gen_tok": _gen_tok,
                })


            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                if _ROLLOUT_COMPACT_RECORD and not active_masks[i]:
                    # Finished trajectories' rows carry active_masks=False and are
                    # dropped by gather_rollout_data(); skip materializing them.
                    # Active rows form a prefix of each trajectory's list, so the
                    # enumerate-based turn_step and the last-active-entry scans in
                    # success_evaluator / filter_group_data are unchanged.
                    continue
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            newly_done = np.logical_and(active_masks, dones)
            is_done = np.logical_or(is_done, dones)

            if self._logprob_prefetch_enabled:
                # Trajectories that finished this turn now have all their rows
                # final; queue them for prefetched old_log_prob on later turns.
                for i in np.nonzero(newly_done)[0]:
                    for step_idx, row in enumerate(total_batch_list[i]):
                        if row['active_masks']:
                            self._logprob_pending.append(((traj_uid[i], step_idx), row))

            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        if _turn_records is not None:
            _print_turn_timing(
                _turn_records, span=(_batch_started, _now()), slot=_current_slot(), rows=batch_size
            )

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
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
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
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
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        # Log-prob prefetch only makes sense for training rollouts (validation
        # never computes old_log_prob). Multimodal runs are excluded: the
        # prefetch sub-batch carries only the four token tensors, so it would
        # compute log probs without multi_modal_inputs and merge WRONG values —
        # the trainer's full-batch path (which passes multi_modal_inputs) must
        # handle those rows. State is cleared per rollout; anything still
        # pending at the end is simply computed by the trainer as usual.
        self._logprob_prefetch_enabled = (
            _ROLLOUT_PREFETCH_LOGPROB and is_train and self.processor is None
        )
        self._logprob_pending = []
        self._prefetched_log_probs = {}

        # Initial observations from the environment
        # One vLLM session for this whole rollout, nested inside whatever wider
        # session the caller holds (_validate holds one around all 413 of them).
        # The contextmanager closes in a finally, so the engine is returned to its
        # slept/offloaded state before the post-rollout (gather/teacher/train)
        # phases -- unless an outer scope is still holding it, which is the point.
        with rollout_session(actor_rollout_wg):
            if self.config.algorithm.filter_groups.enable and is_train:
                # Dynamic Sampling (for DAPO and Dynamic GiGPO)
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.dynamic_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
            else:
                # Vanilla Sampling
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output
