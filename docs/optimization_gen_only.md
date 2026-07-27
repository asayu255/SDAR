# Stage-1 "gen only" — speedup analysis and recommended configuration

Target: `examples/opd_trainer/run_multitask_offpolicy_qwen3_gen_only.sh`
(`verl.trainer.main_opd_offpolicy_gen` → `TeacherTrajectoryGenerator.generate()`).

Companion to `docs/optimization_report.md` (Phase 1: ①–⑤) and
`docs/optimization_phase2.md` (Phase 2: A–E). Those two documents optimize the
**RL training step**, where the rollout is one phase among several
compute-bound training phases. Stage-1 generation is a *different workload* and
several of their conclusions invert here.

Numbers below marked "est." are analytic estimates from the config and the
model shape (Qwen3-1.7B, 28 layers, 8 KV heads × 128) — this branch carries no
Stage-1 profile yet. §6 is the measurement plan that turns them into measured
values.

---

## 1. What the gen-only workload actually is

Per invocation (one task, one frozen teacher), 300 steps of:

| Phase | What runs | Where |
|---|---|---|
| dataloader + preproc | 15 prompts → ×8 → 120 trajectories, tokenize | CPU |
| **multi-turn rollout** | up to `max_steps` turns of (preproc → vLLM generate → decode → `envs.step`) | GPU + CPU |
| `adjust_batch` | pad turn-rows up to a multiple of 240 by **duplicating rows** | CPU |
| **`compute_actor_topk_log_prob`** | one FSDP forward over *every* turn-row, top-20 | GPU |
| accumulate | `keep.to("cpu")`, append to an in-RAM list | CPU/RAM |

and then one `DataProto.concat` + `save_to_disk` at the very end.

Three structural facts drive everything below:

1. **There is no training.** No `update_actor`, no `old_log_prob`, no ref
   policy, no critic (grpo + `use_kl_in_reward=False`). The FSDP actor exists
   *only* to run the top-k forward. The optimizer is built and never stepped.
2. **The teacher weights are frozen for the entire run**, not just within one
   rollout — a strictly stronger version of the invariant that Phase 1's
   mechanism ① exploits.
3. **The three tasks are independent processes with independent models.** The
   script runs them back to back on 3 GPUs each; nothing couples them.

Fact 3 looks like the largest single lever and is the one that does not exist
in the training workload at all — but it does not pay off here; see G1 for the
arithmetic. Facts 1 and 2 are what the adopted mechanisms rest on.

---

## 2. New gen-only mechanisms, ranked

### G1 — Run the three tasks concurrently, 1 GPU each — **NOT ADOPTED (reverted)**

The idea: today Stage 1 runs `alfworld(3 GPU) → search(3 GPU) → webshop(3 GPU)`
sequentially. `tensor_model_parallel_size=1` already, so each GPU holds a
complete engine and the 3-way split is pure data parallelism; the three tasks
are independent processes. So run all three at once, one GPU each.

This was implemented (`0771d52`, `963c176`) and **reverted**. The original
speedup claim rested on "1.7B decode is weight-bandwidth bound, so 40 → 120
trajectories per GPU costs far less than 3×". That reasoning is wrong for this
workload: gen is **prefill-dominated**, not decode-dominated. Every turn
re-prefills the whole history (up to `data.max_prompt_length`) and emits a
short action, and prefill is compute-bound and scales ~linearly with concurrent
sequences. Tripling the per-GPU sequence count roughly triples the per-task GPU
time. (The mechanism-B row in §4 already said gen is prefill-heavy — the two
statements were inconsistent.)

**Break-even.** Per task `i`, let `T_i` be its wall in the sequential 3-GPU
layout and `g` the GPU-busy fraction of it. Concurrently on one GPU the GPU work
takes ~3× while the CPU glue (`envs.step`, tokenization) does not stretch, so
each task takes ~`(1 + 2g)·T_i` and the three overlap:

```
speedup = ΣT_i / ((1 + 2g) · T_max)        r = T_max / ΣT_i
        = 1 / ((1 + 2g) · r)
```

| `r` (the dominant task's share) | break-even `g` | speedup at `g = 0.6` |
|---|---|---|
| 0.7 (alfworld dominates) | `g ≤ 0.21` | 0.65× — **1.5× slower** |
| 0.5 | `g ≤ 0.5` | 0.91× |
| 0.33 (balanced) | always wins | 1.36× |

With per-task caps of alfworld 50 / webshop 15 / search 4 turns, `r ≈ 0.7` is
the realistic case, so concurrency only pays if the GPU is busy less than ~21%
of the gen wall. Phase 1 measured gen SM ≈ 57%, so it is not.

The general form of the error: when per-GPU throughput scales linearly with
GPUs, total GPU-seconds is conserved and rearranging tasks across devices
cannot help. Parallelizing only wins by filling *idle*, and the sequential
layout already gives every task all three GPUs.

**What would make it pay.** `(1 + 2g)` is an upper bound: in the alfworld tail
(turns 16–50) the active set shrinks and the GPUs are underfed, so there the
3× factor does not apply and another task's work would genuinely fill idle.
Whether that is enough is exactly the `g` above, and it is now measurable —
`ROLLOUT_TURN_TIMING=1` prints

```
SHARE  gen(GPU-busy)=XX%  cpu-glue(preproc+decode+envstep, GPU-idle)=YY%
```

and `r` follows from comparing the three tasks' step times. The decisive
experiment is ~30 minutes: alfworld at `+gen.num_trajectories=1200` on 1 GPU
versus on 3 GPUs, comparing step wall.

If it is ever revived, the concurrency machinery in those two commits is worth
reusing: per-task `CUDA_VISIBLE_DEVICES`, per-task logs, PID/exit-status
collection, and the Ray isolation — `RAY_ADDRESS=local` (without it a later
`ray.init()` can attach to a sibling's cluster, see 0 free GPUs and hang until
`ray_wait_register_center_timeout`), per-task `RAY_TMPDIR`,
`include_dashboard=False`, and a pinned `object_store_memory` (the default
"30% of *available* memory" is racy with three simultaneous inits and can
overflow `/dev/shm`). Note also that world size 3 → 1 changes `adjust_batch`'s
divisor from `lcm(16·3, 5·3)=240` to `lcm(16,5)=80`, so the number of
duplicated turn-rows written into each `<task>.pt` changes — see G6.

### G2 — Enlarge the generation batch (est. 1.5–2.5× on alfworld)

`per_task_batch_size=15` is a *PPO* batch size. Stage 1 does no policy update,
so for generation it is a pure throughput knob: the only thing it controls is
how many trajectories are in flight per `generate` call.

This attacks the exact bottleneck Phase 2 identified — the **alfworld tail**.
With 120 trajectories and per-task caps, turns 16–50 run with a shrinking
active set and the GPU empties out. With 480 in flight the tail of one group
overlaps the body of another.

The trajectory total is preserved: `num_trajectories=36000` caps the loop, and
`TaskBalancedSampler` builds `num_batches × per_task_batch_size` indices, so
search still consumes its 4500-prompt pool exactly once as long as
`num_batches × per_task_batch_size = 4500`:

| `per_task_batch_size` | `num_batches` | trajectories/step | steps | total |
|---|---|---|---|---|
| 15 (today) | 300 | 120 | 300 | 36000 |
| 30 | 150 | 240 | 150 | 36000 |
| 60 | 75 | 480 | 75 | 36000 |

No data-prep change is needed: alfworld/webshop have 15 placeholder rows and
`_indices_for_required_size` repeats them to the required count regardless.

Costs: env instances are `per_task_batch_size × env.rollout.n`, so 60 → **480
alfworld env instances** (CPU, RAM, game files) and 480 concurrent sequences of
KV, all on the 3-GPU sequential layout (40 → 160 sequences per GPU). Unlike
G1 this does not trade GPU parallelism away: it feeds the *same* three GPUs a
larger batch, which is exactly what the underfed alfworld tail needs.

Requires editing `expected_multitask_offpolicy_gen_config.yaml` in the same
commit (`data.train_batch_size`, `data.task_balance.per_task_batch_size` are
pinned there) — the intent lock is doing its job; the knob is genuinely
performance-only *on the generation side*, but the same key is a scientific
knob for Stage 2, which is why it is pinned.

Accuracy class: distribution-preserving (③-class). Trajectory count, prompt
pool coverage and sampling temperature are unchanged; only which prompts share
a batch changes.

### G3 — KV headroom: cut the top-k micro-batch *before* raising `gpu_memory_utilization`

0.6 is inherited from the *training* script, where the same GPU must also hold
FSDP gradients, Adam state and `update_actor` activations, none of which exist
in gen-only — which makes "just raise it to 0.85" the obvious move. **It is the
wrong one, and it will OOM.**

What actually bounds the non-vLLM budget is the top-k forward. With
`use_remove_padding=True`, `_forward_micro_batch` materializes `logits_rmpad`
of shape `(nnz, vocab)` before taking the top-k
(`dp_actor.py:218-231`). At `log_prob_micro_batch_size_per_gpu=16` and ~2560
tokens per row that is 40 960 × 151 936 × 2 B ≈ **12.4 GB for the logits
alone**, plus the `logsumexp`/`topk` temporaries and the model itself.

On the 3-GPU layout the FSDP shard adds ~2.3 GB, so the peak non-vLLM working
set is roughly 12.4 + 2.3 ≈ 15 GB — against the 32 GB that
`gpu_memory_utilization=0.6` leaves on an 80 GB card. At 0.85 it would have
12 GB and fail. (G1 would have made this worse still: at world size 1 the shard
is the whole fp32 model, ~6.8 GB.)

The correct order is therefore:
1. Lower `log_prob_micro_batch_size_per_gpu` (16 → 4 puts the logits at
   ~3.1 GB). The top-k forward is compute-bound; 4 × 2560 ≈ 10 k tokens still
   saturates a 1.7B forward, so this costs little.
2. *Then* raise `gpu_memory_utilization`, using the headroom that freed.

Both are ③-class (micro-batch grouping / KV sizing; per-row results
unchanged). Measure the peak with `GPU_PROFILER=1` rather than trusting the
arithmetic above — it is an estimate from tensor shapes, not an observation.

### G4 — Per-task `max_model_len` — **do not apply as first described**

The idea was: `max_model_len=4608` is sized for search/webshop, alfworld's
`task_overrides` cap its prompt at 2048, and Stage 1 runs one task per process,
so alfworld could run at 2560 and get ~1.8× the KV concurrency.

**This is unsafe.** The `task_overrides` apply only to the initial dataset
load. The *per-turn* prompt rebuild inside the rollout tokenizes against the
global `self.config.data.max_prompt_length` = 4096
(`rollout_loop.py:376` and the truncation branch at `:404-414`), not the task
override. So an alfworld turn prompt may legitimately reach 4096 tokens, and a
2560-token engine would reject or truncate it.

verl does not catch this for you: `vllm_rollout_spmd.py:143` takes
`max_model_len` verbatim and only asserts the *model's* context length against
`prompt_length + response_length`.

To make this mechanism available, the per-turn tokenization would have to
become task-aware (use the same override the dataset used). Until then the only
safe version is to measure the actual per-turn prompt-length distribution and
set `max_model_len` above its maximum with margin.

### G5 — Invert ⑤: `actor.fsdp_config.param_offload=True` for gen

Phase 1 adopted `PARAM_OFFLOAD=False` because the training step touches the
FSDP weights in four phases per step. In gen-only the FSDP model is used
**once per step**, for the top-k forward, and sits idle on the GPU for the
entire (minutes-long) rollout otherwise.

Offloading it returns ~2.3 GB/GPU (world size 3) or ~6.8 GB (world size 1) to
KV cache for the whole rollout. The reload cost is one H2D copy of the shard
per step — sub-second against a multi-minute step.

Accuracy class: bit-identical (FSDP placement only), same as in Phase 1.

### G6 — Drop `adjust_batch` from the generation path

`adjust_batch(..., mode="copy")` pads the turn-row batch up to a multiple of
`lcm(log_prob_micro×ws, ppo_micro×ws)` = 240 (at `n_gpus=3`) by **duplicating
randomly chosen rows**. In Stage 1 that has two consequences:

1. Those duplicates are scored by the top-k forward — wasted GPU work, up to
   239 rows/step (est. ~4% of an alfworld step's rows).
2. Those duplicates are **saved into `<task>.pt`**. Stage 2's
   `_load_offpolicy_data` groups rows by `traj_uid`, so the affected
   trajectories carry extra copies of some turn-rows into training.

Neither divisor applies to generation: `update_actor` never runs, and the
top-k call already sets `DataProtoConfig.auto_padding_key=True` for DP
divisibility (`opd_offpolicy_ray_trainer.py:128`). Removing the call is both a
speedup and a data-quality fix.

Accuracy class: **changes the Stage-2 dataset** (removes duplicated rows). It
should be a deliberate, separately-flagged decision, not folded into a
throughput commit.

### G7 — Overlap top-k scoring with the rollout (the gen analog of A)

Phase 2's mechanism A moves `old_log_prob` into the rollout's `envs.step`
windows. Stage 1 has no `old_log_prob` — which is why the script correctly
tells you to leave `ROLLOUT_PREFETCH_LOGPROB` off — but it has the *same shape*
of work in `compute_actor_topk_log_prob`, serialized after the rollout.

Port A to top-k (`ROLLOUT_PREFETCH_TOPK=1`): score finished trajectories'
rows in the GPU-idle `envs.step` windows of later turns. The existing pending
pool / merge machinery in `rollout_loop.py`
(`_prefetch_pending_log_probs`, `take_prefetched_log_probs`) and its CPU tests
transfer directly; only the worker call and the merged key names change.

Ceiling: the whole top-k phase (est. 5–10% of a gen step — ~6 M forward tokens
per alfworld step ≈ 2×10^16 FLOPs), bounded by the available idle window, which
in gen is *larger* than in training because there is no training phase
competing for the GPU.

### G8 — Port mechanism C (env-reset prefetch) to the generator

C is implemented in `RLSDRayTrainer.fit()` / `skillsd_ray_trainer.fit()`, not
in `TeacherTrajectoryGenerator.generate()`. Stage 1 does reset envs every step,
so the mechanism applies; it is "off" only because the generator never calls
it. The overlap window in gen is the top-k phase (and, with G7, whatever is
left of it). alfworld reset = game-file loading, so this is worth real seconds
× 300 steps × 3 tasks.

`filter_groups.enable` is off in this config, so C's dynamic-sampling exclusion
does not apply. Bit-identical.

### G9 — Stop calling `empty_cache()` on every turn

`fsdp_workers.py:681` runs `get_torch_device().empty_cache()` at the end of
*every* `generate_sequences` — i.e. once per turn, up to 50× per step, ~15000×
per alfworld run. Under `free_cache_engine=False` + `ROLLOUT_KEEP_VLLM_AWAKE`
there is nothing to reclaim, and it throws away the caching allocator's blocks
so the next turn re-allocates from scratch. The sharding manager itself
documents this anti-pattern (`fsdp_vllm.py:160-166`: *"Out of vllm scope, we
should avoid empty cache"*).

Gate it on `_rollout_session_active` (skip inside a session, keep the current
behavior outside). Bit-identical; benefits the training path too.

### G10 — Take the top-k from vLLM instead of a second forward (research-grade)

vLLM already computes the full logit vector at every sampled position.
`SamplingParams(logprobs=20)` returns exactly the top-20 the teacher forward is
re-deriving, which would delete the entire `compute_actor_topk_log_prob` phase
**and** the FSDP actor's GPU residency from Stage 1.

Two reasons this is listed last, not first:
- **Numerics.** vLLM's fused kernels and the HF forward do not agree bitwise in
  bf16; the KD targets Stage 2 trains on would change. `temperature=1.0` here,
  so the temperature/logits-processor question is moot, but the kernel
  difference is not. This is a **scientific** change requiring an A/B (top-20
  id agreement rate, log-prob MAE, Stage-2 `teacher_gap`/loss curves).
- **CPU cost.** vLLM materializes per-position logprobs as Python objects;
  20 × up-to-512 tokens × 480 sequences per step is a serialization load that
  can eat the forward it saves. Must be measured, not assumed.

Worth a scoped experiment on a 1000-trajectory pool before committing.

---

## 3. Storage — likely the binding constraint at 36000 trajectories

Not a speed mechanism, but it gates whether the run finishes at all, and the
final `concat` + `save_to_disk` is real wall time.

Rows are padded to `data.max_prompt_length + data.max_response_length` =
4096 + 512 = 4608 **for every task**. The per-task `max_prompt_length` override
that caps alfworld at 2048 does not shrink this: it applies only to the initial
dataset load, while the rollout's per-turn retokenization pads to the global
value (`rollout_loop.py:376` → `torch_functional.py:360` →
`postprocess_data(..., max_length)`). Same finding as G4, and it is also why
Stage 2's `DataProto.concat` over the three tasks' files works at all.

| Key | dtype | bytes/row |
|---|---|---|
| `input_ids`, `attention_mask`, `position_ids` | int64 × 4608 | 3 × 36.9 KB |
| `prompts` | int64 × 4096 | 32.8 KB |
| `responses`, `response_mask` | int64 × 512 | 2 × 4.1 KB |
| `teacher_topk_logprobs` | fp32 × 512 × 20 | 41 KB |
| `teacher_topk_ids` | int64 × 512 × 20 | 82 KB |
| **total** | | **≈ 275 KB** |

36000 alfworld trajectories × an assumed ~20 turn-rows each ≈ 720 k rows ≈
**198 GB**, held entirely in a Python list first and then *doubled* at peak by
`DataProto.concat`. search (4 turns) ≈ 40 GB, webshop ≈ 80 GB. The sequential
layout at least accumulates one task at a time, so peak RAM is the largest task
rather than their sum — one more reason G1 was the wrong trade.

Cheap, additive fixes:
- `teacher_topk_ids` → int32 (vocab 151 936 ≪ 2³¹): −41 KB/row.
- `teacher_topk_logprobs` → fp16/bf16 (values are ≤ 0 log-probs feeding a KL;
  fp32 is not buying accuracy here): −20 KB/row.
- masks → bool: −36 KB/row. token ids → int32: −12 KB/row.
- drop `position_ids` (recomputable from `attention_mask` at load) and
  `prompts` (a prefix of `input_ids`): −37 KB/row.

Together ≈ 275 KB → ≈ 128 KB/row, a **2.1×** reduction. None of these are
implemented.

**Shard incrementally — IMPLEMENTED.** `+gen.shard_every_steps=N` (0 = the old
single-file behavior; the run script sets 10) flushes the buffer to
`<task>_{i:04d}.pt` every N steps and clears it, so peak RAM is one shard's
worth instead of the whole run's, and the doubling `DataProto.concat` at the
end is gone. Stage 2 needs no change — `_load_offpolicy_data` already globs
`*.pt` in `teacher_data_dir`. A crash now costs at most N steps instead of
everything.

Because Stage 2 globs the directory, a rerun first deletes that task's own
`<task>_[0-9]*.pt` and any legacy `<task>.pt` (logged per file): a stale shard
from a longer previous run would otherwise be silently concatenated into the
new dataset. This matches the existing contract that rerunning a task
overwrites its output.

---

## 4. Which existing mechanisms to enable for gen-only

`Y` = enable, `N` = no effect / not applicable, `!` = differs from the training
recommendation.

| # | Mechanism | Knob | gen-only | Why |
|---|---|---|---|---|
| ① | vLLM session | `ROLLOUT_KEEP_VLLM_AWAKE=1` | **Y** | Biggest existing win, and *more* valid here: the teacher is frozen for the whole run, not just one rollout (see G9, and a run-level session is the natural extension). |
| ② | active-only preproc | `ROLLOUT_SKIP_DONE_PREPROC=1` (default on) | **Y** | Keep on. Value scales with the alfworld tail, which gen has in full. |
| ③ | prefix caching | `+...rollout.enable_prefix_caching=True` (already in script) | **Y** | Worth more here than in training: the frozen teacher means the cache stays valid across *all 300 steps*, not just within one rollout. |
| ④ | task interleave | `TASK_BALANCE_INTERLEAVE=1` | **N !** | **No-op.** Stage 1 restricts `task_balance.tasks` to a single task (`main_opd_offpolicy_gen.py:98`), and the interleave only reorders *across* tasks (`main_ppo.py:298-303`). Leave unset. |
| ⑤ | param offload off | `actor.fsdp_config.param_offload` | **INVERT !** | Training wants `False`; gen wants **`True`** — see G5. |
| A | log-prob prefetch | `ROLLOUT_PREFETCH_LOGPROB=1` | **N** | Correct as the script says: no `old_log_prob` phase. Replace with G7 (top-k prefetch). |
| B | CUDA-graph decode | `CUDAGRAPH_CAPTURE_SIZES` | **measure** | The Phase-1/2 note that this needs `VLLM_USE_V1=1` is stale for the pinned `vllm==0.11.0`, which has no V0 engine. Capture sizes should cover the *gen* batch sizes, which are much larger than training's if G2 is on. Gen is also more prefill-heavy than training (every turn re-prefills the history, responses are short), so expect less from decode graphs here — measure before adopting. |
| C | env-reset prefetch | `ENV_RESET_PREFETCH=1` | **N (port)** | Not wired into the generator — see G8. Setting the env var today does nothing. |
| D | retriever query cache | `SEARCH_QUERY_CACHE=1` | **Y (search only)** | High hit rate: the 8 GRPO group members share a question, so turn-1 queries repeat 8×. Requires a deterministic retriever (fixed index). Useless for alfworld/webshop. |
| E1 | parallel tokenization | `ROLLOUT_PREPROC_WORKERS=8` | **Y** | Bit-identical, and preproc is a larger share of a gen step than of a training step (no training phases to hide behind). |
| E2 | active-only decode | `ROLLOUT_DECODE_ACTIVE_ONLY=1` (default on) | **Y** | Keep on. |
| E3 | compact record | `ROLLOUT_COMPACT_RECORD=1` (default on) | **Y** | Keep on. |
| — | profilers | `GPU_PROFILER=1 ROLLOUT_TURN_TIMING=1` | **Y, first run** | Gen has never been profiled; §6. |

Config knobs carried over from the training script that are **dead** in gen and
can be dropped for clarity (no runtime effect):
`actor.optim.*`, `ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu` (except
via `adjust_batch`'s divisor — see G6), all `actor_rollout_ref.ref.*` (no ref
worker is created under grpo + `use_kl_in_reward=False`), and
`enable_gradient_checkpointing=True` (HF skips checkpointing when
`not self.training`, and `compute_topk_log_prob` calls `.eval()`).

---

## 5. Recommended configuration

**Applied.** In the run script:

```bash
export ROLLOUT_KEEP_VLLM_AWAKE=1     # ①
export ROLLOUT_PREPROC_WORKERS=8     # E1
export SEARCH_QUERY_CACHE=1          # D (search block only; needs a fixed index)
# ②/E2/E3 default on; ③ already passed as a Hydra arg
```

plus `+gen.shard_every_steps=10` (§3) and the `_timer` instrumentation (§6).

**Layout: unchanged (sequential, 3 GPUs per task).** G1 was implemented and
reverted — see G1 for why concurrency loses when one task dominates the wall.

**Next, and only with a profile in hand (G3, in this order).** Lower
`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` 16 → 4, *then*
raise `actor_rollout_ref.rollout.gpu_memory_utilization` from 0.6. Doing the
second without the first OOMs the top-k forward — read G3.

**Cheap and independent.**
`actor_rollout_ref.actor.fsdp_config.param_offload=True` (G5, bit-identical):
returns ~2.3 GB/GPU (world size 3) to the KV cache for the whole rollout, since
`fsdp_vllm.py:200-203` offloads the shard immediately after syncing weights into
vLLM and `compute_actor_topk_log_prob` reloads it only for its own forward.

**Then (needs an expectations-file edit).** G2: `per_task_batch_size=30`,
`num_batches=150`.

**Then (code).** G9 (bit-identical, smallest diff) → G8 → G7 → storage (§3).
G4 needs a code change first — see G4.

**Separately, as scientific decisions.** G6 (removes duplicated rows from the
Stage-2 dataset) and G10 (changes the KD targets).

---

## 6. Measurement plan

Stage 1 has no profile on this branch; every "est." above should be replaced.

`TeacherTrajectoryGenerator.generate()` is instrumented with `_timer` phases
**`step` / `gen` / `teacher_topk` / `collect`**, and each step prints its own
breakdown. That instrumentation is what makes the profiler work at all here:
`gpu_profiler`'s sampler is started lazily by the first `push_phase()`, which
only `_timer` issues, and popping `step` (its default
`GPU_PROFILER_BOUNDARY`) is what emits the per-step utilization report. Before
it existed, `GPU_PROFILER=1` was a silent no-op for Stage 1 and
`ROLLOUT_TURN_TIMING`'s `genGPU%` / `DP-IMBALANCE` columns printed `-`.

1. Baseline: `GPU_PROFILER=1 ROLLOUT_TURN_TIMING=1`, alfworld,
   `+gen.num_trajectories=1200` (10 steps). Record: step wall, gen wall,
   top-k-phase wall, `envs.step` wall, gen SM%, and vLLM preemption counts.
   The `step = gen + topk + collect` line gives the first three directly, and
   is the number to check G7 against (how much of `teacher_topk` is
   overlappable) and G3 against (whether `collect` is growing as RAM fills).
2. The KPI is **wall time per 1000 trajectories**, not SM%. Phase 1 §5 already
   proved these move in opposite directions here (removing waste *lowers* util
   and *raises* throughput); gen amplifies that.
3. Watch vLLM's preemption / "cache full" log lines when adopting G2/G3 —
   the failure mode of over-subscribing KV is recompute thrash that looks like
   a plain slowdown.
4. Accuracy gate for the ③-class mechanisms (G2): the trajectory count,
   per-task success rate and mean episode length of the generated pool should
   track the baseline within sampling noise. For G6/G10, gate on Stage 2:
   `sdar/teacher_gap` and the loss curve.
