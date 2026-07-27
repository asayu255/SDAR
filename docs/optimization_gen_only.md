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

Fact 3 is the largest single lever and it does not exist in the training
workload at all.

---

## 2. New gen-only mechanisms, ranked

### G1 — Run the three tasks concurrently, 1 GPU each (est. 2.5–3× wall)

Today: `alfworld(3 GPU) → search(3 GPU) → webshop(3 GPU)`, sequential.
`tensor_model_parallel_size=1` already, so each GPU holds a complete engine and
the 3-way split is pure data parallelism.

Instead run all three at once, one GPU per task (`trainer.n_gpus_per_node=1`,
`CUDA_VISIBLE_DEVICES=<i>`, `&` + `wait`).

Why this is nearly free: per-GPU trajectory count goes 40 → 120, but 1.7B
decode is **weight-bandwidth bound**, so the per-token cost of a decode step is
almost flat in batch size over this range. Meanwhile the wall time of a task no
longer waits on the other two. Also removes the cross-GPU collective in every
`generate_sequences` dispatch and makes DP imbalance structurally impossible
(one task per device — Phase 1's ④ solved the same problem the hard way).

Caveats, all real:
- KV cache: 120 concurrent sequences on one GPU. At alfworld's actual prompt
  length (`max_prompt_length=2048`) ≈ 114 KB/token × ~2000 tokens ≈ 230 MB per
  sequence → ~27 GB (est.). Needs G3 (`gpu_memory_utilization`) and G4
  (per-task `max_model_len`), or vLLM will preempt and you lose the win.
- CPU/RAM: 3 env fleets alive at once (alfworld game loading is the heavy one),
  and the search task hits the retriever alone rather than sharing a window.
- Ray: three `ray.init()` instances on one host. Give each a distinct
  `RAY_TMPDIR`/temp dir so the GCS/dashboard ports and object store do not
  collide.

Accuracy class: not bit-identical (DP world size changes vLLM batch
composition), same distribution-preserving class as ③.
Side benefit: `adjust_batch`'s divisor drops from `lcm(16·3, 5·3)=240` to
`lcm(16,5)=80`, so fewer duplicated rows (see G6).

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
KV. Combines with G1 only up to what one GPU's KV can hold — pick the pair
together, e.g. G1 + `per_task_batch_size=30`.

Requires editing `expected_multitask_offpolicy_gen_config.yaml` in the same
commit (`data.train_batch_size`, `data.task_balance.per_task_batch_size` are
pinned there) — the intent lock is doing its job; the knob is genuinely
performance-only *on the generation side*, but the same key is a scientific
knob for Stage 2, which is why it is pinned.

Accuracy class: distribution-preserving (③-class). Trajectory count, prompt
pool coverage and sampling temperature are unchanged; only which prompts share
a batch changes.

### G3 — `gpu_memory_utilization` 0.6 → 0.80–0.85

0.6 is inherited from the *training* script, where the same GPU must also hold
FSDP gradients, Adam state and `update_actor` activations. In gen-only none of
those exist. The non-vLLM resident set is the FSDP shard (fp32, ~2.3 GB/GPU at
world size 3; ~6.8 GB at world size 1) plus the top-k forward's activations.

Everything freed goes to KV cache → more concurrent sequences → this is what
makes G1 and G2 land instead of preempting.

### G4 — Per-task `max_model_len` (est. 1.8× KV capacity for alfworld)

`max_model_len=4608` is `4096 + 512`, sized for search/webshop. But alfworld's
`task_overrides` cap its prompt at 2048, and Stage 1 runs **one task per
process** — so alfworld can run at `max_model_len=2560`. vLLM sizes its
per-sequence block budget from this, so alfworld gets ~1.8× the concurrency at
the same memory. alfworld is the task with the longest tail, so this is the
task where concurrency matters most.

Pure speedup, truncates nothing (the task's own prompt cap is 2048).

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

Per saved turn-row, at alfworld's shape (prompt 2048 + response 512 = 2560):

| Key | dtype | bytes/row |
|---|---|---|
| `input_ids`, `attention_mask`, `position_ids` | int64 × 2560 | 3 × 20.5 KB |
| `prompts` | int64 × 2048 | 16.4 KB |
| `responses`, `response_mask` | int64 × 512 | 2 × 4.1 KB |
| `teacher_topk_logprobs` | fp32 × 512 × 20 | 41 KB |
| `teacher_topk_ids` | int64 × 512 × 20 | 82 KB |
| **total** | | **≈ 209 KB** |

36000 alfworld trajectories × an assumed ~20 turn-rows each ≈ 720 k rows ≈
**150 GB in one `.pt`**, held entirely in a Python list first and then
*doubled* at peak by `DataProto.concat`. search (4 turns) lands around 30 GB.

Cheap, additive fixes:
- `teacher_topk_ids` → int32 (vocab 151 936 ≪ 2³¹): −41 KB/row.
- `teacher_topk_logprobs` → fp16/bf16 (values are ≤ 0 log-probs feeding a KL;
  fp32 is not buying accuracy here): −20 KB/row.
- masks → bool: −36 KB/row. token ids → int32: −12 KB/row.
- drop `position_ids` (recomputable from `attention_mask` at load) and
  `prompts` (a prefix of `input_ids`): −37 KB/row.

Together ≈ 209 KB → ≈ 62 KB/row, a **3.4×** reduction.

- **Shard incrementally.** Write `<task>_shard{i}.pt` every N steps instead of
  accumulating. Stage 2 already globs `*.pt`
  (`opd_offpolicy_ray_trainer.py:195`), so sharded output loads with no Stage-2
  change — and a crash stops costing the whole run.

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
| B | CUDA-graph decode | `CUDAGRAPH_CAPTURE_SIZES` | **measure** | The Phase-1/2 note that this needs `VLLM_USE_V1=1` is stale for the pinned `vllm==0.11.0`, which has no V0 engine. Capture sizes should cover the *gen* batch sizes, which are much larger than training's if G1/G2 are on. Gen is also more prefill-heavy than training (every turn re-prefills the history, responses are short), so expect less from decode graphs here — measure before adopting. |
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

**Now (no code change).** Add to the script:

```bash
export ROLLOUT_KEEP_VLLM_AWAKE=1     # ①
export ROLLOUT_PREPROC_WORKERS=8     # E1
export SEARCH_QUERY_CACHE=1          # D (search block only; needs a fixed index)
# ②/E2/E3 default on; ③ already passed as a Hydra arg
```

and change these Hydra args (performance-only, not in the expectations file):

```
actor_rollout_ref.rollout.gpu_memory_utilization=0.85     # G3 (was 0.6)
actor_rollout_ref.actor.fsdp_config.param_offload=True    # G5 (was False)
actor_rollout_ref.rollout.max_model_len=2560              # G4, alfworld block only
```

**Next (script-only, biggest win).** G1: three tasks concurrently at
`trainer.n_gpus_per_node=1`, one `CUDA_VISIBLE_DEVICES` each, distinct
`RAY_TMPDIR`, `&` + `wait`.

**Then (needs an expectations-file edit).** G2: `per_task_batch_size=30`,
`num_batches=150`.

**Then (code).** G9 (bit-identical, smallest diff) → G8 → G7 → storage (§3).

**Separately, as scientific decisions.** G6 (removes duplicated rows from the
Stage-2 dataset) and G10 (changes the KD targets).

---

## 6. Measurement plan

Stage 1 has no profile on this branch; every "est." above should be replaced.

1. Baseline: `GPU_PROFILER=1 ROLLOUT_TURN_TIMING=1`, alfworld,
   `+gen.num_trajectories=1200` (10 steps). Record: step wall, gen wall,
   top-k-phase wall, `envs.step` wall, gen SM%, and vLLM preemption counts.
2. The KPI is **wall time per 1000 trajectories**, not SM%. Phase 1 §5 already
   proved these move in opposite directions here (removing waste *lowers* util
   and *raises* throughput); gen amplifies that.
3. Watch vLLM's preemption / "cache full" log lines when adopting G1/G2/G3 —
   the failure mode of over-subscribing KV is recompute thrash that looks like
   a plain slowdown.
4. Accuracy gate for the ③-class mechanisms (G1, G2): the trajectory count,
   per-task success rate and mean episode length of the generated pool should
   track the baseline within sampling noise. For G6/G10, gate on Stage 2:
   `sdar/teacher_gap` and the loss curve.
