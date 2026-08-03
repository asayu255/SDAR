# SDAR 3-task Multitask — GPU Utilization / Throughput Optimization Report

Branch: `claude/festive-gates-fxercy`
Setup: SDAR multitask RL (alfworld + search + webshop), Qwen3-1.7B, 2 GPUs,
FSDP + colocated vLLM, `env.max_steps` 50 (per-task caps alfworld=50 / search=4 /
webshop=15), GRPO group_size=8, mixed batch 384 (128/task).
Goal: raise GPU utilization / throughput **without changing accuracy**.

---

## 1. Executive summary

| Metric (per training step) | Before¹ | After² | Δ |
|---|---|---|---|
| GPU util (wall-weighted SM) | 82.6% | ~88.8% | **+6.2 pp** |
| step wall time | 3784 s | ~3290 s | **−13%** |
| throughput / GPU | ~847 tok/s | ~905 tok/s | **+7%** |
| **gen (rollout) wall** | 977 s | **~520 s** | **−47%** |
| gen per-token | 0.708 ms | 0.496 ms | **−30%** |
| gen-phase DP imbalance | ~20 pp (40/60) | ~8 pp | **−60%** |
| peak mem reserved | ~50 GB | ~51 GB | flat |
| accuracy (val/success, teacher-gap, pg-loss) | — | **unchanged** | ✓ |

¹ earliest full measurement (retriever offline, no session/interleave).
² final config (retriever online + adopted mechanisms below). Step throughput
carries ±5–7% step-to-step variance from per-step token counts (5.8–6.9M).

**Headline:** the rollout (`gen`) phase — the only under-utilized part — was cut
~47% in wall time; training phases were already compute-bound (~97% SM) and were
left untouched. All gains are accuracy-preserving.

---

## 2. How we found the bottleneck (measurement infra)

Two opt-in, zero-overhead-when-off instruments were added and kept:

- **`verl/utils/gpu_profiler.py`** — background NVML sampler that tags every
  sample with the current `_timer` phase and reports, per phase: SM%, memory-BW%,
  idle%, mem, power, clock, PCIe, and **per-GPU** SM. Enabled by `GPU_PROFILER=1`.
- **per-turn rollout timing** (`rollout_loop.py`, `ROLLOUT_TURN_TIMING=1`) —
  per turn: active count, preproc / gen / decode / env.step seconds, the SM util
  *during generation* (`genGPU%`), per-GPU split, and a `DP-IMBALANCE` summary.

Findings:
- Only `gen` is under-utilized; `old_log_prob`/`teacher_forward`/`ref`/
  `update_actor` run at 90–97% SM (compute-bound, already optimal).
- `gen` idle decomposed into: CPU "glue" between generate calls (preproc / decode
  / env.step, GPU idle), per-turn redundant vLLM weight-sync/wake-sleep, and
  data-parallel load imbalance from mixing tasks across the 2 GPUs.

---

## 3. Adopted mechanisms and measured effect

| # | Mechanism | Flag / knob | Measured effect | Accuracy |
|---|---|---|---|---|
| ① | **vLLM session** (keep engine awake, sync frozen weights **once** per rollout instead of every turn) | `ROLLOUT_KEEP_VLLM_AWAKE=1` | gen wall 738→562 s, gen/tok 0.708→0.541 (**−24%**); throughput 877→932 (+6.3%) | bit-identical (frozen weights; RNG preserved) |
| ② | **active-only preprocess** (skip tokenization of finished trajectories) | `ROLLOUT_SKIP_DONE_PREPROC=1` (default on) | preproc 95.5→37 s/step (**−61%**) | bit-identical (finished rows dropped downstream) |
| ③ | **vLLM prefix caching made configurable** | `+...rollout.enable_prefix_caching=True` | marginal (already on for the spmd path); persists across turns under ① | lossless KV reuse |
| ④ | **Fix1 – interleaved task layout** (round-robin instead of contiguous task blocks → balanced DP split) | `TASK_BALANCE_INTERLEAVE=1` | DP imbalance 20→8 pp, gen util 51→57%, gen/tok 0.541→0.496 (**−8%**) | bit-identical (reorder only; uid groups unchanged) |
| ⑤ | **FSDP param-offload off** (≥64 GB headroom) | `PARAM_OFFLOAD=False` | small; avoids per-phase weight gather | bit-identical (placement only) |
| — | **retriever online** (operational, not code) | `SEARCH_URL=…` | env.step 159→28 s/step (**−131 s**) + valid search training | — |
| — | run-script env knobs (seed/epochs/test_freq/search_url/max_model_len/skills/offload/micro-batch) | env vars | reproducible, conflict-free overrides | — |

Compounded gen-phase result: **gen/tok 0.708 → 0.496 (−30%)**, **gen wall
977 → ~520 s (−47%)** (① + ② + ④ + retriever).

---

## 4. Rejected / not-adopted mechanisms

| Mechanism | Why rejected | Status |
|---|---|---|
| **Fix2 – per-task generation** (one generate call per task) | Achieved its goal (homogeneous per-task util) but the extra per-call overhead **negated Fix1's gen speedup**: Fix1-only gen/tok 0.496 vs Fix1+Fix2 0.53–0.57, and peak mem 51→55.6 GB. No throughput gain. | **removed from branch** |
| `OPTIMIZER_OFFLOAD=False` (GPU Adam) | `update_actor` is 97% SM compute-bound → no speedup; only added resident memory and drove peak mem 51→60 GB with KV preemption. | kept as an env knob, **left default (True)** |
| `cudagraph_capture_sizes` | V1 `CompilationConfig` feature; under `VLLM_USE_V1=0` it does not apply. | not used |
| Async / continuous-batching rollout | The only path to raise `gen` util further, but requires de-vectorizing the env packages + a token-level async engine (multi-week). | **deferred**; design kept in `docs/async_rollout_project.md`, scaffolding in `agent_system/multi_turn_rollout/async_rollout_core.py` (+ CPU test). Not in the production path. |

---

## 5. Key insight: utilization ≠ throughput

The isolation experiment produced the central lesson of this work:

- **vLLM session (①)** *lowered* gen SM% (60→51%) yet *sped up* gen 24% — the
  previous "high" util included redundant weight-sync busy-work. Removing waste
  lowers the util number but increases throughput.
- **Fix1 (④)** *raised* gen SM% (51→57%) and *also* sped gen 8% — here the
  imbalance was real idle, and filling it helped.
- The remaining `gen` SM ~57% is **not recoverable idle**: it is small-model
  (1.7B) autoregressive decode, which is memory-bandwidth / critical-path bound
  (wall set by the longest trajectory's sequential token-by-token decode).
- Crucially, the 3 tasks share one alfworld-gated rollout wall, so the multitask
  run already produces ~3 tasks' worth of trajectories in ~1 task's wall time —
  the low gen util is largely **cosmetic**, and throughput is near its ceiling.

**Conclusion:** 100% GPU util is structurally impossible (and not the right
target) for this workload. The meaningful KPI is throughput / wall time, which we
improved; further gen gains require the deferred async rollout.

---

## 6. Accuracy preservation

The rollout engine only chooses *which trajectories are sampled*; all training
log-probs (`old_log_prob` / `teacher_forward` / `ref`) are **recomputed** by the
FSDP actor on the sampled tokens (`vllm ... logprobs=0`). So preserving the
sampling distribution preserves training.
- ②, ④, ⑤ and the session weight-sync are **bit-identical** (finished-row /
  reorder / placement / frozen-weight changes); RNG state is preserved across the
  session.
- ③ (prefix caching) and vLLM batch-composition changes are lossless /
  distribution-preserving (not bit-identical, the same standard already in use).
- Observed: `val/*success_rate`, `sdar/teacher_gap`, `actor/pg_loss`,
  `grad_norm`, `entropy_loss` tracked the baseline across steps; initial val
  identical (same seed).

---

## 7. Final production configuration

```bash
export ROLLOUT_KEEP_VLLM_AWAKE=1     # ① session (biggest wall-time win)
export ROLLOUT_SKIP_DONE_PREPROC=1   # ② active-only preproc (default on)
export PARAM_OFFLOAD=False           # ⑤ bit-identical
export TASK_BALANCE_INTERLEAVE=1     # ④ Fix1 (DP balance, gen −8%)
# enable_prefix_caching=True is set in the run script (③)
# OPTIMIZER_OFFLOAD unset (=True), ROLLOUT_PER_TASK_GEN removed (Fix2 dropped)
export SEARCH_URL=http://<retriever-host>:8000/retrieve   # retriever online
# measurement (optional): GPU_PROFILER=1 ROLLOUT_TURN_TIMING=1
bash examples/sdar_trainer/run_multitask_qwen3_1_7b_no_preprocess.sh vllm
```

## 8. Remaining headroom
- `gen` SM ~57% is decode-bound; only **async continuous-batching** (deferred,
  see `docs/async_rollout_project.md`) can overlap the alfworld long-tail decode
  with other work to push it higher.
- `update_actor` (≈46% of step) and `teacher_forward` are compute-bound at ~97%;
  no accuracy-safe wall-time left there.

---

## 9. Phase 2 (follow-up branch)

The deferred headroom above is partially claimed by the Phase 2 mechanisms
(finished-trajectory log-prob prefetch, env-reset prefetch, active-only
decode/record, CUDA-graph knobs) — see `docs/optimization_phase2.md`. Two more
were prototyped there and have since been removed: a retriever query cache and a
parallel prompt tokenizer.

---

## 10. Phase 3 — the actor update, and the host that made it visible

Phases 1–2 left `update_actor` alone on the reading in §8 that it was
compute-bound at ~93–97% SM. That reading does not survive the move to a 2-GPU
host without NVLink: `update_actor` carries by far the highest PCIe traffic of
any phase (≈10.5 GB/s TX, 9.0 GB/s RX vs ≈2.0/1.9 during `gen`), and NCCL
collectives count as SM-busy, so SM% cannot distinguish computing from
communicating. Same lesson as §5, one layer down.

### Mechanisms

| Mechanism | Knob (default) | What it removes | Accuracy class |
|---|---|---|---|
| **ZeRO-2** — keep parameters gathered from forward through backward | `actor.fsdp_config.sharding_strategy` = `shard_grad_op` (default; `null` restores ZeRO-3) | 3 all-gathers per layer per micro-batch under gradient checkpointing → 1 | arithmetic-neutral (placement) |
| **`no_sync` gradient accumulation** — reduce once per mini-batch | `actor.no_sync_grad_accum` = `True` (default) | 12 reduce-scatters per mini-batch → 1 at `ppo_mini_batch=60 / micro=5`; under ZeRO-2 also the per-micro-batch re-gather | **not** bit-identical: partial sums reduce in a different order (identical expectation) |
| **FSDP forward prefetch** | `actor.fsdp_config.forward_prefetch` = `True` (default) | serialization of the next unit's all-gather behind the current unit's compute | scheduling only |
| **Resident reference policy** — `ref.fsdp_config.param_offload` decides again, instead of FSDP `CPUOffload` being forced on | `ref.fsdp_config.param_offload` = `False` (default) | a full model fetched over PCIe per micro-batch (measured 7.6–8.9 GB/s sustained through the reference forward) | placement only (the ref runs under `no_grad`); costs `param_bytes / world_size` |
| **Deferred metric reads** — keep logger-only scalars as 0-d GPU tensors until the end of `update_policy` | always on | several hundred forced host↔device syncs per step | identical values, except `actor/kl_loss` and `actor/sdl_loss`, which were assigned rather than appended and so reported only the last micro-batch |
| **Exact token-mean under dynamic bsz** | `+actor.dynamic_bsz_token_scale=True` (**off**) | nothing — it makes `use_dynamic_bsz` safe to turn on by removing the objective's dependence on the packing | changes the loss under dynamic bsz (to the correct one) |

Everything except the last row is on by default. `dynamic_bsz_token_scale` stays
off because it is the one entry that changes the loss rather than how it is
computed, and it only applies when `use_dynamic_bsz` is on — which the multitask
recipes do not use.

Scope of the defaults: `sharding_strategy`, `forward_prefetch` and
`no_sync_grad_accum` live under `actor_rollout_ref.actor` in
`verl/trainer/config/ppo_trainer.yaml`, so they change the **actor** update for
every recipe that composes that file. The ref, critic and reward_model workers
have their own `fsdp_config` blocks, do not carry these keys, and keep the mesh
default. That separation is not a nicety for the ref: a recipe is free to ask for
`param_offload=True` there, and ZeRO-2 exists to keep parameters resident, so the
two would cancel — the combination `get_sharding_strategy` warns about.

Two consequences worth stating plainly. `no_sync_grad_accum=True` means a run
started after this change will not reproduce an earlier run's gradients bit for
bit; set it to `False` when that matters. And ZeRO-2 raises peak memory by
roughly the unsharded parameter size minus its shard, so a model that only just
fits under ZeRO-3 needs `sharding_strategy=null`.

Both sit on the gradient path, so a comparison run should pin them next to its
scientific knobs rather than trusting the default to stay put.

### Instruments

- **worker-side stage phases** (`actor.fwd` / `actor.bwd` / `actor.task_metrics`
  / `actor.optim`) — the driver's `_timer` cannot see inside `update_policy`
  because that runs in a worker; rank 0 pushes its own phases so the
  `update_actor` bucket gets an interior. `GPU_PROFILER_SYNC_PHASES=1` makes the
  split exact at the cost of serializing overlap.
- **`timing_s/update_actor_worker`** — `timing_s/update_actor` is a blocking
  `ray.get`, so it also covers serializing a few-hundred-MB batch into the object
  store and back. The difference between the two is transport.
- **NVLink and driver-CPU columns, cumulative table, per-sample trace** — see the
  module docstring of `verl/utils/gpu_profiler.py`. On a node *with* NVLink the
  collectives are invisible to the PCIe counters, so `nvlink_mb_s` is the only
  place they show up at all; the cumulative table is the only place periodic
  phases (validation, checkpointing) show their real share of a run.

Separately, `verl/utils/flops_counter.py` had no entry for either host's GPU, so
`perf/mfu/actor` divided by `inf` and logged exactly `0.000` every step — a
metric that reads as broken rather than missing. RTX A6000 and RTX PRO 6000
Blackwell (incl. the 300W Max-Q bin) are now in the table.

### Host memory, which is what actually killed the runs

Two fixes outside the GPU entirely, both required to finish a run on a 256GB box:

- **Lazy environment construction** (`LazyEnvManager`) — the val envs are Ray
  actors that sit idle until the first validation, and are never touched at all
  with `test_freq <= 0`; the train envs are dead weight in a `val_only` run. In
  multitask that is 252 of 492 actors either way, each holding its own copy of the
  environment's data.
- **WebShop worker JVM cap and session pruning** — see
  `docs/webshop_worker_memory.md`. ~1.8 GB/step of host RAM, which is what the OOM
  killer was reacting to around step 24.

### Turning them off

No run script has to opt in — the defaults are in `ppo_trainer.yaml`, and
`tests/trainer/test_speedup_defaults.py` pins them, since flipping one is a
one-character edit no other test would notice. Note that the three actor keys
now *exist* in the config, so a run script must write them without Hydra's `+`
append form, which refuses a key that is already there.

To restore the pre-Phase-3 behaviour, per run or in the yaml:

```bash
    actor_rollout_ref.actor.fsdp_config.sharding_strategy=null \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=False \
    actor_rollout_ref.actor.no_sync_grad_accum=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
```

The one to reach for first is `no_sync_grad_accum=False`, when a run has to
reproduce earlier gradients bit for bit; then `sharding_strategy=null`, if the
model no longer fits.
