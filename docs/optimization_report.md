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
