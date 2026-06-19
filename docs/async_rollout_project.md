# Async Continuous-Batching Rollout — Project Scope

Status: PROPOSED (formal scope). No production code beyond the measurement infra
and the orchestration core (`async_rollout_core.py`, CPU-tested) has been written.
Branch: `claude/festive-gates-fxercy`.

## 1. Objective & success criteria
- Raise the **gen (rollout) phase** GPU utilization from ~60% → ~90%, and overall
  wall-weighted util from ~87% → **~92–95%**.
- **Accuracy unchanged** in the distributional sense (same policy/sampling/env →
  same training distribution). Not bit-identical (continuous batching changes FP
  ordering); verified by N-step sync-vs-async metric-curve match.
- Fully **flag-gated**: `actor_rollout_ref.rollout.mode=sync` (or a collector flag)
  restores the current path instantly.

## 2. Root cause (evidence)
GPU-util timeline splits cleanly into a spiky region and a pinned-100% region.
Mapped to phases via the profiler:
- **Pinned 100%** = training (old_log_prob 96% / teacher_forward 89% / ref 95% /
  update_actor 97%) — compute-bound, already saturated. Not a target.
- **Spiky** = the **rollout/gen** phase: lock-step turns of
  `generate (GPU) → env.step + preproc + decode (CPU, GPU idle)` × ~50 turns, and
  the generation itself only ~65% util. This is the sole lever.

Lock-step is enforced by (a) a single batched `generate_sequences` per turn and
(b) **vectorized** env stepping. Fix = per-sequence independent progression +
continuous batching so the engine always has work and CPU/env latency of some
sequences overlaps generation of others.

## 3. Feasibility findings (de-risking)
- **Loss recomputes log-probs** on the sampled tokens (`old_log_prob`/`teacher`/
  `ref` are FSDP recomputes; vLLM runs with `logprobs=0`). So the rollout engine
  only determines *which trajectories are sampled* → accuracy is distributional.
- **Envs are internally per-instance**, only wrapped by a vectorized fan-out:
  - `search`: list of independent `SearchEnv` + internal ThreadPoolExecutor/asyncio.
  - `alfworld`: one worker **process per instance** (`num_processes = env_num*group_n`).
  - `webshop`: one **Ray actor per instance** (`env_worker.remote(...)`).
  → Subset/single async stepping can be **exposed over existing instances**; no
  rewrite of env logic. search is trivially async already.
- **verl async server is chat-completion only** (`AsyncvLLMServer.chat_completion`),
  which would force a re-tokenization round-trip (an accuracy risk). We therefore
  need a **token-level async generate** path that returns the exact generated
  token ids (no re-tokenization).
- The **orchestration core** (`async_rollout_core.collect_async`) and its CPU
  equivalence test already exist and pass — they prove the async schedule collects
  byte-identical trajectories to the sync loop for a deterministic policy+env.

## 4. Target architecture
Driver-side async collector (asyncio), N trajectory coroutines:
```
for each trajectory (coroutine), until done or max_steps:
    prompt_ids = preprocess_single_sample(obs)          # exact, unchanged
    resp_ids   = await token_async_generate(req_id, prompt_ids, sampling)  # WS1
    action_txt = tokenizer.decode(resp_ids)
    obs, r, d  = await env.step_one(traj_id, action_txt)  # WS2
    record(traj_id, prompt_ids, resp_ids, r, d, ...)
```
- vLLM **AsyncLLM** continuously batches the concurrent `token_async_generate`
  requests across coroutines → GPU stays full; env/CPU latency overlaps generation.
- Per-instance async env step (WS2) lets each trajectory advance independently.
- Records are reassembled into the **exact DataProto** that the existing
  `gather_rollout_data` consumes → everything downstream (teacher pass, advantage,
  loss, logging) is unchanged.

## 5. Workstreams
- **WS1 — token-level async generation (worker/engine).** Expose
  `AsyncLLM.generate(TokensPrompt(prompt_token_ids), SamplingParams, request_id)`
  returning generated token ids, through the worker + a manager method. Build
  `SamplingParams` identically to `vllm_rollout` (do_sample/temperature/top_p/top_k).
  Manage wake/sleep with FSDP colocation. **HIGH risk / new infra.**
- **WS2 — per-instance async env stepping.** Add `step_one`/`step_subset` (async)
  to search → alfworld → webshop, over the already-independent instances. Make
  per-instance history/state (`memory`, `pre_text_obs`) per-trajectory. **MEDIUM.**
- **WS3 — async collector.** Build on `async_rollout_core`; wire WS1+WS2; reassemble
  to the existing DataProto; preserve GRPO uid grouping. **MEDIUM (reuses core).**
- **WS4 — integration / accuracy / rollback.** `rollout.mode`/flag selection,
  accuracy gates, gpu_profiler measurement, sync fallback. **LOW–MEDIUM.**

## 6. Accuracy preservation & verification (central constraint)
Invariants:
| Element | Guarantee |
|---|---|
| Sampling distribution | identical SamplingParams + weights per request |
| Generated tokens | token-level (WS1) → exact ids, **no re-tokenization** |
| Env transitions | same env instance + same action → same transition |
| GRPO grouping | uid groups preserved (async core) |
| Teacher/adv/loss/logging | downstream unchanged (recompute on tokens) |
| Difference | continuous-batch FP variance + RNG order → **not bit-identical, distributionally equal** |

Verification ladder (each gates the next):
1. **CPU equivalence test** (done): async core ≡ sync trajectories on mocks.
2. **GPU determinism spot-check**: per-request seed fixed; compare a few sequences.
3. **N-step learning-curve match** (N≈20–50, same seed, sync vs async):
   `val/*success_rate`, `actor/pg_loss`, `sdar/teacher_gap`, `actor/entropy_loss`
   overlap within seed noise.
Per-request seed = f(global_seed, traj_uid) for reproducibility. Flag-gated rollback.

## 7. Milestones (each independently verifiable)
- **M0 (DONE)**: measurement infra (gpu_profiler, per-turn timing), accuracy audit,
  orchestration core + CPU equivalence test.
- **M1**: WS1 token-async in isolation — single-turn token-in/token-out matches sync
  `generate_sequences` (same prompt/seed). *GPU.*
- **M2**: WS2 for **search** (per-instance async step) matches vectorized stepping.
- **M3 (DECISION GATE)**: WS3 end-to-end async collector on a **search-only**
  multitask subset → measure real gen-util gain + accuracy-curve match. Decide
  whether the util gain justifies M4 before doing the expensive env work.
- **M4**: WS2 for **alfworld + webshop**; full 3-task async; target gen-util ~90%.
- **M5**: hardening — env crash/abort handling, engine request abort, checkpoint/
  resume, validation-path async, long-run stability.

## 8. Risks & mitigations
- vLLM AsyncLLM × colocated FSDP wake/sleep correctness (WS1) → prototype isolated
  in M1; pin vLLM version.
- Real util ceiling uncertain (engine/env overhead may cap below 90%) → **M3
  measurement gate before M4**.
- Per-instance env perf/correctness → start with search (easiest), spike
  alfworld/webshop early.
- No GPU in the dev environment → all GPU milestones require the user's hands-on
  iteration loop; deliverables are code + tests, validated against shared logs.
- Reproducibility/debuggability → per-request seeds, structured logging, the CPU
  equivalence test as a permanent regression gate.

## 9. Rough effort
M1 ~2–4 GPU-days · M2 ~1–2 · M3 ~3–5 (incl. accuracy validation) · M4 ~1–2 weeks
(alfworld/webshop) · M5 ongoing. Order of magnitude: **several weeks of
GPU-iterated work.**

## 10. Inputs needed
- vLLM version (V0 `AsyncLLMEngine` vs V1 `AsyncLLM`; available async API surface).
- GPU model & count (memory budget; 2 vs 3).
- Confirmation we may modify the search/alfworld/webshop env packages (add
  subset-step APIs).
- (Agreed) distributional equivalence is the accepted accuracy standard.

## 11. Recommendation
Run **M1 → M3 as a de-risking spike first** (token-async + search-only end-to-end),
measure the actual util gain, then commit to M4 (alfworld/webshop). This avoids
investing in the broader env work before the util payoff is proven on GPU.
