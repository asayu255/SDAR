# SDAR Multitask Speedup — Phase 2 Mechanisms (A–E)

Branch: `claude/festive-gates-fxercy-optimizations-pqd3v2` (builds on
`claude/festive-gates-fxercy`; see `docs/optimization_report.md` for Phase 1:
vLLM session ①, active-only preproc ②, prefix caching ③, task interleave ④,
param-offload ⑤).

Phase 1 established that the remaining rollout wall time is dominated by the
**alfworld tail** (per-task caps search=4 / webshop=15 / alfworld=50 leave
turns 16–50 running 1/3 of the batch at ~57% SM) plus residual CPU glue
(preproc / decode / env.step) between generate calls. Phase 2 attacks exactly
those, with every mechanism either **bit-identical** or in the same
distribution-preserving class as ③. All are env-var knobs; every default-off
knob reproduces the Phase 1 behavior exactly when unset.

| # | Mechanism | Knob (default) | What it overlaps / removes | Accuracy class |
|---|---|---|---|---|
| A | **Finished-trajectory log-prob prefetch**: while `envs.step()` runs in a background thread (GPU idle), the driver runs `compute_log_prob` on a bounded chunk of rows whose trajectories already finished (search from ~turn 4, webshop from ~turn 15). The trainer's `old_log_prob` phase reuses the prefetched rows and computes only the remainder. | `ROLLOUT_PREFETCH_LOGPROB=1` (off), chunk via `ROLLOUT_PREFETCH_LOGPROB_CHUNK` (64) | Moves `old_log_prob` work into the rollout's GPU-idle env.step windows (~35 tail turns) | Same weights (frozen during rollout), same per-row computation; micro-batch grouping differs from the monolithic phase → **③-class** |
| B | **CUDA-graph decode knobs**: `CUDAGRAPH_CAPTURE_SIZES` passes `cudagraph_capture_sizes` through to vLLM (`enforce_eager=False` is already set); requires the V1 engine (`VLLM_USE_V1=1`) — under V0 the CompilationConfig is ignored (the Phase 1 finding). 1.7B decode is launch-overhead-sensitive, so graphs target the decode-bound tail directly. | `VLLM_USE_V1=1 CUDAGRAPH_CAPTURE_SIZES='[8,16,32,64,128,256,384]'` (both unset) | Kernel-launch gaps in autoregressive decode | Sampling-distribution-preserving → **③-class** |
| C | **Env-reset prefetch**: right after a rollout's data is collected the train envs are idle; the trainer peeks the next dataloader batch and launches `envs.reset()` (CPU / subprocess / HTTP: alfworld game loading etc.) in a background thread, overlapping `old_log_prob`/`teacher`/`ref`/`update_actor`. Reset still runs exactly once per rollout in the same order, so stateful schedules (alfworld's game-file iterator) are unchanged; a kwargs mismatch fails loudly instead of double-resetting. | `ENV_RESET_PREFETCH=1` (off) | Rollout-start reset latency | **bit-identical** |
| D | **Retriever query cache**: class-level LRU keyed by (url, topk, query). RL rollouts re-issue identical queries (GRPO group members share questions; epochs revisit the same subset); a deterministic retriever returns the byte-identical passage set, so hits skip the HTTP round trip. Only successful lookups are cached. | `SEARCH_QUERY_CACHE=1` (off), size via `SEARCH_QUERY_CACHE_SIZE` (100k) | search `env.step` HTTP latency | **bit-identical** iff the retriever is deterministic (fixed index) |
| E1 | **Parallel prompt tokenization**: the per-row `apply_chat_template` + tokenize loop runs on a thread pool with per-thread tokenizer clones (HF fast tokenizers release the GIL in Rust but mutate shared truncation/padding state, hence the clones). Verified byte-identical against the sequential loop. Multimodal batches stay sequential. | `ROLLOUT_PREPROC_WORKERS=8` (0 = off) | Remaining preproc CPU (~37 s/step after ②) | **bit-identical** |
| E2 | **Active-only decode**: decode only generated rows; finished rows' scattered filler is pad-only, which `batch_decode(skip_special_tokens=True)` renders as `''` anyway — fill `''` directly. | `ROLLOUT_DECODE_ACTIVE_ONLY=1` (**on**; set 0 for A/B) | Wasted decode of pad rows (2/3 of batch in the tail) | **bit-identical** |
| E3 | **Compact per-turn recording**: skip appending `active_masks=False` rows to `total_batch_list`/`total_infos`. Active rows form a prefix of each trajectory's list, so `turn_step` (enumerate), the last-active-entry scans in `success_evaluator`, and `filter_group_data` all see identical data; `gather_rollout_data` dropped these rows anyway. | `ROLLOUT_COMPACT_RECORD=1` (**on**; set 0 for A/B) | Per-turn dict materialization + memory for finished rows | **bit-identical** |

## Where the code lives

- A: `agent_system/multi_turn_rollout/rollout_loop.py`
  (`_prefetch_pending_log_probs`, pending-pool bookkeeping in
  `vanilla_multi_turn_loop`, `take_prefetched_log_probs`) +
  `agent_system/multi_turn_rollout/utils.py::compute_log_prob_with_prefetch` +
  the `old_log_prob` phase in `verl/trainer/ppo/skillsd_ray_trainer.py` /
  `rlsd_ray_trainer.py`.
- B: knobs in `examples/sdar_trainer/run_multitask_qwen3.sh` (the
  `cudagraph_capture_sizes` passthrough already exists in
  `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`).
- C: `TrajectoryCollector.prefetch_env_reset` / `_reset_envs` + the peek-ahead
  loop in both trainers' `fit()`.
- D: `agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`.
- E: `preprocess_batch` / `_run_full_preprocess`, the decode and record blocks
  of `vanilla_multi_turn_loop`.

## Operational notes

- **A + `ROLLOUT_KEEP_VLLM_AWAKE`**: the prefetched `compute_log_prob` runs
  while the vLLM session is open. The FSDP forward is eval-mode and consumes no
  RNG, so the session's generation RNG stream is untouched; memory coexists the
  same way the normal `old_log_prob` phase already does under
  `free_cache_engine=False`. The prefetch call is issued synchronously between
  turns, so it never contends with `generate_sequences` on the worker actors.
- **A sizing**: gains cap at the total env.step/decode window
  (≈35 tail turns × a few seconds). If `ROLLOUT_PREFETCH_LOGPROB_CHUNK` is too
  large the turn extends past env.step; the work still comes off the
  `old_log_prob` phase, but the overlap is lost. The per-turn timing
  (`ROLLOUT_TURN_TIMING=1`) shows the envstep bucket absorbing the prefetch.
- **A accounting**: prefetched GPU time is attributed to the `gen` timer, so
  expect `gen` to rise and `old_log_prob` to fall; judge by step wall time.
- **C + dynamic sampling**: reset prefetch is disabled when
  `algorithm.filter_groups.enable` (dynamic sampling re-resets within a step).
- **D**: only enable against a deterministic retriever (fixed index, no
  sampling); the cache returns byte-identical observations on hits.

## Tests

`tests/ray_cpu/test_rollout_speedup_mechanisms.py` (CPU-only) gates:
prefetch-merge equivalence incl. `adjust_batch` row duplication and
all/none-prefetched edges; the rollout pending pool; env-reset prefetch
consume/mismatch semantics incl. cross-env isolation; compact-record
equivalence; and the query cache. The E1 parallel tokenization path was
verified byte-identical against the sequential loop with a real HF fast
tokenizer (full-batch and active-mask paths).
