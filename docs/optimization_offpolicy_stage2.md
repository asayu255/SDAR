# Off-policy Stage 2 — profile of a real run, and what was changed

Target: `examples/opd_trainer/run_multitask_offpolicy_qwen3_nogen.sh`
(`verl.trainer.main_opd_offpolicy` → `OffPolicyOPDRayTrainer.fit`).

Companion to `docs/optimization_report.md` (Phase 1), `docs/optimization_phase2.md`
(Phase 2, the rollout) and `docs/optimization_gen_only.md` (Stage 1). Those three
optimize workloads that generate. **Stage 2 generates nothing during training** —
it draws from a fixed teacher pool and runs one `update_actor` per step — so
almost none of their conclusions carry over, and the numbers here are measured on
this workload rather than argued from it.

This is a record of one run, not a proposal. Everything under "measured" comes
from that run's log; everything else is labelled.

---

## 0. The run these numbers come from

3 × A6000 (47.5 GiB), Qwen3-1.7B student, 300 steps, one node.
Teacher pool: 36000 trajectories × 3 tasks, 333 GiB on disk, 90 shards.

The profiled window is steps 122–133, with `GPU_PROFILER=1` and the cumulative
table at step 125 covering the whole run so far.

**The run predates two changes that are now on the branch.** It has the hard-label
CE term (`f85b931`) but not the pool dtype narrowing (`9716ff5`) and not the
per-task loss weighting (`32d1278`): its log carries no `task_loss/token_share/*`
and no `actor/*_weighted`, and `perf/cpu_memory_used_gb: 441.8` matches the
un-narrowed pool (283.5 GiB resident + ~160 GiB of env workers) rather than the
narrowed one. So the memory work in §5 is **not** reflected in the step times
below, and the loss the run optimizes weights tasks by their token share rather
than equally.

---

## 1. Measured

### Per step

| | wall/step |
|---|---|
| mean over 10 non-checkpoint steps | **492.5 s** |
| min / max | 461.8 / 529.6 s |
| the checkpoint step (125) | 663.1 s |
| cumulative mean over 125 steps | 499.7 s → **41.6 h for 300 steps** |

### Where the time goes

Driver-side phases, cumulative over 125 steps:

| phase | wall | share | SM% |
|---|---|---|---|
| `update_actor` | 61315 s | **98.2%** | 89.5 |
| `save_checkpoint` (every 25 steps) | 1000 s | 1.6% | **2.8** |
| `(idle/other)` | 150 s | **0.24%** | 16.5 |

Worker-side phases inside `update_actor` (rank 0, one representative step):

| phase | wall | share of update_actor | SM% |
|---|---|---|---|
| `actor.fwd` | ~125 s | 25–28% | 90 |
| `actor.bwd` | ~330 s | 65–74% | 89 |
| `actor.task_metrics` | ~20 s | 4.0–4.8% | 93 |
| `actor.optim` | ~1 s | 0.1–0.3% | 98 |

### Efficiency

- `perf/mfu/actor` = **0.206** (mean over the window, 0.203–0.209)
- SM utilization during `update_actor` = **89.5%**
- `timing_s/update_actor` − `timing_s/update_actor_worker` ≈ 1.5–2.0 s (transport)
- `perf/max_memory_allocated_gb` = 40.79, reserved 52.44
- `global_seqlen/balanced_min` == `balanced_max` on every step

---

## 2. The finding

**SM utilization is 89.5%. There is essentially no idle to reclaim.** The driver
window between steps — the thing `OFFPOLICY_BATCH_PREFETCH` exists to hide — is
0.24% of the run. `_balance_batch` splits tokens across DP ranks exactly.
`actor.optim` is 0.2%. Every place one would normally look is already tight.

The gap is elsewhere: **MFU 0.206 with SM 89.5%**. The GPU is busy; what it is
busy with is thin. Two causes, both measured:

**Gradient-checkpointing recompute.** `actor.bwd / actor.fwd = 2.64`. Without
activation checkpointing that ratio is ~2 (backward is about twice the forward).
The excess is the recomputed forward: `(2.64 − 2) × 125 s ≈ 80 s`, i.e. **~16% of
the step is the same forward run a second time**. That work is real GPU time and
is deliberately not in the MFU numerator — verl's `estimate_flops` counts
`6·N·tokens + attention`, forward and backward only (`verl/utils/flops_counter.py:175-184`).
So 0.206 is the efficiency of the work that is *kept*.

**Small GEMMs.** `ppo_micro_batch_size_per_gpu = 2` against `hidden_size = 2048`.
A micro-batch is 2 rows, mean ~835 valid tokens each, so the M dimension of every
matmul is ~1670. A 1.7B model at that shape does not saturate an A6000's tensor
cores no matter how busy the SM counter looks.

The general lesson, and the reason this took a worker-side profiler to see: **a
high SM number does not mean the work is useful.** The single largest waste in
this run was inside `actor.bwd` — the phase with 89% utilization — not in any
idle window.

---

## 3. Wasted work found by reading the code

Two of these were found only because the profile pointed at `actor.fwd` /
`actor.bwd` and forced a line-by-line read of `_forward_micro_batch`.

### 3.1 The top-k forward ran over prompt rows and threw them away — FIXED (`153003f`)

`_forward_micro_batch` ran `logsumexp`, `topk` and `gather` over **every**
unpadded row — prompt included — and then kept only `[:, -response_length-1:-1]`.
Measured on this run, `prompt_length/mean = 630` against
`response_length/mean = 205`, so **roughly three quarters of that vocab-sized work
(151936 wide) was discarded**.

Both stages go through it. Stage 1 scores the teacher's own top-k; Stage 2 gathers
the student's log-probs at the pool's teacher ids, and that one sits **inside the
autograd graph** — the scatter/gather kernels ran at full size in the backward
too, producing zeros. Stage 2 is the arm's entire step, so the saving lands
directly on step time.

`response_row_selection` picks the surviving rows up front. Verified
value-identical against the all-rows path over 500 random layouts (batch size,
sequence length, prompt/response split, left and right padding).

It also removed a `(bs, seqlen, k)` zero tensor that the student path built to
scatter `topk_ids` into and then gathered back through the same index map — the
same mapping computed the long way round.

### 3.2 Per-task metrics cost a device sync per micro-batch — NOT FIXED

`actor.task_metrics` is 4.0–4.8% of `update_actor` at 93% SM. The values
themselves are deferred GPU tensors (no `.item()`), but
`iter_task_row_masks` (`verl/trainer/ppo/metric_utils.py:122`) calls
`torch.unique(task_ids).tolist()` — **one host sync per micro-batch**, ~712 per
step. The task set is a fixed three, so iterating `range(len(task_id_names))` and
doing mask arithmetic only would remove the sync entirely.

Estimated at 2–4% of the step, deliberately conservative: some of what that phase
measures is the backward's queued reduce-scatter draining, which would simply move
rather than disappear.

### 3.3 Checkpointing writes at ~100 MB/s — NOT FIXED

197.6 s to write ~20 GB (bf16 model + fp32 optimizer state across 3 ranks) is
about **100 MB/s**, at 2.8% SM. On local NVMe the same write is seconds. Amortized
over `save_freq=25` it is 1.6% of the run — about 40 minutes over 300 steps.

The target is `/opt/home/ohara/checkpoints`; if that is network storage, moving it
to a local disk (and copying out asynchronously) is the whole fix.

---

## 4. Correctness defects found while profiling

None of these are performance problems. They are recorded here because they were
found by the same reading, and because **each one fails silently** — no exception,
no metric moves.

| | status | what it did |
|---|---|---|
| Stage-1 `adjust_batch` padding was saved into the pool | fixed `066254e` | duplicated turn-rows trained on twice; +10% optimizer steps for the same real data |
| `actor/teacher_kl_loss` was assigned, not appended | fixed `066254e` | logged only the *last* micro-batch, which after `_balance_batch` is often entirely padding |
| Retriever retry exhaustion | fixed `8a34290` | the error string was substituted for the retrieved document and trained on as if it were one |
| Resume restarted the trajectory draw sequence | fixed `db29c95` | resuming at step 125/300 re-trained draws 1–175 and never saw the last 125, breaking "36000 trajectories = exactly one epoch" |
| `num_mini_batches` asserted divisibility | fixed `994b294` | a short final mini-batch is the normal case (`adjust_batch` rounds to 160, mini-batch is 60); the on-policy arm died at step 2 on this |

The resume one is worth singling out. `_save_checkpoint` **does** persist
`train_dataloader` state, and `_load_checkpoint` restores it — which is exactly
what made this invisible. That loader drives Stage 1 and validation. The Stage-2
training loop iterates a plain Python generator with a seeded RNG and per-task
cursors, and nothing saved its position. The appearance of "the dataloader is
checkpointed" covered a data source that was not.

---

## 5. Implemented, by accuracy class

Sorted by what each does to the numbers, because that is what decides whether two
runs remain comparable.

### Bit-identical

| | commit | effect |
|---|---|---|
| top-k over response rows only | `153003f` | removes ~75% of the vocab-sized work in fwd and bwd (§3.1) |
| batch prefetch (`OFFPOLICY_BATCH_PREFETCH`) | `d520692` | driver window is 0.24% of the run — **confirmed working**, nothing left to hide |
| `fsdp_config.forward_prefetch` | `d520692` | all-gather issue order only |

### Arithmetic-neutral (memory layout and scheduling)

| | commit | effect |
|---|---|---|
| pool held per shard, never concatenated | `84984c9` | load peak from ~2× the pool to `resident + one shard` (~9 GiB) |
| dead columns dropped at load (`prompts`, `response_mask`) | `066254e` | −37 KB of 275 KB per row |
| storage dtypes narrowed (int32 / uint8) | `9716ff5` | **283 → ~149 GiB resident**; `teacher_topk_logprobs` deliberately left fp32 |
| ZeRO-2 (`sharding_strategy=shard_grad_op`) | `d520692` | parameters stay gathered from forward through backward |

The narrowing was written **in response to an OOM kill**: the first launch died at
`494.32 GB / 503.46 GB (98.2%)` during `init_workers` — 283.5 GiB pool + ~160 GiB
of env workers (webshop alone runs ~250 Ray actors at 0.64 GB) + the three model
workers coming up. The pool was the one tenant whose size was negotiable. Values
are unchanged: the vocab (151936) and the padded sequence length (4608) both fit
int32, `attention_mask` is 0/1, and `_prepare_batch` restores every column to its
compute dtype on the per-step batch before anything reads it.

**This is not in the step times in §1** — the profiled run predates it (§0).

### Changes the floating-point association

| | commit | note |
|---|---|---|
| `no_sync_grad_accum` | `d520692` | one gradient reduce per mini-batch; pinned in the expectations file precisely *because* it is on the gradient path |

### Changes the objective

| | commit | note |
|---|---|---|
| hard-label CE summed with the top-k KL | `f85b931` | in the profiled run |
| equal per-task share of the loss | `32d1278` | **not** in the profiled run |

---

## 6. Not implemented, with the arithmetic

Quantified during this analysis, deliberately not applied to a run in flight.

**Free vLLM's 14.4 GiB.** `gpu_memory_utilization=0.3` × 48 GiB is resident for all
300 steps and used only by validation (2 rollouts, at steps 150 and 300), because
`free_cache_engine=False`. Releasing it requires `free_cache_engine=True` **and**
`enforce_eager=True` — `vllm_rollout_spmd.py:107` asserts they cannot be combined
with CUDA graphs. That 14.4 GiB is what makes either of the next two possible;
they are **mutually exclusive on memory** and should be compared with a 5-step probe.

- **`ppo_micro_batch_size_per_gpu` 2 → 4.** Doubles the M dimension of every GEMM,
  which is the direct answer to §2. Estimated **−15 to −30%**.
- **Gradient checkpointing off.** Removes the ~16% recompute measured in §2, and
  changes no arithmetic at all. Estimated **−13 to −16%**.

**Dynamic batching by token count** is the largest lever on paper — packing
micro-batches to a token budget instead of a fixed 2 rows would raise the mean
rows/micro-batch by ~5× — but `use_dynamic_bsz` changes the micro-batch
aggregation arithmetic and `normalize_loss_by_task` refuses it by assertion. It is
an objective change, not a throughput knob.

**Send the pool narrow to the workers.** `_prepare_batch` restores int64 on the
driver and then ships ~1 GB per step; restoring worker-side would halve that.
Worth ~0.3% — listed for completeness, not recommended.

---

## 7. Lessons

**SM utilization and MFU measure different things.** 89.5% busy with MFU 0.206 was
the whole story of this run. Chasing the idle 10% would have been worth at most a
few percent; the discarded three quarters of the top-k forward was hiding inside
the busy 90%.

**Profile the phase you cannot see.** From the driver, `update_actor` is one
opaque 500-second bucket — 98.2% of the run in a single row. Splitting it worker-side
(`actor.fwd` / `bwd` / `task_metrics` / `optim`) is what produced §2 and §3.
Getting there needed two prerequisites first: worker-side phase pushes, and an
A6000 entry in `get_device_flops` — without it the unknown-device fallback left
`flops = inf` and **every MFU in every previous log read exactly 0.000**. Those
zeros were "not measured", not "bad".

**The bwd/fwd ratio is a free measurement.** 2.64 against a theoretical 2.0 gives
the gradient-checkpointing recompute share directly, with no extra instrumentation.

**Silent failure modes clustered around "the data".** Four of the five defects in
§4 corrupt what the model trains on while every metric keeps moving normally:
padding rows trained twice, a retriever error string trained as a document, a
resumed run replaying the first half of the pool, a loss metric reporting only its
last micro-batch. The profiler found none of them — reading the code to explain the
profile did.

---

## Appendix A — reproducing the measurement

```bash
GPU_PROFILER=1 \
GPU_PROFILER_INTERVAL=0.3 \
GPU_PROFILER_ROLLUP_EVERY=25 \
bash examples/opd_trainer/run_multitask_offpolicy_qwen3_nogen.sh
```

Do **not** set `GPU_PROFILER_SYNC_PHASES=1` for a run whose wall time matters: it
synchronizes at every actor stage boundary, making the phase split exact at the
cost of serializing what the run overlaps. `GPU_PROFILER=1` alone is an NVML poll
on a background thread and a table per step.

To measure the size of the window batch prefetch hides, one run needs
`OFFPOLICY_BATCH_PREFETCH=0` — with it on (the script default) that time is inside
`update_actor`, and `(idle/other)` reads 0.24% whether or not the mechanism helps.

## Appendix B — the pool, measured

`python3 scripts/inspect_teacher_pool.py <pool>`:

| task | trajectories | rows | padding | r_k (rows/traj) | resp tok/row | resp share |
|---|---|---|---|---|---|---|
| alfworld | 36,000 | 927,780 | 1.0% | 25.52 | 207.7 | 72.1% |
| search | 36,000 | 94,860 | 7.7% | 2.43 | 131.3 | 4.3% |
| webshop | 36,000 | 280,620 | 3.1% | 7.55 | 228.8 | 23.5% |

Bit-identity spot check on the padding rows: **1800/1800** matched a row in their
own trajectory, so what the loader drops is provably duplicated data.

`resp share` is what a plain token-mean gives each task, and is set almost entirely
by `r_k` — the 50/15/4-turn episode caps — rather than by response length, which
varies by less than 2×. The profiled run trains under exactly this weighting;
`32d1278` (not in it) makes all three 1/3.
