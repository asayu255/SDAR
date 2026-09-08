# The 12 speedup items, and why each was taken or left

Every figure here is reproduced by `scripts/opd_speedup_evidence.py`, so the next
series recomputes rather than re-derives:

```
scripts/opd_speedup_evidence.py --run RUN.log --turns TURNTABLE.log [--ab OTHER.log]
```

The baseline it is all measured against: a 480.5 s step (last-30 median) made of
gen 267.9 (56%), update_actor 191.1 (40%), old_log_prob 11.4.

## Taken

### 3.1 ppo_micro_batch_size_per_gpu 5 -> 10

The reason is the per-micro-batch parameter all-gather. This host has no NVLink:
`nvidia-smi topo -m` reports SYS between GPU0 and GPU1, on different NUMA nodes.
`SHARD_GRAD_OP` gathers the full parameter set once per micro-batch, so each rank
receives 1.89 GiB of the 3.78 GiB bf16 parameters, and at the measured 564
micro-batches per rank per step that is **1,066 GiB per rank per step**. Holding
that inside the observed 362.7 ms per micro-batch needs 5,336 MB/s sustained.

Measured `rxpci` during the update phase: **4,569 MB/s (GPU0) and 5,145 MB/s
(GPU1) mean, peaks near 20,000**. Reconstructing the volume from the mean gives
~940 GiB against the predicted 1,066 GiB, agreement within 12%, so the gather is
essentially all of the update phase's PCIe traffic. GPU utilisation over the same
window averaged 78-81%, i.e. ~41 s idle out of 206 s in-micro, the same order as
the gather's wire time.

Result, step 1 against step 1 on the same arm: **update 258.6 -> 225.0 s (-33.6),
step 645.3 -> 602.1 s (-43.3)**, gen and old_log_prob unchanged. The prediction
was -20 to -27 s, so this came in above it.

Class: floating-point summation order only. The objective is invariant under
`normalize_loss_by_task=true` with `use_dynamic_bsz=false` and `ppo_epochs=1`,
because the per-task row weights carry the whole normalisation and the
world-size and accumulation factors cancel. All four locks carry the same value.

A prior decision had to be overturned to take this. The lock recorded the
10 -> 5 halving as a memory fix, citing `teacher_cache/gb 16.3`. In the current
configuration that cache measures **3.673 with `device_gb 0.000`**, i.e. host RAM
and not device, and vLLM sleeps before the update: 95.5 GB during the rollout,
30.0 GB between phases, 53.6/45.6 GB of 97.9 during the update. There was 44 GB
free where the original OOM was recorded.

## Left, on measurement

### 3.5 diagnostics frequency

**A correction is recorded here.** An unpaired comparison over the last 60 steps
read as 215 s against 195 s and looked like a 20 s periodic cost. That comparison
is dominated by the +-20 s step-to-step swing. Testing a *periodic* overhead
requires pairing each multiple-of-5 step against its own neighbours, and doing
that gives **median +3.5 s with 9 of 17 pairs positive**, i.e. indistinguishable
from zero, or +0.71 s amortised. An earlier group-split estimate of -1.1 s agrees
on the conclusion. `paired_periodic_test` in the script is the correct form.

### 3.9 KV budget 0.6 -> 0.68

Decided by the rollout window, not by the KV maths: physical occupancy during
generation peaks at **95.5 GB of 97.9 on rank 0, leaving 2.4 GB**. The proposal
assumed 6-11 GB. There is no room to spend.

## Left, because the premise did not survive measurement

The turn table decomposes the gen phase over **7,500 turn rows in 150 tables**,
and table totals match `timing_s/gen` to 0.99, so it is a decomposition of the
whole phase and not a sample of it:

| component | share of the gen phase |
|---|---|
| inside the engine | **72.1 %** |
| envstep | 18.6 % |
| preproc | 5.5 % |
| tchWait | 3.4 % |
| decode | 0.4 % |

### 3.7 shortening the turn round-trip

The premise was that time outside the engine is at least equal to time inside.
Measured it is 27.9 % against 72.1 %, and the addressable part of that is
envstep plus preproc, so the ceiling is ~24 % of gen and ~13 % of the step.
Downgraded rather than refuted: the ceiling is real, it is just not the
co-dominant term the design assumed.

### 3.8 the in-window frozen forward

tchWait is 3.4 % of gen, ~1.9 % of the step, ~9 s. The turn table that measured
it also predates `9ee1fe9`, which reordered the teacher join specifically to
reduce it, so 9 s is an upper bound on the current code.

## Left, because they change the objective or the definition

### 3.2 dynamic batch size -- HELD, and coupled to 3.3

`use_dynamic_bsz=true` breaks one of the two preconditions the invariance
argument above rests on, which is why the existing assert blocks it. The "two
places of code" work is precisely the work of restoring invariance: the loss is a
weighted row sum whose weights come from step-level token totals, and a weighted
row sum does not depend on how the rows are split, so the restoration is exact
rather than approximate. Held on its own; **required if 3.3 is taken** (below).

### 3.10 fp8 KV cache

Lossy, so unlike 3.6 it does not even claim to preserve the sampling
distribution. Left.

### 3.11 overlapping the next rollout with the update

A 1-step off-policy training loop is a different algorithm, not a knob.

## Retired rather than rejected

### 3.0 the three-step probe run

All four of its targets were answered without it, so neither a probe run nor a
stop was needed:

| target | how it was answered instead |
|---|---|
| 1 gen decomposition | an existing turn table in a 2026-09-06 log, config-matched on response_length 512, train_batch_size 45, Qwen3-1.7B and ROLLOUT_KEEP_VLLM_AWAKE=1 |
| 2 KV / rollout window | `nvidia-smi` on the live run: 95.5 GB, 2.4 GB free |
| 3 per-rank update memory | `nvidia-smi` across a phase transition: 53.6 / 45.6 GB of 97.9 |
| 4 what the per-micro-batch cost is | `dmon -s t` rxpci plus the all-gather volume above |

Note for target 3: `perf/update_peak_allocated_gb` reported 87.5 and
`max_memory_reserved_gb` 109.9, above the card's physical 97.9, because torch's
counters include vLLM's sleeping CuMemAllocator pool. Physical occupancy has to
be read from `nvidia-smi`.

## Still open

### 3.3 gradient checkpointing off -- measure the WORST micro-batch, not the mean

Bit-identical in principle (recompute against store), so it does not touch the
sampling. The memory question is not settled by an average, because activations
with checkpointing off scale with the **tokens in a micro-batch**, and that
distribution is wide:

| | tokens |
|---|---|
| average row | 610 prompt + 143 response = **753** |
| longest row | 3,354 prompt + 512 response = **3,866** |
| ratio | **5.13x** |
| micro-batch 10, average | 7,531 |
| micro-batch 10, worst case | **38,660** |

A +13 GB estimate taken from the average is therefore the wrong statistic; ten
maximal rows landing in one micro-batch is roughly five times that. rank 1's
headroom is ~37-41 GB, so the worst case can exceed it, which is consistent with
the earlier record of an OOM at *step 1's first micro-batch* specifically.
`BALANCE_MINIBATCH_COLUMNS` equalises the sums within a column and does not bound
the largest micro-batch, and a peak sampled from one step's update phase cannot
rule the worst case in or out.

**A token budget is the only thing that bounds it**, which is item 3.2.
`ppo_max_token_len_per_gpu=9216` is already configured and inert while dynamic
bsz is off. The coherent pairs are therefore: take 3.3 and 3.2 first goes in, or
take neither.

### 3.4 FSDP2, to stop the re-gather inside a mini-batch

Same safety class as 3.1 (execution mechanics, not sampling), but the design
itself marks it non-bit-identical and its own experiment, and it carries
checkpoint-compatibility risk. Held.

### 3.6 n-gram speculative decoding -- the rejection reason was withdrawn

This was first left out on the ground that it changes which tokens are sampled.
That ground does not hold, and the record should say so plainly:

- Rejection sampling preserves the target distribution. vLLM's `RejectionSampler`
  "strictly follows" 2211.17192; the n-gram proposer supplies no draft
  probabilities, i.e. a deterministic draft, for which accept-with-p plus the
  recovered distribution reproduces the target exactly.
- What changes is *which* tokens are drawn, which is the same class as
  `enable_prefix_caching=True` and `enforce_eager=False` -- same distribution,
  not bit-identical. **Both of those are already in every arm and neither is
  pinned**, i.e. the project already treats that class as a performance knob.
- The sibling scripts' phrase "its own experiment on every arm at once, not a
  knob to flip here" means it must go on all arms together, not that it is
  forbidden. That is the same treatment 3.1 received.
- Their recorded blocker is stale for this stack. It was a V0 issue:
  `SpecDecodeWorker` does not implement `sleep()`, and the engine is built with
  `enable_sleep_mode=True`. This stack is `core=v1`, where spec decode lives in
  `v1/spec_decode` and sleep is supported, `speculative_config` composes through
  `engine_kwargs.vllm`, and `SpeculativeConfig` builds on CPU with
  `method=ngram`.

It is also the only large lever left. 72 % of the engine's own time sits at
<= 40 % of peak concurrency, costing 65 ms/seq/turn against the head's 24. That
tail is ~29 % of the step, and SM during generation measures 60-73 %, i.e. the
engine is busy in a memory-bound regime rather than idling on the pump -- which
is the regime speculation addresses.

Two conditions remain, and only the second is unverified:

1. Exactness needs temperature 1 with no top_p or top_k. Training samples at
   temperature 1.0 and sets neither, and `acceptance_method` is pinned to
   `rejection_sampler` rather than left to a default. vLLM's own docstring says
   spec decode does not support top_p/top_k, which is the one case avoided here.
2. Coexistence with sleep mode and the rollout pump has to be confirmed from a
   startup log. Config-only, so the failure mode is a startup crash.

Acceptance rate is unmeasured and sets the size of the gain. It is observable
from step 1: `disable_log_stats=False`, so `SpecDecodingStats` reports
`num_drafts`, `num_accepted_tokens` and `num_accepted_tokens_per_pos`, the last
of which also says whether `num_speculative_tokens` is set sensibly.
