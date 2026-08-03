# Per-task loss weighting in multitask runs

`actor.normalize_loss_by_task` (off by default) makes every task contribute
`1/num_tasks` of **every** loss term, instead of its share of the batch's
response tokens.

## The problem

Every term of the actor loss is a `token-mean` over the whole mixed batch, so a
task's contribution equals its share of the batch's response tokens. On the
alfworld / webshop / search mixture that share is decided by the 50 / 15 / 4-turn
episode caps and by nothing anyone chose — measured at roughly **69 / 27 / 4 %**.
Search brings a third of the prompts and 4% of the gradient.

The imbalance is in the rows as well as the tokens (a row is one *turn*, and
alfworld runs up to 50 of them per trajectory), so `loss_agg_mode` cannot fix it:
`seq-mean-token-mean` equalises turns, not tasks.

## What it does

Writing `s_t` for task `t`'s token share and `M_t(·)` for a token-mean over that
task's rows, the four terms of the SDAR objective behave like this:

```
single-task run t :  L_t = 1.0·M_t(pg) − 0.001·M_t(ent) + c_t·M_t(kl) + 0.01·M_t(sdar)
multitask, today  :  L   = Σ_t  s_t · L_t          s ≈ 0.69 / 0.27 / 0.04
multitask, on     :  L   = Σ_t (1/3) · L_t   =   (L_alfworld + L_webshop + L_search) / 3
```

So the weighted multitask loss is the **mean of the three single-task losses**.
The relative weighting between tasks matches the single-task setup exactly; what
remains is a global `1/3` that is identical for every task and absorbs into the
learning rate. Each task still receives the same data per step it did before —
15 prompts × group 8 — only the normalisation denominator changes.

Because one weight column is applied to every term, each coefficient keeps its
meaning and only the split across tasks moves. Leaving one term unweighted would
break the identity for the whole objective, which is why the actor's guard is a
whitelist: a term that is not routed through `_task_weighted` trips it.

## What it fixes, beyond the split

`kl_loss_coef_by_task = {alfworld: 0.01, search: 0.001, webshop: 0.01}` replicates
the single-task recipes (`run_alfworld_qwen3.sh` = 0.01, `run_search_qwen3.sh` =
0.001, `run_webshop_qwen3.sh` = 0.01), i.e. an intended ratio of **10 : 10 : 1**.

It was not arriving that way. `agg_loss_with_sample_weights` divides by the
*whole batch's* token count, so a per-row coefficient comes out multiplied by the
task's token share:

| task | intended `c_t` | × token share | effective | intended ratio | actual ratio |
|---|---|---|---|---|---|
| alfworld | 0.01 | × 0.69 | 6.9e-3 | 10 | **172** |
| webshop | 0.01 | × 0.27 | 2.7e-3 | 10 | **67** |
| search | 0.001 | × 0.04 | 4e-5 | 1 | **1** |

Search's reference-KL leash was roughly 17× looser than asked for. With the
weighting on, the normalisation and the per-task coefficient multiply into one
row weight, the effective value is `c_t / 3`, and the ratio is 10 : 10 : 1 again.
`tests/trainer/test_task_weighted_terms.py` measures both the distortion and its
removal.

## Consequences to expect

- **Effective learning rate per task changes.** Search's gradient share goes from
  ~4% to 33% (about 8×), alfworld's from ~69% to 33% (about 0.5×). The total
  gradient magnitude stays O(1), so the learning rate does not need retuning, but
  `actor/grad_norm` and `actor/pg_loss/<task>` are worth watching for the first
  tens of steps.
- **`adjust_batch`'s duplicate rows get weight 0**, and are excluded from the
  task token totals. Today the plain token-mean counts a duplicated row twice.
  The originals keep a full weight, so no turn loses its signal — but this is a
  change independent of the task split.
- **Not bit-identical.** The loss itself changes; a run cannot switch mid-flight
  and stay comparable with its earlier steps.

## Metrics

`attach_task_loss_weights` logs what it is correcting, so the split is visible
whether or not the weighting is on:

- `task_loss/token_share/<task>` — the share the plain token-mean gives
- `task_loss/rows/<task>`, `task_loss/padding_rows`

The plain loss metrics (`actor/pg_loss`, `actor/kl_loss`, `actor/sdl_loss`,
`sdar/loss`) stay **unweighted token-means** in both modes, so a dashboard does
not change scale between arms; the weighted values are logged separately as
`*_weighted`. Per-task diagnostics (`actor/pg_loss/<task>` and friends) are
unweighted token-means too, which is what makes them comparable against a
single-task run.

## Constraints

Asserted in `update_policy`: incompatible with `use_dynamic_bsz` (its micro-batch
scaling assumes an unweighted token-mean), requires
`ulysses_sequence_parallel_size == 1` and `ppo_epochs == 1`. The multitask
recipes satisfy all three.

## Enabling it

```bash
    actor_rollout_ref.actor.normalize_loss_by_task=True \
```

Pin it in the run's expectations file in the same commit — it decides what the
experiment is. Both arms of a comparison must agree on it, or their difference
carries the task weighting as well as the objective.

Implementation: `verl/trainer/ppo/task_loss_weights.py` (byte-identical with the
pure-OPD and offline-KD branches, so all arms share one implementation) and
`_task_weighted` in `verl/workers/actor/dp_actor.py`.
