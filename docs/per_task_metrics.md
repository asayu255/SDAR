# Per-task metrics (multitask runs)

Multitask runs (`examples/sdar_trainer/run_multitask_qwen3.sh`, ...) train on
alfworld / search / webshop at the same time, so a single logged number mixes all three tasks. Every
metric that can be attributed to individual rollout rows is therefore logged a
second time per task.

## Naming

The per-task metric is the overall metric name with the task appended as the last
path segment:

```
critic/score/mean            -> critic/score/mean/alfworld
                                critic/score/mean/search
                                critic/score/mean/webshop
```

The overall metric keeps its name, and both live in the same wandb section, so
`critic/score/mean*` shows the aggregate next to its breakdown.

Tasks are bucketed by `task_name` (falling back to `env_kwargs["task_name"]`) and
normalized by substring: `alfworld_eval` -> `alfworld`, `webshop_train` ->
`webshop`. A task name that matches none of the canonical tasks keeps its own
bucket. Runs without any `task_name` (all single-task scripts) log exactly what
they logged before — no per-task keys are emitted.

## What is split

Driver side (`verl/trainer/ppo/`):

- everything in `compute_data_metrics` — `critic/score|rewards|advantages|returns|values`,
  `response_length/*`, `prompt_length/*`, `episode/reward|length|tool_call_count`
  (`compute_opd_data_metrics` for OPD)
- `actor/entropy_loss`
- `episode/valid_action_ratio`
- `actor/reward_kl_penalty`
- `rlsd/teacher_student_gap_*`, `skillsd/teacher_student_gap_*`
- `perf/total_num_tokens`, `perf/throughput` (per-task throughputs sum to the overall one)
- validation: `val/{task}/test_score`, `val/{task}/tool_call_count/mean`, which
  aggregate a task that spans several data sources (search covers nq, hotpotqa, ...)

Worker side (`verl/workers/`): the trainer tags each row with an integer
`task_ids` column (`RayPPOTrainer._attach_task_ids`) plus a `task_id_names`
lookup in `meta_info`, which lets the actor and critic re-aggregate their losses
over the rows of one task:

- `actor/pg_loss`, `actor/pg_clipfrac`, `actor/ppo_kl`, `actor/pg_clipfrac_lower`
- `actor/entropy_loss`, `actor/kl_loss`, `actor/kl_coef` (when task-weighted)
- `actor/sdl_loss`, `actor/teacher_kl_loss`, the `sdar/*` loss metrics
- `critic/vf_loss`, `critic/vf_clipfrac`, `critic/vpred_mean`

These re-aggregations run under `torch.no_grad()` on tensors that were already
computed for the loss; they are diagnostics and do not change the objective, the
gradients, or the optimizer step.

## What is not split

- Timings (`timing_s/*`, `timing_per_token_ms/*`, `perf/time_per_step`) — one step
  runs all tasks together, so wall-clock cannot be attributed to a task.
- Global constants: coefficients and schedules (`actor/kl_coef` when a single
  scalar, `rlsd/lambda`, `skillsd/sdl_lambda`, `actor/lr`, ...), `actor/grad_norm`,
  `perf/mfu`, memory, and the `training/*` and `global_seqlen/*` bookkeeping.
- Success rates: the env manager computes them over the whole rollout and copies
  the value onto every row, so a row slice cannot recover a per-task value. They
  are already reported per task as `episode/{task}_success_rate` and
  `val/{task}_success_rate`.
