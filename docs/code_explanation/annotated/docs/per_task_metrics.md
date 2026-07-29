<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Per-task metrics (multitask runs)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Multitask runs (`examples/sdar_trainer/run_multitask_qwen3.sh`,
`examples/opd_trainer/run_multitask_qwen3.sh`, ...) train on alfworld / search /
webshop at the same time, so a single logged number mixes all three tasks. Every
metric that can be attributed to individual rollout rows is therefore logged a
second time per task.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Naming

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The per-task metric is the overall metric name with the task appended as the last
path segment:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
critic/score/mean            -> critic/score/mean/alfworld
                                critic/score/mean/search
                                critic/score/mean/webshop
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The overall metric keeps its name, and both live in the same wandb section, so
`critic/score/mean*` shows the aggregate next to its breakdown.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Tasks are bucketed by `task_name` (falling back to `env_kwargs["task_name"]`) and
normalized by substring: `alfworld_eval` -> `alfworld`, `webshop_train` ->
`webshop`. A task name that matches none of the canonical tasks keeps its own
bucket. Runs without any `task_name` (all single-task scripts) log exactly what
they logged before — no per-task keys are emitted.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## What is split

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Driver side (`verl/trainer/ppo/`):

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- everything in `compute_data_metrics` — `critic/score|rewards|advantages|returns|values`,
  `response_length/*`, `prompt_length/*`,
  `episode/reward|length|response_tokens|tool_call_count`
  (`compute_opd_data_metrics` for OPD)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `actor/entropy_loss`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `episode/valid_action_ratio`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `actor/reward_kl_penalty`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `rlsd/teacher_student_gap_*`, `skillsd/teacher_student_gap_*`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `perf/total_num_tokens`, `perf/throughput` (per-task throughputs sum to the overall one)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- validation: `val/{task}/test_score`, `val/{task}/tool_call_count/mean`, which
  aggregate a task that spans several data sources (search covers nq, hotpotqa, ...)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Worker side (`verl/workers/`): the trainer tags each row with an integer
`task_ids` column (`RayPPOTrainer._attach_task_ids`) plus a `task_id_names`
lookup in `meta_info`, which lets the actor and critic re-aggregate their losses
over the rows of one task:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `actor/pg_loss`, `actor/pg_clipfrac`, `actor/ppo_kl`, `actor/pg_clipfrac_lower`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `actor/entropy_loss`, `actor/kl_loss`, `actor/kl_coef` (when task-weighted)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `actor/sdl_loss`, `actor/teacher_kl_loss`, the `sdar/*` loss metrics
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `critic/vf_loss`, `critic/vf_clipfrac`, `critic/vpred_mean`

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
These re-aggregations run under `torch.no_grad()` on tensors that were already
computed for the loss; they are diagnostics and do not change the objective, the
gradients, or the optimizer step.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Turn-level vs sample-level lengths

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
A row of the batch is one env turn, so `response_length/*` and `prompt_length/*`
are **per-turn** token counts and `episode/length/*` is the **number of turns** in
a trajectory. `episode/response_tokens/mean|max|min` is the sample-level length:
the rows sharing a `traj_uid` are summed, giving the tokens one whole trajectory
generated across its turns (`compute_trajectory_response_tokens`). It is split
per task like every other row-derived metric.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Prompt tokens are deliberately not summed the same way — every turn re-sends the
history, so a trajectory total would count the same context many times over.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## What is not split

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Timings (`timing_s/*`, `timing_per_token_ms/*`, `perf/time_per_step`) — one step
  runs all tasks together, so wall-clock cannot be attributed to a task.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Global constants: coefficients and schedules (`actor/kl_coef` when a single
  scalar, `rlsd/lambda`, `skillsd/sdl_lambda`, `actor/lr`, ...), `actor/grad_norm`,
  `perf/mfu`, memory, and the `training/*` and `global_seqlen/*` bookkeeping.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Success rates: the env manager computes them over the whole rollout and copies
  the value onto every row, so a row slice cannot recover a per-task value. They
  are already reported per task as `episode/{task}_success_rate` and
  `val/{task}_success_rate`.
