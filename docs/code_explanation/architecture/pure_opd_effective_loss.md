# Pure OPD の実効 Loss

## Entry point の上書き

`OPDTaskRunner.run()` は Hydra compose 後、worker 作成前に actor config を
次のように上書きする。

```text
use_teacher_kl_loss = True
teacher_kl_loss_coef = algorithm.opd.kl_loss_coef
teacher_kl_loss_type = algorithm.opd.kl_loss_type
teacher_kl_topk = algorithm.opd.topk
pg_loss_coef = 0
entropy_coeff = 0
use_kl_loss = False
use_sdl_loss = False
use_sdar_loss = False
algorithm.use_kl_in_reward = False
```

その後に `expected_multitask_config.yaml` と比較するため、expectations は raw
default ではなく、上書き後の effective config を固定する。

## Actor 内の scalar loss

`DataParallelPPOActor.update_policy()` の加算順は概念上、次のとおりである。

```text
policy_loss
  = pg_loss * pg_loss_coef
  - entropy_loss * entropy_coeff
  + optional reference KL
  + optional SDL
  + optional SDAR
  + teacher_kl_loss * teacher_kl_loss_coef
```

Pure OPD の実効値を代入すると、前5項はゼロまたは分岐未到達になり、

```text
effective_loss = token_mean(
    teacher_KL(student_log_prob, detached_teacher_log_prob),
    response_mask
) * teacher_kl_loss_coef
```

だけが残る。teacher Tensor は detach/no-grad、student Tensor は gradient
付きなので、backward は student 側だけへ流れる。

## Invalid-action penalty の判定

config では `use_invalid_action_penalty=True` と task 別 coefficient が有効である。
しかし Pure OPD `fit()` では `compute_reward()` の出力を
`batch.batch["token_level_scores"]` に保存した後、軽量 metrics と rollout dump
に使うだけで、advantage/returns を計算しない。actor の Pure OPD 分岐も
`token_level_scores` を選択せず、teacher KL の加算式へ渡さない。

したがって、この source commit では invalid-action penalty は
「reward Tensor と、それを読む monitoring/dump を変えるが、Pure OPD scalar
loss には入らない」に該当する。通常 PPO/GRPO 経路では意味を持ち得るが、
Pure OPD thin loop の gradient 信号ではない。
