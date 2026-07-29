<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Recipe: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> Open-Source Algorithm Implementation & Expriement Running: [Yuxuan Tong](https://tongyx361.github.io/), [Guangming Sheng](https://hk.linkedin.com/in/guangming-sheng-b50640211)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
🏠 [Homepage](https://dapo-sia.github.io/) | 📝 [Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf) | 🤗 [Datasets&Models@HF](https://huggingface.co/collections/BytedTsinghua-SIA/dapo-67d7f1517ee33c8aed059da0) | 🐱 [Code@GitHub](https://github.com/volcengine/verl/tree/gm-tyx/puffin/main/recipe/dapo) | 🐱 [Repo@GitHub](https://github.com/BytedTsinghua-SIA/DAPO)


<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> We propose the **D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization (DAPO) algorithm. By making our work publicly available, we provide the broader research community and society with practical access to scalable reinforcement learning, enabling all to benefit from these advancements. Applying DAPO training to Qwen2.5-32B base model proves to outperform the previous state-of-the-art DeepSeek-R1-Zero-Qwen-32B on AIME 2024, achieving **50%** accuracy with **50%** less training steps.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
>
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> ![dapo-main-result](https://dapo-sia.github.io/static/images/score.png)

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Quickstart

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Prepare the datasets **on the Ray cluster**:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
bash prepare_dapo_data.sh # This downloads the datasets to ${HOME}/verl/data by default
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Submit the job to the Ray cluster **from any machine**:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
cd verl # Repo root
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./verl/trainer/runtime_env.yaml"
bash recipe/dapo/run_dapo_qwen2.5_32b.sh
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Reproduction Runs

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| Setup                                        | AIME 2024 Acc. | Training Script                                                  | Training Record                                                                           |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| -------------------------------------------- | -------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| DAPO w/o Token-level Loss & Dynamic Sampling | 44%            | [run_dapo_early_qwen2.5_32b.sh](./run_dapo_early_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| DAPO w/o Dynamic Sampling                    | 50%            | [run_dapo_wo_ds_qwen2.5_32b.sh](./run_dapo_wo_ds_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| DAPO                                         | 52%            | [run_dapo_qwen2.5_32b.sh](./run_dapo_qwen2.5_32b.sh)             | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Configuration

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> [!NOTE]
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> Most experiments in the paper, including the best-performant one, are run without Overlong Filtering because it's somehow overlapping with Overlong Reward Shaping in terms of properly learning from the longest outputs. So we don't implement it here.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Separated Clip Epsilons (-> Clip-Higher)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
An example configuration:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
`clip_ratio_low` and `clip_ratio_high` specify the $\varepsilon_{\text {low }}$ and $\varepsilon_{\text {high }}$ in the DAPO objective.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Core relevant code:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Dynamic Sampling (with Group Filtering)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
An example configuration:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc # score / seq_reward / seq_final_reward / ...
    max_num_gen_batches: 10 # Non-positive values mean no upper limit
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Setting `filter_groups.enable` to `True` will filter out groups whose outputs' `metric` are all the same, e.g., for `acc`, groups whose outputs' accuracies are all 1 or 0.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The trainer will repeat sampling with `gen_batch_size` until there are enough qualified groups for `train_batch_size` or reaching the upper limit specified by `max_num_gen_batches`.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Core relevant code:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```python
prompt_bsz = self.config.data.train_batch_size
if num_prompt_in_batch < prompt_bsz:
    print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
    num_gen_batches += 1
    max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        print(f'{num_gen_batches=} < {max_num_gen_batches=}. Keep generating...')
        continue
    else:
        raise ValueError(
            f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
        )
else:
    # Align the batch
    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
    batch = batch[:traj_bsz]
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Flexible Loss Aggregation Mode (-> Token-level Loss)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
An example configuration:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean" # / "seq-mean-token-sum" / "seq-mean-token-mean"
    # NOTE: "token-mean" is the default behavior
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Setting `loss_agg_mode` to `token-mean` will mean the (policy gradient) loss across all the tokens in all the sequences in a mini-batch.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Core relevant code:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```python
if loss_agg_mode == "token-mean":
    loss = verl_F.masked_mean(loss_mat, loss_mask)
elif loss_agg_mode == "seq-mean-token-sum":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
    loss = torch.mean(seq_losses)  # seq-mean
elif loss_agg_mode == "seq-mean-token-mean":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
    loss = torch.mean(seq_losses)  # seq-mean
else:
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Overlong Reward Shaping

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
An example configuration:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```yaml
data:
  max_response_length: 20480 # 16384 + 4096
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Setting `overlong_buffer.enable` to `True` will penalize the outputs whose lengths are overlong but still within the hard context limit.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Specifically, the penalty increases linearly from `0` to `overlong_buffer.penalty_factor` when the length of the output exceeds the `max_response_length` by `0` to `overlong_buffer.len` tokens.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Core relevant code:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```python
if self.overlong_buffer_cfg.enable:
    overlong_buffer_len = self.overlong_buffer_cfg.len
    expected_len = self.max_resp_len - overlong_buffer_len
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
```
