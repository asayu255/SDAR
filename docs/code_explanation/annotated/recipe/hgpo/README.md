<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# HGPO: Hierarchy-of-Groups Policy Optimization for Long-horizon Agentic Tasks ICLR 2026

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This repository provides a **HGPO (Hierarchy-of-Groups Policy Optimization for Long-horizon Agentic Tasks)** recipe for verl-agent, used for multi-turn agentic RL.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
![Motivation](illustration.png)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
*Motivation: Figure (a) compares trajectory-wise and stepwise policy optimization frameworks. Given two example group trajectories, Figure (b) illustrates trajectory-level and step-level grouping with their corresponding advantage estimations. Best viewed in color.*

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
![HGPO](HGPO.png)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
*Overview of HGPO. The LLM-based agent interacts with a set of environments initialized from the same state $\bm{s}_{0}$, producing four group trajectories (states with the same color are identical). HGPO comprises two key components: context-aware hierarchical grouping and adaptive weighted advantage computation. For illustration, consider the state $\bm{s}_{2}$ (purple). First, HGPO assigns $\bm{s}_{2}$ into three hierarchical groups according to its historical contexts. Then, it computes the final advantage estimate by adaptively aggregating the weighted advantages from these groups.*

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Scripts (recipe/hgpo)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
All scripts live under `recipe/hgpo/`, organized by model size and environment:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| Script | Description | Wandb logs
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
|--------|-------------|----------|
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `run_qwen2.5_1.5b_alfworld_train.sh` | AlfWorld training, Qwen2.5-1.5B| [![wandb](wandb_log.svg)](https://api.wandb.ai/links/hs827083890-nanyang-technological-university-singapore/wqg929x9)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `run_qwen2.5_1.5b_alfworld_eval.sh` | AlfWorld evaluation, Qwen2.5-1.5B | [![wandb](wandb_log.svg)](https://api.wandb.ai/links/hs827083890-nanyang-technological-university-singapore/wjw9osa8)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `run_qwen2.5_7b_alfworld_train.sh` | AlfWorld training, Qwen2.5-7B| [![wandb](wandb_log.svg)](https://wandb.ai/hs827083890-nanyang-technological-university-singapore/Qwen2-5_7b_Alfworld_train_open/reports/Qwen2-5_7B_Alfworld_train--VmlldzoxNTkyODEyMA)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `run_qwen2.5_7b_alfworld_eval.sh` | AlfWorld evaluation, Qwen2.5-7B| [![wandb](wandb_log.svg)](https://wandb.ai/hs827083890-nanyang-technological-university-singapore/Qwen2-5_7b_Alfworld_eval_open/reports/Qwen2-5_7B_Alfworld_eval--VmlldzoxNTkyODE5Nw)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `run_qwen2.5_1.5b_webshop_train.sh` | WebShop training, Qwen2.5-1.5B| [![wandb](wandb_log.svg)](https://wandb.ai/hs827083890-nanyang-technological-university-singapore/Qwen2-5_1-5b_webshop_train_open/reports/Qwen2-5_1-5B_WebShop_train--VmlldzoxNTkyNzkxNA)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `run_qwen2.5_1.5b_webshop_eval.sh` | WebShop evaluation, Qwen2.5-1.5B| [![wandb](wandb_log.svg)](https://wandb.ai/hs827083890-nanyang-technological-university-singapore/Qwen2-5_1-5b_webshop_eval_open/reports/Qwen2-5_1-5b_WebShop_eval--VmlldzoxNTkyODAyMg)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `run_qwen2.5_7b_webshop_train.sh` | WebShop training, Qwen2.5-7B| [![wandb](wandb_log.svg)](https://wandb.ai/hs827083890-nanyang-technological-university-singapore/Qwen2-5_7b_webshop_train_open/reports/Qwen2-5_7B_WebShop_train--VmlldzoxNTkyODE0OQ)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `run_qwen2.5_7b_webshop_eval.sh` | WebShop evaluation, Qwen2.5-7B| [![wandb](wandb_log.svg)](https://wandb.ai/hs827083890-nanyang-technological-university-singapore/Qwen2-5_7b_webshop_eval_open/reports/Qwen2-5_7B_WebShop_eval--VmlldzoxNTkyODE2NQ)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **Training scripts:** Set `history_length`, `group_size`, `mode`, `weight_type`, `length_weight_alpha`, `base_group`, etc. Experiment names are auto-generated (e.g. `k2_hgpo_length_alpha1.0_baseGroup_False`).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **Eval scripts:** Fill in `eval_experiment_names` in the script (matching training `experiment_name`). The script parses `history_length` from the name (e.g. `k2`→2, `k4`→4), runs evaluation for each of `seeds=(123 456 789)`, and writes logs to `logs/<checkpoint_dir>/output_seed{seed}.log`.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Environment variables

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Scripts rely on the following environment variables (set as needed):

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `HF_HOME`: Hugging Face cache directory
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `WANDB_API_KEY`: WandB API key (optional)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `WANDB_DIR`: WandB log directory (optional)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `CUDA_VISIBLE_DEVICES`: Visible GPUs
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `CHECKPOINTS_DIR`: Checkpoint root directory; used by both training and evaluation

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Example:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
export HF_HOME=/path/to/hf
export WANDB_API_KEY=your_key
export WANDB_DIR=/path/to/wandb
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CHECKPOINTS_DIR=/path/to/checkpoints
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Algorithm parameters (algorithm.hgpo)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| Parameter | Description | Typical values |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
|-----------|-------------|----------------|
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `mode` | Within-group advantage normalization | `mean_norm` / `mean_std_norm` |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `weight_type` | Within-group weight type | `length` (step-length weighting) |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `length_weight_alpha` | Weight is L^alpha; alpha=0 is uniform | 1.0 |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `base_group` | Use episode advantage as initial group in aggregation | true / false |

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Use together with env options such as `env.history_length` and `env.rollout.n` (rollouts per group).

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Reproducing experiments

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Environment and data

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Install and configure AlfWorld / WebShop (see [agent_system/environments](../../agent_system/environments)).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Data is used only to set batch size and format. Prepare text data and generate parquet first:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
python3 -m examples.data_preprocess.prepare --mode 'text' --train_data_size 16 --val_data_size 128
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Paths are set in the scripts via `data.train_files` / `data.val_files`; defaults are `$HOME/data/verl-agent/text/train.parquet` and `$HOME/data/verl-agent/text/test.parquet`.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### AlfWorld

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Training (1.5B, 2 GPUs):**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
bash recipe/hgpo/run_qwen2.5_1.5b_alfworld_train.sh
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Training (7B, 4 GPUs):**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
bash recipe/hgpo/run_qwen2.5_7b_alfworld_train.sh
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Evaluation:** Edit `eval_experiment_names` in the corresponding eval script (e.g. add `k2_hgpo_length_alpha1.0_baseGroup_False`), then run:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
# 1.5B
bash recipe/hgpo/run_qwen2.5_1.5b_alfworld_eval.sh

# 7B
bash recipe/hgpo/run_qwen2.5_7b_alfworld_eval.sh
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
In AlfWorld eval scripts, `val_out` controls the validation set: `val_out=True` for in-domain, `val_out=False` for out-of-domain (some scripts use the variable name `eval_out` with the same meaning).

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### WebShop

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Training:**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
# 1.5B, 2 GPUs
bash recipe/hgpo/run_qwen2.5_1.5b_webshop_train.sh

# 7B, 4 GPUs
bash recipe/hgpo/run_qwen2.5_7b_webshop_train.sh
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Evaluation:** Similarly, set `eval_experiment_names` in the eval script (e.g. `k2_hgpo_length_step30_alpha1.0`), then run:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
bash recipe/hgpo/run_qwen2.5_1.5b_webshop_eval.sh
# or
bash recipe/hgpo/run_qwen2.5_7b_webshop_eval.sh
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Upstream dependencies (recipe self-contained parts)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This recipe is self-contained under `recipe/hgpo/` for HGPO logic and trainer extensions when submitting to upstream verl-agent:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| File | Description |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
|------|-------------|
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `hgpo/core_hgpo.py` | HGPO advantage computation (self-contained) |
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
| `hgpo/hgpo_ray_trainer.py` | PPO Ray trainer with HGPO support; `adjust_batch()` runs after `compute_advantage()` |

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
If not included upstream, you may also need:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **`agent_system/environments/env_manager.py`**: AlfWorld branch should use `config.trainer.val_out` to select in-domain vs out-of-domain validation.


<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Related links

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [verl-agent](https://github.com/langfengQ/verl-agent)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [GiGPO paper](https://arxiv.org/abs/2505.10978)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bibtex
@inproceedings{
he2026hierarchyofgroups,
title={Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks},
author={Shuo He and Lang Feng and Qi Wei and Xin Cheng and Lei Feng and Bo An},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=T8Dev99qnz}
}
```
