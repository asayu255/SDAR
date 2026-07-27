set -x

# Data prep only — builds the shared multitask prompt parquets and nothing else.
#
# run_multitask_offpolicy_qwen3.sh runs this same step inline before Stage 1, so
# this script exists for the flows that skip generation and would otherwise have
# no train.parquet / test.parquet:
#   * run_multitask_offpolicy_qwen3_nogen.sh (Stage 2 on an existing teacher pool)
#   * backfill_topk_from_sft.sh (top-20 backfill over SFT trajectories)
#
# Pure pandas: no GPU, no Ray, no ALFWORLD_DATA / retriever, no model download.
# Only --search_dir is read (the preprocessed SearchR1 corpus); alfworld and
# webshop rows are placeholders whose episodes come from their environments.
#
# EVERY parameter is a literal argument below, matching the house rule of the
# other scripts in this directory: one place to read or edit a setting.
#
# The literals are cross-checked by the expectations files in this directory
# (per_task_batch_size=15, val_per_task_size=126, total_training_steps=300,
# data.seed=1), so they must stay in sync with them.
#
# Sizing (--total_training_steps 300 x --per_task_batch_size 15):
#   train: search 4500 real prompts, alfworld/webshop 15 placeholder rows each
#   test:  126 rows per task (env.multitask.val_per_task_batch_size)
# 4500 search prompts / 15 per batch = 300 batches x env.rollout.n 8 = 36000
# trajectories, i.e. exactly the Stage-1 gen.num_trajectories target, consumed
# in a single pass over the prompt pool.
#
# Rerunning is safe and deterministic: --seed 1 fixes the sample, and the
# parquets are overwritten in place.

python3 -m examples.data_preprocess.prepare_sdar_multitask \
    --search_dir "$HOME/data/searchR1_processed_direct" \
    --local_dir "$HOME/data/verl-agent/sdar_multitask" \
    --total_training_steps 300 \
    --per_task_batch_size 15 \
    --env_train_per_task_size 15 \
    --val_per_task_size 126 \
    --seed 1
