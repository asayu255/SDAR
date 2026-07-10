set -x
# Backfill teacher top-20 onto SFT-generated trajectories, producing off-policy KD
# data WITHOUT re-running the (expensive) teacher rollouts. Reads the SFT pool
# (gen.collect_topk=False, no top-k) and writes an off-policy pool with top-20.
#
# Prereq: the SFT Stage-1 data must already exist at $sft_traj_dir/<task>.pt
# (produced by examples/sft_trainer/run_multitask_sft_qwen3.sh).
# Output at $out_traj_dir/<task>.pt is then usable as off-policy KD's
# algorithm.opd.teacher_data_dir.

# Per-task teachers: MUST be the same checkpoints that generated the SFT data.
teacher_alfworld=${TEACHER_ALFWORLD:-/opt/home/ohara/checkpoints/teachers/alfworld_step300}
teacher_search=${TEACHER_SEARCH:-/opt/home/ohara/checkpoints/teachers/search_step300}
teacher_webshop=${TEACHER_WEBSHOP:-/opt/home/ohara/checkpoints/teachers/webshop_step300}

sft_traj_dir=${SFT_TRAJ_DIR:-$HOME/data/verl-agent/sdar_multitask/teacher_traj_sft}
out_traj_dir=${TEACHER_TRAJ_DIR:-$HOME/data/verl-agent/sdar_multitask/teacher_traj}
opd_topk=${OPD_TOPK:-20}
chunk_size=${BACKFILL_CHUNK_SIZE:-4096}

model_path="Qwen/Qwen3-1.7B"
log_prob_micro_per_gpu=${LOG_PROB_MICRO_PER_GPU:-16}

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

declare -A teacher_for=( [alfworld]=$teacher_alfworld [search]=$teacher_search [webshop]=$teacher_webshop )
for task in alfworld search webshop; do
    python3 -m verl.trainer.main_opd_offpolicy_backfill \
        +backfill.task=$task \
        +backfill.teacher_path=${teacher_for[$task]} \
        +backfill.in_dir=$sft_traj_dir \
        +backfill.out_dir=$out_traj_dir \
        +backfill.topk=$opd_topk \
        +backfill.chunk_size=$chunk_size \
        actor_rollout_ref.model.path=$model_path \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_fused_kernels=False \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_per_gpu \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.n_gpus_per_node=3 \
        trainer.nnodes=1 \
        trainer.ray_wait_register_center_timeout=600 $@
done
