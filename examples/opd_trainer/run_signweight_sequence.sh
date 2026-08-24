set -uo pipefail

# The sign-weighting experiment, mid-point first:
#
#   1. position arm  -> step 150         5. position arm  150 -> 300
#   2. validate position @ 150           6. validate position @ 300
#   3. target arm    -> step 150         7. target arm    150 -> 300
#   4. validate target @ 150             8. validate target @ 300
#      + validate the CONTROL @ 150/300 (instance logs) between 4 and 5
#
# Mid-point first because the primary comparison is against the control's
# 150-step numbers: this ordering has the first comparable result after
# ~1.5 days and both arms compared at 150 after ~3, instead of every number
# arriving at the end of day 5.
#
# HOW 150 IS REACHED. trainer.stop_after_steps=150, NOT total_training_steps=150.
# total_training_steps is pinned in the intent lock and also sets the LR
# schedule (warmup is 10% of total), so a 150-step run would refuse to start
# and, if forced, would put a different LR trajectory into steps 15-30 than the
# control had. stop_after_steps only decides when this process exits; the
# continuation resumes from the step-150 checkpoint with schedule, data order
# and objective identical to a straight 300-step run interrupted by a crash.
#
# Serial on purpose: two GPUs, each arm wants both. RUN IT DETACHED:
#   nohup bash examples/opd_trainer/run_signweight_sequence.sh > ~/logs/seq.log 2>&1 &
#
# Before running:
#   conda activate sdar-multitask
#   export WANDB_API_KEY=...        (this script refuses to start without it)
#   export RAY_memory_usage_threshold=0.98
#   export RAY_memory_monitor_refresh_ms=500
#
# RESUMABLE. Every phase is skipped when its output already exists (training: the
# phase's checkpoint; validation: its per-instance jsonl), and training resumes
# from its own last checkpoint. Re-running this same command after a crash or a
# deliberate stop continues instead of starting over.
#
# Run a subset by naming phases:
#   bash examples/opd_trainer/run_signweight_sequence.sh train_position_150
#   bash examples/opd_trainer/run_signweight_sequence.sh val_control
#
# The teacher-indexed arm (run_multitask_signweight_target_teachertopk_qwen3.sh)
# has phases here but is deliberately NOT in the default list: its support comes
# from the teacher, which is a different objective, so it does not belong in a
# sequence whose point is that its arms are comparable. Train and evaluate it to
# 150 with:
#   bash examples/opd_trainer/run_signweight_sequence.sh \
#       train_target_teachertopk_150 val_target_teachertopk_150
#
# Overridable from the environment:
#   SEARCH_URL   the retriever (default below)
#   LOG_DIR      where the per-phase logs go
#   MID_STEPS    the pause point (default 150; must be a save_freq multiple)
#   TOTAL_STEPS  the step a training arm is considered finished at

SEARCH_URL=${SEARCH_URL:-http://100.86.45.30:8001/retrieve}
LOG_DIR=${LOG_DIR:-$HOME/logs}
MID_STEPS=${MID_STEPS:-150}
TOTAL_STEPS=${TOTAL_STEPS:-300}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Profiling stays OFF for the sequence no matter what the launching shell has
# exported. The per-step stage tables (and their CUDA-event overhead) are for
# diagnosing one run, not for a multi-day sequence, and a GPU_PROFILER=1 left
# over in the shell would silently switch them back on. Run a single arm by
# hand with GPU_PROFILER=1 if a profile is ever wanted.
export GPU_PROFILER=0
export GPU_PROFILER_SYNC_PHASES=0

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "FATAL: WANDB_API_KEY is not set. A multi-day run that logs to console only" >&2
    echo "       is a multi-day run whose curves nobody can read afterwards." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# arm -> script, checkpoint directory, experiment name (= instance-log subdir).
# One place, because a mismatch does not announce itself: resume_mode is "auto",
# so an arm pointed at another arm's directory does not overwrite it, it
# CONTINUES it and reports the result under its own name.
script_of() {
    case "$1" in
        control) echo "$REPO/examples/opd_trainer/run_multitask_qwen3.sh" ;;
        *)       echo "$REPO/examples/opd_trainer/run_multitask_signweight_$1_qwen3.sh" ;;
    esac
}
ckpt_of() {
    case "$1" in
        control) echo "$HOME/checkpoints/verl_agent_opd_multitask" ;;
        *)       echo "$HOME/checkpoints/verl_agent_opd_signweight_$1_multitask" ;;
    esac
}
exp_of() {
    case "$1" in
        control) echo "opd_multitask_qwen3_1.7b" ;;
        *)       echo "opd_multitask_signweight_$1_qwen3_1.7b" ;;
    esac
}

banner() {
    echo
    echo "==============================================================="
    echo "  $* "
    echo "  $(date -Is)"
    echo "==============================================================="
}

# Ray does not always take its cluster down with the process that started it.
tidy_ray() {
    ray stop --force >/dev/null 2>&1 || true
    sleep 10
}

preflight() {
    # max_retries=null means the client WAITS for the retriever rather than
    # training on its error text, so a retriever that is down looks like a hang.
    local host_port=${SEARCH_URL#*://}
    host_port=${host_port%%/*}
    local host=${host_port%%:*}
    local port=${host_port##*:}
    if timeout 5 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
        echo "[preflight] retriever reachable at $host:$port"
    else
        echo "[preflight] WARNING: cannot reach the retriever at $host:$port." >&2
        echo "[preflight] max_retries=null means the run will WAIT rather than fail." >&2
    fi
}

train_arm() {
    local mode=$1 stop=$2
    local ckpt log
    ckpt=$(ckpt_of "$mode")
    log="$LOG_DIR/opd_signweight_${mode}_train_to${stop}.log"

    # Done when this phase's checkpoint exists -- or a LATER one does, since a
    # finished 300 makes "get to 150" moot on a resumed sequence.
    if [ -d "$ckpt/global_step_$stop" ] || [ -d "$ckpt/global_step_$TOTAL_STEPS" ]; then
        banner "SKIP training $mode -> $stop (checkpoint already exists)"
        return 0
    fi
    banner "TRAIN $mode -> step $stop (of $TOTAL_STEPS)"
    echo "  log:        $log"
    echo "  checkpoints:$ckpt"
    local extra=()
    if [ "$stop" -lt "$TOTAL_STEPS" ]; then
        extra+=("trainer.stop_after_steps=$stop")
    fi
    # test_freq=-1: no mid-run validation; the val phases below do it from the
    # checkpoints, so re-evaluating never costs training time.
    bash "$(script_of "$mode")" \
        env.search.search_url="$SEARCH_URL" \
        trainer.test_freq=-1 \
        "${extra[@]}" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "FATAL: training $mode exited $rc; see $log" >&2
        return "$rc"
    fi
    if [ ! -d "$ckpt/global_step_$stop" ]; then
        echo "FATAL: training $mode returned 0 but $ckpt/global_step_$stop is missing." >&2
        return 1
    fi
}

val_arm() {
    local mode=$1 step=$2
    local ckpt marker log
    ckpt=$(ckpt_of "$mode")
    marker="$HOME/val_instances/$(exp_of "$mode")/val_step$step.jsonl"
    log="$LOG_DIR/opd_signweight_${mode}_val_$step.log"

    if [ -f "$marker" ]; then
        banner "SKIP validation $mode @ $step ($marker already exists)"
        return 0
    fi
    if [ ! -d "$ckpt/global_step_$step" ]; then
        echo "FATAL: $ckpt/global_step_$step does not exist; train $mode first." >&2
        [ "$mode" = "control" ] && echo "       (the control was trained in an earlier experiment; check the path)" >&2
        return 1
    fi
    banner "VALIDATE $mode @ step $step"
    echo "  log:     $log"
    echo "  instance rows: $marker"
    bash "$(script_of "$mode")" \
        env.search.search_url="$SEARCH_URL" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$ckpt/global_step_$step" \
        trainer.val_only=True \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "FATAL: validation $mode @ $step exited $rc; see $log" >&2
        return "$rc"
    fi
    if [ ! -f "$marker" ]; then
        echo "WARNING: validation ran but $marker was not written; the paired analysis" >&2
        echo "         needs it -- check trainer.val_instance_log_dir in the run script." >&2
    fi
    tidy_ray
}

phase_train_position_150() { train_arm position "$MID_STEPS"; }
phase_val_position_150()   { val_arm position "$MID_STEPS"; }
phase_train_target_150()   { train_arm target "$MID_STEPS"; }
phase_val_target_150()     { val_arm target "$MID_STEPS"; }
# The control's instance rows: its original run predates val_instance_log_dir,
# and without them the treatment rows have nothing to pair against.
phase_val_control()        { val_arm control "$MID_STEPS" && val_arm control "$TOTAL_STEPS"; }
# The teacher-indexed arm. NOT part of the default sequence and not comparable
# with the three arms that are: it takes its top-k support from the teacher, so
# it optimises a different lower bound on the same KL. Run it by name.
phase_train_target_teachertopk_150() { train_arm target_teachertopk "$MID_STEPS"; }
phase_val_target_teachertopk_150()   { val_arm target_teachertopk "$MID_STEPS"; }
phase_train_position_300() { train_arm position "$TOTAL_STEPS"; }
phase_val_position_300()   { val_arm position "$TOTAL_STEPS"; }
phase_train_target_300()   { train_arm target "$TOTAL_STEPS"; }
phase_val_target_300()     { val_arm target "$TOTAL_STEPS"; }

PHASES=("$@")
if [ ${#PHASES[@]} -eq 0 ]; then
    PHASES=(
        train_position_150 val_position_150
        train_target_150 val_target_150
        val_control
        train_position_300 val_position_300
        train_target_300 val_target_300
    )
fi

preflight
banner "SEQUENCE: ${PHASES[*]}"

for phase in "${PHASES[@]}"; do
    if ! declare -F "phase_$phase" >/dev/null; then
        echo "FATAL: unknown phase '$phase'." >&2
        echo "       known: train_position_150 val_position_150 train_target_150 val_target_150" >&2
        echo "              val_control train_position_300 val_position_300 train_target_300 val_target_300" >&2
        echo "              train_target_teachertopk_150 val_target_teachertopk_150 (not in the default run)" >&2
        exit 2
    fi
    # Before EVERY phase, the first included: a cluster left over from an
    # earlier shell keeps that shell's environment (GPU_PROFILER and all), and
    # workers inherit from the raylet, not from this script.
    tidy_ray
    if ! "phase_$phase"; then
        echo >&2
        echo "SEQUENCE STOPPED at '$phase'. Fix it, then re-run the same command --" >&2
        echo "finished phases are skipped and training resumes from its checkpoint." >&2
        exit 1
    fi
done

banner "SEQUENCE COMPLETE"
echo "wandb project: verl_agent_opd_signweight_multitask"
echo
echo "Per-instance validation rows, for the PAIRED comparison:"
for mode in position target control target_teachertopk; do
    echo "  $mode: \$HOME/val_instances/$(exp_of "$mode")/val_step{$MID_STEPS,$TOTAL_STEPS}.jsonl"
done
