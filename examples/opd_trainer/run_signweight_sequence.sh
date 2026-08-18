set -uo pipefail

# The whole sign-weighting experiment, in the order it has to run.
#
#   1. position arm, 300 steps
#   2. target arm, 300 steps
#   3. position arm, validation at 150 and 300
#   4. target arm, validation at 150 and 300
#
# Serial on purpose: two GPUs, and each arm wants both. Roughly 70-75h per
# training arm at the measured 35-37h/150 steps, so the two trainings alone are
# about six days. RUN IT DETACHED -- tmux, or:
#
#   nohup bash examples/opd_trainer/run_signweight_sequence.sh > ~/logs/seq.log 2>&1 &
#
# Before running:
#   conda activate sdar-multitask
#   export WANDB_API_KEY=...        (this script refuses to start without it)
#   export RAY_memory_usage_threshold=0.98
#   export RAY_memory_monitor_refresh_ms=500
#
# GPU_PROFILER=1 / ROLLOUT_TURN_TIMING=1 are deliberately NOT set here. They are
# right for a shakedown run and wrong for a six-day one: the per-turn table is
# written every turn of every step.
#
# RESUMABLE. Every phase is skipped if its output already exists, and training
# resumes from its own last checkpoint (resume_mode defaults to auto, and the two
# arms have separate directories). So after a crash, a machine reboot, or a
# deliberate stop, re-running this same command picks up where it left off
# instead of starting the sequence again.
#
# Run a subset by naming phases:
#   bash examples/opd_trainer/run_signweight_sequence.sh train_position
#   bash examples/opd_trainer/run_signweight_sequence.sh val_position val_target
#
# Overridable from the environment:
#   SEARCH_URL   the retriever (default below)
#   LOG_DIR      where the per-phase logs go
#   VAL_STEPS    which checkpoints to evaluate (default "150 300")
#   TOTAL_STEPS  the step a training phase is considered finished at

SEARCH_URL=${SEARCH_URL:-http://100.86.45.30:8000/retrieve}
LOG_DIR=${LOG_DIR:-$HOME/logs}
VAL_STEPS=${VAL_STEPS:-"150 300"}
TOTAL_STEPS=${TOTAL_STEPS:-300}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "FATAL: WANDB_API_KEY is not set. A six-day run that logs to console only" >&2
    echo "       is a six-day run whose curves nobody can read afterwards." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# arm -> script, checkpoint directory. Kept in one place because a mismatch here
# is the failure that does not announce itself: trainer.resume_mode is "auto", so
# an arm pointed at another arm's directory does not overwrite it, it CONTINUES
# it and reports the result under its own name.
script_of() {
    echo "$REPO/examples/opd_trainer/run_multitask_signweight_$1_qwen3.sh"
}
ckpt_of() {
    echo "$HOME/checkpoints/verl_agent_opd_signweight_$1_multitask"
}

banner() {
    echo
    echo "==============================================================="
    echo "  $* "
    echo "  $(date -Is)"
    echo "==============================================================="
}

# Ray does not always take its cluster down with the process that started it, and
# the next phase then fails to bind or, worse, attaches to a half-dead cluster
# holding the GPUs. Between phases, not before the first: a running job elsewhere
# on this host is not this script's to kill.
tidy_ray() {
    ray stop --force >/dev/null 2>&1 || true
    sleep 10
}

preflight() {
    # env.search.max_retries=null means the client WAITS for the retriever instead
    # of giving up -- which is right (an exhausted budget puts the error text into
    # the <information> block the model trains on), and which also means a
    # retriever that is simply down looks exactly like a hang. Say so up front.
    local host_port=${SEARCH_URL#*://}
    host_port=${host_port%%/*}
    local host=${host_port%%:*}
    local port=${host_port##*:}
    if timeout 5 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
        echo "[preflight] retriever reachable at $host:$port"
    else
        echo "[preflight] WARNING: cannot reach the retriever at $host:$port." >&2
        echo "[preflight] max_retries=null means the run will WAIT rather than fail," >&2
        echo "[preflight] so a rollout that never progresses is the symptom." >&2
    fi
}

train_arm() {
    local mode=$1
    local ckpt done_marker log
    ckpt=$(ckpt_of "$mode")
    done_marker="$ckpt/global_step_$TOTAL_STEPS"
    log="$LOG_DIR/opd_signweight_${mode}_train.log"

    if [ -d "$done_marker" ]; then
        banner "SKIP training $mode -- $done_marker already exists"
        return 0
    fi
    banner "TRAIN $mode -> $TOTAL_STEPS steps"
    echo "  log:        $log"
    echo "  checkpoints:$ckpt"
    # test_freq=-1: no mid-run validation. The evaluation phases below do it from
    # the checkpoints instead, so a validation pass never sits in the middle of a
    # training step's timing, and re-evaluating costs no training time.
    bash "$(script_of "$mode")" \
        env.search.search_url="$SEARCH_URL" \
        trainer.test_freq=-1 \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "FATAL: training $mode exited $rc; see $log" >&2
        return "$rc"
    fi
    if [ ! -d "$done_marker" ]; then
        echo "FATAL: training $mode returned 0 but $done_marker is missing." >&2
        return 1
    fi
}

val_arm() {
    local mode=$1
    local ckpt step log
    ckpt=$(ckpt_of "$mode")
    for step in $VAL_STEPS; do
        if [ ! -d "$ckpt/global_step_$step" ]; then
            echo "FATAL: $ckpt/global_step_$step does not exist; train $mode first." >&2
            return 1
        fi
        log="$LOG_DIR/opd_signweight_${mode}_val_$step.log"
        banner "VALIDATE $mode @ step $step"
        echo "  log: $log"
        # val_only is checked on its own rather than under val_before_train, so
        # this evaluates the checkpoint and stops. It used to fall through to
        # TRAINING from the checkpoint, which looks like it worked right up until
        # the numbers never appear.
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
        tidy_ray
    done
}

phase_train_position() { train_arm position; }
phase_train_target()   { train_arm target; }
phase_val_position()   { val_arm position; }
phase_val_target()     { val_arm target; }

PHASES=("$@")
if [ ${#PHASES[@]} -eq 0 ]; then
    PHASES=(train_position train_target val_position val_target)
fi

preflight
banner "SEQUENCE: ${PHASES[*]}"

first=1
for phase in "${PHASES[@]}"; do
    if ! declare -F "phase_$phase" >/dev/null; then
        echo "FATAL: unknown phase '$phase'." >&2
        echo "       known: train_position train_target val_position val_target" >&2
        exit 2
    fi
    [ "$first" -eq 1 ] || tidy_ray
    first=0
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
for mode in position target; do
    echo "  $mode:  \$HOME/val_instances/opd_multitask_signweight_${mode}_qwen3_1.7b/val_step{150,300}.jsonl"
done
echo "  control: \$HOME/val_instances/opd_multitask_qwen3_1.7b/val_step{150,300}.jsonl"
echo
echo "The control's rows have to exist for the pairing to be worth anything. If"
echo "that arm was run before val_instance_log_dir was added, re-validate it:"
echo "  bash examples/opd_trainer/run_multitask_qwen3.sh \\"
echo "    env.search.search_url=$SEARCH_URL \\"
echo "    trainer.resume_mode=resume_path \\"
echo "    trainer.resume_from_path=\$HOME/checkpoints/verl_agent_opd_multitask/global_step_150 \\"
echo "    trainer.val_only=True"
