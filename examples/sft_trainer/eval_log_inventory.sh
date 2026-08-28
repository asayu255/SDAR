#!/usr/bin/env bash
# What configuration produced each of these logs, and what did it cost?
#
#   bash examples/sft_trainer/eval_log_inventory.sh /tmp/eval_*.log
#
# judge_eval.sh compares TWO logs and assumes you picked a pair that differs in
# one thing. Picking that pair is the part that goes wrong: a directory of
# eval_*.log files carries the configuration only in the filename, and a
# filename is a claim, not a measurement -- /tmp/eval_pump_ms4.log turned out to
# be V0 + multi-step + pump, so comparing it against a V1 run measured two
# changes at once and attributed the result to one of them.
#
# So this reads what each run ACTUALLY reported about itself and prints it as a
# table. Two rows that differ in one column are a valid pair for judge_eval.sh;
# two rows that differ in three are not an experiment.
set -uo pipefail

[ "$#" -gt 0 ] || { sed -n '2,4p' "$0"; exit 1; }

_NORMDIR=$(mktemp -d)
trap 'rm -rf "$_NORMDIR"' EXIT
# Strip Ray's "(Actor pid=N) " prefix once into a file, then grep the file --
# never `sed | grep -q`, which returns 141 under pipefail when grep exits first.
_clean() {
    local out="$_NORMDIR/$(printf '%s' "$1" | md5sum | cut -c1-16)"
    [ -s "$out" ] || sed 's/^([^)]*) //' "$1" > "$out"
    printf '%s' "$out"
}

_first() { grep -m1 -o "$2" "$1" || true; }

printf '%-26s %-6s %-5s %-6s %-6s %-6s %-8s %-9s %-8s %s\n' \
    LOG PUMP CORE STEPS WIDTH DEPTH BATCHES WALL ALL/LAST SCORE
printf '%-26s %-6s %-5s %-6s %-6s %-6s %-8s %-9s %-8s %s\n' \
    -------------------------- ------ ----- ------ ------ ------ -------- --------- -------- -----

for log in "$@"; do
    [ -r "$log" ] || continue
    f=$(_clean "$log")
    # Only files that are actually evaluation runs.
    grep -q 'val-pipeline\]' "$f" || continue

    pump=off
    grep -q 'rollout-pump.*driving .* ranks as a pool' "$f" && pump=ON
    grep -q 'rollout-pump.*staying on the blocking path\|rollout-pump.*handshake failed' "$f" && pump=REFUSED

    # STEPS is only knowable from the [rollout-engine] line. A run older than
    # that line reports no core, and defaulting its steps to 1 would state as
    # measured the very thing that cannot be read -- which is how a log gets
    # mistaken for a valid control. No engine line, no engine columns.
    # The [eval] config line, when the run wrote one: it appears on line two,
    # before Ray starts, so a run that died in startup is still identifiable.
    # Everything below falls back to the in-run lines for older logs.
    _cfg=$(_first "$f" '\[eval\] config *:.*')
    core=$(_first "$f" 'core=[A-Za-z0-9]*'); core=${core#core=}
    if [ -n "$core" ]; then
        steps=$(_first "$f" 'num_scheduler_steps=[0-9]*'); steps=${steps#num_scheduler_steps=}
        steps=${steps:-1}   # =<default> or =absent both mean ordinary scheduling
    else
        steps="?"
    fi
    depth=$(_first "$f" 'VAL_PIPELINE_DEPTH=[0-9]*'); depth=${depth#VAL_PIPELINE_DEPTH=}
    if [ -z "$depth" ] && [ -n "$_cfg" ]; then
        depth=$(printf '%s' "$_cfg" | grep -o 'depth=[0-9]*' | cut -d= -f2)
    fi
    width=$(printf '%s' "$_cfg" | grep -o 'search=[0-9]*' | cut -d= -f2)

    # The last report, final or periodic -- and whether it was the final one,
    # because a partial run's wall is not comparable to a finished one's.
    last=$(grep -o 'val-pipeline\] \(final\|after [0-9]*\): [0-9]* batches over [0-9.]*s' "$f" | tail -1)
    batches=$(printf '%s' "$last" | grep -o '[0-9]* batches' | cut -d' ' -f1)
    wall=$(printf '%s' "$last" | grep -o 'over [0-9.]*s' | sed 's/over //')
    # No [val-pipeline] report yet -- it prints every VAL_PIPELINE_REPORT_EVERY
    # batches -- but the WALL lines are per batch and are already there. Count
    # those rather than printing "?", which reads as "this log is unreadable"
    # when the run is simply young.
    if [ -z "$batches" ]; then
        batches=$(grep -c 'ms/row last' "$f")
        wall="not-yet"   # fits the column; the "*" already says it is running
    fi
    case "$last" in *final:*) ;; *) batches="${batches:-?}*" ;; esac  # * = still going

    # last= AND all=. all= is cumulative from batch 1, so on a young run it is
    # mostly the ~70 s of env-manager construction and the first CUDA graph
    # capture: a run three batches in read 112 against a finished run's 67 and
    # the difference was startup. last= is a moving window and sheds that.
    msline=$(grep -o 'ms/row last[0-9]*=[0-9]* all=[0-9]*' "$f" | tail -1)
    msrow=$(printf '%s' "$msline" | grep -o 'all=[0-9]*' | cut -d= -f2)
    mslast=$(printf '%s' "$msline" | grep -o 'last[0-9]*=[0-9]*' | cut -d= -f2)
    [ -n "$mslast" ] && msrow="${msrow}/${mslast}"
# THE SCORES COME FROM scripts/val_scores.py, not from a grep in this file.
#
# ray_trainer prints pprint(f"Initial validation metrics: {val_metrics}") --
# pprint of an f-STRING, so the dict is one string broken at 80 columns
# wherever that falls, a key can end one line and its value begin the next, the
# break moves with the dict's contents, and a log can hold more than one block.
# Three shell extractions got that wrong in three different ways and each time
# a key went silently missing, which reads as "the run did not score it".
_SCORES="$(dirname "$0")/../../scripts/val_scores.py"
    score=$(python3 "$_SCORES" "$log" 2>/dev/null \
        | awk '$1 == "val/success_rate" {print $2}' | tail -1)

    printf '%-26s %-6s %-5s %-6s %-6s %-6s %-8s %-9s %-8s %s\n' \
        "$(basename "$log")" "$pump" "${core:-?}" "$steps" "${width:-?}" "${depth:-?}" \
        "${batches:-?}" "${wall:-?}" "${msrow:-?}" "${score:-unfinished}"
done

cat <<'NOTE'

  A "*" on BATCHES means the run had not printed [val-pipeline] final: -- it was
  killed or is still going, so its WALL covers only those batches.
  STEPS is num_scheduler_steps: 1 is ordinary scheduling, >1 is V0 multi-step.
  Pick two rows differing in ONE column, then: judge_eval.sh <control> <candidate>
NOTE
