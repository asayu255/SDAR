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

printf '%-26s %-6s %-5s %-6s %-6s %-8s %-9s %-8s %s\n' \
    LOG PUMP CORE STEPS DEPTH BATCHES WALL MS/ROW SCORE
printf '%-26s %-6s %-5s %-6s %-6s %-8s %-9s %-8s %s\n' \
    -------------------------- ------ ----- ------ ------ -------- --------- -------- -----

for log in "$@"; do
    [ -r "$log" ] || continue
    f=$(_clean "$log")
    # Only files that are actually evaluation runs.
    grep -q 'val-pipeline\]' "$f" || continue

    pump=off
    grep -q 'rollout-pump.*driving .* ranks as a pool' "$f" && pump=ON
    grep -q 'rollout-pump.*staying on the blocking path\|rollout-pump.*handshake failed' "$f" && pump=REFUSED

    core=$(_first "$f" 'core=[A-Za-z0-9]*'); core=${core#core=}
    steps=$(_first "$f" 'num_scheduler_steps=[0-9]*'); steps=${steps#num_scheduler_steps=}
    depth=$(_first "$f" 'VAL_PIPELINE_DEPTH=[0-9]*'); depth=${depth#VAL_PIPELINE_DEPTH=}

    # The last report, final or periodic -- and whether it was the final one,
    # because a partial run's wall is not comparable to a finished one's.
    last=$(grep -o 'val-pipeline\] \(final\|after [0-9]*\): [0-9]* batches over [0-9.]*s' "$f" | tail -1)
    batches=$(printf '%s' "$last" | grep -o '[0-9]* batches' | cut -d' ' -f1)
    wall=$(printf '%s' "$last" | grep -o 'over [0-9.]*s' | sed 's/over //')
    case "$last" in *final:*) ;; *) batches="${batches:-?}*" ;; esac  # * = still going

    msrow=$(grep -o 'ms/row last[0-9]*=[0-9]* all=[0-9]*' "$f" | tail -1 | grep -o 'all=[0-9]*' | cut -d= -f2)
    score=$(grep -o "'val/success_rate': [0-9.]*" "$f" | tail -1 | grep -o '[0-9.]*$')

    printf '%-26s %-6s %-5s %-6s %-6s %-8s %-9s %-8s %s\n' \
        "$(basename "$log")" "$pump" "${core:-?}" "${steps:-1}" "${depth:-?}" \
        "${batches:-?}" "${wall:-?}" "${msrow:-?}" "${score:-unfinished}"
done

cat <<'NOTE'

  A "*" on BATCHES means the run had not printed [val-pipeline] final: -- it was
  killed or is still going, so its WALL covers only those batches.
  STEPS is num_scheduler_steps: 1 is ordinary scheduling, >1 is V0 multi-step.
  Pick two rows differing in ONE column, then: judge_eval.sh <control> <candidate>
NOTE
