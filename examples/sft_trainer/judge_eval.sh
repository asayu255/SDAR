#!/usr/bin/env bash
# Decide whether a change to the evaluation is worth keeping.
#
#   bash examples/sft_trainer/judge_eval.sh <control.log> <candidate.log>
#   bash examples/sft_trainer/judge_eval.sh <one.log>            # just read one
#
# WHY THIS IS A SCRIPT AND NOT A LIST OF GREPS. Every line it looks for is
# printed by a different instrument with its own format, and a grep that matches
# nothing looks exactly like a grep that matches zero occurrences -- which is how
# this arm concluded twice that a change "did nothing". Each check below says
# NOT FOUND when the line is absent, which is a different answer from 0.
#
# WHAT NEEDS A FINISHED RUN, AND WHAT DOES NOT. Only the scores do, and only
# when the change reached the tokens. Everything else is readable within a few
# minutes:
#
#   did the pump engage?   first seconds     ([rollout-pump])
#   did PARTIAL fall?      first report      ([gpu-residency]; set
#                                             VAL_PIPELINE_REPORT_EVERY=5 to
#                                             get one in a couple of minutes)
#   did generation change? as batches land   ([val-hash], compared by index)
#   the scores             only at the end
#
# So: if PARTIAL did not fall, stop the run -- the scores are moot. If it fell
# AND the digests match the control's, generation is unchanged, the scores
# cannot move, and no finished run is needed either. The VERDICT section at the
# bottom says which of those three you are in.
#
# A wandb chart shows the spikes but cannot answer any of this. It samples every
# 15 s against this instrument's 0.3 s, which aliases a sub-second signal and
# inflated PARTIAL from 8.0% to 14.6% on the same run; it does not separate a
# node that is idle from one card working while two wait; and it never sees the
# tokens.
#
# THE TWO-PART RULE. A change to the rollout is kept only if BOTH hold:
#
#   1. the number it was aimed at moves, and
#   2. the scores do not.
#
# Speed that moves the scores is not speed, it is a different experiment. The
# pump and the generate merge both reshape the batch, and a reshaped batch
# changes reduction order, which changes tokens (measured: 28 of 30 batches
# differed, ~0.7% of rows). So utilisation alone cannot decide either of them.
set -uo pipefail

CONTROL="${1:-}"
CANDIDATE="${2:-}"
[ -n "$CONTROL" ] || { sed -n '2,4p' "$0"; exit 1; }
for f in "$CONTROL" ${CANDIDATE:+"$CANDIDATE"}; do
    [ -r "$f" ] || { echo "cannot read $f" >&2; exit 1; }
done

# Ray prefixes worker output with "(SFTMultiTaskTaskRunner pid=NNN) ". Strip it
# so the patterns below can anchor on the instrument's own tag -- ONCE, into a
# file, which every check below then greps directly.
#
# NOT `sed ... | grep -q`, which is what this used to be. Under `set -o
# pipefail` an early-exiting grep (-q, -m1) closes the pipe the moment it
# matches, sed dies of SIGPIPE, and the pipeline reports 141 -- so a pattern
# that MATCHED reads as absent. It only misfires when the match is early and
# the file is long, which is every real log and no test log: this script
# reported "pump off (no [rollout-pump] line)" for two runs that both printed
# that line in their first seconds, and "TOO EARLY. No residency report yet."
# directly under the residency report it had just printed. Checks that read to
# EOF (grep | tail, grep -c) were unaffected, which is why half the output was
# right and half was inverted.
_NORMDIR=$(mktemp -d)
trap 'rm -rf "$_NORMDIR"' EXIT
_clean() {
    local out="$_NORMDIR/$(printf '%s' "$1" | md5sum | cut -c1-16)"
    [ -s "$out" ] || sed 's/^([^)]*) //' "$1" > "$out"
    printf '%s' "$out"
}

rule() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

# --- 0. is there a run in this file at all? --------------------------------- #
# Every check below reports an ABSENCE as a finding -- "pump off", "NOT FOUND",
# "batches hashed: 0" -- and those readings are only meaningful about a run that
# got far enough to produce the line. A run killed during startup produces the
# same absences as a run that measured something and disagreed, and the report
# read as the second: "pump off", "NOT FOUND", "0 batches", one line each, with
# nothing saying the process had died forty minutes earlier.
_reached() {
    grep -qE '\[rollout-engine\]|\[rollout-pump\]|\[val-pipeline\]|\[val-batching\]|val-hash|\[gpu-residency\]' "$1"
}

preflight() {
    local log="$1" label="$2"
    local f; f=$(_clean "$log")
    printf '%-28s ' "$label"
    # The intended configuration is on line two, before Ray, so it survives a
    # run that died in startup -- which is exactly when you need it.
    local cfg; cfg=$(grep -m1 -o 'search=[^-]*' "$f" | sed 's/ *$//')
    if _reached "$f"; then
        echo "reached validation.  ${cfg:-(no [eval] config line)}"
    else
        echo "DID NOT REACH VALIDATION.  ${cfg:-(no [eval] config line)}"
        printf '%-28s %s, last written %s\n' "  file" \
            "$(wc -c < "$log" | tr -d ' ') bytes" \
            "$(date -r "$log" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo '?')"
        # The LAST line, `set -x` traces included -- a `+ python3 -c ...` trace
        # is the command that was running when it died, which is the one fact
        # worth having. Filtering the traces out reports the last thing that
        # succeeded instead of the thing that did not.
        printf '%-28s %s\n' "  stopped at" "$(tail -1 "$log" | cut -c1-90)"
        printf '%-28s %s\n' "  " \
            "no traceback means it was killed, not raised -- try: dmesg -T | grep -i 'killed process'"
    fi
}

# --- 1. did the path under test actually engage? ---------------------------- #
# The failure this catches: the pool refuses, the run falls back to the blocking
# path, and everything downstream measures the control twice. That happened for
# a whole run once, because eos_token_id was a list.
engaged() {
    local log="$1" label="$2"
    printf '%-28s ' "$label"
    local f; f=$(_clean "$log")
    if grep -q 'rollout-pump.*driving .* ranks as a pool' "$f"; then
        echo "PUMP ON  -- $(grep -m1 -o 'driving [0-9]* ranks as a pool' "$f")"
    elif grep -q 'rollout-pump.*staying on the blocking path\|rollout-pump.*handshake failed' "$f"; then
        echo "PUMP REFUSED (this log measures the blocking path, not the pump):"
        grep -m1 'rollout-pump.*\(staying on the blocking path\|handshake failed\)' "$f" | sed 's/^/    /'
    elif grep -q 'rollout-pump.*the pool was never used' "$f"; then
        echo "PUMP NEVER USED -- every generate took the blocking path"
    elif ! _reached "$f"; then
        # NOT "pump off". A log with no [rollout-pump] line in it because the
        # run never got as far as the engine is not a log of a run with the
        # pump off, and this printed the second for the first: a run killed in
        # startup was reported as "pump off (no [rollout-pump] line)", which
        # reads as a measured configuration. Same error as naming a slot state
        # from a trace with no slot column.
        echo "DID NOT REACH VALIDATION -- nothing here is a measurement"
    else
        echo "pump off (no [rollout-pump] line)"
    fi
    printf '%-28s ' "  slots"
    grep -m1 -o 'VAL_PIPELINE_DEPTH=[0-9]*: [0-9]* slot(s)' "$f" || echo "NOT FOUND"
}

# --- 2. the number the change was aimed at ---------------------------------- #
# EMPTY is every card idle and falls as p^depth. PARTIAL is a rank idle inside a
# collective call and only the pump reaches it. They are printed together
# precisely so nobody optimises one and reports the other.
residency() {
    local log="$1"
    local line
    local f; f=$(_clean "$log")
    if ! grep -q 'gpu-residency' "$f"; then
        echo "  NOT FOUND -- GPU_PROFILER=0, or the run has not reached its first report"
        return
    fi
    grep 'gpu-residency.*sampled' "$f" | tail -1 | sed 's/^/  /'
    # The duty-cycle line, which is the number the PASS criteria below are
    # written around. It was being dropped: 'gpu-residency.*EMPTY' matches this
    # line AND the "-> EMPTY is ..." verdict printed after it, and `tail -1`
    # kept the verdict -- so every reading of an engine setting was made
    # without the one number an engine setting can move.
    grep 'gpu-residency\] EMPTY' "$f" | tail -1 | sed 's/^/  /'
    grep 'gpu-residency\] ->' "$f" | tail -1 | sed 's/^/  /'
}

# --- 3. the scores, which must NOT move ------------------------------------- #
scores() {
    local log="$1"
    local found
    # THE SCORES COME FROM scripts/val_scores.py; see the note there. pprint of
    # an f-string wraps the dict at 80 columns mid-pair, the break moves with
    # the contents, and a log can hold more than one block -- three shell
    # extractions got it wrong in three different ways.
    found=$(python3 "$(dirname "$0")/../../scripts/val_scores.py" "$log" 2>/dev/null \
        | awk '/^  val\// {print $1 ": " $2}' | tail -40)
    if [ -z "$found" ]; then
        echo "  NOT FOUND -- the run has not finished validating"
        return
    fi
    echo "$found" | sed "s/'//g" | sort -u | sed 's/^/  /'
}

# --- 4. row-level generation, which is what "the scores did not move" rests on #
# Scores are means over thousands of rows and hide a change of a few rows.
# [val-hash] is a sha1 of each batch's response ids: same tokens, same digest.
hashes() {
    local log="$1"
    grep -c 'val-hash' "$(_clean "$log")" | sed 's/^/  batches hashed: /'
}

rule "0. is there a run in these files?"
preflight "$CONTROL" "control:"
[ -n "$CANDIDATE" ] && preflight "$CANDIDATE" "candidate:"

rule "1. did the path under test engage?"
engaged "$CONTROL" "control:"
[ -n "$CANDIDATE" ] && engaged "$CANDIDATE" "candidate:"

rule "2. GPU residency -- EMPTY (depth) and PARTIAL (pump)"
echo "control:"; residency "$CONTROL"
if [ -n "$CANDIDATE" ]; then echo "candidate:"; residency "$CANDIDATE"; fi
cat <<'NOTE'
  PASS for an ENGINE setting (multi-step, async scheduling): the duty cycle --
                     "All 3 cards had work X% of the time, reading Y%" -- rises.
                     Y is the only number those settings can move. EMPTY and
                     PARTIAL are not theirs, and node util is dominated by them,
                     so node util alone will read as "no effect" either way.
  PASS for the pump: PARTIAL falls toward 0 and EMPTY stays put.
  PASS for depth:    EMPTY falls, PARTIAL does not move.
  Read the in-run line, not a wandb chart: wandb samples every 15 s and this
  every 0.3 s, and the 15 s grid inflates PARTIAL by aliasing a sub-second signal.
NOTE

rule "3. scores -- these must NOT move"
echo "control:"; scores "$CONTROL"
if [ -n "$CANDIDATE" ]; then echo "candidate:"; scores "$CANDIDATE"; fi

DIFFERING=""
COMPARED=0
if [ -n "$CANDIDATE" ]; then
    rule "4. per-batch generation digests"
    echo "control:";   hashes "$CONTROL"
    echo "candidate:"; hashes "$CANDIDATE"
    # Joined on batch index, not diffed as sorted sets, so a run that was killed
    # early still answers: the overlap of the two logs is what gets compared.
    # This is the check that does NOT need a finished run.
    _digests() { sed -n 's/.*val-hash\] batch#\([0-9]*\) .*sha1 \([0-9a-f]*\).*/\1 \2/p' "$(_clean "$1")" | sort -n -u; }
    read -r COMPARED DIFFERING <<<"$(join <(_digests "$CONTROL") <(_digests "$CANDIDATE") \
        | awk '{n++; if ($2 != $3) d++} END {print n+0, d+0}')"
    echo "  batches present in BOTH logs: $COMPARED"
    echo "  of those, responses differ:   $DIFFERING"
    cat <<'NOTE'
  This is the early answer. If the overlap is non-trivial and nothing differs,
  the change did not reach the tokens and the scores cannot move -- no finished
  run is needed. If batches differ, only the scores can say whether it matters,
  and those print once, at the end.
NOTE
fi

# --- 5. the only speed number that survives a different batch mix ----------- #
# NOT the turn table's TOTAL row. That is per batch, and a batch is alfworld
# (126 rows to 50 turns), search (252 rows to 4) or webshop (126 rows to 15), so
# `tail -3` on two logs compares three different batches -- seen in practice at
# 100k against 632k prompt tokens, a 6x spread that swamps any real difference.
# ms/row divides it out, which is why the WALL line carries it.
rule "5. speed -- ms/row, normalised across the batch mix"
_msrow() { grep -o 'ms/row last[0-9]*=[0-9]* all=[0-9]*' "$(_clean "$1")"; }
for f in "$CONTROL" ${CANDIDATE:+"$CANDIDATE"}; do
    printf '%-24s ' "$(basename "$f"):"
    line=$(_msrow "$f" | tail -1)
    if [ -n "$line" ]; then
        echo "$line   (over $(_msrow "$f" | wc -l | tr -d ' ') batches)"
    else
        echo "NOT FOUND -- no [rollout-turn-timing] WALL line yet"
    fi
done
# The two lines above are each run's own last reading, and a run that is 57
# batches further along has eaten a different slice of the task mix -- which is
# the trap this whole section exists to avoid, left in place one line further
# down. `all=` is cumulative, so the Kth reading IS the run's ms/row over its
# first K batches: take K = the shorter run's length and the two numbers cover
# the same batches. This is the only pair here that may be subtracted.
if [ -n "$CANDIDATE" ]; then
    _k_ctl=$(_msrow "$CONTROL" | wc -l); _k_can=$(_msrow "$CANDIDATE" | wc -l)
    K=$(( _k_ctl < _k_can ? _k_ctl : _k_can ))
    if [ "$K" -gt 0 ]; then
        A=$(_msrow "$CONTROL"   | sed -n "${K}p" | grep -o 'all=[0-9]*' | cut -d= -f2)
        B=$(_msrow "$CANDIDATE" | sed -n "${K}p" | grep -o 'all=[0-9]*' | cut -d= -f2)
        printf '  same-prefix (first %s batches of each): control all=%s  candidate all=%s' "$K" "$A" "$B"
        [ -n "$A" ] && [ -n "$B" ] && [ "$A" -gt 0 ] \
            && printf '   -> %+.1f%%' "$(awk -v a="$A" -v b="$B" 'BEGIN{print 100*(b-a)/a}')"
        echo
    fi
fi
cat <<'NOTE'
  Read "all=", not "last=": the last-20 window moves with whichever tasks
  happened to be in it. Compare on the same-prefix line, not on the two
  per-run lines: the mix is not uniform in time, so two runs at different
  batch counts are two different experiments.
NOTE

rule "6. wall"
for f in "$CONTROL" ${CANDIDATE:+"$CANDIDATE"}; do
    printf '%-24s ' "$(basename "$f"):"
    _f=$(_clean "$f")
    grep -o 'val-pipeline\] final:.*over [0-9.]*s' "$_f" | tail -1 \
        || grep -o 'val-pipeline\] after [0-9]*:.*over [0-9.]*s' "$_f" | tail -1 \
        || echo "NOT FOUND"
done
# Same correction as section 5. Each [val-pipeline] report carries the wall so
# far, so the two runs can be read at a batch count they both reached.
if [ -n "$CANDIDATE" ]; then
    _wall_at() { sed -n 's/.*val-pipeline\] \(final\|after [0-9]*\): \([0-9]*\) batches over \([0-9.]*\)s.*/\2 \3/p' "$(_clean "$1")"; }
    N=$(join <(_wall_at "$CONTROL" | sort -k1,1) <(_wall_at "$CANDIDATE" | sort -k1,1) \
        | sort -n -k1,1 | tail -1)
    if [ -n "$N" ]; then
        set -- $N
        printf '  same-prefix (%s batches each): control %ss  candidate %ss' "$1" "$2" "$3"
        awk -v a="$2" -v b="$3" 'BEGIN{if (a>0) printf "   -> %+.1f%%", 100*(b-a)/a}'
        echo
    else
        echo "  same-prefix: no batch count reported by BOTH runs yet"
        echo "  (VAL_PIPELINE_REPORT_EVERY differed between them -- the reports"
        echo "   have to land on a common multiple to be paired.)"
    fi
fi

# --- what can be decided right now ------------------------------------------ #
rule "VERDICT"
_have() { grep -q "$2" "$(_clean "$1")"; }

if [ -z "$CANDIDATE" ]; then
    echo "  One log given -- this is a reading, not a comparison. Pass a control"
    echo "  log as the first argument to judge anything."
elif ! _have "$CANDIDATE" 'gpu-residency.*EMPTY'; then
    echo "  TOO EARLY. No residency report yet. It prints every"
    echo "  VAL_PIPELINE_REPORT_EVERY batches (default 25); set it to 5 for a"
    echo "  first read within a couple of minutes."
elif [ "${COMPARED:-0}" -gt 0 ] && [ "${DIFFERING:-0}" -eq 0 ]; then
    echo "  DECIDABLE NOW, and no finished run is needed:"
    echo "    $COMPARED batches overlap and every digest matches, so generation"
    echo "    is unchanged and the scores cannot move. Judge on section 2 alone."
elif [ "${DIFFERING:-0}" -gt 0 ]; then
    echo "  NEEDS A FINISHED RUN, but only for the scores:"
    echo "    $DIFFERING of $COMPARED overlapping batches generated different"
    echo "    tokens. Section 2 already tells you whether it was worth it; the"
    echo "    scores tell you whether you are allowed to keep it."
    echo "    If section 2 shows no gain, stop the run now -- the scores are moot."
else
    echo "  Residency is readable (section 2) but no batch overlaps yet, so"
    echo "  nothing can be said about generation. Give it a few more batches."
fi
