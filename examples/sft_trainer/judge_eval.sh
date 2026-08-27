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
# so the patterns below can anchor on the instrument's own tag.
_clean() { sed 's/^([^)]*) //' "$1"; }

rule() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

# --- 1. did the path under test actually engage? ---------------------------- #
# The failure this catches: the pool refuses, the run falls back to the blocking
# path, and everything downstream measures the control twice. That happened for
# a whole run once, because eos_token_id was a list.
engaged() {
    local log="$1" label="$2"
    printf '%-28s ' "$label"
    if _clean "$log" | grep -q 'rollout-pump.*driving .* ranks as a pool'; then
        echo "PUMP ON  -- $(_clean "$log" | grep -m1 -o 'driving [0-9]* ranks as a pool')"
    elif _clean "$log" | grep -q 'rollout-pump.*staying on the blocking path\|rollout-pump.*handshake failed'; then
        echo "PUMP REFUSED (this log measures the blocking path, not the pump):"
        _clean "$log" | grep -m1 'rollout-pump.*\(staying on the blocking path\|handshake failed\)' | sed 's/^/    /'
    elif _clean "$log" | grep -q 'rollout-pump.*the pool was never used'; then
        echo "PUMP NEVER USED -- every generate took the blocking path"
    else
        echo "pump off (no [rollout-pump] line)"
    fi
    printf '%-28s ' "  slots"
    _clean "$log" | grep -m1 -o 'VAL_PIPELINE_DEPTH=[0-9]*: [0-9]* slot(s)' || echo "NOT FOUND"
}

# --- 2. the number the change was aimed at ---------------------------------- #
# EMPTY is every card idle and falls as p^depth. PARTIAL is a rank idle inside a
# collective call and only the pump reaches it. They are printed together
# precisely so nobody optimises one and reports the other.
residency() {
    local log="$1"
    local line
    line=$(_clean "$log" | grep 'gpu-residency.*EMPTY' | tail -1)
    if [ -z "$line" ]; then
        echo "  NOT FOUND -- GPU_PROFILER=0, or the run has not reached its first report"
        return
    fi
    _clean "$log" | grep 'gpu-residency.*sampled' | tail -1 | sed 's/^/  /'
    echo "$line" | sed 's/^/  /'
}

# --- 3. the scores, which must NOT move ------------------------------------- #
scores() {
    local log="$1"
    local found
    found=$(_clean "$log" | grep -oE "'val/[^']*(test_score|success_rate)': [0-9.]+" | tail -40)
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
    _clean "$log" | grep -c 'val-hash' | sed 's/^/  batches hashed: /'
}

rule "1. did the path under test engage?"
engaged "$CONTROL" "control:"
[ -n "$CANDIDATE" ] && engaged "$CANDIDATE" "candidate:"

rule "2. GPU residency -- EMPTY (depth) and PARTIAL (pump)"
echo "control:"; residency "$CONTROL"
if [ -n "$CANDIDATE" ]; then echo "candidate:"; residency "$CANDIDATE"; fi
cat <<'NOTE'
  PASS for the pump: PARTIAL falls toward 0 and EMPTY stays put.
  PASS for depth:    EMPTY falls, PARTIAL does not move.
  Read the in-run line, not a wandb chart: wandb samples every 15 s and this
  every 0.3 s, and the 15 s grid inflates PARTIAL by aliasing a sub-second signal.
NOTE

rule "3. scores -- these must NOT move"
echo "control:"; scores "$CONTROL"
if [ -n "$CANDIDATE" ]; then echo "candidate:"; scores "$CANDIDATE"; fi

if [ -n "$CANDIDATE" ]; then
    rule "4. per-batch generation digests"
    echo "control:";   hashes "$CONTROL"
    echo "candidate:"; hashes "$CANDIDATE"
    differing=$(diff \
        <(_clean "$CONTROL"   | grep -o 'val-hash.*' | sort) \
        <(_clean "$CANDIDATE" | grep -o 'val-hash.*' | sort) \
        | grep -c '^<' || true)
    echo "  batches whose responses differ: $differing"
    cat <<'NOTE'
  A non-zero count is not by itself a rejection -- it is the question the scores
  answer. It says the change reached the tokens, so "the scores did not move"
  has to come from the scores rather than from an assumption that nothing could
  have moved them.
NOTE
fi

rule "5. wall"
for f in "$CONTROL" ${CANDIDATE:+"$CANDIDATE"}; do
    printf '%-28s ' "$(basename "$f"):"
    _clean "$f" | grep -o 'val-pipeline\] final:.*over [0-9.]*s' | tail -1 \
        || _clean "$f" | grep -o 'val-pipeline\] after [0-9]*:.*over [0-9.]*s' | tail -1 \
        || echo "NOT FOUND"
done
