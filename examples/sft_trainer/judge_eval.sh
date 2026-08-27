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

DIFFERING=""
COMPARED=0
if [ -n "$CANDIDATE" ]; then
    rule "4. per-batch generation digests"
    echo "control:";   hashes "$CONTROL"
    echo "candidate:"; hashes "$CANDIDATE"
    # Joined on batch index, not diffed as sorted sets, so a run that was killed
    # early still answers: the overlap of the two logs is what gets compared.
    # This is the check that does NOT need a finished run.
    _digests() { _clean "$1" | sed -n 's/.*val-hash\] batch#\([0-9]*\) .*sha1 \([0-9a-f]*\).*/\1 \2/p' | sort -n -u; }
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

rule "5. wall"
for f in "$CONTROL" ${CANDIDATE:+"$CANDIDATE"}; do
    printf '%-28s ' "$(basename "$f"):"
    _clean "$f" | grep -o 'val-pipeline\] final:.*over [0-9.]*s' | tail -1 \
        || _clean "$f" | grep -o 'val-pipeline\] after [0-9]*:.*over [0-9.]*s' | tail -1 \
        || echo "NOT FOUND"
done

# --- what can be decided right now ------------------------------------------ #
rule "VERDICT"
_have() { _clean "$1" | grep -q "$2"; }

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
