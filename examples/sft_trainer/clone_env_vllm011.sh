#!/usr/bin/env bash
# Clone the working conda environment and put vLLM 0.11.0 in the copy.
#
#   bash examples/sft_trainer/clone_env_vllm011.sh                 # clone $CONDA_DEFAULT_ENV
#   bash examples/sft_trainer/clone_env_vllm011.sh sdar-multitask sdar-vllm011
#   DRY_RUN=1 bash examples/sft_trainer/clone_env_vllm011.sh       # print, do nothing
#
# WHY A CLONE AND NOT AN UPGRADE IN PLACE. vLLM 0.11.0 needs torch 2.8, and the
# environment running today has vLLM 0.8.5, which needs torch 2.6. pip will
# upgrade torch underneath everything compiled against 2.6 -- flash-attn,
# flashinfer, xformers, apex, any custom CUDA extension. Those do not fail at
# install time, they fail at import or, worse, at the first kernel launch. A
# clone means the working environment is still there when that happens.
#
# WHAT environment.yml IS NOT. It pins vllm==0.11.0 and torch==2.8.0, but it
# describes an environment named `verl-agent` against internal channel mirrors
# (sankuai.com), not the one in use here. It is a vendored artifact, not a
# record of this machine's environment that happened to drift. Do not "restore"
# from it.
#
# WHY BOTHER. async_scheduling -- vLLM V1 overlapping its scheduler with the
# forward pass -- does not exist before 0.10.2, and it is the only lever left
# for the last ~9 points of GPU utilisation, the part where all three cards are
# busy and still read 91%. See docs/eval_gpu_util_status.md.
#
# WHAT IT COSTS. 0.8.5 -> 0.11.0 changes kernels and reduction order, so every
# score measured before it becomes non-comparable. Run it as one experiment
# against one re-measured baseline, not mixed with anything else.
set -uo pipefail

SRC="${1:-${CONDA_DEFAULT_ENV:-}}"
DST="${2:-${SRC}-vllm011}"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"

[ -n "$SRC" ] || { echo "no source env: pass one, or activate the env to clone" >&2; exit 1; }
command -v conda >/dev/null || { echo "conda not on PATH" >&2; exit 1; }

_run() { conda run --no-capture-output -n "$1" "${@:2}"; }

echo "[env] source      : $SRC"
echo "[env] destination : $DST"
echo "[env] vllm        : $VLLM_VERSION"

if conda env list | awk '{print $1}' | grep -qx "$DST"; then
    echo "[env] $DST already exists. Remove it first, or pass another name:" >&2
    echo "        conda env remove -n $DST" >&2
    exit 1
fi

# A clone hardlinks conda packages but COPIES pip ones, and a torch+vllm
# site-packages is tens of gigabytes. Running out halfway leaves a broken clone.
AVAIL_KB=$(df -Pk "$(conda info --base)" | awk 'NR==2 {print $4}')
echo "[env] free on the conda volume: $((AVAIL_KB/1024/1024)) GiB (want >= 40)"
if [ "${AVAIL_KB:-0}" -lt 41943040 ]; then
    echo "[env] WARNING: under 40 GiB free. A half-written clone is worse than none." >&2
fi

echo
echo "[env] BEFORE, in $SRC:"
_run "$SRC" python - <<'PY' || true
import importlib
for name in ("torch", "vllm", "transformers", "flash_attn", "flashinfer", "xformers"):
    try:
        m = importlib.import_module(name)
        print(f"  {name:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  {name:14s} -- {type(e).__name__}: {e}")
PY

if [ "${DRY_RUN:-0}" != "0" ]; then
    echo
    echo "[env] DRY_RUN=1, stopping. Would run:"
    echo "        conda create -y -n $DST --clone $SRC"
    echo "        conda run -n $DST pip install 'vllm==$VLLM_VERSION'"
    exit 0
fi

echo
echo "[env] cloning (minutes, not seconds) ..."
conda create -y -n "$DST" --clone "$SRC" || { echo "[env] clone failed; $SRC is untouched" >&2; exit 1; }

# Recorded before pip is allowed to resolve, so what it changed is readable
# afterwards rather than inferred.
FREEZE_BEFORE=$(mktemp); FREEZE_AFTER=$(mktemp)
_run "$DST" python -m pip freeze > "$FREEZE_BEFORE" 2>/dev/null

echo
echo "[env] installing vllm==$VLLM_VERSION into $DST ..."
if ! _run "$DST" python -m pip install "vllm==$VLLM_VERSION"; then
    echo "[env] pip failed. $SRC is untouched; remove the clone with:" >&2
    echo "        conda env remove -n $DST" >&2
    exit 1
fi
_run "$DST" python -m pip freeze > "$FREEZE_AFTER" 2>/dev/null

echo
echo "[env] what pip changed:"
diff <(sort "$FREEZE_BEFORE") <(sort "$FREEZE_AFTER") | grep -E '^[<>]' | sed 's/^/    /' | head -40
rm -f "$FREEZE_BEFORE" "$FREEZE_AFTER"

echo
echo "[env] AFTER, in $DST -- an import error here is the point of the clone:"
_run "$DST" python - <<'PY'
import importlib, inspect, sys

ok = True
for name in ("torch", "vllm", "transformers", "flash_attn", "flashinfer", "xformers"):
    try:
        m = importlib.import_module(name)
        print(f"  {name:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        # Not fatal on its own: flashinfer and xformers are optional for this
        # path. torch and vllm are not.
        print(f"  {name:14s} -- {type(e).__name__}: {e}")
        if name in ("torch", "vllm"):
            ok = False

try:
    from vllm import LLM
    params = set(inspect.signature(LLM.__init__).parameters)
    have = "async_scheduling" in params
    print(f"  async_scheduling  {'AVAILABLE on LLM()' if have else 'ABSENT -- this build cannot do it'}")
    if not have:
        ok = False
except Exception as e:
    print(f"  async_scheduling  -- could not check: {type(e).__name__}: {e}")
    ok = False

sys.exit(0 if ok else 2)
PY
STATUS=$?

echo
if [ "$STATUS" -ne 0 ]; then
    echo "[env] The clone is NOT usable as it stands (see above)."
    echo "      $SRC is untouched. Throw the clone away with:"
    echo "        conda env remove -n $DST"
    exit "$STATUS"
fi

cat <<NEXT
[env] $DST looks importable. Before spending an evaluation on it:

    conda activate $DST
    python -m pytest tests/utils tests/ray_cpu -q      # the CPU suite still passes
    ray stop --force
    bash examples/sft_trainer/eval_checkpoints.sh <step> 2>&1 | tee /tmp/eval_v011.log

Then read, in this order:

    grep -m1 'rollout-engine' /tmp/eval_v011.log      # version and engine core in force
    grep 'EMPTY is'          /tmp/eval_v011.log       # what the remaining idle is

Turn async_scheduling on only in the NEXT run after that one, so the version
change and the flag are not measured together:

    bash examples/sft_trainer/eval_checkpoints.sh <step> -- \\
      +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=true

And re-measure the baseline in this environment before comparing any score to
one from the 0.8.5 environment. Different kernels, different reduction order.
NEXT
