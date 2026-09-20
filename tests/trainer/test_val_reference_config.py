"""A val-only run scores with the recipe the comparison table was built on.

Every arm of that table was scored at VAL_PIPELINE_DEPTH=3 with
gpu_memory_utilization=0.6 (logs/val2026/*.log), and both knobs change how
batches are formed, so a number taken with other values is not a row in it. The
val-only branch therefore sets them itself.

The branch is shell, so it is RUN here rather than read: the block is extracted
and executed with the two variables it depends on, and the test asserts what it
exported and appended. Reading the source for a literal would pass on a script
that assembles the flags and never reaches them.
"""

import glob
import os
import re
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = sorted(
    p for p in glob.glob(os.path.join(REPO, "examples", "**", "*.sh"), recursive=True)
    if "VAL_ONLY_ARGS=(" in open(p, errors="ignore").read()
)


def _block(path):
    """The `if [ "${VAL_ONLY:-0}" = "1" ]; then ... fi` branch, on its own."""
    src = open(path, errors="ignore").read()
    start = src.index('if [ "${VAL_ONLY:-0}" = "1" ]; then')
    end = src.index("\nfi\n", start) + len("\nfi\n")
    return "VAL_ONLY_ARGS=()\n" + src[start:end]


def _run(path, env):
    """Execute the branch and report what it left behind."""
    script = _block(path) + (
        '\nprintf "DEPTH=%s\\n" "${VAL_PIPELINE_DEPTH:-unset}"\n'
        'printf "V1=%s\\n" "${VLLM_USE_V1:-unset}"\n'
        'printf "ASYNC=%s\\n" "${ROLLOUT_ASYNC_GENERATE:-unset}"\n'
        'printf "ARG=%s\\n" "${VAL_ONLY_ARGS[@]}"\n'
    )
    e = dict(os.environ, VAL_ONLY="1")
    e.update(env)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e)
    return out


def test_every_val_only_branch_exists_to_be_tested():
    assert len(SCRIPTS) >= 10, f"only found {len(SCRIPTS)} scripts with a val-only branch"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: os.path.basename(p))
def test_val_only_takes_the_reference_scoring_recipe(path, tmp_path):
    # VAL_CKPT must look like a real checkpoint: the branch refuses one without actor/.
    ck = tmp_path / "global_step_150"
    (ck / "actor").mkdir(parents=True)
    out = _run(path, {"VAL_CKPT": str(ck), "VAL_PIPELINE_DEPTH": "1"})
    assert out.returncode == 0, out.stderr[-600:]
    assert "DEPTH=3" in out.stdout, f"depth not forced: {out.stdout!r}"
    assert "ARG=actor_rollout_ref.rollout.gpu_memory_utilization=0.6" in out.stdout, out.stdout
    assert "ARG=trainer.val_only=True" in out.stdout
    # the sweep's speculative decoding, added or overridden whatever the caller did
    spec = "actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config"
    for key, val in (("method", "ngram"), ("num_speculative_tokens", "4"),
                     ("prompt_lookup_min", "2"), ("prompt_lookup_max", "5"),
                     ("acceptance_method", "rejection_sampler")):
        assert f"ARG=++{spec}.{key}={val}" in out.stdout, (key, out.stdout)
    # spec decode and sleep() both need the V1 engine; 0.8.5 hosts default to V0
    assert "V1=1" in out.stdout, out.stdout
    # and the score has to reproduce
    assert "ASYNC=0" in out.stdout, out.stdout


@pytest.mark.parametrize("path", SCRIPTS[:1], ids=lambda p: os.path.basename(p))
def test_the_opt_out_leaves_both_to_the_caller(path, tmp_path):
    ck = tmp_path / "global_step_150"
    (ck / "actor").mkdir(parents=True)
    out = _run(path, {"VAL_CKPT": str(ck), "VAL_PIPELINE_DEPTH": "1", "VAL_REFERENCE_CONFIG": "0"})
    assert out.returncode == 0, out.stderr[-600:]
    assert "DEPTH=1" in out.stdout
    # the opt-out says so in its own line, so look at the ARGS, not the whole output
    args = [l for l in out.stdout.splitlines() if l.startswith("ARG=")]
    assert not any("gpu_memory_utilization" in a for a in args), args
    assert not any("speculative_config" in a for a in args), args
    # the opt-out is about the recipe, not about reproducibility
    assert "ASYNC=0" in out.stdout, out.stdout


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: os.path.basename(p))
def test_the_recipe_beats_a_trailing_override(path):
    """VAL_ONLY_ARGS must come AFTER "$@", or a launcher's own value would win."""
    src = open(path, errors="ignore").read()
    line = next(l for l in src.splitlines() if '"${VAL_ONLY_ARGS[@]}"' in l and '"$@"' in l)
    assert line.index('"$@"') < line.index('"${VAL_ONLY_ARGS[@]}"'), line


def test_a_training_run_is_untouched(tmp_path):
    """VAL_ONLY unset: no branch, no args, and the caller's depth stands."""
    path = SCRIPTS[0]
    script = _block(path) + '\nprintf "DEPTH=%s\\n" "${VAL_PIPELINE_DEPTH:-unset}"\n' \
                            'printf "N=%s\\n" "${#VAL_ONLY_ARGS[@]}"\n'
    e = dict(os.environ, VAL_PIPELINE_DEPTH="1")
    e.pop("VAL_ONLY", None)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e)
    assert out.returncode == 0, out.stderr[-400:]
    assert "DEPTH=1" in out.stdout and "N=0" in out.stdout
