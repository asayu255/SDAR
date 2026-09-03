"""Two different arms must not write into one checkpoint tree.

Checkpoints land in ``trainer.default_local_dir/global_step_N/actor``.
``trainer.experiment_name`` is NOT in that path -- it names the wandb run and
nothing on disk. So two invocations that differ in experiment_name but share
default_local_dir do two things, both silent:

* they overwrite each other's checkpoints step for step, and
* ``trainer.resume_mode`` defaults to ``auto``, which reads
  ``default_local_dir/latest_checkpointed_iteration.txt`` -- so the second arm to
  start RESUMES FROM THE FIRST ARM'S WEIGHTS instead of from the base model.

This is what happens when a run script is created by copying another and editing
the parts that are obviously about identity. The student-indexed arm's script was
created that way and had exactly this defect; it was caught by reading the script
before running it, not by anything in the suite.

Text-level and per-invocation on purpose: the mistake is made in the shell script,
a script may launch several python3 commands with different settings, and Hydra is
never reached to complain about any of it.
"""

import pathlib
import re

import pytest

_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples/opd_trainer"
_SCRIPTS = sorted(_DIR.glob("run_*.sh"))

# Directories the operator has decided to share, listed exhaustively so the check
# stays live for every script that is NOT on this list.
#
# All three off-policy scripts train into one directory under three experiment
# names. Two of them (the Stage 1 + Stage 2 script and the Stage 2 -only one)
# predate this check. The student-indexed arm was given its own tree when this
# hazard was found, and then put back here on the operator's instruction --
# whether the control's checkpoints under that path still matter is theirs to
# know, not this file's.
#
# What the exemption does NOT do is make the consequences go away, and the run
# script says so at the assignment: under resume_mode=auto the arm that starts
# second resumes from whatever is already there, and its own saves overwrite the
# other's at the same step number. This list is a record of an accepted cost, not
# a statement that the cost is zero.
_KNOWN_SHARED = {
    "/opt/home/ohara/checkpoints/verl_agent_opd_offpolicy_multitask": {
        "run_multitask_offpolicy_qwen3.sh",
        "run_multitask_offpolicy_qwen3_nogen.sh",
        "run_multitask_offpolicy_studenttopk_qwen3.sh",
    }
}


def _invocations(text):
    """Each ``python3 -m ...`` command in the script, as its own chunk.

    A script may launch several -- Stage 1 once per task and then Stage 2 -- with
    different settings, so reading the file as one blob and taking the last match
    attributes one invocation's directory to another's experiment name.
    """
    parts = re.split(r"^\s*python3\s+-m\s", text, flags=re.MULTILINE)
    return parts[1:]


def _setting(chunk, key):
    found = re.findall(rf"^\s*\+?{re.escape(key)}=(\S+?)\s*\\?$", chunk, re.MULTILINE)
    return found[-1] if found else None


def _training_invocations():
    """(script name, checkpoint dir, experiment name) for everything that saves."""
    out = []
    for path in _SCRIPTS:
        for chunk in _invocations(path.read_text()):
            local_dir = _setting(chunk, "trainer.default_local_dir")
            if local_dir is None:
                continue  # generation writes a pool, not checkpoints
            out.append((path.name, local_dir, _setting(chunk, "trainer.experiment_name")))
    return out


def test_there_are_training_invocations_to_check():
    """Guard against the parse silently matching nothing after a refactor."""
    found = _training_invocations()
    assert len(found) >= 3, f"expected several training invocations, parsed {found}"


def test_one_checkpoint_directory_never_holds_two_experiments():
    by_dir = {}
    for script, local_dir, experiment in _training_invocations():
        by_dir.setdefault(local_dir, {}).setdefault(experiment, set()).add(script)

    shared = {
        local_dir: {e: sorted(s) for e, s in names.items()}
        for local_dir, names in by_dir.items()
        if len(names) > 1
    }
    # Drop the one that predates this check, and only if it is still exactly the
    # pair that was documented -- a third script joining it is a new defect.
    for local_dir, expected_scripts in _KNOWN_SHARED.items():
        entry = shared.get(local_dir)
        if entry is None:
            continue
        actual = {s for scripts in entry.values() for s in scripts}
        if actual == expected_scripts:
            del shared[local_dir]

    assert not shared, (
        "these invocations write checkpoints into one directory under different "
        "experiment names, so they overwrite each other and resume from each other "
        f"under resume_mode=auto: {shared}"
    )


def test_a_script_that_shares_a_tree_says_so_where_the_value_is_set():
    """An exemption in this file is invisible to whoever runs the script.

    Sharing a checkpoint directory is allowed -- the operator may know the other
    arm's checkpoints are gone, or may intend to move them -- but it changes what
    happens on start, and the person typing the command is the one who has to
    steer around it. So a script on the shared list has to carry the two
    consequences in its own header: that resume_mode=auto will pick up whatever is
    already there, and that its saves overwrite the other arm's.
    """
    for local_dir, scripts in _KNOWN_SHARED.items():
        for name in scripts:
            path = _DIR / name
            if not path.exists():
                continue
            header = path.read_text()
            assert "resume_mode" in header, (
                f"{name} shares {local_dir} but never mentions resume_mode; whoever "
                "runs it cannot know it may resume from another arm's weights"
            )
            assert "overwrite" in header.lower(), (
                f"{name} shares {local_dir} but does not say its saves overwrite the "
                "other arm's checkpoints at the same step"
            )


@pytest.mark.parametrize(
    "script", [p.name for p in _SCRIPTS if "offpolicy" in p.name], ids=lambda n: n
)
def test_an_off_policy_training_arm_pins_its_own_intent_lock(script):
    """The lock is what makes an arm's knobs checkable, and a copied script that
    kept the original's lock would be checked against the wrong expectations.

    Scoped to the off-policy scripts: run_multitask_qwen3.sh (the on-policy arm)
    pins no expected_config at all, which is a real gap but not one to discover
    through an unrelated test going red.
    """
    for chunk in _invocations((_DIR / script).read_text()):
        if _setting(chunk, "trainer.default_local_dir") is None:
            continue
        assert _setting(chunk, "trainer.expected_config"), (
            f"{script} has a training invocation that pins no expected_config"
        )
