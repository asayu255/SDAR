"""The weighted arms must differ from the control in the weighting and nothing else.

A control/treatment pair is only a pair if one thing changed. The composed
config is what the run actually saw, so that is what is compared here rather
than the scripts' text -- a knob moved in ppo_trainer.yaml, or an injection in
main_opd, would not show up in a diff of the arguments.
"""

import os
import re

import pytest

pytest.importorskip("torch")
hydra = pytest.importorskip("hydra")
yaml = pytest.importorskip("yaml")

from hydra import compose, initialize_config_dir

from tests.trainer.test_run_script_overrides_compose import _overrides

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "verl", "trainer", "config")
CONTROL = "examples/opd_trainer/run_multitask_qwen3.sh"
ARMS = [
    ("position", "examples/opd_trainer/run_multitask_signweight_position_qwen3.sh"),
    ("target", "examples/opd_trainer/run_multitask_signweight_target_qwen3.sh"),
]

# Everything that names the run rather than defining it. Sharing any of these
# between two arms is its own bug, not a missing difference.
IDENTITY = {
    "trainer.expected_config",
    "trainer.project_name",
    "trainer.experiment_name",
    "trainer.default_local_dir",
    "trainer.val_instance_log_dir",
}


def _flat(cfg, prefix=""):
    from omegaconf import DictConfig

    out = {}
    for key, value in cfg.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, DictConfig):
            out.update(_flat(value, prefix=f"{dotted}."))
        else:
            out[dotted] = value
    return out


def _effective(script, home="/opt/home/tester", extra=()):
    os.environ["HOME"] = home
    overrides = list(_overrides(script)) + list(extra)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return _flat(compose(config_name="ppo_trainer", overrides=overrides))


def _differing(a, b):
    return {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}


@pytest.mark.parametrize("mode,script", ARMS)
def test_each_arm_differs_from_the_control_only_in_the_weighting(mode, script):
    control = _effective(CONTROL)
    arm = _effective(script)
    differing = _differing(control, arm)

    allowed = set(IDENTITY)
    allowed |= {k for k in differing if k.startswith("algorithm.opd.sign_weight.")}
    allowed |= {k for k in differing if k.startswith("actor_rollout_ref.actor.sign_weight.")}
    # Micro-batch sizes are how a run is fitted onto the GPUs it got: the per-task
    # loss weights multiply by gradient_accumulation, so the objective is
    # invariant to the split and only the reduction order moves. Matching them is
    # still right for a pair, which is why they are equal today -- the exemption
    # exists so a host change does not have to be a science change.
    allowed |= {k for k in differing if "micro_batch_size" in k}

    assert differing <= allowed, sorted(differing - allowed)
    assert any(k.startswith("algorithm.opd.sign_weight.") for k in differing), "the arms are identical"


@pytest.mark.parametrize("mode,script", ARMS)
def test_the_arms_do_not_share_a_checkpoint_directory(mode, script):
    """Not merely allowed to differ -- required to.

    The checkpoint path is default_local_dir/global_step_N and
    trainer.resume_mode defaults to "auto", so a shared directory does not just
    overwrite: the arm started second RESUMES FROM the arm started first and
    reports it under its own name.
    """
    control = _effective(CONTROL)
    arm = _effective(script)
    assert control["trainer.default_local_dir"] != arm["trainer.default_local_dir"]
    assert control["trainer.val_instance_log_dir"] != arm["trainer.val_instance_log_dir"]


def test_the_two_weighted_arms_differ_only_in_the_mode():
    """position and target are one mechanism spent two ways.

    If anything else moved between them, a difference in their results would not
    be attributable to where the weight is applied.
    """
    a = _effective(ARMS[0][1])
    b = _effective(ARMS[1][1])
    differing = _differing(a, b) - IDENTITY
    # Composed, not injected: main_opd copies these onto the actor at startup, so
    # the actor-side mirror does not exist yet at this point. The lock is what
    # checks the copy arrived, since it validates the config AFTER injection.
    assert differing == {"algorithm.opd.sign_weight.mode"}, sorted(differing)


@pytest.mark.parametrize("mode,script", ARMS)
def test_the_lock_pins_every_knob_of_the_mechanism(mode, script):
    """A knob the lock does not pin is one a future edit can move silently."""
    arm = _effective(script)
    lock_path = os.path.join(REPO, arm["trainer.expected_config"])
    lock = yaml.safe_load(open(lock_path))
    for key in (
        "enable",
        "mode",
        "agree_weight",
        "agree_neg_weight",
        "disagree_weight",
        "deadzone",
        "base_path",
    ):
        dotted = f"algorithm.opd.sign_weight.{key}"
        assert dotted in lock, f"{lock_path} does not pin {dotted}"
        assert str(lock[dotted]).lower() == str(arm[dotted]).lower(), dotted
    assert lock["algorithm.opd.sign_weight.mode"] == mode


def test_target_mode_is_locked_to_a_neutral_conflict_weight():
    """A conflict factor in target mode multiplies a probability, so it has to
    say which way to push, and one number cannot: pulling back toward the
    objecting teachers means lowering a token the on-task teacher raised and
    raising one it lowered. The trainer refuses any other value; the lock is what
    keeps a script from trying."""
    arm = _effective(ARMS[1][1])
    assert float(arm["algorithm.opd.sign_weight.disagree_weight"]) == 1.0


def test_the_comparison_would_notice_a_drifting_arm():
    """Negative control: the check above must be able to fail.

    A parser that silently dropped the overrides would compose two copies of the
    bare defaults, find them equal, and report success while checking nothing.
    """
    control = _effective(CONTROL)
    drifted = _effective(ARMS[1][1], extra=["actor_rollout_ref.actor.optim.lr=5e-6"])
    assert "actor_rollout_ref.actor.optim.lr" in _differing(control, drifted)
