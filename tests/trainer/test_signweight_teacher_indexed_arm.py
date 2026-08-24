"""The teacher-indexed weighted arm, and the silent no-op it was built on.

Until this arm existed, the sign weighting was wired to student_indexed_topk in
four places, and only one of them announced itself. With the support taken from
the teacher instead, ``fwd_teacher_topk_logprobs`` stayed None, the actor's
weighting block was skipped by its ``is not None`` guard, and the run trained
plain OPD -- after the driver had already spent three extra frozen forwards a
step (about a quarter of the wall clock) building a cache nothing read. Nothing
raised; the only symptom was ``sign_weight/*`` missing from wandb, which is a
thing you notice a day in.

These tests hold the two halves of the fix: the actor must be able to build the
weights from either support, and the frozen models the weights read must keep
their hidden states regardless of who chose that support.
"""

import os

import pytest

pytest.importorskip("torch")
hydra = pytest.importorskip("hydra")
yaml = pytest.importorskip("yaml")

from hydra import compose, initialize_config_dir

from tests.trainer.test_run_script_overrides_compose import _overrides

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "verl", "trainer", "config")

STUDENT_ARM = "examples/opd_trainer/run_multitask_signweight_target_qwen3.sh"
TEACHER_ARM = "examples/opd_trainer/run_multitask_signweight_target_teachertopk_qwen3.sh"
CONTROL = "examples/opd_trainer/run_multitask_qwen3.sh"

# What the teacher-indexed arm is allowed to change relative to the 1.25/0.75
# student-indexed target arm. Anything else appearing here is an unintended
# difference between two runs that will be discussed side by side.
INTENDED = {
    "actor_rollout_ref.actor.student_indexed_topk",
    "algorithm.opd.sign_weight.agree_weight",
    "algorithm.opd.sign_weight.agree_neg_weight",
}
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


def _composed(script, home="/opt/home/tester"):
    os.environ["HOME"] = home
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="ppo_trainer", overrides=_overrides(script))


def _injected(script):
    from verl.trainer.main_opd import inject_opd_config

    cfg = _composed(script)
    inject_opd_config(cfg)
    return cfg


def test_the_arm_differs_from_the_student_indexed_target_only_as_intended():
    a = _flat(_composed(STUDENT_ARM))
    b = _flat(_composed(TEACHER_ARM))
    keys = set(a) | set(b)
    differing = {k for k in keys if a.get(k, object()) != b.get(k, object())} - IDENTITY
    assert differing == INTENDED, sorted(differing)


def test_the_support_really_is_the_teachers():
    cfg = _composed(TEACHER_ARM)
    assert cfg.actor_rollout_ref.actor.student_indexed_topk is False


def test_the_frozen_models_still_keep_their_hidden_states():
    """The bug this arm was built on.

    ppo_trainer.yaml ties ref.student_indexed_topk to the ACTOR's support choice
    by interpolation, but the two are independent: whoever picks the support,
    the base policy and the off-task teachers have to be resolvable at it, and
    that is what the ref-side flag buys. Left interpolated, this arm cached
    nothing and the weighting silently did not run.
    """
    cfg = _injected(TEACHER_ARM)
    assert cfg.actor_rollout_ref.actor.student_indexed_topk is False
    assert cfg.actor_rollout_ref.ref.student_indexed_topk is True
    assert cfg.actor_rollout_ref.ref.response_only_logits is True


def test_the_forcing_does_not_reach_an_arm_without_the_weighting():
    """The control has no sign weighting, so nothing has to be scored at ids
    chosen later, and it must not pay for the caching that would enable it."""
    cfg = _injected(CONTROL)
    assert "sign_weight" not in cfg.algorithm.opd
    assert cfg.actor_rollout_ref.ref.student_indexed_topk == (
        cfg.actor_rollout_ref.actor.student_indexed_topk
    )


def test_the_weighting_reaches_the_actor_with_the_one_sided_table():
    cfg = _injected(TEACHER_ARM)
    sign = cfg.actor_rollout_ref.actor.sign_weight
    assert sign.enable is True
    assert sign.mode == "target"
    assert float(sign.agree_weight) == 1.5
    # 1.0, not a mirror of agree_weight: agreement to lower is recorded and not
    # acted on. A future edit that "restores symmetry" changes the arm.
    assert float(sign.agree_neg_weight) == 1.0
    assert float(sign.disagree_weight) == 1.0


def test_the_arm_passes_its_own_intent_lock():
    from verl.utils.expected_config import check_expected_config

    cfg = _injected(TEACHER_ARM)
    lock = os.path.join(REPO, cfg.trainer.expected_config)
    assert check_expected_config(cfg, lock) == [], check_expected_config(cfg, lock)


def test_the_lock_pins_the_support_on_both_sides():
    """The actor's choice and the ref's caching are now separate values, so a
    lock that pinned only one of them would let the other move silently -- which
    is precisely the failure mode this arm exposed."""
    cfg = _composed(TEACHER_ARM)
    lock = yaml.safe_load(open(os.path.join(REPO, cfg.trainer.expected_config)))
    assert lock["actor_rollout_ref.actor.student_indexed_topk"] is False
    assert lock["actor_rollout_ref.ref.student_indexed_topk"] is True


def test_the_two_target_locks_differ_only_in_the_four_scientific_values():
    student = yaml.safe_load(
        open(os.path.join(REPO, "examples/opd_trainer/expected_multitask_signweight_target_config.yaml"))
    )
    teacher = yaml.safe_load(
        open(
            os.path.join(
                REPO,
                "examples/opd_trainer/expected_multitask_signweight_target_teachertopk_config.yaml",
            )
        )
    )
    keys = set(student) | set(teacher)
    differing = {k for k in keys if student.get(k, object()) != teacher.get(k, object())}
    assert differing == {
        "trainer.experiment_name",
        "actor_rollout_ref.actor.student_indexed_topk",
        "actor_rollout_ref.ref.student_indexed_topk",
        "algorithm.opd.sign_weight.agree_weight",
        "algorithm.opd.sign_weight.agree_neg_weight",
        "actor_rollout_ref.actor.sign_weight.agree_weight",
        "actor_rollout_ref.actor.sign_weight.agree_neg_weight",
    }, sorted(differing)


def test_the_arm_does_not_share_a_checkpoint_directory():
    """resume_mode is "auto", so an arm pointed at another arm's directory does
    not overwrite it -- it CONTINUES it, and reports the result under its own
    name."""
    mine = _composed(TEACHER_ARM)
    for other in (STUDENT_ARM, CONTROL):
        theirs = _composed(other)
        assert mine.trainer.default_local_dir != theirs.trainer.default_local_dir
        assert mine.trainer.experiment_name != theirs.trainer.experiment_name
        assert mine.trainer.val_instance_log_dir != theirs.trainer.val_instance_log_dir
