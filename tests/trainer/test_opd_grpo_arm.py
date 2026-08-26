"""The OPD+GRPO arm: what it adds to pure OPD, and what it must not change.

The arm is defined by one difference -- a policy gradient in the loss next to the
per-task teacher KL -- so almost every test here is about the things that stayed
the same. Four gates:

* the config injection shares its distillation half with pure OPD and differs
  only on the policy-gradient side;
* each run script satisfies its own intent lock, checked through the same
  ``enforce_expected_config`` the entry point calls;
* each script differs from its pure-OPD control only in the objective and the
  run's identity;
* the trainer inherits teacher routing and the whole loop, overriding exactly the
  two hooks that carry the objective.

Plus the actor-side arithmetic that makes the joint loss well-posed: with
per-task normalisation on, the policy-gradient term has to be aggregated by the
same row weights as the teacher KL, or ``pg_loss_coef`` stops being the ratio
between them.
"""

import functools
import os
import re

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
hydra = pytest.importorskip("hydra")
yaml = pytest.importorskip("yaml")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from tests.trainer.test_run_script_overrides_compose import _overrides

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "verl", "trainer", "config")

# Each OPD+GRPO arm and the pure-OPD arm it is the counterpart of.
PAIRS = [
    (
        "plain",
        "examples/opd_grpo_trainer/run_multitask_qwen3.sh",
        "examples/opd_trainer/run_multitask_qwen3.sh",
    ),
    (
        "signweight_target",
        "examples/opd_grpo_trainer/run_multitask_signweight_target_qwen3.sh",
        "examples/opd_trainer/run_multitask_signweight_target_qwen3.sh",
    ),
    # The teacher-indexed 1.5/0.5 arm, i.e. the one
    # docs/multitask_signweight_teachertopk_150step_report.md is about. Its
    # pure-OPD half is the only arm in this sequence with a completed validation,
    # so this pair is the one whose control already has numbers.
    (
        "signweight_target_teachertopk",
        "examples/opd_grpo_trainer/run_multitask_signweight_target_teachertopk_qwen3.sh",
        "examples/opd_trainer/run_multitask_signweight_target_teachertopk_qwen3.sh",
    ),
]

# Everything that names the run rather than defining it.
IDENTITY = {
    "trainer.expected_config",
    "trainer.project_name",
    "trainer.experiment_name",
    "trainer.default_local_dir",
    "trainer.val_instance_log_dir",
}

# The objective itself: the only non-identity keys an OPD+GRPO arm may differ
# from its pure-OPD control in.
OBJECTIVE = {
    "actor_rollout_ref.actor.pg_loss_coef",
    # A diagnostic column, not a term in the loss -- but pure OPD switches it off
    # (nothing reads it there) and this arm switches it on (the drift check does),
    # so it shows up in the diff and is listed rather than silently allowed.
    "actor_rollout_ref.rollout.return_rollout_log_probs",
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
    overrides = list(_overrides(script))
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="ppo_trainer", overrides=overrides)


# --------------------------------------------------------------------------- #
# The config injection
# --------------------------------------------------------------------------- #
def _minimal_config(**opd):
    """A config with just enough in it for either injection to run."""
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "opd": {
                    "teacher_paths": {"alfworld": "/t/a", "search": "/t/s", "webshop": "/t/w"},
                    **opd,
                },
                "use_kl_in_reward": True,
                "adv_estimator": "grpo",
            },
            "actor_rollout_ref": {
                "actor": {"pg_loss_coef": 1.0, "entropy_coeff": 0.01},
                "ref": {"response_only_logits": True},
            },
        }
    )
    return cfg


def test_the_grpo_injection_keeps_the_policy_gradient_that_pure_opd_zeroes():
    """The whole difference between the arms, stated as a test.

    Pure OPD force-injects the three settings that keep any non-teacher signal
    out of the loss. OPD+GRPO is defined by NOT doing that, so the check is that
    the values the script asked for survive it.
    """
    from verl.trainer.main_opd import inject_opd_config
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    pure = _minimal_config()
    inject_opd_config(pure)
    assert pure.actor_rollout_ref.actor.pg_loss_coef == 0
    assert pure.actor_rollout_ref.actor.entropy_coeff == 0
    assert pure.algorithm.use_kl_in_reward is False

    grpo = _minimal_config()
    inject_opd_grpo_config(grpo)
    assert grpo.actor_rollout_ref.actor.pg_loss_coef == 1.0
    assert grpo.actor_rollout_ref.actor.entropy_coeff == 0.01
    assert grpo.algorithm.use_kl_in_reward is True


def test_the_two_injections_agree_on_everything_about_the_teacher():
    """Distillation is configured once, by a function both arms call.

    If these ever diverge, an A/B between the arms is comparing two teacher
    setups as well as two objectives, and nothing would say so.
    """
    from verl.trainer.main_opd import inject_opd_config
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    opd_knobs = dict(kl_loss_type="topk_kl", kl_loss_coef=2.0, topk=32, normalize_loss_by_task=True)
    pure = _minimal_config(**opd_knobs)
    grpo = _minimal_config(**opd_knobs)
    inject_opd_config(pure)
    inject_opd_grpo_config(grpo)

    for key in (
        "use_teacher_kl_loss",
        "teacher_kl_loss_type",
        "teacher_kl_loss_coef",
        "teacher_kl_topk",
        "normalize_loss_by_task",
        "use_kl_loss",
        "use_sdl_loss",
        "use_sdar_loss",
    ):
        assert pure.actor_rollout_ref.actor[key] == grpo.actor_rollout_ref.actor[key], key


def test_the_grpo_injection_refuses_a_run_with_no_pg_loss_coef():
    """It is the ratio between the two terms, i.e. the number the experiment is
    about. A default here would produce a run that trains fine and answers a
    different question than the one asked."""
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    cfg = _minimal_config()
    with open_dict(cfg):
        del cfg.actor_rollout_ref.actor.pg_loss_coef
    with pytest.raises(AssertionError, match="pg_loss_coef"):
        inject_opd_grpo_config(cfg)


def test_the_grpo_injection_carries_the_sign_weight_settings_to_the_actor():
    """The weights are built inside the actor's forward, so the settings have to
    reach the actor config -- on this arm exactly as on the pure one."""
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    cfg = _minimal_config(
        kl_loss_type="topk_kl",
        sign_weight={"enable": True, "mode": "target", "base_path": "Qwen/Qwen3-1.7B"},
    )
    inject_opd_grpo_config(cfg)
    assert cfg.actor_rollout_ref.actor.sign_weight.enable is True
    assert cfg.actor_rollout_ref.actor.sign_weight.mode == "target"
    # The frozen models the weights read are scored at ids the training forward
    # picks, so they have to keep their hidden states.
    assert cfg.actor_rollout_ref.ref.student_indexed_topk is True


# --------------------------------------------------------------------------- #
# Scripts against their locks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,script,_control", PAIRS)
def test_the_script_satisfies_its_own_intent_lock(name, script, _control):
    """The lock's whole purpose is to fail in seconds instead of hours, and it
    only does that if the script it guards actually passes it. Run through the
    same injection and the same checker the entry point uses."""
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config
    from verl.utils.expected_config import enforce_expected_config

    cfg = _composed(script)
    inject_opd_grpo_config(cfg)

    lock = cfg.trainer.expected_config
    assert lock.startswith("examples/opd_grpo_trainer/"), lock
    n = enforce_expected_config(cfg, os.path.join(REPO, lock), tag=f"test:{name}")
    assert n > 0


@pytest.mark.parametrize("name,script,_control", PAIRS)
def test_the_lock_pins_the_policy_gradient_as_live(name, script, _control):
    """A lock that agreed with pg_loss_coef=0 would pin the pure-OPD arm under
    this arm's name -- the exact mislabeling the locks exist to prevent."""
    cfg = _composed(script)
    pinned = yaml.safe_load(open(os.path.join(REPO, cfg.trainer.expected_config)))
    assert float(pinned["actor_rollout_ref.actor.pg_loss_coef"]) != 0
    # ...and the advantages the gradient reads have to be group-relative.
    assert pinned["algorithm.adv_estimator"] == "grpo"


@functools.lru_cache(maxsize=None)
def _lock(script):
    """The intent lock a script points at, parsed.

    Cached: composing a script through Hydra is not cheap and the comparisons
    below ask for the same handful of locks repeatedly.
    """
    return yaml.safe_load(open(os.path.join(REPO, _composed(script).trainer.expected_config)))


@pytest.mark.parametrize("name,script,control", PAIRS)
def test_the_lock_differs_from_its_pure_opd_counterpart_only_in_the_objective(name, script, control):
    """The locks are the machine-readable statement of what each arm IS, so the
    difference between a pair's two locks is the difference between the arms.
    Anything beyond pg_loss_coef and the run's name here means a knob was
    changed while nobody was looking at it -- the locks' whole job is to make
    that impossible to do quietly."""
    grpo, pure = _lock(script), _lock(control)
    differing = {k for k in set(grpo) | set(pure) if grpo.get(k, object()) != pure.get(k, object())}
    assert differing == {
        "actor_rollout_ref.actor.pg_loss_coef",
        "trainer.project_name",
        "trainer.experiment_name",
    }, sorted(differing)


def test_the_two_grpo_target_locks_differ_only_in_the_four_scientific_values():
    """The GRPO half of the invariant
    ``test_signweight_teacher_indexed_arm.py::test_the_two_target_locks_differ_only_in_the_four_scientific_values``
    pins on the pure-OPD side: the support and the weight table, and nothing
    else. Asserted separately rather than assumed from the derivation, because
    "the GRPO arms were derived from the pure ones" stops being true the first
    time somebody edits one of them."""
    student = _lock("examples/opd_grpo_trainer/run_multitask_signweight_target_qwen3.sh")
    teacher = _lock("examples/opd_grpo_trainer/run_multitask_signweight_target_teachertopk_qwen3.sh")
    differing = {
        k for k in set(student) | set(teacher) if student.get(k, object()) != teacher.get(k, object())
    }
    assert differing == {
        "trainer.experiment_name",
        "actor_rollout_ref.actor.student_indexed_topk",
        "actor_rollout_ref.ref.student_indexed_topk",
        "algorithm.opd.sign_weight.agree_weight",
        "algorithm.opd.sign_weight.agree_neg_weight",
        "actor_rollout_ref.actor.sign_weight.agree_weight",
        "actor_rollout_ref.actor.sign_weight.agree_neg_weight",
    }, sorted(differing)


def test_the_arms_do_not_share_a_lock_with_pure_opd():
    """Each arm points at its own file. Sharing one would make a change to the
    objective un-pinnable: the same lock cannot say pg_loss_coef is 0 and 1."""
    locks = set()
    for _name, script, control in PAIRS:
        for path in (script, control):
            locks.add(_composed(path).trainer.expected_config)
    assert len(locks) == 2 * len(PAIRS), sorted(locks)


# --------------------------------------------------------------------------- #
# Each arm against its pure-OPD control
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,script,control", PAIRS)
def test_the_arm_differs_from_pure_opd_only_in_the_objective(name, script, control):
    """The composed config is compared, not the scripts' text: a knob moved in
    ppo_trainer.yaml would not show up in a diff of the arguments."""
    grpo = _flat(_composed(script))
    pure = _flat(_composed(control))
    differing = {k for k in set(grpo) | set(pure) if grpo.get(k, "<absent>") != pure.get(k, "<absent>")}

    allowed = IDENTITY | OBJECTIVE
    assert differing <= allowed, sorted(differing - allowed)
    assert "actor_rollout_ref.actor.pg_loss_coef" in differing, "the arms are identical"


@pytest.mark.parametrize("name,script,control", PAIRS)
def test_the_arm_does_not_share_a_checkpoint_directory_with_its_control(name, script, control):
    """trainer.resume_mode defaults to auto, so a shared default_local_dir means
    one arm silently resumes from the other's checkpoint."""
    grpo = _composed(script)
    pure = _composed(control)
    assert grpo.trainer.default_local_dir != pure.trainer.default_local_dir


@pytest.mark.parametrize("name,script,_control", PAIRS)
def test_the_script_runs_the_grpo_entrypoint(name, script, _control):
    """A recipe that pins pg_loss_coef=1.0 and then calls main_opd would have it
    injected straight back to 0, and the lock would catch it -- but only after
    the arm failed to start. Cheaper to read the command."""
    text = open(os.path.join(REPO, script)).read()
    assert re.search(r"^python3 -m verl\.trainer\.main_opd_grpo", text, re.M), script
    assert not re.search(r"^python3 -m verl\.trainer\.main_opd\s", text, re.M), script


# --------------------------------------------------------------------------- #
# Trainer wiring
# --------------------------------------------------------------------------- #
def test_the_grpo_trainer_inherits_the_opd_loop_and_overrides_only_the_objective():
    """Teacher routing, the hidden-state cache, the sign-weight pass, env-reset
    prefetch and stop_after_steps are the parts that must stay identical for the
    A/B to mean anything. A second fit() is how they drift while both still look
    right, so the subclass must not define one."""
    from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    assert issubclass(OPDGRPORayTrainer, OPDRayTrainer)

    overridden = {
        name
        for name in vars(OPDGRPORayTrainer)
        if not name.startswith("__") and hasattr(OPDRayTrainer, name)
    }
    assert overridden == {"_reward_and_advantage", "_data_metrics", "progress_desc"}, overridden

    # The loop itself, and everything it calls around the hooks, is inherited.
    for shared in ("fit", "init_workers", "compute_teacher_log_probs", "compute_sign_weight_cache",
                   "_teacher_prefetch_chunk", "_save_checkpoint"):
        assert getattr(OPDGRPORayTrainer, shared) is getattr(OPDRayTrainer, shared), shared


def test_pure_opd_scores_the_reward_for_monitoring_and_never_for_the_loss():
    """The base hook must survive a reward-manager failure: nothing downstream of
    it reads the score, so taking the run down would trade a whole run for a
    metric. The GRPO override must NOT swallow it -- there the same score is the
    policy gradient's only signal."""
    import inspect

    from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    base = inspect.getsource(OPDRayTrainer._reward_and_advantage)
    assert "except Exception" in base

    grpo = inspect.getsource(OPDGRPORayTrainer._reward_and_advantage)
    assert "except Exception" not in grpo
    # ...and it does compute the two things pure OPD skips.
    assert "compute_log_prob_with_prefetch" in grpo
    assert "compute_advantage" in grpo


def test_the_opd_loop_calls_the_hooks_rather_than_the_pure_opd_bodies():
    """The hooks only buy anything if fit() goes through them."""
    import inspect

    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    fit = inspect.getsource(OPDRayTrainer.fit)
    assert "self._reward_and_advantage(" in fit
    assert "self._data_metrics(" in fit
    # the advantage-free metrics are reachable only through the hook
    assert "compute_opd_data_metrics(batch=batch)" not in fit


# --------------------------------------------------------------------------- #
# The joint loss: both terms weighted the same way
# --------------------------------------------------------------------------- #
def test_per_task_weighting_now_accepts_a_policy_gradient():
    """Pure OPD asserted pg_loss_coef==0 here because it only knew how to weight
    the teacher KL. The actor weights the policy-gradient and entropy terms too
    now, so the assertion would only stand in the way of this arm."""
    from verl.workers.actor.dp_actor import check_task_weighting_supported

    cfg = OmegaConf.create(
        {
            "use_dynamic_bsz": False,
            "ppo_epochs": 1,
            "loss_agg_mode": "token-mean",
            "policy_loss": {"loss_mode": "vanilla"},
            "use_kl_loss": False,
            "use_sdl_loss": False,
            "use_sdar_loss": False,
        }
    )
    check_task_weighting_supported(cfg, use_teacher_kl_loss=True, ulysses_sequence_parallel_size=1)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"use_dynamic_bsz": True}, "use_dynamic_bsz"),
        ({"ppo_epochs": 2}, "one pass"),
        ({"loss_agg_mode": "seq-mean-token-mean"}, "weighted"),
        ({"policy_loss": {"loss_mode": "gspo"}}, "vanilla"),
        ({"use_kl_loss": True}, "use_kl_loss"),
    ],
)
def test_per_task_weighting_still_refuses_what_would_silently_undo_it(mutation, message):
    """The row weights carry the WHOLE normalisation, so anything that re-scales
    or re-aggregates the loss afterwards does not produce a wrong-looking number
    -- it produces a plausible one with the task weighting quietly gone."""
    from verl.workers.actor.dp_actor import check_task_weighting_supported

    cfg = OmegaConf.create(
        {
            "use_dynamic_bsz": False,
            "ppo_epochs": 1,
            "loss_agg_mode": "token-mean",
            "policy_loss": {"loss_mode": "vanilla"},
            "use_kl_loss": False,
            "use_sdl_loss": False,
            "use_sdar_loss": False,
            **mutation,
        }
    )
    with pytest.raises(AssertionError, match=message):
        check_task_weighting_supported(cfg, use_teacher_kl_loss=True, ulysses_sequence_parallel_size=1)


def test_the_split_policy_loss_reproduces_the_aggregated_one():
    """compute_policy_loss now delegates its clipping to
    compute_policy_loss_per_token, so the weighted path reuses the objective
    rather than reimplementing it. That is only safe while the two agree."""
    from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_per_token

    torch.manual_seed(0)
    bs, t = 6, 11
    old_log_prob = torch.randn(bs, t)
    log_prob = old_log_prob + 0.1 * torch.randn(bs, t)
    advantages = torch.randn(bs, t)
    response_mask = torch.ones(bs, t)
    response_mask[:, -3:] = 0

    kwargs = dict(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
        clip_ratio_c=3.0,
    )
    whole, clipfrac, ppo_kl, clip_lower = compute_policy_loss(loss_agg_mode="token-mean", **kwargs)
    per_token, clipfrac2, ppo_kl2, clip_lower2 = compute_policy_loss_per_token(**kwargs)

    torch.testing.assert_close(
        whole, agg_loss(loss_mat=per_token, loss_mask=response_mask, loss_agg_mode="token-mean")
    )
    torch.testing.assert_close(clipfrac, clipfrac2)
    torch.testing.assert_close(ppo_kl, ppo_kl2)
    torch.testing.assert_close(clip_lower, clip_lower2)


def test_the_row_weights_give_each_task_an_equal_share_of_the_policy_gradient():
    """What per-task normalisation is FOR, applied to the policy gradient.

    The batch is deliberately lopsided: one task holds most of the tokens, so a
    plain token-mean would hand it most of the gradient. Under the row weights
    each task contributes the same amount regardless of how many tokens it
    brought.
    """
    from verl.trainer.ppo.core_algos import agg_loss_by_task_weights

    bs, t = 6, 10
    response_mask = torch.zeros(bs, t)
    # task 0: rows 0-3, 8 tokens each; task 1: rows 4-5, 2 tokens each
    task_ids = torch.tensor([0, 0, 0, 0, 1, 1])
    response_mask[:4, :8] = 1
    response_mask[4:, :2] = 1
    loss_mat = torch.ones(bs, t)

    # One weight per row: 1 / (num_tasks * that task's token count).
    row_weights = torch.zeros(bs)
    for task in (0, 1):
        rows = task_ids == task
        row_weights[rows] = 1.0 / (2 * response_mask[rows].sum())

    total = agg_loss_by_task_weights(loss_mat=loss_mat, loss_mask=response_mask, row_weights=row_weights)
    torch.testing.assert_close(total, torch.tensor(1.0))

    per_task = [
        agg_loss_by_task_weights(
            loss_mat=loss_mat[task_ids == task],
            loss_mask=response_mask[task_ids == task],
            row_weights=row_weights[task_ids == task],
        )
        for task in (0, 1)
    ]
    torch.testing.assert_close(per_task[0], per_task[1])

    # The token-mean it replaces does NOT: 32 of the 36 tokens are task 0's.
    token_share = response_mask[task_ids == 0].sum() / response_mask.sum()
    assert token_share > 0.85


def test_the_weighted_pg_term_is_the_row_sum_and_not_a_token_mean():
    """The two aggregations must not be confused: token-mean divides by the
    batch's token count, which is exactly the quantity the row weights exist to
    stop deciding the answer."""
    from verl.trainer.ppo.core_algos import agg_loss, agg_loss_by_task_weights

    torch.manual_seed(1)
    loss_mat = torch.randn(4, 7)
    response_mask = torch.ones(4, 7)
    uniform = torch.full((4,), 1.0 / response_mask.sum())

    weighted = agg_loss_by_task_weights(loss_mat=loss_mat, loss_mask=response_mask, row_weights=uniform)
    token_mean = agg_loss(loss_mat=loss_mat, loss_mask=response_mask, loss_agg_mode="token-mean")
    # With uniform weights of 1/total_tokens the two coincide...
    torch.testing.assert_close(weighted, token_mean)

    # ...and with anything else they must not, or the weighting is being ignored.
    skewed = uniform.clone()
    skewed[0] *= 4
    assert not torch.isclose(
        agg_loss_by_task_weights(loss_mat=loss_mat, loss_mask=response_mask, row_weights=skewed),
        token_mean,
    )
