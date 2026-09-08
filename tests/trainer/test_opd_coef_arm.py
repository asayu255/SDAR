# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The three arms, and the claim that they differ in nothing but b.

The comparison is only worth running if that is true, and it is the kind of
thing that stops being true one careless edit later -- so it is checked here
rather than asserted in a comment.
"""

import os
import re

import pytest

from verl.trainer.main_opd import KL_COEF_BY_TASK_BOX, validate_kl_coef_by_task
from verl.utils.expected_config import load_expectations

ARMS = ("control", "uniform", "redistribute", "pushback", "cross", "cross2")
EXPECT = "examples/opd_grpo_trainer/expected_multitask_opd_coef_{arm}_config.yaml"
SCRIPT = "examples/opd_grpo_trainer/run_multitask_opd_coef_qwen3.sh"

# The design's numbers (docs/opd_coefficient_arm_design.md, step 300, N=8).
B = {
    "control": None,
    "uniform": {"alfworld": 1.110833, "search": 1.110833, "webshop": 1.110833},
    "redistribute": {"alfworld": 1.076431, "search": 1.191101, "webshop": 0.5},
    "pushback": None,          # no static coefficient: the online gate replaces it
    "cross": None,
    # v2: same pure OPD+GRPO underneath, so the static coefficient is null too
    "cross2": None,             # MOPD v1: pure OPD+GRPO plus the cross-task gate
}
CALIBRATION_D = {"alfworld": 0.4563, "search": 0.8859, "webshop": 0.4084}


def _flat(arm):
    os.environ.setdefault("RUN_TAG_SUFFIX", "")
    return load_expectations(EXPECT.format(arm=arm))


def _is_coef_key(key):
    # the parent key AND its per-task children, on both sides of the injection
    return "kl_loss_coef_by_task" in key


def _is_pushback_key(key):
    return "pushback" in key


# Speculative decoding is pinned per FAMILY, not per arm: the cross family
# (control + cross) samples with it, the coefficient family (uniform,
# redistribute, pushback) without, because the live pushback run and the August
# pure OPD+GRPO baseline both sampled without it. It changes which tokens are
# drawn, so it has to be identical within a comparison -- which is exactly what
# test_speculative_decoding_is_uniform_within_each_family asserts. Here it is
# allowed to differ from control so the coefficient arms are not reported as
# having drifted.
SPEC_ROOT = "actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config"
CROSS_FAMILY = ("control", "cross", "cross2")


def _is_spec_key(key):
    return key == SPEC_ROOT or key.startswith(SPEC_ROOT + ".")


def _is_cross_key(key):
    # the cross gate's own knobs, and the self gate it DECLARES off (the
    # control lock does not mention pushback at all, so "null" differs from
    # "absent" and has to be allowed here -- and checked below)
    return "cross_gate" in key or "pushback" in key


# ---------------------------------------------------------------------------
# 1. the arms differ in b and in their own name, and in nothing else


@pytest.mark.parametrize("arm", ("uniform", "redistribute", "pushback", "cross"))
def test_the_lock_files_differ_in_nothing_but_the_arms_own_knob(arm):
    control, other = _flat("control"), _flat(arm)
    allowed = {"trainer.experiment_name"}
    own_or_family = lambda k: _is_spec_key(k) or own(k)
    own = {"pushback": _is_pushback_key, "cross": _is_cross_key,
           "cross2": _is_cross_key}.get(arm, _is_coef_key)
    differing = {
        k for k in set(control) | set(other)
        if control.get(k, "<absent>") != other.get(k, "<absent>")
    }
    unexpected = {k for k in differing if k not in allowed and not own_or_family(k)}
    assert not unexpected, f"{arm} differs from control outside its own knob: {sorted(unexpected)}"
    # and it really does differ -- a test that passes because both files are
    # identical would be worse than no test.
    assert any(own(k) for k in differing)
    if arm in ("pushback", "cross", "cross2"):
        # the static coefficient stays null on this arm, on both sides
        assert other["actor_rollout_ref.actor.teacher_kl_loss_coef_by_task"] is None
        assert other["algorithm.opd.kl_loss_coef_by_task"] is None
    if arm in ("cross", "cross2"):
        # pure OPD+GRPO underneath: the self gate is declared OFF, on both sides
        assert other["algorithm.opd.pushback_control"] is None
        assert other["actor_rollout_ref.actor.teacher_kl_pushback"] is None
        assert other["algorithm.opd.cross_gate.enable"] is True
        assert other["actor_rollout_ref.actor.teacher_kl_cross_gate.enable"] is True
        # the pre-fixed conditions (design doc §8), pinned on both sides
        for side in ("algorithm.opd.cross_gate", "actor_rollout_ref.actor.teacher_kl_cross_gate"):
            assert other[f"{side}.eps_cross"] == 0.0
            assert other[f"{side}.rho"] == 10000.0
            assert other[f"{side}.lambda_max"] == 0.2
            assert other[f"{side}.ema_decay"] == 0.8
            assert other[f"{side}.window_steps"] == 8


def test_the_control_arm_leaves_the_key_unset_rather_than_setting_it_to_one():
    control = _flat("control")
    assert control["actor_rollout_ref.actor.teacher_kl_loss_coef_by_task"] is None
    assert control["algorithm.opd.kl_loss_coef_by_task"] is None
    assert not any(k.endswith((".alfworld", ".search", ".webshop")) and _is_coef_key(k)
                   for k in control)


@pytest.mark.parametrize("arm", ("uniform", "redistribute"))
def test_the_pinned_coefficients_are_the_design_numbers_on_both_sides(arm):
    flat = _flat(arm)
    for task, want in B[arm].items():
        assert flat[f"actor_rollout_ref.actor.teacher_kl_loss_coef_by_task.{task}"] == want
        assert flat[f"algorithm.opd.kl_loss_coef_by_task.{task}"] == want


@pytest.mark.parametrize("arm", ARMS)
def test_no_arm_validates_inside_the_training_run(arm):
    """The loop validates BEFORE it saves (opd_ray_trainer.py:1579 then :1586),
    and the validation pass has exhausted this host's RAM. An OOM at step 150
    would therefore cost the step-150 checkpoint as well as the 24 hours that
    produced it -- for a pass that can be recomputed from a checkpoint. So the
    arms train only, and the evaluations are separate VAL_ONLY runs.

    Pinned rather than trusted, because re-enabling it costs a whole run and the
    failure arrives 24 hours after the mistake."""
    flat = _flat(arm)
    assert flat["trainer.test_freq"] == -1
    # ...which means the checkpoints have to be there to evaluate from
    assert flat["trainer.save_freq"] == 25
    assert flat["trainer.total_training_steps"] % flat["trainer.save_freq"] == 0
    assert (flat["trainer.total_training_steps"] // 2) % flat["trainer.save_freq"] == 0, (
        "the 150-step waypoint must land on a save boundary"
    )


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_pins_the_readout_and_the_step_count(arm):
    flat = _flat(arm)
    assert flat["algorithm.opd.task_diag"] is True
    assert flat["actor_rollout_ref.actor.teacher_kl_task_diag"] is True
    assert flat["trainer.total_training_steps"] == 300
    # The arms are only comparable on one machine: sampled rollouts make the GPU
    # count part of the experiment, not a performance knob.
    assert flat["trainer.n_gpus_per_node"] == 2
    # unchanged from the control recipe this arm is derived from
    assert flat["actor_rollout_ref.actor.teacher_kl_loss_coef"] == 0.01
    assert flat["algorithm.opd.normalize_loss_by_task"] is True
    assert flat["actor_rollout_ref.actor.cross_teacher_kl_weight.enable"] is False


# ---------------------------------------------------------------------------
# 2. the rule's own arithmetic, on the numbers the files pin


def test_the_redistributed_vector_holds_the_calibration_budget():
    """sum_j q_j b_j = 1, with q the gradient-norm shares. The property the
    whole design is built on, checked against the pinned values rather than
    against a note in a document."""
    d = CALIBRATION_D
    total = sum(d.values())
    q = {k: v / total for k, v in d.items()}
    b = B["redistribute"]
    assert sum(q[k] * b[k] for k in d) == pytest.approx(1.0, abs=5e-4)
    # the linear budget is the same statement in unnormalised form
    assert sum(b[k] * d[k] for k in d) == pytest.approx(sum(d.values()), rel=5e-4)


def test_the_uniform_arm_does_not_hold_that_budget_and_is_not_meant_to():
    d = CALIBRATION_D
    b = B["uniform"]
    ratio = sum(b[k] * d[k] for k in d) / sum(d.values())
    assert ratio == pytest.approx(1.110833, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. the run script


def _script():
    return open(SCRIPT).read()


@pytest.mark.parametrize("arm", ARMS)
def test_the_script_offers_exactly_these_arms(arm):
    s = _script()
    assert re.search(rf"^\s+{arm}\)\s*$", s, re.M), f"ARM={arm} has no case branch"


def test_the_script_refuses_an_unknown_arm():
    s = _script()
    assert "ARM must be control | uniform | redistribute" in s
    assert "exit 1" in s.split("esac")[0]


@pytest.mark.parametrize("arm", ("uniform", "redistribute"))
def test_the_script_passes_the_pinned_vector(arm):
    """Parsed and compared as numbers, so a lost decimal cannot pass as a match."""
    branch = _script().split(f"    {arm})")[1].split(";;")[0]
    m = re.search(r"kl_loss_coef_by_task=\{([^}]*)\}", branch)
    assert m, f"{arm}: no coefficient vector in its branch"
    got = {k.strip(): float(v) for k, v in
           (pair.split(":") for pair in m.group(1).split(","))}
    assert got == B[arm]
    assert got == {t: _flat(arm)[f"algorithm.opd.kl_loss_coef_by_task.{t}"] for t in got}


def test_the_script_selects_the_lock_by_arm_and_turns_the_readout_on():
    s = _script()
    assert 'expected_multitask_opd_coef_${ARM}_config.yaml' in s
    assert "+algorithm.opd.task_diag=True" in s
    assert '"${OPD_COEF_ARGS[@]}"' in s


def test_the_control_branch_says_null_rather_than_saying_nothing():
    """Omitting the key leaves it at <<MISSING>>, which the lock rejects.

    The lock pins it to null on all three arms on purpose -- the control has to
    DECLARE that it did not use per-task coefficients -- so the control branch
    has to say null. This assertion used to require the opposite, and the arm
    would have died at startup.
    """
    branch = _script().split("    control)")[1].split(";;")[0]
    assert "+algorithm.opd.kl_loss_coef_by_task=null" in branch
    assert "OPD_COEF_ARGS=()" not in branch


def test_the_script_and_the_lock_agree_on_the_step_count():
    """The total is the cosine schedule's denominator, not a duration knob: an
    arm declared over 150 steps and one over 300 are at different learning
    rates at every shared step number."""
    s = _script()
    assert "trainer.total_training_steps=300" in s
    assert "trainer.total_epochs=300" in s
    assert "--total_training_steps 300" in s, "data prep must match"
    # and validation does NOT run inside training -- see the next test
    assert "trainer.test_freq=-1" in s


# ---------------------------------------------------------------------------
# 4. the startup gate


@pytest.mark.parametrize("arm", ("uniform", "redistribute"))
def test_startup_validation_accepts_the_arms(arm):
    assert validate_kl_coef_by_task(B[arm]) is not None


def test_startup_validation_accepts_unset():
    assert validate_kl_coef_by_task(None) is None


@pytest.mark.parametrize("bad", [
    {"alfworld": 1.6},          # above the box
    {"alfworld": 0.4},          # below it
    {"alfworld": float("nan")},
    {"alfworld": float("inf")},
    {"alfworld": "1.0"},        # a string coefficient would reach the row tensor
    {"alfworld": True},         # bool is an int in python, and is not a coefficient
    {},                         # an empty map is not the uniform arm
])
def test_startup_validation_refuses_what_would_become_a_different_arm(bad):
    with pytest.raises(ValueError):
        validate_kl_coef_by_task(bad)


def test_the_box_is_the_designs_box_and_not_a_mean_one_constraint():
    assert KL_COEF_BY_TASK_BOX == (0.5, 1.5)
    # sum(b) = 2.767 for the redistributed arm: a mean-1 check would reject it
    assert sum(B["redistribute"].values()) == pytest.approx(2.767532, abs=1e-5)
    assert validate_kl_coef_by_task(B["redistribute"]) is not None


# ---------------------------------------------------------------------------
# 5. the arms as the shell actually launches them: real args -> Hydra compose
#    -> config injection -> intent lock.
#
# Diffing the lock files against each other cannot catch a key the SCRIPT never
# passes, and that is exactly how the control arm came to fail its own lock:
# every file-level test passed while the arm died at startup.

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _real_overrides(arm, tmp_path):
    """The override list the shell hands to main_opd_grpo, via a python3 shim.

    Reading the script's text would be a second parser. Running it with a stub
    python3 is the script's own answer, and it costs nothing: every python3 call
    it makes -- the model check, the data prep, the trainer -- is stubbed.
    """
    import subprocess

    shim = tmp_path / "bin"
    shim.mkdir()
    dump = tmp_path / "args.txt"
    (shim / "python3").write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-m" ] && [ "$2" = "verl.trainer.main_opd_grpo" ]; then\n'
        '  shift 2; printf "%s\\n" "$@" > "$ARGDUMP"; exit 0\n'
        "fi\nexit 0\n"
    )
    (shim / "python3").chmod(0o755)
    env = dict(os.environ, PATH=f"{shim}:{os.environ['PATH']}", ARM=arm, ARGDUMP=str(dump))
    subprocess.run(["bash", SCRIPT], env=env, cwd=ROOT,
                   capture_output=True, check=True, timeout=300)
    return [line for line in dump.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("arm", ARMS)
def test_the_arm_the_shell_launches_passes_its_own_intent_lock(arm, tmp_path):
    import pathlib

    from hydra import compose, initialize_config_dir

    from verl.trainer.main_opd import inject_distillation_config
    from verl.utils.expected_config import check_expected_config

    overrides = _real_overrides(arm, tmp_path)
    lock = next(o.split("=", 1)[1] for o in overrides
                if o.startswith("+trainer.expected_config="))
    overrides = [o for o in overrides if not o.startswith("+trainer.expected_config=")]
    assert f"opd_coef_{arm}_config.yaml" in lock

    cfgdir = str(pathlib.Path(ROOT, "verl/trainer/config").resolve())
    os.environ.setdefault("RUN_TAG_SUFFIX", "")
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    inject_distillation_config(cfg)
    mismatches = check_expected_config(cfg, os.path.join(ROOT, lock))
    assert not mismatches, [(k, g, w) for k, g, w in mismatches]

    # and the actor ends up with exactly this arm's b
    got = cfg.actor_rollout_ref.actor.get("teacher_kl_loss_coef_by_task", "MISSING")
    if B[arm] is None:
        assert got is None, "control must reach the actor as None, i.e. the original path"
    else:
        assert dict(got) == B[arm]
    assert cfg.actor_rollout_ref.actor.teacher_kl_task_diag is True


def test_speculative_decoding_is_uniform_within_each_family():
    """Both arms of a comparison must sample the same way, or the comparison is
    between two sampling processes rather than two objectives.

    Rejection sampling preserves the target distribution, but which tokens get
    drawn still changes, so this is not covered by the performance-knob
    exemption. The cross family carries it; the coefficient family does not,
    because the live pushback run and the August baseline (wandb ktrcnege) both
    sampled without it.
    """
    locks = {a: _flat(a) for a in ARMS}
    on = {a: locks[a].get(SPEC_ROOT + ".method") for a in CROSS_FAMILY}
    assert set(on.values()) == {"ngram"}, f"cross family disagrees on spec decode: {on}"
    for a in CROSS_FAMILY:
        assert locks[a][SPEC_ROOT + ".num_speculative_tokens"] == 4
        assert locks[a][SPEC_ROOT + ".acceptance_method"] == "rejection_sampler"
    for a in ARMS:
        if a in CROSS_FAMILY:
            continue
        assert locks[a][SPEC_ROOT] is None, (
            f"{a} is in the coefficient family and must declare spec decode null, "
            f"got {locks[a][SPEC_ROOT]!r}"
        )
    # and the two families really do differ on it, or this test is vacuous
    assert locks["control"].get(SPEC_ROOT + ".method") != locks["pushback"].get(SPEC_ROOT + ".method")


def test_cross_and_cross2_differ_in_the_gate_basis_and_nothing_else():
    """The v1/v2 pair is only a comparison if one pinned value separates them."""
    a, b = _flat("cross"), _flat("cross2")
    differing = {
        k for k in set(a) | set(b)
        if a.get(k, "<absent>") != b.get(k, "<absent>")
    }
    expected = {
        "trainer.experiment_name",
        "algorithm.opd.cross_gate.gate_version",
        "actor_rollout_ref.actor.teacher_kl_cross_gate.gate_version",
    }
    assert differing <= expected, f"cross2 differs beyond the gate basis: {sorted(differing - expected)}"
    assert a["algorithm.opd.cross_gate.gate_version"] == 1
    assert b["algorithm.opd.cross_gate.gate_version"] == 2
    # and the conditions the design holds fixed really are fixed
    for k in ("eps_cross", "rho_rel", "lambda_max", "ema_decay", "window_steps",
              "min_prompts", "min_pg_prompts", "min_tokens", "max_staleness", "split_seed"):
        assert a[f"algorithm.opd.cross_gate.{k}"] == b[f"algorithm.opd.cross_gate.{k}"], k
    for side in ("algorithm.opd", "actor_rollout_ref.actor.teacher_kl"):
        pass
    assert b["algorithm.opd.pushback_control"] is None
    assert b["algorithm.opd.kl_loss_coef_by_task"] is None
