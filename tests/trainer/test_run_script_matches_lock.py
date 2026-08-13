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
"""The OPD+GRPO run script must satisfy its own intent lock.

``main_opd_grpo`` checks the lock at startup, which is the design -- fail in
seconds rather than hours. But that check only ever runs on the machine with the
GPUs, so a script/lock disagreement introduced here is invisible until someone
tries to launch. Every knob the lock pins is a *scientific* setting, so the
failure mode is not a slow run, it is a run that does not start at all.

Composed the way an actual launch composes it: the script's literal arguments as
Hydra overrides on ``ppo_trainer.yaml``, then ``inject_opd_grpo_config`` to turn
``algorithm.opd.*`` into the actor keys the lock reads. ``$HOME`` is expanded
first, because the shell expands it before Hydra ever sees the argument -- a
checker that skips that step validates a string no run ever passes.
"""

import os
import shlex

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

try:
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config
    from verl.utils.expected_config import check_expected_config
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ARM_DIR = os.path.join(_REPO_ROOT, "examples", "opd_grpo_trainer")
_CONFIG_DIR = os.path.join(_REPO_ROOT, "verl", "trainer", "config")

# Every arm launched through main_opd_grpo, as (run script, intent lock). The
# sign-weighting arm is a copy of the plain one plus algorithm.opd.sign_weight.*,
# and a copy is exactly the thing that drifts, so it is checked against its own
# lock by the same machinery rather than trusted to have stayed in step.
_ARMS = {
    "plain": (
        os.path.join(_ARM_DIR, "run_multitask_qwen3.sh"),
        os.path.join(_ARM_DIR, "expected_multitask_config.yaml"),
    ),
    "signweight": (
        os.path.join(_ARM_DIR, "run_multitask_signweight_qwen3.sh"),
        os.path.join(_ARM_DIR, "expected_multitask_signweight_config.yaml"),
    ),
}
_SCRIPT, _LOCK = _ARMS["plain"]


def _overrides(home, script=None):
    """The literal Hydra arguments of the script's trainer invocation.

    Read the way the shell hands them to python, not the way they look in the
    file: ``$HOME`` expanded, and quoting removed via ``shlex`` so that
    ``...coef_by_task='{alfworld:0.1,...}'`` reaches Hydra as a dict rather than
    as a string that happens to contain braces. Skipping either step validates
    arguments no run ever passes.
    """
    body = open(script or _SCRIPT).read().split("python3 -m verl.trainer.main_opd_grpo", 1)[1]
    out = []
    for line in body.splitlines():
        line = line.strip().rstrip("\\").strip().replace('"$@"', "").strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:  # end of the argument list
            break
        words = shlex.split(line.replace("$HOME", home))
        assert len(words) == 1, f"expected one argument per line, got {words!r}"
        out.append(words[0])
    return out


def _effective_config(home, script=None):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=_overrides(home, script))
    return inject_opd_grpo_config(cfg)


@pytest.mark.parametrize("arm", sorted(_ARMS))
@pytest.mark.parametrize("home", ["/opt/home/ohara", "/home/someone-else"])
def test_the_run_script_satisfies_the_intent_lock(arm, home, monkeypatch):
    """Zero mismatches, on every arm and on more than one home.

    Two homes because the lock pins the teachers as ``$HOME/...`` and
    ``load_expectations`` expands it: with a single home a lock that had the
    literal path baked in would pass here and fail on the other machine.
    """
    script, lock = _ARMS[arm]
    monkeypatch.setenv("HOME", home)
    mismatches = check_expected_config(_effective_config(home, script), lock)
    assert mismatches == [], [f"{k}: config={got!r} expected={want!r}" for k, got, want in mismatches]


def test_the_two_arms_differ_only_in_the_sign_weighting(monkeypatch):
    """The comparison this branch exists to make.

    The sign-weighting arm is only a control-and-treatment pair with the plain
    arm if nothing else moved: same teachers, same data, same batch sizes, same
    eval protocol. Diffing the two effective configs is the check that says so,
    and it fails the moment someone tunes one script and not the other.
    """
    from omegaconf import OmegaConf

    home = "/opt/home/ohara"
    monkeypatch.setenv("HOME", home)
    plain = OmegaConf.to_container(_effective_config(home, _ARMS["plain"][0]), resolve=True)
    signed = OmegaConf.to_container(_effective_config(home, _ARMS["signweight"][0]), resolve=True)

    def flat(d, prefix=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{prefix}{k}"
            out.update(flat(v, f"{key}.")) if isinstance(v, dict) else out.update({key: v})
        return out

    a, b = flat(plain), flat(signed)
    differing = {k for k in set(a) | set(b) if a.get(k, "<missing>") != b.get(k, "<missing>")}
    allowed = {k for k in differing if k.startswith("algorithm.opd.sign_weight.")}
    # the run identity has to differ too, or the two arms would overwrite each
    # other's checkpoints and metrics
    allowed |= {"trainer.project_name", "trainer.experiment_name", "trainer.expected_config"}
    assert differing == allowed, sorted(differing - allowed)
    assert any(k.startswith("algorithm.opd.sign_weight.") for k in differing), "the arms are identical"


def test_the_lock_still_catches_a_drifting_run_script(monkeypatch):
    """Negative control: the check above must be able to fail.

    A parser that silently dropped the arguments -- a reformatted script, a
    line-continuation moved -- would compose the bare defaults and, if those
    happened to agree, report success while checking nothing.
    """
    home = "/opt/home/ohara"
    monkeypatch.setenv("HOME", home)
    cfg = _effective_config(home)
    cfg.algorithm.opd.kl_loss_type = "low_var_kl" if cfg.algorithm.opd.kl_loss_type == "topk_kl" else "topk_kl"
    assert any(k == "algorithm.opd.kl_loss_type" for k, _, _ in check_expected_config(cfg, _LOCK))


def test_the_script_passes_every_key_the_lock_pins_for_this_arm(monkeypatch):
    """The knobs this branch's ports moved, asserted by name.

    The blanket check above would also pass if a key were dropped from BOTH the
    script and the lock. These are the ones whose absence would silently change
    the experiment rather than stop it.
    """
    home = "/opt/home/ohara"
    monkeypatch.setenv("HOME", home)
    from omegaconf import OmegaConf

    cfg = _effective_config(home)
    expected = {
        # per-task equal share, on both loss terms
        "actor_rollout_ref.actor.normalize_loss_by_task": True,
        # the GRPO policy gradient is live -- this is what distinguishes the arm
        # from pure OPD, and what the weighting had to be extended to cover
        "actor_rollout_ref.actor.pg_loss_coef": 1.0,
        "actor_rollout_ref.actor.use_teacher_kl_loss": True,
        "actor_rollout_ref.actor.teacher_kl_loss_type": "topk_kl",
        # gradient-path speedups
        "actor_rollout_ref.actor.fsdp_config.sharding_strategy": "shard_grad_op",
        "actor_rollout_ref.actor.no_sync_grad_accum": True,
        # retry policy decides what enters the trajectory
        "env.search.max_retries": None,
        "env.search.timeout": 600,
        # the plain arm must NOT carry the weighting, or it is not a control
        "algorithm.opd.sign_weight": None,
    }
    for key, want in expected.items():
        assert OmegaConf.select(cfg, key) == want, key


def test_the_sign_weight_script_passes_the_knobs_that_define_that_arm(monkeypatch):
    """Same idea for the new arm: a key dropped from both script and lock would
    pass the blanket check while silently turning the arm back into its control."""
    home = "/opt/home/ohara"
    monkeypatch.setenv("HOME", home)
    from omegaconf import OmegaConf

    cfg = _effective_config(home, _ARMS["signweight"][0])
    expected = {
        "algorithm.opd.sign_weight.enable": True,
        "algorithm.opd.sign_weight.mode": "position",
        "algorithm.opd.sign_weight.agree_weight": 1.25,
        "algorithm.opd.sign_weight.disagree_weight": 0.75,
        "algorithm.opd.sign_weight.deadzone": 0.1,
        "algorithm.opd.sign_weight.base_path": "Qwen/Qwen3-1.7B",
        # the shifts are measured against the model the teachers came from, so
        # the base path and the student's init are the same checkpoint
        "actor_rollout_ref.model.path": "Qwen/Qwen3-1.7B",
        # the shared support the four models are compared on only exists here
        "algorithm.opd.kl_loss_type": "topk_kl",
    }
    for key, want in expected.items():
        assert OmegaConf.select(cfg, key) == want, key
