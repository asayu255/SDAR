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
    "signweight_target": (
        os.path.join(_ARM_DIR, "run_multitask_signweight_target_qwen3.sh"),
        os.path.join(_ARM_DIR, "expected_multitask_signweight_target_config.yaml"),
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

    A ``#`` line ENDS the argument list, because that is what it does in the
    shell: the previous line's ``\\`` joins the two, so the ``#`` starts a comment
    that runs to the newline and the command stops there -- everything below is
    dropped from the invocation and then executed as a command of its own,
    ``"$@"`` (every override the caller passed) included. This used to ``continue``
    past such a line, which is the exact failure the docstring above warns about:
    for months it composed a config from arguments the shell never delivered, and
    reported the signweight arm as matching its lock while that arm could not
    start at all.
    """
    body = open(script or _SCRIPT).read().split("python3 -m verl.trainer.main_opd_grpo", 1)[1]
    out = []
    for line in body.splitlines():
        line = line.strip().rstrip("\\").strip().replace('"$@"', "").strip()
        if line.startswith("#"):  # the shell stops here; so do we
            break
        if not line:
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
def test_no_arm_hides_a_comment_inside_its_argument_list(arm):
    """A ``#`` between two continued lines silently truncates the command.

    The lock check above would also catch it -- the dropped
    ``total_training_steps`` stops matching the pinned 300 -- but it reports a
    config mismatch, which points at the lock rather than at the eleven lines of
    prose sitting in the middle of the invocation. This says what is actually
    wrong. It is a real failure, not a style rule: everything below the comment,
    ``"$@"`` included, is dropped from the python call and then run as a shell
    command, so the caller's own overrides never reach the trainer.
    """
    script, _ = _ARMS[arm]
    body = open(script).read().split("python3 -m verl.trainer.main_opd_grpo", 1)[1]
    # Where the command ends is decided by the backslashes, not by the content:
    # a line ending in one continues, and the first line that does not is the
    # command's last. Comments below that point are ordinary prose.
    offenders = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            offenders.append(f"{stripped[:60]!r}")
        if not stripped.endswith("\\"):
            break
    assert not offenders, (
        f"{os.path.basename(script)} has comments inside the trainer argument list, which "
        f"the shell reads as end-of-command: {offenders}. Move them into the file header."
    )


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


@pytest.mark.parametrize("arm", ["signweight", "signweight_target"])
def test_each_weighted_arm_differs_from_the_control_only_in_the_weighting(arm, monkeypatch):
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
    signed = OmegaConf.to_container(_effective_config(home, _ARMS[arm][0]), resolve=True)

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
    allowed |= {
        "trainer.project_name",
        "trainer.experiment_name",
        "trainer.expected_config",
        "trainer.default_local_dir",
        # Placement, not science: the global batch sizes are pinned, and the
        # per-task loss weights already divide by the DP world size, so the
        # objective is the same on any GPU count (bit-level reduction order
        # aside). Allowed to differ because arms get scheduled on whatever is
        # free -- but when an arm is used as another's CONTROL, match it anyway,
        # so the pair differs in the weighting alone.
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
    }
    # Same exemption the lock files state for themselves ("performance knobs
    # (micro batch sizes, offload, prefix caching, ...) are intentionally NOT
    # listed"): they are how a run is fitted onto the GPUs it got. The per-task
    # loss weights already multiply by gradient_accumulation, so the objective is
    # invariant to the split; only the reduction order moves. Matching them is
    # still the right thing for a control/treatment pair, and the comment beside
    # n_gpus_per_node says so.
    allowed |= {k for k in differing if "micro_batch_size" in k}
    # Subset, not equality: an allowance the arms happen to agree on (both on the
    # same GPU count, say) is not a failure. What must not happen is a difference
    # outside the list.
    assert differing <= allowed, sorted(differing - allowed)
    assert any(k.startswith("algorithm.opd.sign_weight.") for k in differing), "the arms are identical"
    # Not merely allowed to differ -- required to. The checkpoint path is
    # default_local_dir/global_step_N and trainer.resume_mode defaults to "auto",
    # so a shared directory does not just overwrite: the arm started second
    # RESUMES FROM the arm started first and reports it under its own name.
    assert a["trainer.default_local_dir"] != b["trainer.default_local_dir"], (
        "arms must not share a checkpoint directory; resume_mode=auto would make "
        "the second one continue the first"
    )


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
        # lm_head on the response rows only -- same arithmetic, different GEMM
        # shape, and it sits on both the student's gradient and the teacher's
        # targets, so the two must move together across compared runs
        "actor_rollout_ref.actor.response_only_logits": True,
        "actor_rollout_ref.ref.response_only_logits": True,
        # retry policy decides what enters the trajectory
        "env.search.max_retries": None,
        "env.search.timeout": 600,
        # the plain arm must NOT carry the weighting, or it is not a control
        "algorithm.opd.sign_weight": None,
    }
    for key, want in expected.items():
        assert OmegaConf.select(cfg, key) == want, key


@pytest.mark.parametrize("arm,mode", [("signweight", "position"), ("signweight_target", "target")])
def test_the_sign_weight_script_passes_the_knobs_that_define_that_arm(arm, mode, monkeypatch):
    """Same idea for the new arms: a key dropped from both script and lock would
    pass the blanket check while silently turning the arm back into its control.

    ``mode`` is asserted per arm because it is the one knob whose value decides
    which of the two claims the run is evidence for -- a target-mode result filed
    under the position-mode name would be a different paper.
    """
    home = "/opt/home/ohara"
    monkeypatch.setenv("HOME", home)
    from omegaconf import OmegaConf

    cfg = _effective_config(home, _ARMS[arm][0])
    expected = {
        "algorithm.opd.sign_weight.enable": True,
        "algorithm.opd.sign_weight.mode": mode,
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
