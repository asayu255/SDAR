"""Every override a run script passes must actually compose.

A run script is only correct if Hydra accepts it, and one mistake in particular
is invisible on inspection: `+key=value` means *append a new key*, so it raises

    ConfigCompositionException: Could not append to config.
    An item is already at 'actor_rollout_ref.actor.response_only_logits'.

as soon as that key gains a default in ppo_trainer.yaml. The trap is that Hydra
allows `+` when the existing value is null -- which is why
`+actor_rollout_ref.actor.fsdp_config.sharding_strategy=...` has always worked --
so adding a *non-null* default to a key some script passes with `+` breaks that
script, in a different file, at startup. Nothing else in the suite catches it:
the config loads fine, the expectations file agrees, and the arm simply refuses
to launch.

Two gates:
  * compose the multitask arms' scripts against the real config, end to end;
  * a static sweep over every example script for the same pattern, so a script
    nobody is running today still fails loudly the moment the default lands.
"""

import os
import re

import pytest

pytest.importorskip("torch")
hydra = pytest.importorskip("hydra")
yaml = pytest.importorskip("yaml")

from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "verl", "trainer", "config")
# main_opd and main_ppo both use config_name="ppo_trainer".
CONFIG_NAME = "ppo_trainer"

COMPOSED_SCRIPTS = [
    "examples/opd_trainer/run_multitask_qwen3.sh",
    "examples/opd_trainer/run_multitask_signweight_position_qwen3.sh",
    "examples/opd_trainer/run_multitask_signweight_target_qwen3.sh",
    "examples/sdar_trainer/run_multitask_qwen3.sh",
]


def _overrides(path):
    """The Hydra arguments of the script's trainer invocation.

    The command is one backslash-continued line, so read until a line that does
    not continue. `$@` is the caller's trailing overrides and `$HOME` is expanded
    by the shell before Hydra sees it, so do both here too.
    """
    text = open(os.path.join(REPO, path)).read()
    marker = re.search(r"python3 -m verl\.trainer\.main_\w+", text)
    if marker is None:
        return None
    tail = text[marker.start():]
    lines = []
    for raw in tail.splitlines():
        lines.append(raw)
        if not raw.rstrip().endswith("\\"):
            break
    cmd = " ".join(line.rstrip().rstrip("\\").strip() for line in lines)
    cmd = re.sub(r"python3 -m verl\.trainer\.main_\w+", "", cmd).replace('"$@"', "")

    def _unquote(tok):
        # The shell removes a matching pair of surrounding quotes before Hydra
        # sees the value, and for a structured value that is the difference
        # between a dict and a string: '{alfworld:0.1,...}' composes as a nested
        # config, "'{alfworld:0.1,...}'" as one opaque token whose sub-keys then
        # read as missing.
        key, _, value = tok.partition("=")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        return f"{key}={value}"

    return [_unquote(os.path.expandvars(tok)) for tok in cmd.split() if "=" in tok]


@pytest.mark.parametrize("script", COMPOSED_SCRIPTS)
def test_the_script_composes(script):
    if not os.path.exists(os.path.join(REPO, script)):
        pytest.skip(f"{script} not present on this branch")
    overrides = _overrides(script)
    assert overrides, f"no overrides parsed from {script}"
    try:
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            compose(config_name=CONFIG_NAME, overrides=overrides)
    except ConfigCompositionException as e:
        pytest.fail(f"{script} does not compose:\n{e}")


def _schema_value(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def test_no_script_appends_over_an_existing_non_null_default():
    """The static form of the same rule, swept over every example script.

    `+` is legal on a key the schema leaves null (that is how the optional FSDP
    knobs are passed) and illegal on one that already has a value.
    """
    schema = yaml.safe_load(open(os.path.join(CONFIG_DIR, CONFIG_NAME + ".yaml")))
    offenders = []
    for dirpath, _, filenames in os.walk(os.path.join(REPO, "examples")):
        for name in filenames:
            if not name.endswith(".sh"):
                continue
            path = os.path.join(dirpath, name)
            body = open(path).read()
            for line in body.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for match in re.finditer(r"\+((?:\w+\.)+\w+)=", stripped):
                    key = match.group(1)
                    present, value = _schema_value(schema, key)
                    if present and value is not None:
                        offenders.append((os.path.relpath(path, REPO), key, value))

    assert not offenders, (
        "these scripts use '+' on a key that already has a non-null default, "
        "so Hydra refuses to compose them:\n"
        + "\n".join(f"  {s}: +{k}=  (schema default: {v!r})" for s, k, v in sorted(set(offenders)))
    )


def test_the_composed_pure_opd_arm_uses_the_student_indexed_support():
    """Which 20 tokens the coarse-grained KL is exact on is a property of the
    experiment, not a tuning knob: the two settings are different (both valid)
    lower bounds on the same reverse KL, so arms compared across them compare
    nothing. Pinned in expected_multitask_config.yaml and asserted here against
    the config the script actually composes."""
    overrides = _overrides("examples/opd_trainer/run_multitask_qwen3.sh")
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

    assert cfg.actor_rollout_ref.actor.student_indexed_topk is True
    # The teacher follows the actor -- one support, not two.
    assert cfg.actor_rollout_ref.ref.student_indexed_topk is True
    # ...and it needs the response-only row map on both sides.
    assert cfg.actor_rollout_ref.actor.response_only_logits is True
    assert cfg.actor_rollout_ref.ref.response_only_logits is True


def test_the_default_is_on_and_the_teacher_follows_it():
    """Default-ON, so a script that does not mention it still gets the tighter
    bound; ref mirrors the actor so the two halves cannot disagree."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        base = compose(config_name=CONFIG_NAME, overrides=[])
        assert base.actor_rollout_ref.actor.student_indexed_topk is True
        assert base.actor_rollout_ref.ref.student_indexed_topk is True

        off = compose(
            config_name=CONFIG_NAME, overrides=["actor_rollout_ref.actor.student_indexed_topk=False"]
        )
        assert off.actor_rollout_ref.ref.student_indexed_topk is False


def test_no_script_exports_expandable_segments():
    """vLLM's sleep/wake allocates through CuMemAllocator, which asserts outright
    on expandable segments (pytorch#147851) -- the engine refuses to build. Every
    script here runs vLLM with enable_sleep_mode, so this is not a tuning
    trade-off, it is an incompatibility, and it costs a full worker init to find
    out at runtime."""
    import glob

    offenders = []
    for path in glob.glob(os.path.join(REPO, "examples", "**", "*.sh"), recursive=True):
        for line in open(path):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "PYTORCH_CUDA_ALLOC_CONF" in stripped and "expandable_segments:True" in stripped:
                offenders.append(os.path.relpath(path, REPO))
    assert not offenders, f"expandable_segments is incompatible with vLLM sleep mode: {offenders}"
