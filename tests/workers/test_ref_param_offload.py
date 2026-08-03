"""``ref.fsdp_config.param_offload`` now decides, so every recipe must say what it wants.

The reference / teacher policy used to be pinned to FSDP ``CPUOffload`` no matter
what that key said, which made it dead config. Now it is live -- which means a
recipe that leaves it unset silently changes behaviour relative to before this
branch: the ref shards stay resident instead of being fetched from host memory
per all-gather. That is the point of the change on a PCIe-bound host, but it is
not something a recipe should get by accident, because it costs
``param_bytes / world_size`` of GPU memory.

So: any run script that builds a reference policy has to set the key explicitly.
The condition for building one is the same in every entry point --
``algorithm.use_kl_in_reward or actor_rollout_ref.actor.use_kl_loss`` (see
``verl/trainer/main_ppo.py``) -- and both are plain Hydra overrides in the run
scripts, so this can be read straight off the text.

Megatron recipes are exempt: they build their ref through
``verl/workers/megatron_workers.py``, which this change does not touch.
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
# Every directory that holds runnable recipes. recipe/ matters as much as
# examples/ here -- PRIME inherits ppo_trainer.yaml, so it picks up the default.
RECIPE_DIRS = [REPO / "examples", REPO / "recipe", REPO / "tests"]

_KEY = "actor_rollout_ref.ref.fsdp_config.param_offload"


def _override(text: str, key: str):
    """Value of a ``key=value`` Hydra override, or None. Handles the ``+key=`` form."""
    m = re.search(rf"(?<![\w.]){re.escape(key)}=([^\s\\]+)", text)
    return m.group(1) if m else None


def _builds_an_fsdp_reference_policy(text: str) -> bool:
    if "megatron" in text:  # built by megatron_workers.py, which is untouched
        return False
    return (
        _override(text, "actor_rollout_ref.actor.use_kl_loss") == "True"
        or _override(text, "algorithm.use_kl_in_reward") == "True"
    )


def _scripts():
    for directory in RECIPE_DIRS:
        for script in sorted(directory.rglob("*.sh")):
            yield script


def test_every_recipe_with_a_reference_policy_pins_its_placement():
    missing = []
    for script in _scripts():
        text = script.read_text()
        if not _builds_an_fsdp_reference_policy(text):
            continue
        if _override(text, _KEY) is None:
            missing.append(str(script.relative_to(REPO)))
    assert not missing, (
        f"{_KEY} is now honoured rather than forced on, so these recipes would "
        f"change the ref's memory placement without saying so: {missing}"
    )


def test_the_scan_actually_finds_recipes():
    """A regex that matches nothing would make the test above vacuously true."""
    found = [s for s in _scripts() if _builds_an_fsdp_reference_policy(s.read_text())]
    assert len(found) > 10, f"only {len(found)} recipes matched; the override scan is broken"
