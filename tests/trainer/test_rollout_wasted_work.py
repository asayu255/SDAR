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
"""Two things the rollout stopped doing for nobody.

* ``rollout_log_probs`` -- vLLM is asked for the sampled token's log-prob and a
  Python loop over every generated token assembles the column. Its only consumer
  is the rollout-vs-actor drift check, which needs an ``old_log_prob`` phase to
  compare against; the off-policy arms have none, so the column was built every
  turn and never read.
* ``empty_cache()`` at the end of ``generate_sequences`` -- it forces a device
  synchronize and returns cached blocks to the driver, and it ran once per turn
  (~50x per rollout) even while the vLLM session held the engine awake, which is
  the state the session exists to keep.

Both are ports from the pure-OPD arm. The risk in the first is a reader that
assumed the column is always there, so that is what is tested here; the second is
a call that must not happen inside a session.
"""

import inspect
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "examples" / "sft_trainer" / "run_multitask_sft_qwen3.sh"

# Files that PRODUCE the column rather than read it.
_PRODUCERS = ("vllm_rollout_spmd.py", "vllm_rollout.py", "sglang_rollout.py")


def _reader_lines():
    """Every line outside the rollout backends that names the column."""
    out = []
    for path in sorted((REPO / "verl").rglob("*.py")) + sorted((REPO / "recipe").rglob("*.py")):
        if path.name in _PRODUCERS:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if "rollout_log_probs" in line:
                out.append((path, n, line))
    return out


def test_every_reader_of_the_column_checks_that_it_is_there():
    """With return_rollout_log_probs=False the key is absent from the batch, so an
    unguarded read is a KeyError in the middle of a run rather than at startup."""
    readers = _reader_lines()
    assert readers, "no readers found -- the search is looking in the wrong place"

    for path, lineno, line in readers:
        # A read is safe if it is the membership test itself, or sits under one.
        if 'in batch.batch.keys()' in line or "in batch.batch" in line:
            continue
        window = path.read_text().splitlines()[max(0, lineno - 6) : lineno]
        assert any("rollout_log_probs" in w and " in " in w for w in window), (
            f"{path.relative_to(REPO)}:{lineno} reads rollout_log_probs without a "
            f"membership check above it: {line.strip()}"
        )


def test_the_sft_run_script_turns_the_column_off():
    """This arm's loop has no old_log_prob phase; validation only scores rollouts."""
    lines = [
        line for line in RUN_SCRIPT.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any("rollout.return_rollout_log_probs=False" in line for line in lines)


def test_the_default_keeps_producing_it():
    """Arms that do run the drift check must be unaffected by the knob existing."""
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    config = OmegaConf.load(REPO / "verl" / "trainer" / "config" / "ppo_trainer.yaml")

    assert OmegaConf.select(config, "actor_rollout_ref.rollout.return_rollout_log_probs") is True


def test_the_kv_cache_release_is_skipped_inside_a_rollout_session():
    """empty_cache forces a device synchronize. Calling it per turn while the
    session deliberately holds vLLM awake pays that sync ~50x a rollout to release
    blocks the engine immediately takes back."""
    pytest.importorskip("torch")
    try:
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
    except Exception as e:  # pragma: no cover - environment without full deps
        pytest.skip(f"verl import unavailable: {e}")

    source = inspect.getsource(ActorRolloutRefWorker.generate_sequences)

    guard = re.search(
        r"if not getattr\(self, \"_rollout_session_active\", False\):\s*\n\s*"
        r"get_torch_device\(\)\.empty_cache\(\)",
        source,
    )
    assert guard, "generate_sequences releases the KV cache without checking for an open session"
