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
"""A lock file must match the run script it locks — checked here, not on the GPU.

``test_expected_config.py`` checks the lock against a config assembled from its
own keys, which catches a typo but not a lock that disagrees with the script.
That disagreement is cheap to make (the point of the lock is that changing one
requires changing the other) and expensive to discover: the run aborts at
startup, after the queue wait.

So this composes the script's *actual* config and validates the lock against it.
Neither half is reimplemented:

* the overrides come from running the script under a stub ``python3`` on PATH,
  so bash does its own variable expansion, defaulting and quoting;
* the config comes from Hydra composing ``ppo_trainer.yaml`` with those
  overrides, so ``+key=`` and the inline dict/list forms parse exactly as they
  will at run time.

The entry point's config injection is replayed too, because the lock is
validated after it — that is what lets the lock pin
``actor_rollout_ref.actor.sdar_loss_coef`` and not just its
``algorithm.sdar.sdar_coef`` source.
"""

import os
import pathlib
import stat
import subprocess

import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

try:
    from verl.utils.expected_config import check_expected_config, load_expectations
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "verl" / "trainer" / "config"

# (run script, entry-point module, lock file). One row per locked experiment.
LOCKED_RUNS = [
    (
        "examples/sdar_trainer/run_multitask_qwen3.sh",
        "verl.trainer.main_sdar",
        "examples/sdar_trainer/expected_multitask_config.yaml",
    ),
]


def _stub_python3(tmp_path, entry_module):
    """A `python3` that prints the training command's arguments and runs nothing.

    The script calls python3 three times -- a tokenizer preflight, the data prep,
    and the trainer. Only the last one's arguments are wanted, and none of the
    three should actually execute (the first two hit the network and the disk).
    """
    shim = tmp_path / "python3"
    shim.write_text(
        "#!/bin/bash\n"
        "for a in \"$@\"; do\n"
        f'  if [ "$a" = "{entry_module}" ]; then\n'
        "    shift 2\n"
        '    printf "%s\\n" "$@"\n'
        "    exit 0\n"
        "  fi\n"
        "done\n"
        "exit 0\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return shim


def _overrides_from(script, entry_module, tmp_path):
    _stub_python3(tmp_path, entry_module)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    result = subprocess.run(
        ["bash", str(REPO / script)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    overrides = [line for line in result.stdout.splitlines() if line.strip()]
    assert overrides, (
        f"no trainer arguments recovered from {script}; the stub never saw "
        f"{entry_module} (stderr tail: {result.stderr[-400:]})"
    )
    return overrides


def _inject_entry_point_config(cfg, entry_module):
    """Replay what the entry point writes into the config before the trainer runs.

    Kept in step with the module by hand, which is the same coupling the lock
    itself has: if an injection changes and this does not, the pinned effective
    values stop matching and this test says so.
    """
    if entry_module == "verl.trainer.main_sdar":
        sdar = cfg.algorithm.get("sdar", {})
        with open_dict(cfg):
            cfg.actor_rollout_ref.actor.use_sdar_loss = True
            cfg.actor_rollout_ref.actor.sdar_loss_coef = sdar.get("sdar_coef", 0.1)
            cfg.actor_rollout_ref.actor.sdar_gate_beta = sdar.get("gate_beta", 5.0)
            if sdar.get("kl_loss_coef_by_task", None) is not None:
                cfg.actor_rollout_ref.actor.kl_loss_coef_by_task = sdar.get("kl_loss_coef_by_task")
    return cfg


def _effective_config(script, entry_module, tmp_path):
    overrides = _overrides_from(script, entry_module, tmp_path)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    return _inject_entry_point_config(cfg, entry_module)


@pytest.mark.parametrize("script, entry_module, lock", LOCKED_RUNS)
def test_the_lock_matches_the_script_it_locks(script, entry_module, lock, tmp_path):
    cfg = _effective_config(script, entry_module, tmp_path)
    mismatches = check_expected_config(cfg, str(REPO / lock))
    assert not mismatches, "\n".join(
        f"  {key}: script gives {got!r}, {lock} expects {want!r}"
        for key, got, want in mismatches
    )


@pytest.mark.parametrize("script, entry_module, lock", LOCKED_RUNS)
def test_the_script_actually_passes_its_lock(script, entry_module, lock):
    """A lock nothing points at protects nothing."""
    text = (REPO / script).read_text()
    assert f"trainer.expected_config={lock}" in text, (
        f"{script} does not pass +trainer.expected_config={lock}, so the lock is "
        f"never read at run time"
    )


@pytest.mark.parametrize("script, entry_module, lock", LOCKED_RUNS)
def test_a_drifted_script_would_be_caught(script, entry_module, lock, tmp_path):
    """Negative control: the comparison has to be able to fail.

    Without this, a lock whose keys had all silently gone missing from the config
    would still pass the test above if the mismatch list were built wrong.
    """
    cfg = _effective_config(script, entry_module, tmp_path)
    expectations = load_expectations(str(REPO / lock))
    drift_key = next(k for k, v in expectations.items() if not isinstance(v, (dict, list)))
    OmegaConf.update(cfg, drift_key, "___DRIFTED___", force_add=True)

    mismatched = {key for key, _, _ in check_expected_config(cfg, str(REPO / lock))}
    assert mismatched == {drift_key}, mismatched


@pytest.mark.parametrize("script, entry_module, lock", LOCKED_RUNS)
def test_the_lock_stays_out_of_the_machine_knobs(script, entry_module, lock):
    """Pinning a throughput setting would make tuning a host a two-file edit.

    The two gradient-path exceptions are named explicitly, so adding a third has
    to be a deliberate edit here rather than a drive-by line in the yaml.
    """
    allowed_perf_keys = {
        "actor_rollout_ref.actor.fsdp_config.sharding_strategy",
        "actor_rollout_ref.actor.no_sync_grad_accum",
        "actor_rollout_ref.actor.use_dynamic_bsz",
    }
    forbidden = (
        "micro_batch_size",
        "gpu_memory_utilization",
        "max_model_len",
        "param_offload",
        "optimizer_offload",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "n_gpus_per_node",
        "nnodes",
        "default_local_dir",
        "train_files",
        "val_files",
        "save_freq",
        "use_fused_kernels",
    )
    offenders = [
        key
        for key in load_expectations(str(REPO / lock))
        if key not in allowed_perf_keys and any(f in key for f in forbidden)
    ]
    assert not offenders, (
        f"{lock} pins machine/throughput settings, so tuning a host would require "
        f"editing it: {offenders}"
    )
