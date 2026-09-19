"""A val-only run is deterministic, and records how it scored (verl/utils/val_scoring.py)."""

import json
import os
import subprocess

import pytest
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_the_nondeterministic_pair_is_refused():
    from verl.utils.val_scoring import check_deterministic_scoring

    with pytest.raises(AssertionError, match="nondeterministic"):
        check_deterministic_scoring({"ROLLOUT_ASYNC_GENERATE": "1", "ROLLOUT_KEEP_VLLM_AWAKE": "1"})


@pytest.mark.parametrize("env", [
    {"ROLLOUT_ASYNC_GENERATE": "0", "ROLLOUT_KEEP_VLLM_AWAKE": "1"},   # the adopted scoring config
    {"ROLLOUT_ASYNC_GENERATE": "1", "ROLLOUT_KEEP_VLLM_AWAKE": "0"},   # either alone is byte-identical
    {},                                                                # code defaults: both off
])
def test_the_deterministic_configurations_pass(env):
    from verl.utils.val_scoring import check_deterministic_scoring

    assert check_deterministic_scoring(env).startswith("deterministic scoring")


def test_the_explicit_way_out_is_loud():
    from verl.utils.val_scoring import check_deterministic_scoring

    msg = check_deterministic_scoring({"ROLLOUT_ASYNC_GENERATE": "1", "ROLLOUT_KEEP_VLLM_AWAKE": "1",
                                       "VAL_ALLOW_NONDETERMINISTIC": "1"})
    assert msg.startswith("NONDETERMINISTIC")


def test_the_scoring_config_is_written_beside_the_dump(tmp_path):
    from verl.utils.val_scoring import write_scoring_config

    cfg = OmegaConf.create({
        "actor_rollout_ref": {"rollout": {"engine_kwargs": {"vllm": {"speculative_config": {"method": "ngram"}}},
                                          "enable_prefix_caching": True, "enforce_eager": False,
                                          "val_kwargs_by_task": {"webshop": {"temperature": 0.4}}},
                              "model": {"path": "Qwen/Qwen3-1.7B"}},
        "data": {"max_response_length": 512, "val_files": "x.parquet"},
        "trainer": {"n_gpus_per_node": 4, "resume_from_path": "/c/global_step_300"},
    })
    path = write_scoring_config(cfg, str(tmp_path / "dump"), {"ROLLOUT_ASYNC_GENERATE": "0",
                                                             "ROLLOUT_KEEP_VLLM_AWAKE": "1"})
    rec = json.load(open(path))
    assert rec["ROLLOUT_ASYNC_GENERATE"] == "0" and rec["ROLLOUT_KEEP_VLLM_AWAKE"] == "1"
    assert rec["speculative_config"] == {"method": "ngram"}
    assert rec["enable_prefix_caching"] is True and rec["enforce_eager"] is False
    assert rec["resume_from_path"] == "/c/global_step_300"


def test_the_guard_runs_in_build_and_fit_for_val_only():
    src = open(os.path.join(REPO, "verl/trainer/main_opd.py")).read()
    i = src.index("enforce_expected_config(config, expect_file")
    tail = src[i:i + 900]
    assert "check_deterministic_scoring()" in tail and "write_scoring_config(" in tail
    assert 'config.trainer.get("val_only", False)' in tail


def _launcher_env_for(tmp_path, val_only: bool, extra=None):
    """Run the control launcher's VAL_ONLY block in isolation and report ROLLOUT_ASYNC_GENERATE."""
    script = os.path.join(REPO, "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh")
    text = open(script).read()
    start = text.index("VAL_ONLY_ARGS=()")
    end = text.index("\nfi\n", start) + 4
    block = text[start:end]
    env = {"PATH": os.environ["PATH"], "ROLLOUT_ASYNC_GENERATE": "1", **(extra or {})}
    if val_only:
        # The block checks VAL_CKPT/actor exists, so point it at a real (empty) one.
        ckpt = tmp_path / "global_step_300"
        (ckpt / "actor").mkdir(parents=True)
        env.update({"VAL_ONLY": "1", "VAL_CKPT": str(ckpt)})
    out = subprocess.run(["bash", "-c", block + '\necho "ASYNC=$ROLLOUT_ASYNC_GENERATE"'], env=env,
                         capture_output=True, text=True)
    return out.stdout.strip().splitlines()[-1]


def test_the_launcher_turns_async_off_for_val_only(tmp_path):
    assert _launcher_env_for(tmp_path, val_only=True) == "ASYNC=0"


def test_the_launcher_leaves_training_alone(tmp_path):
    assert _launcher_env_for(tmp_path, val_only=False) == "ASYNC=1"


def test_the_launcher_honours_the_explicit_way_out(tmp_path):
    assert _launcher_env_for(tmp_path, val_only=True, extra={"VAL_ALLOW_NONDETERMINISTIC": "1"}) == "ASYNC=1"
