"""The tied_opsd launcher composes, satisfies its own lock, and defines the arm the design asks for."""
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("hydra")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = "examples/opd_grpo_trainer/run_multitask_tied_opsd_qwen3.sh"


def _cfg(monkeypatch):
    from hydra import compose, initialize_config_dir

    from tests.trainer.test_run_script_overrides_compose import _overrides
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    monkeypatch.setenv("RUN_TAG_SUFFIX", "")
    monkeypatch.delenv("LAM", raising=False)
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "verl", "trainer", "config")):
        cfg = compose(config_name="ppo_trainer", overrides=list(_overrides(SCRIPT)))
    inject_opd_grpo_config(cfg)
    return cfg


def test_launcher_satisfies_its_lock_and_defines_the_arm(monkeypatch):
    from verl.utils.expected_config import check_expected_config

    cfg = _cfg(monkeypatch)
    assert cfg.trainer.expected_config.endswith("expected_multitask_tied_opsd_config.yaml")
    assert check_expected_config(cfg, os.path.join(REPO, cfg.trainer.expected_config)) == []
    alg, actor = cfg.algorithm, cfg.actor_rollout_ref.actor
    assert alg.tied_opsd.enable and actor.tied_opsd
    assert dict(alg.opd.teacher_paths) == {} and not actor.use_teacher_kl_loss and not actor.use_sdar_loss
    assert actor.normalize_loss_by_task and alg.adv_estimator == "grpo"
    assert not alg.progress_rank.enable and not alg.oci_slots.enable and not alg.opsd.enable
    assert (alg.tied_opsd.topk, alg.tied_opsd.refresh_every, alg.tied_opsd.retention, alg.tied_opsd.guard) == (
        20, 50, 0.8, True)
    assert alg.oci_slots.search_flow_path.endswith("route_hints_v2.json")
    assert int(cfg.env.rollout.n) == 8


def test_injection_refuses_a_teacher_left_in(monkeypatch):
    from omegaconf import open_dict

    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    cfg = _cfg(monkeypatch)
    with open_dict(cfg):
        cfg.algorithm.opd.teacher_paths = {"alfworld": "/x"}
    with pytest.raises(AssertionError, match="teacher_paths"):
        inject_opd_grpo_config(cfg)
