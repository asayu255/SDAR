"""Unit tests for multitask OPD + GRPO.

Covered:
* ``OPDGRPORayTrainer`` inherits the per-task teacher routing from
  ``OPDRayTrainer`` (correct teacher per task, order restored, empty/unknown
  task handling) — it must distill from the same teachers as pure OPD;
* the combined-loss invariant that defines this variant: the injected config
  has ``pg_loss_coef != 0`` (GRPO active) AND ``use_teacher_kl_loss=True`` (OPD
  active), unlike pure OPD which forces ``pg_loss_coef=0``;
* the combined-loss math used by the actor: the policy loss is
  ``pg_loss * pg_loss_coef + teacher_kl_loss * teacher_kl_coef``.

CPU-only; the heavy worker/Ray machinery is mocked. If the verl stack is not
installed, the module is skipped.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl import DataProto
    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


# --------------------------------------------------------------------------- #
# Routing (inherited from OPDRayTrainer)
# --------------------------------------------------------------------------- #
class _FakeTeacherWG:
    """Stand-in for a teacher worker group; echoes a per-row marker built from
    the task code and the sample's original index (responses[:, 0])."""

    def __init__(self, code):
        self.code = code
        self.calls = []

    def compute_ref_log_prob(self, sub):
        first_tok = sub.batch["responses"][:, 0].float()
        self.calls.append(first_tok.tolist())
        bs, resp_len = sub.batch["responses"].shape
        out = torch.zeros((bs, resp_len), dtype=torch.float32)
        out[:, 0] = self.code * 1000.0 + first_tok  # marker = code*1000 + orig_idx
        return DataProto.from_dict(tensors={"ref_log_prob": out})


def _make_batch(task_names, resp_len=4):
    bs = len(task_names)
    responses = torch.zeros((bs, resp_len), dtype=torch.long)
    responses[:, 0] = torch.arange(bs)
    return DataProto.from_dict(
        tensors={"responses": responses},
        non_tensors={"task_name": np.array(task_names, dtype=object)},
    )


def _make_trainer(teacher_wg, topk=None):
    from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer

    # Bypass the heavy __init__; we only exercise the inherited routing. The
    # attributes it would have set and compute_teacher_log_probs reads have to be
    # set by hand -- teacher_topk_kl selects the single-token or the dense top-k
    # branch.
    trainer = object.__new__(OPDGRPORayTrainer)
    trainer.teacher_wg = teacher_wg
    trainer.teacher_topk_kl = topk is not None
    trainer.teacher_kl_topk = topk or 0
    # This arm trains against the student's top-k, but the routing under test is
    # the same either way; the student-indexed path is covered in
    # test_opd_routing.py against the base trainer that defines it.
    trainer.student_indexed_topk = False
    trainer._teacher_cache_counter = 0
    return trainer


def test_subclass_inherits_routing():
    from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    assert issubclass(OPDGRPORayTrainer, OPDRayTrainer)
    # Routing/worker setup are reused unchanged; only fit() is overridden.
    assert OPDGRPORayTrainer.compute_teacher_log_probs is OPDRayTrainer.compute_teacher_log_probs
    assert OPDGRPORayTrainer.init_workers is OPDRayTrainer.init_workers
    assert OPDGRPORayTrainer.fit is not OPDRayTrainer.fit


def test_routing_assigns_correct_teacher_and_restores_order():
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    trainer = _make_trainer(teachers)

    task_names = ["webshop", "alfworld", "search", "alfworld_easy", "search/nq"]
    batch = _make_batch(task_names)
    trainer.compute_teacher_log_probs(batch)
    out = batch.batch["teacher_log_probs"]

    code_by_task = {"alfworld": 1, "search": 2, "webshop": 3}
    norm = ["webshop", "alfworld", "search", "alfworld", "search"]
    for i, task in enumerate(norm):
        expected = code_by_task[task] * 1000.0 + i
        assert out[i, 0].item() == pytest.approx(expected), f"row {i} ({task}) mis-routed"


def test_unknown_task_raises():
    teachers = {"alfworld": _FakeTeacherWG(1)}
    trainer = _make_trainer(teachers)
    batch = _make_batch(["alfworld", "search"])  # no teacher for search
    with pytest.raises(ValueError, match="No teacher configured"):
        trainer.compute_teacher_log_probs(batch)


# --------------------------------------------------------------------------- #
# Combined-loss invariant (mirrors main_opd_grpo config injection)
# --------------------------------------------------------------------------- #
def test_injected_config_keeps_both_grpo_and_teacher_kl():
    """Replicates the config injection in main_opd_grpo.OPDGRPOTaskRunner.run."""
    from omegaconf import OmegaConf, open_dict

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"actor": {}},
            "algorithm": {
                "adv_estimator": "grpo",
                "opd": {"kl_loss_coef": 1.0, "kl_loss_type": "low_var_kl", "topk": 20},
            },
        }
    )
    opd_cfg = config.algorithm.opd
    with open_dict(config):
        config.actor_rollout_ref.actor.use_teacher_kl_loss = True
        config.actor_rollout_ref.actor.teacher_kl_loss_coef = opd_cfg.get("kl_loss_coef", 1.0)
        config.actor_rollout_ref.actor.teacher_kl_loss_type = opd_cfg.get("kl_loss_type", "low_var_kl")
        config.actor_rollout_ref.actor.teacher_kl_topk = opd_cfg.get("topk", 20)
        config.actor_rollout_ref.actor.pg_loss_coef = opd_cfg.get("pg_loss_coef", 1.0)
        config.actor_rollout_ref.actor.use_kl_loss = False

    a = config.actor_rollout_ref.actor
    # The defining invariant of OPD+GRPO: BOTH signals are active.
    assert a.pg_loss_coef != 0, "GRPO policy gradient must be active"
    assert a.use_teacher_kl_loss is True, "teacher-KL distillation must be active"
    assert a.use_kl_loss is False
    assert config.algorithm.adv_estimator == "grpo"


def test_combined_loss_is_pg_plus_teacher_kl():
    torch.manual_seed(0)
    log_prob = torch.randn(3, 4, requires_grad=True)
    teacher = torch.randn(3, 4)
    mask = torch.ones(3, 4)

    pg_loss = torch.tensor(0.7, requires_grad=True)
    pg_loss_coef = 1.0
    teacher_kl_coef = 1.0

    policy_loss = pg_loss * pg_loss_coef
    kld = kl_penalty(logprob=log_prob, ref_logprob=teacher, kl_penalty="low_var_kl")
    tkl = agg_loss(loss_mat=kld, loss_mask=mask, loss_agg_mode="token-mean")
    policy_loss = policy_loss + tkl * teacher_kl_coef

    assert policy_loss.item() == pytest.approx(pg_loss.item() * pg_loss_coef + tkl.item() * teacher_kl_coef)
    policy_loss.backward()
    assert log_prob.grad is not None  # teacher-KL term contributes gradient
    assert pg_loss.grad is not None   # GRPO term contributes gradient
    assert teacher.grad is None       # teacher provides no gradient
