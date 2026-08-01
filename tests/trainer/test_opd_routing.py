"""Unit tests for multitask OPD (on-policy distillation).

Covered:
* per-task teacher routing in ``OPDRayTrainer.compute_teacher_log_probs``
  (correct teacher per task, order restored to original batch order,
  unknown / empty task handling);
* the teacher-KL loss math used by the actor (gradient flows through the
  student only; pure-distillation loss equals the teacher KL term).

These exercise CPU-only logic; the heavy worker/Ray machinery is mocked.
If the verl stack (numpy/torch/...) is not installed, the module is skipped.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl import DataProto
    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty, topk_kl_per_token
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
class _FakeTeacherWG:
    """Stand-in for a teacher worker group.

    ``compute_ref_log_prob`` echoes a per-row marker built from the task code and
    the sample's original index (stored in responses[:, 0]) so the test can later
    assert that each row received the right teacher *and* landed back in its
    original position.
    """

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

    def compute_ref_topk_log_prob(self, sub):
        """Same marker scheme for the dense top-k branch (kl_loss_type=topk_kl)."""
        first_tok = sub.batch["responses"][:, 0].float()
        self.calls.append(first_tok.tolist())
        bs, resp_len = sub.batch["responses"].shape
        k = sub.meta_info["topk_k"]
        logprobs = torch.zeros((bs, resp_len, k), dtype=torch.float32)
        ids = torch.zeros((bs, resp_len, k), dtype=torch.long)
        logprobs[:, 0, 0] = self.code * 1000.0 + first_tok
        ids[:, 0, 0] = self.code * 1000 + first_tok.long()
        return DataProto.from_dict(
            tensors={"teacher_topk_logprobs": logprobs, "teacher_topk_ids": ids}
        )


def _make_batch(task_names, resp_len=4):
    bs = len(task_names)
    # responses[i, 0] == i  -> lets us recover the original index after routing.
    responses = torch.zeros((bs, resp_len), dtype=torch.long)
    responses[:, 0] = torch.arange(bs)
    return DataProto.from_dict(
        tensors={"responses": responses},
        non_tensors={"task_name": np.array(task_names, dtype=object)},
    )


def _make_trainer(teacher_wg, topk=None):
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    # Bypass the heavy __init__; we only exercise compute_teacher_log_probs. The
    # attributes it would have set and this method reads have to be set by hand --
    # teacher_topk_kl selects the single-token or the dense top-k branch.
    trainer = object.__new__(OPDRayTrainer)
    trainer.teacher_wg = teacher_wg
    trainer.teacher_topk_kl = topk is not None
    trainer.teacher_kl_topk = topk or 0
    return trainer


def test_routing_assigns_correct_teacher_and_restores_order():
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    trainer = _make_trainer(teachers)

    # Deliberately interleaved + alias forms to exercise normalization.
    task_names = ["webshop", "alfworld", "search", "alfworld_easy", "search/nq"]
    batch = _make_batch(task_names)

    assert trainer.compute_teacher_log_probs(batch) is None  # writes into batch
    out = batch.batch["teacher_log_probs"]

    code_by_task = {"alfworld": 1, "search": 2, "webshop": 3}
    norm = ["webshop", "alfworld", "search", "alfworld", "search"]
    for i, task in enumerate(norm):
        expected = code_by_task[task] * 1000.0 + i  # right teacher AND right position
        assert out[i, 0].item() == pytest.approx(expected), f"row {i} ({task}) mis-routed"


def test_empty_task_slice_is_skipped():
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    trainer = _make_trainer(teachers)
    batch = _make_batch(["alfworld", "search"])  # no webshop samples

    trainer.compute_teacher_log_probs(batch)
    out = batch.batch["teacher_log_probs"]

    assert teachers["webshop"].calls == []  # never invoked
    assert out[0, 0].item() == pytest.approx(1000.0)  # alfworld, idx 0
    assert out[1, 0].item() == pytest.approx(2001.0)  # search, idx 1


def test_topk_routing_assigns_correct_teacher_and_restores_order():
    """kl_loss_type=topk_kl takes a separate branch, with its own output tensors."""
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    trainer = _make_trainer(teachers, topk=20)

    task_names = ["webshop", "alfworld", "search", "alfworld_easy", "search/nq"]
    batch = _make_batch(task_names)

    trainer.compute_teacher_log_probs(batch)

    assert "teacher_log_probs" not in batch.batch  # the single-token branch stayed out
    logprobs = batch.batch["teacher_topk_logprobs"]
    ids = batch.batch["teacher_topk_ids"]
    assert logprobs.shape == (5, 4, 20)
    assert ids.shape == (5, 4, 20)

    code_by_task = {"alfworld": 1, "search": 2, "webshop": 3}
    norm = ["webshop", "alfworld", "search", "alfworld", "search"]
    for i, task in enumerate(norm):
        expected = code_by_task[task] * 1000.0 + i  # right teacher AND right position
        assert logprobs[i, 0, 0].item() == pytest.approx(expected), f"row {i} ({task}) mis-routed"
        assert ids[i, 0, 0].item() == int(expected)


def test_unknown_task_raises():
    teachers = {"alfworld": _FakeTeacherWG(1)}
    trainer = _make_trainer(teachers)
    batch = _make_batch(["alfworld", "search"])  # no teacher for search

    with pytest.raises(ValueError, match="No teacher configured"):
        trainer.compute_teacher_log_probs(batch)


# --------------------------------------------------------------------------- #
# Teacher-KL loss math (mirrors the dp_actor branch)
# --------------------------------------------------------------------------- #
def test_teacher_kl_gradient_flows_through_student_only():
    torch.manual_seed(0)
    log_prob = torch.randn(2, 5, requires_grad=True)
    teacher = torch.randn(2, 5)  # leaf, no grad — as returned by the teacher worker
    mask = torch.ones(2, 5)

    kld = kl_penalty(logprob=log_prob, ref_logprob=teacher, kl_penalty="low_var_kl")
    loss = agg_loss(loss_mat=kld, loss_mask=mask, loss_agg_mode="token-mean")
    loss.backward()

    assert log_prob.grad is not None
    assert torch.isfinite(loss)
    assert teacher.grad is None  # teacher provides no gradient


def test_pure_distillation_loss_equals_teacher_kl():
    torch.manual_seed(0)
    log_prob = torch.randn(3, 4, requires_grad=True)
    teacher = torch.randn(3, 4)
    mask = torch.ones(3, 4)

    # Pure OPD: policy_loss starts at zero (pg_loss_coef=0, entropy_coeff=0) and
    # only the teacher-KL term is added.
    pg_loss_coef = 0.0
    entropy_coeff = 0.0
    teacher_kl_coef = 1.0

    policy_loss = torch.zeros((), dtype=log_prob.dtype)
    assert pg_loss_coef == 0.0 and entropy_coeff == 0.0  # invariants

    kld = kl_penalty(logprob=log_prob, ref_logprob=teacher, kl_penalty="low_var_kl")
    tkl = agg_loss(loss_mat=kld, loss_mask=mask, loss_agg_mode="token-mean")
    policy_loss = policy_loss + tkl * teacher_kl_coef

    assert policy_loss.item() == pytest.approx(tkl.item())


# --------------------------------------------------------------------------- #
# Dense top-k (+tail) reverse-KL math
# --------------------------------------------------------------------------- #
def _logsoftmax_at(logits, ids):
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    return logits.gather(-1, ids) - lse


def test_topk_kl_zero_for_identical_distributions():
    torch.manual_seed(0)
    bs, resp, V, k = 2, 3, 50, 8
    logits = torch.randn(bs, resp, V)
    vals, ids = torch.topk(logits, k, dim=-1)
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    topk_lp = vals - lse  # same teacher & student distribution
    kl = topk_kl_per_token(topk_lp.clone(), topk_lp.clone())
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_topk_kl_nonnegative_and_student_only_gradient():
    torch.manual_seed(1)
    bs, resp, V, k = 2, 4, 60, 10
    s_logits = torch.randn(bs, resp, V, requires_grad=True)
    t_logits = torch.randn(bs, resp, V)

    # Teacher's top-k support; student log-probs gathered at the SAME ids.
    _, t_ids = torch.topk(t_logits, k, dim=-1)
    teacher_topk = _logsoftmax_at(t_logits, t_ids)  # leaf, no grad
    student_topk = _logsoftmax_at(s_logits, t_ids)  # carries grad

    kl = topk_kl_per_token(student_topk, teacher_topk)
    assert torch.all(kl >= -1e-5)  # reverse KL >= 0 up to tail approximation

    kl.sum().backward()
    assert s_logits.grad is not None
    assert t_logits.grad is None  # teacher provides no gradient
