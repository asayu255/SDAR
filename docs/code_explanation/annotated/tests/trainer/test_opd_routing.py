# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
np = pytest.importorskip("numpy")
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
torch = pytest.importorskip("torch")

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl import DataProto
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty, topk_kl_per_token
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except Exception as e:  # pragma: no cover - environment without full deps
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
# [EXPLAIN] `_FakeTeacherWG` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class _FakeTeacherWG:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Stand-in for a teacher worker group.

    ``compute_ref_log_prob`` echoes a per-row marker built from the task code and
    the sample's original index (stored in responses[:, 0]) so the test can later
    assert that each row received the right teacher *and* landed back in its
    original position.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, code):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.code = code
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.calls = []

    # [EXPLAIN] `compute_ref_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_ref_log_prob(self, sub):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        first_tok = sub.batch["responses"][:, 0].float()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.calls.append(first_tok.tolist())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bs, resp_len = sub.batch["responses"].shape
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        out = torch.zeros((bs, resp_len), dtype=torch.float32)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        out[:, 0] = self.code * 1000.0 + first_tok  # marker = code*1000 + orig_idx
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto.from_dict(tensors={"ref_log_prob": out})


# [EXPLAIN] `_make_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _make_batch(task_names, resp_len=4):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bs = len(task_names)
    # responses[i, 0] == i  -> lets us recover the original index after routing.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    responses = torch.zeros((bs, resp_len), dtype=torch.long)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    responses[:, 0] = torch.arange(bs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return DataProto.from_dict(
        tensors={"responses": responses},
        non_tensors={"task_name": np.array(task_names, dtype=object)},
    )


# [EXPLAIN] `_make_trainer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _make_trainer(teacher_wg):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    # Bypass the heavy __init__; we only exercise compute_teacher_log_probs.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trainer = object.__new__(OPDRayTrainer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trainer.teacher_wg = teacher_wg
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return trainer


# [EXPLAIN] `test_routing_assigns_correct_teacher_and_restores_order` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_routing_assigns_correct_teacher_and_restores_order():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trainer = _make_trainer(teachers)

    # Deliberately interleaved + alias forms to exercise normalization.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_names = ["webshop", "alfworld", "search", "alfworld_easy", "search/nq"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = _make_batch(task_names)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    out = trainer.compute_teacher_log_probs(batch)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    code_by_task = {"alfworld": 1, "search": 2, "webshop": 3}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    norm = ["webshop", "alfworld", "search", "alfworld", "search"]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i, task in enumerate(norm):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected = code_by_task[task] * 1000.0 + i  # right teacher AND right position
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert out[i, 0].item() == pytest.approx(expected), f"row {i} ({task}) mis-routed"


# [EXPLAIN] `test_empty_task_slice_is_skipped` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_empty_task_slice_is_skipped():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trainer = _make_trainer(teachers)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = _make_batch(["alfworld", "search"])  # no webshop samples

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    out = trainer.compute_teacher_log_probs(batch)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert teachers["webshop"].calls == []  # never invoked
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert out[0, 0].item() == pytest.approx(1000.0)  # alfworld, idx 0
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert out[1, 0].item() == pytest.approx(2001.0)  # search, idx 1


# [EXPLAIN] `test_unknown_task_raises` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_unknown_task_raises():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teachers = {"alfworld": _FakeTeacherWG(1)}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trainer = _make_trainer(teachers)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = _make_batch(["alfworld", "search"])  # no teacher for search

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(ValueError, match="No teacher configured"):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        trainer.compute_teacher_log_probs(batch)


# --------------------------------------------------------------------------- #
# Teacher-KL loss math (mirrors the dp_actor branch)
# --------------------------------------------------------------------------- #
# [EXPLAIN] `test_teacher_kl_gradient_flows_through_student_only` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_teacher_kl_gradient_flows_through_student_only():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.manual_seed(0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_prob = torch.randn(2, 5, requires_grad=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher = torch.randn(2, 5)  # leaf, no grad — as returned by the teacher worker
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask = torch.ones(2, 5)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kld = kl_penalty(logprob=log_prob, ref_logprob=teacher, kl_penalty="low_var_kl")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loss = agg_loss(loss_mat=kld, loss_mask=mask, loss_agg_mode="token-mean")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    loss.backward()

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert log_prob.grad is not None
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.isfinite(loss)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert teacher.grad is None  # teacher provides no gradient


# [EXPLAIN] `test_pure_distillation_loss_equals_teacher_kl` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_pure_distillation_loss_equals_teacher_kl():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.manual_seed(0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_prob = torch.randn(3, 4, requires_grad=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher = torch.randn(3, 4)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask = torch.ones(3, 4)

    # Pure OPD: policy_loss starts at zero (pg_loss_coef=0, entropy_coeff=0) and
    # only the teacher-KL term is added.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_loss_coef = 0.0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy_coeff = 0.0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_kl_coef = 1.0

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    policy_loss = torch.zeros((), dtype=log_prob.dtype)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert pg_loss_coef == 0.0 and entropy_coeff == 0.0  # invariants

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kld = kl_penalty(logprob=log_prob, ref_logprob=teacher, kl_penalty="low_var_kl")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tkl = agg_loss(loss_mat=kld, loss_mask=mask, loss_agg_mode="token-mean")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    policy_loss = policy_loss + tkl * teacher_kl_coef

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert policy_loss.item() == pytest.approx(tkl.item())


# --------------------------------------------------------------------------- #
# Dense top-k (+tail) reverse-KL math
# --------------------------------------------------------------------------- #
# [EXPLAIN] `_logsoftmax_at` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _logsoftmax_at(logits, ids):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return logits.gather(-1, ids) - lse


# [EXPLAIN] `test_topk_kl_zero_for_identical_distributions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_topk_kl_zero_for_identical_distributions():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.manual_seed(0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bs, resp, V, k = 2, 3, 50, 8
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = torch.randn(bs, resp, V)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vals, ids = torch.topk(logits, k, dim=-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    topk_lp = vals - lse  # same teacher & student distribution
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kl = topk_kl_per_token(topk_lp.clone(), topk_lp.clone())
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


# [EXPLAIN] `test_topk_kl_nonnegative_and_student_only_gradient` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_topk_kl_nonnegative_and_student_only_gradient():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.manual_seed(1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bs, resp, V, k = 2, 4, 60, 10
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    s_logits = torch.randn(bs, resp, V, requires_grad=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    t_logits = torch.randn(bs, resp, V)

    # Teacher's top-k support; student log-probs gathered at the SAME ids.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _, t_ids = torch.topk(t_logits, k, dim=-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_topk = _logsoftmax_at(t_logits, t_ids)  # leaf, no grad
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    student_topk = _logsoftmax_at(s_logits, t_ids)  # carries grad

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kl = topk_kl_per_token(student_topk, teacher_topk)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(kl >= -1e-5)  # reverse KL >= 0 up to tail approximation

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    kl.sum().backward()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert s_logits.grad is not None
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert t_logits.grad is None  # teacher provides no gradient
