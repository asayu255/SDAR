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

    def __init__(self, code, world_size=1):
        self.code = code
        self.calls = []
        # student_indexed_topk pads explicitly to this instead of letting the DP
        # dispatch do it, so that padding rows can carry a -1 cache id.
        self.world_size = world_size

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
        if "teacher_cache_ids" in sub.batch.keys():
            # The real worker caches the hidden states here and hands back only
            # the row count; the values are resolved later in the actor.
            self.cached = sub.batch["teacher_cache_ids"].tolist()
            return DataProto.from_dict(tensors={"teacher_scored": torch.ones(bs, 1, dtype=torch.bool)})
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


def _make_trainer(teacher_wg, topk=None, student_indexed=False):
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    # Bypass the heavy __init__; we only exercise compute_teacher_log_probs. The
    # attributes it would have set and this method reads have to be set by hand --
    # teacher_topk_kl selects the single-token or the dense top-k branch, and
    # student_indexed_topk decides whether the teacher's own top-k comes back at
    # all or only a cache key does.
    trainer = object.__new__(OPDRayTrainer)
    trainer.teacher_wg = teacher_wg
    trainer.teacher_topk_kl = topk is not None
    trainer.teacher_kl_topk = topk or 0
    trainer.student_indexed_topk = bool(topk is not None and student_indexed)
    trainer._teacher_cache_counter = 0
    # This file is about the ROUTING, which is the same on every arm; the
    # cross-teacher planes have their own file. __init__ always sets the flag,
    # so False here is the plain-OPD trainer rather than a hole in the stub.
    trainer.cross_teacher_enabled = False
    trainer.base_wg = None
    # See test_sign_weight_routing._make_trainer: 0 keeps the worker's own row
    # bound, which is the behaviour every test in this file was written against.
    trainer._post_rollout_token_budget = 0
    # The window path's own budget, off by default -- the 4-row bound there
    # was bought with an OOM beside a live KV pool.
    trainer._window_forward_token_budget = 0
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


# --------------------------------------------------------------------------- #
# Teacher prefetch (ROLLOUT_PREFETCH_TEACHER)
# --------------------------------------------------------------------------- #
def _make_batch_with_ids(task_names, traj_uids, turn_steps, resp_len=4):
    """A batch that also carries the (traj_uid, turn_step) identity a prefetch keys on."""
    batch = _make_batch(task_names, resp_len=resp_len)
    batch.non_tensor_batch["traj_uid"] = np.array(traj_uids, dtype=object)
    batch.non_tensor_batch["turn_step"] = np.array(turn_steps, dtype=object)
    return batch


def _prefetch_entry(marker, resp_len=4, k=20):
    """A prefetched top-k result carrying `marker`, in the same slot the fakes use."""
    logprobs = torch.zeros((resp_len, k), dtype=torch.float32)
    ids = torch.zeros((resp_len, k), dtype=torch.long)
    logprobs[0, 0] = marker
    ids[0, 0] = int(marker)
    return logprobs, ids


def _make_row(idx, task, resp_len=4, seqlen=6):
    """A rollout row as the loop queues it: four model inputs plus its task."""
    responses = torch.zeros(resp_len, dtype=torch.long)
    responses[0] = idx
    return {
        "input_ids": torch.zeros(seqlen, dtype=torch.long),
        "attention_mask": torch.ones(seqlen, dtype=torch.long),
        "position_ids": torch.arange(seqlen, dtype=torch.long),
        "responses": responses,
        "task_name": task,
    }


def test_prefetch_chunk_routes_each_row_to_its_task_teacher():
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    trainer = _make_trainer(teachers, topk=20)

    chunk = [
        (("t0", 0), _make_row(0, "webshop")),
        (("t1", 0), _make_row(1, "alfworld")),
        (("t1", 1), _make_row(2, "alfworld_easy")),  # alias normalises to alfworld
        (("t2", 3), _make_row(3, "search/nq")),
    ]
    out = trainer._teacher_prefetch_chunk(chunk)

    assert set(out) == {("t0", 0), ("t1", 0), ("t1", 1), ("t2", 3)}
    code_by_key = {("t0", 0): 3, ("t1", 0): 1, ("t1", 1): 1, ("t2", 3): 2}
    idx_by_key = {("t0", 0): 0, ("t1", 0): 1, ("t1", 1): 2, ("t2", 3): 3}
    for key, (logprobs, ids, _cache_id) in out.items():
        expected = code_by_key[key] * 1000.0 + idx_by_key[key]
        assert logprobs[0, 0].item() == pytest.approx(expected), f"{key} mis-routed"
    # one call per task present, and alfworld's two rows travelled together
    assert teachers["alfworld"].calls == [[1.0, 2.0]]
    assert teachers["search"].calls == [[3.0]]
    assert teachers["webshop"].calls == [[0.0]]


def test_the_student_indexed_prefetch_brings_back_only_a_cache_key():
    """The support is the student's, so the teacher's own top-k is resolved
    nowhere downstream. Shipping (rows, response_length, k) log-probs plus the
    same again in int64 ids was ~860 MB a step of pure transport."""
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    trainer = _make_trainer(teachers, topk=20, student_indexed=True)

    chunk = [
        (("t0", 0), _make_row(0, "webshop")),
        (("t1", 0), _make_row(1, "alfworld")),
        (("t1", 1), _make_row(2, "alfworld_easy")),
    ]
    out = trainer._teacher_prefetch_chunk(chunk)

    assert set(out) == {("t0", 0), ("t1", 0), ("t1", 1)}
    # Plain ints, and distinct: the key is what locates the row's hidden states.
    assert all(isinstance(v, int) for v in out.values())
    assert len(set(out.values())) == 3
    # The teachers still ran -- this removes the return trip, not the forward.
    assert teachers["alfworld"].calls == [[1.0, 2.0]]
    assert teachers["webshop"].calls == [[0.0]]


def test_the_student_indexed_batch_carries_keys_instead_of_the_top_k_columns():
    """Merged into the batch: one int64 per row, not two (bs, response_length, k)
    tensors -- and every row gets one, prefetch hit or miss."""
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    trainer = _make_trainer(teachers, topk=20, student_indexed=True)

    batch = _make_batch_with_ids(
        ["alfworld", "alfworld", "search"], traj_uids=["a", "a", "b"], turn_steps=[0, 1, 0]
    )
    trainer.compute_teacher_log_probs(batch, prefetched={("a", 0): 11})

    assert "teacher_topk_logprobs" not in batch.batch.keys()
    assert "teacher_topk_ids" not in batch.batch.keys()
    keys = batch.batch["teacher_cache_ids"]
    assert keys[0].item() == 11                       # the prefetched row kept its key
    assert (keys > 0).all()                           # ...and the misses got fresh ones
    assert len(set(keys.tolist())) == 3


def test_prefetched_rows_are_not_rescored():
    """A hit fills the row and keeps it out of the per-task teacher call."""
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    trainer = _make_trainer(teachers, topk=20)

    batch = _make_batch_with_ids(
        task_names=["alfworld", "alfworld", "search"],
        traj_uids=["a", "a", "b"],
        turn_steps=[0, 1, 0],
    )
    # Row 1 only. 77777 cannot collide with a fake teacher's code*1000 + idx.
    prefetched = {("a", 1): _prefetch_entry(77777.0)}

    trainer.compute_teacher_log_probs(batch, prefetched=prefetched)
    logprobs = batch.batch["teacher_topk_logprobs"]

    assert logprobs[1, 0, 0].item() == pytest.approx(77777.0)  # came from the prefetch
    assert logprobs[0, 0, 0].item() == pytest.approx(1000.0)   # alfworld teacher, idx 0
    assert logprobs[2, 0, 0].item() == pytest.approx(2002.0)   # search teacher, idx 2
    # the alfworld teacher saw row 0 alone -- row 1 was already done
    assert teachers["alfworld"].calls == [[0.0]]


def test_a_fully_prefetched_task_skips_its_teacher_entirely():
    teachers = {"alfworld": _FakeTeacherWG(1), "search": _FakeTeacherWG(2), "webshop": _FakeTeacherWG(3)}
    trainer = _make_trainer(teachers, topk=20)

    batch = _make_batch_with_ids(["search", "alfworld"], ["s", "a"], [0, 0])
    prefetched = {("s", 0): _prefetch_entry(55555.0)}

    trainer.compute_teacher_log_probs(batch, prefetched=prefetched)

    assert teachers["search"].calls == []       # never invoked
    assert teachers["alfworld"].calls == [[1.0]]
    assert batch.batch["teacher_topk_logprobs"][0, 0, 0].item() == pytest.approx(55555.0)


def test_adjust_batch_duplicates_reuse_one_prefetched_entry():
    """adjust_batch copies rows; a copy shares its original's key, so both hit."""
    teachers = {"alfworld": _FakeTeacherWG(1)}
    trainer = _make_trainer(teachers, topk=20)

    # rows 0 and 2 are the same (traj_uid, turn_step) -- 2 is the appended copy.
    batch = _make_batch_with_ids(
        task_names=["alfworld", "alfworld", "alfworld"],
        traj_uids=["a", "a", "a"],
        turn_steps=[0, 1, 0],
    )
    prefetched = {("a", 0): _prefetch_entry(99999.0)}

    trainer.compute_teacher_log_probs(batch, prefetched=prefetched)
    logprobs = batch.batch["teacher_topk_logprobs"]

    assert logprobs[0, 0, 0].item() == pytest.approx(99999.0)
    assert logprobs[2, 0, 0].item() == pytest.approx(99999.0)
    assert teachers["alfworld"].calls == [[1.0]]  # only the un-prefetched row


def test_a_stale_prefetch_key_is_ignored():
    """Keys with no row in this batch (filtered groups, a later step) change nothing."""
    teachers = {"alfworld": _FakeTeacherWG(1)}
    trainer = _make_trainer(teachers, topk=20)

    batch = _make_batch_with_ids(["alfworld"], ["a"], [0])
    trainer.compute_teacher_log_probs(batch, prefetched={("gone", 7): _prefetch_entry(1.0)})

    assert teachers["alfworld"].calls == [[0.0]]
    assert batch.batch["teacher_topk_logprobs"][0, 0, 0].item() == pytest.approx(1000.0)


def test_batch_without_turn_identity_ignores_the_prefetch():
    """Validation batches carry no traj_uid/turn_step; the merge must stay off."""
    teachers = {"alfworld": _FakeTeacherWG(1)}
    trainer = _make_trainer(teachers, topk=20)

    batch = _make_batch(["alfworld"])  # no traj_uid / turn_step
    trainer.compute_teacher_log_probs(batch, prefetched={("a", 0): _prefetch_entry(1.0)})

    assert teachers["alfworld"].calls == [[0.0]]  # scored normally
    assert batch.batch["teacher_topk_logprobs"][0, 0, 0].item() == pytest.approx(1000.0)


def test_prefetch_metrics_report_the_hit_rate():
    teachers = {"alfworld": _FakeTeacherWG(1)}
    trainer = _make_trainer(teachers, topk=20)

    batch = _make_batch_with_ids(["alfworld"] * 4, ["a"] * 4, [0, 1, 2, 3])
    prefetched = {("a", 0): _prefetch_entry(1.0), ("a", 2): _prefetch_entry(2.0)}
    metrics = {}

    trainer.compute_teacher_log_probs(batch, prefetched=prefetched, metrics=metrics)

    assert metrics["teacher_prefetch/rows"] == 2
    assert metrics["teacher_prefetch/hit_rate"] == pytest.approx(0.5)


def test_single_token_branch_also_merges_the_prefetch():
    """kl_loss_type != topk_kl stores one value per token, not a (logprob, id) pair."""
    teachers = {"alfworld": _FakeTeacherWG(1)}
    trainer = _make_trainer(teachers)  # topk=None -> single-token estimator

    batch = _make_batch_with_ids(["alfworld", "alfworld"], ["a", "a"], [0, 1])
    row = torch.zeros(4, dtype=torch.float32)
    row[0] = 42424.0

    trainer.compute_teacher_log_probs(batch, prefetched={("a", 1): row})
    out = batch.batch["teacher_log_probs"]

    assert out[1, 0].item() == pytest.approx(42424.0)
    assert out[0, 0].item() == pytest.approx(1000.0)
    assert teachers["alfworld"].calls == [[0.0]]
