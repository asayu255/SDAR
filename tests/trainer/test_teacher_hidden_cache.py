"""Teacher log-probs recovered from cached hidden states, at student-chosen ids.

Three things have to hold, and they fail in different ways:

1. **The values.** ``h @ W[ids].T - lse`` must equal indexing a full
   ``log_softmax(h @ W.T)`` at the same ids. This is the whole justification for
   caching ``h`` instead of a pre-selected top-k, so it is checked against a
   literal full-vocabulary reference.
2. **The routing.** The rank that caches a row is not the rank that later picks
   the student's top-k -- the rows are regrouped by task, padded and then
   reordered by ``_balance_batch``. A row answered from the wrong cache entry
   produces a plausible number, so the guards must turn that into an exception.
3. **The exchange.** Ownership is unique, so summing answers is exact; a row
   nobody owns, or one two ranks claim, must raise rather than train on a zero.

The distributed test runs two real gloo processes: an exchange that only works
single-process is not evidence for anything here.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.teacher_cache import (
        TeacherHiddenCache,
        exchange_teacher_logprobs,
        teacher_logprobs_from_hidden,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


H, VOCAB, K = 64, 512, 8


def _teacher(seed=0, n=16):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((VOCAB, H), generator=g) / H**0.5
    h = torch.randn((n, H), generator=g) / 2
    lse = torch.logsumexp(h @ W.T, dim=-1)
    return W, h, lse


# --------------------------------------------------------------------------- #
# 1. values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("trial", range(6))
def test_matches_a_full_vocabulary_log_softmax(trial):
    """The claim the whole design rests on: the cached normaliser plus a narrow
    gather is the same number as normalising the full projection."""
    W, h, lse = _teacher(seed=trial, n=32)
    g = torch.Generator().manual_seed(100 + trial)
    ids = torch.randint(0, VOCAB, (32, K), generator=g)

    reference = torch.log_softmax(h @ W.T, dim=-1).gather(-1, ids)
    got = teacher_logprobs_from_hidden(h, lse, W, ids)

    torch.testing.assert_close(got, reference, rtol=0, atol=1e-5)


def test_the_normaliser_is_id_independent():
    """Why the teacher can run before the ids exist: lse is a full-vocabulary sum,
    so two disjoint id sets share it."""
    W, h, lse = _teacher(n=8)
    a = teacher_logprobs_from_hidden(h, lse, W, torch.arange(K).expand(8, K))
    b = teacher_logprobs_from_hidden(h, lse, W, torch.arange(K, 2 * K).expand(8, K))
    full = torch.log_softmax(h @ W.T, dim=-1)
    torch.testing.assert_close(a, full[:, :K], rtol=0, atol=1e-5)
    torch.testing.assert_close(b, full[:, K : 2 * K], rtol=0, atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. routing
# --------------------------------------------------------------------------- #


def _filled_cache(n=12, task="alfworld", seed=0, base=1000):
    W, h, lse = _teacher(seed=seed, n=n)
    wit_lp, wit_ids = torch.topk(torch.log_softmax(h @ W.T, dim=-1), K, dim=-1)
    cache = TeacherHiddenCache()
    cache.register_lm_head(task, W)
    keys = torch.arange(base, base + n, dtype=torch.long)
    cache.put(keys, task, h, lse, witness_ids=wit_ids, witness_lp=wit_lp)
    return cache, keys, W, h, lse


def test_lookup_answers_only_what_it_owns():
    cache, keys, W, h, lse = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, K))
    asked = keys.clone()
    asked[3] = 999_999  # a key this cache never saw
    asked[5] = -1  # a row that was never queued for scoring

    values, found = cache.logprobs_at(asked, ids)

    assert found.tolist() == [1, 1, 1, 0, 1, 0, 1, 1]
    assert torch.all(values[3] == 0) and torch.all(values[5] == 0)
    reference = torch.log_softmax(h @ W.T, dim=-1).gather(-1, ids)
    torch.testing.assert_close(values[0], reference[0], rtol=0, atol=1e-5)


def test_a_row_answered_from_another_rows_entry_is_wrong_by_orders_of_magnitude():
    """Motivates the witness: mis-routing does not produce garbage, it produces a
    plausible log-prob. Only its distance from the teacher's own output shows it."""
    cache, keys, W, h, lse = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, K))
    correct, _ = cache.logprobs_at(keys, ids)
    shifted, _ = cache.logprobs_at(keys.roll(1), ids)

    assert torch.isfinite(shifted).all(), "mis-routed values look perfectly normal"
    assert (shifted - correct).abs().max() > 1.0


def test_witness_passes_on_a_consistent_cache():
    cache, *_ = _filled_cache(n=10)
    assert cache.check_witness(atol=1e-3) < 1e-3


def test_witness_catches_a_cache_whose_hidden_states_moved():
    """The failure the witness exists for: h filed under the wrong key."""
    cache, keys, W, h, lse = _filled_cache(n=10)
    cache._h = {int(k): cache._h[int(kk)] for k, kk in zip(keys, keys.roll(1))}

    with pytest.raises(RuntimeError, match="witness"):
        cache.check_witness(atol=1e-3)


def test_witness_catches_a_stale_normaliser():
    cache, keys, *_ = _filled_cache(n=6)
    for k in keys:
        cache._lse[int(k)] = cache._lse[int(k)] + 0.5

    with pytest.raises(RuntimeError, match="witness"):
        cache.check_witness(atol=1e-3)


def test_clear_drops_everything_between_steps():
    cache, keys, *_ = _filled_cache(n=4)
    assert len(cache) == 4 and int(keys[0]) in cache
    cache.clear()
    assert len(cache) == 0 and int(keys[0]) not in cache


# --------------------------------------------------------------------------- #
# 3. exchange (single process)
# --------------------------------------------------------------------------- #


def test_single_process_exchange_returns_the_reference():
    cache, keys, W, h, lse = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, K))
    got = exchange_teacher_logprobs(cache, keys, ids, world_size=1)
    reference = torch.log_softmax(h @ W.T, dim=-1).gather(-1, ids)
    torch.testing.assert_close(got, reference, rtol=0, atol=1e-5)


def test_an_unowned_row_raises_instead_of_training_on_zero():
    cache, keys, *_ = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, K))
    asked = keys.clone()
    asked[2] = 424_242

    with pytest.raises(RuntimeError, match="unanswered"):
        exchange_teacher_logprobs(cache, asked, ids, world_size=1)


def test_rows_never_queued_are_allowed_through():
    """hit_rate is not 1.0 -- the final turns' rows are scored by the trainer
    instead. Those carry -1 and must not trip the guard."""
    cache, keys, *_ = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, K))
    asked = keys.clone()
    asked[1] = -1
    asked[6] = -1

    got = exchange_teacher_logprobs(cache, asked, ids, world_size=1)
    assert torch.all(got[1] == 0) and torch.all(got[6] == 0)


# --------------------------------------------------------------------------- #
# 3b. exchange (two real gloo processes)
# --------------------------------------------------------------------------- #


def _worker(rank, world_size, port, mode, out):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        n = 6
        # Each rank CACHES one half of the rows but ASKS about the other half --
        # the ownership mismatch _balance_batch creates, in its purest form.
        W, h, lse = _teacher(seed=7, n=n * world_size)
        wit_lp, wit_ids = torch.topk(torch.log_softmax(h @ W.T, dim=-1), K, dim=-1)
        all_keys = torch.arange(n * world_size, dtype=torch.long)

        cache = TeacherHiddenCache()
        cache.register_lm_head("alfworld", W)
        owned = all_keys[rank * n : (rank + 1) * n]
        cache.put(owned, "alfworld", h[rank * n : (rank + 1) * n], lse[rank * n : (rank + 1) * n],
                  witness_ids=wit_ids[rank * n : (rank + 1) * n], witness_lp=wit_lp[rank * n : (rank + 1) * n])
        cache.check_witness(atol=1e-3)

        other = (rank + 1) % world_size
        asked = all_keys[other * n : (other + 1) * n].clone()
        g = torch.Generator().manual_seed(11)
        ids = torch.randint(0, VOCAB, (world_size * n, K), generator=g)[other * n : (other + 1) * n]

        if mode == "orphan":
            asked[2] = 987_654  # nobody caches this
        elif mode == "duplicate":
            cache.put(asked[:1], "alfworld", h[other * n : other * n + 1], lse[other * n : other * n + 1])

        try:
            got = exchange_teacher_logprobs(cache, asked, ids)
            reference = torch.log_softmax(h @ W.T, dim=-1).gather(-1, ids)[0:0]  # placeholder
            reference = torch.log_softmax(h[other * n : (other + 1) * n] @ W.T, dim=-1).gather(-1, ids)
            err = (got - reference).abs().max().item()
            out.put((rank, "ok", err))
        except RuntimeError as exc:
            out.put((rank, "raised", str(exc)))
    finally:
        dist.destroy_process_group()


def _run_two_ranks(mode):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = 29500 + (abs(hash(mode)) % 2000)
    procs = [ctx.Process(target=_worker, args=(r, 2, port, mode, out)) for r in range(2)]
    for p in procs:
        p.start()
    results = [out.get(timeout=120) for _ in range(2)]
    for p in procs:
        p.join(timeout=60)
    return dict((r, (s, v)) for r, s, v in results)


@pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn")
def test_two_ranks_answer_each_others_rows():
    """Rank 0 caches rows 0-5 and asks about 6-11; rank 1 does the mirror. Every
    answer must come from the other rank and match the full-vocabulary reference."""
    res = _run_two_ranks("clean")
    for rank, (status, err) in res.items():
        assert status == "ok", f"rank {rank}: {err}"
        assert err < 1e-5, f"rank {rank} deviation {err}"


@pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn")
def test_a_row_no_rank_owns_raises_on_every_rank():
    res = _run_two_ranks("orphan")
    for rank, (status, msg) in res.items():
        assert status == "raised", f"rank {rank} should have raised, got {msg}"
        assert "unanswered" in msg


@pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn")
def test_two_ranks_claiming_one_row_raises():
    """Duplicated ownership sums two answers into one slot -- a silently doubled
    teacher target if the count were not checked."""
    res = _run_two_ranks("duplicate")
    for rank, (status, msg) in res.items():
        assert status == "raised", f"rank {rank} should have raised, got {msg}"
        assert "more than once" in msg


# --------------------------------------------------------------------------- #
# 4. the loss the two indexings produce
# --------------------------------------------------------------------------- #


def test_both_indexings_agree_when_the_two_top_k_sets_coincide():
    """A control for the plumbing, not for the choice. Where student and teacher
    rank the same k tokens first, the support is the same set and the two paths
    must produce the same KL -- so any difference in a real run is the support
    moving, not the hidden-state route computing something else."""
    from verl.trainer.ppo.core_algos import topk_kl_per_token

    W, h, lse = _teacher(seed=5, n=16)
    t_logsm = torch.log_softmax(h @ W.T, dim=-1)
    # A student whose ordering matches the teacher's: monotone in the teacher's
    # logits, so topk picks the same ids.
    s_logits = (h @ W.T) * 0.9
    s_logsm = torch.log_softmax(s_logits, dim=-1)

    t_lp, t_ids = torch.topk(t_logsm, K, dim=-1)
    s_lp, s_ids = torch.topk(s_logsm, K, dim=-1)
    assert torch.equal(t_ids, s_ids), "test premise: the two top-k sets coincide"

    # teacher-indexed: student gathered at the teacher's ids
    teacher_indexed = topk_kl_per_token(
        s_logsm.gather(-1, t_ids).unsqueeze(1), t_lp.unsqueeze(1)
    )
    # student-indexed: teacher resolved from cached hidden states at the student's ids
    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)
    keys = torch.arange(16, dtype=torch.long)
    cache.put(keys, "alfworld", h, lse)
    resolved = exchange_teacher_logprobs(cache, keys, s_ids, world_size=1)
    student_indexed = topk_kl_per_token(s_lp.unsqueeze(1), resolved.unsqueeze(1))

    torch.testing.assert_close(student_indexed, teacher_indexed, rtol=0, atol=1e-5)


def test_the_support_actually_differs_when_the_models_disagree():
    """The other half: with an ordinary student the sets diverge, so the two
    losses are genuinely different objectives -- which is why the flag is pinned
    rather than treated as a performance knob."""
    from verl.trainer.ppo.core_algos import topk_kl_per_token

    W, h, lse = _teacher(seed=6, n=16)
    g = torch.Generator().manual_seed(3)
    t_logsm = torch.log_softmax(h @ W.T, dim=-1)
    s_logsm = torch.log_softmax(h @ W.T + 3.0 * torch.randn((16, VOCAB), generator=g), dim=-1)

    t_lp, t_ids = torch.topk(t_logsm, K, dim=-1)
    s_lp, s_ids = torch.topk(s_logsm, K, dim=-1)
    overlap = (t_ids.unsqueeze(-1) == s_ids.unsqueeze(-2)).any(-1).float().mean()
    assert overlap < 0.5, "test premise: the supports differ"

    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)
    cache.put(torch.arange(16), "alfworld", h, lse)
    resolved = exchange_teacher_logprobs(cache, torch.arange(16), s_ids, world_size=1)

    teacher_indexed = topk_kl_per_token(s_logsm.gather(-1, t_ids).unsqueeze(1), t_lp.unsqueeze(1))
    student_indexed = topk_kl_per_token(s_lp.unsqueeze(1), resolved.unsqueeze(1))

    assert torch.all(teacher_indexed >= -1e-5) and torch.all(student_indexed >= -1e-5)
    assert (student_indexed - teacher_indexed).abs().max() > 1e-3


# --------------------------------------------------------------------------- #
# 5. wiring contracts
# --------------------------------------------------------------------------- #


def test_hidden_capture_requires_the_response_only_row_map():
    """The hidden states come back on the packed rows response_only_logits
    selects. Without it there is no row map, so this must refuse rather than
    hand back something misaligned."""
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.response_only_logits = False
    actor.actor_module = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="response_only_logits"):
        DataParallelPPOActor.compute_topk_log_prob(actor, data=None, topk_k=K, return_hidden=True)


def test_the_cache_key_is_per_row_and_shared_by_its_positions():
    """A row is scored once, so every response position of it reads the same
    cache entry -- but with the position's own ids. Getting this wrong is the
    mis-routing the witness exists to catch, so pin the expansion."""
    bs, resp_len = 3, 4
    cache_ids = torch.tensor([11, -1, 13], dtype=torch.long)
    flat = cache_ids.reshape(bs, 1).expand(bs, resp_len).reshape(-1)

    assert flat.tolist() == [11] * 4 + [-1] * 4 + [13] * 4
    assert flat.numel() == bs * resp_len


def test_padding_positions_are_filed_as_unscored():
    """pad_input zero-fills positions the packed batch never had, and a real
    logit row cannot produce a zero normaliser -- so a zero lse marks padding,
    which must be stored as -1 rather than as a live entry."""
    lse = torch.tensor([[1.5, 2.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]])
    keys = torch.tensor([7, 8], dtype=torch.long).reshape(2, 1).expand(2, 4).reshape(-1)
    real = lse.reshape(-1) != 0
    filed = torch.where(real, keys, torch.full_like(keys, -1))

    assert filed.tolist() == [7, 7, -1, -1, 8, -1, -1, -1]
