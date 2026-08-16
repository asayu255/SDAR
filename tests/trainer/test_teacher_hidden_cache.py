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
        assert_rows_were_owned_once,
        exchange_teacher_logprobs,
        teacher_logprobs_from_hidden,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


H, VOCAB, K = 64, 512, 8


def _teacher(seed=0, n=16, temperature=1.0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((VOCAB, H), generator=g) / H**0.5
    h = torch.randn((n, H), generator=g) / 2
    # The forward divides before it normalises, so lse belongs to the scaled
    # logits while h stays raw -- exactly the split the cache has to carry.
    lse = torch.logsumexp(h @ W.T / temperature, dim=-1)
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


@pytest.mark.parametrize("temperature", [0.7, 1.3])
def test_a_non_unit_temperature_scales_the_logits_not_just_the_normaliser(temperature):
    """``lse`` comes from logits the forward already divided; ``h`` does not.

    Subtracting one from the other without redoing the division is silent at
    T=1 -- the run's pinned value -- and wrong at every other temperature, so the
    scaling is carried on the entry rather than assumed.
    """
    W, h, lse = _teacher(seed=3, n=32, temperature=temperature)
    ids = torch.randint(0, VOCAB, (32, K), generator=torch.Generator().manual_seed(4))

    reference = torch.log_softmax(h @ W.T / temperature, dim=-1).gather(-1, ids)
    torch.testing.assert_close(
        teacher_logprobs_from_hidden(h, lse, W, ids, temperature=temperature), reference, rtol=0, atol=1e-5
    )
    # A per-row temperature is the same number, so a mixed batch stays legal.
    torch.testing.assert_close(
        teacher_logprobs_from_hidden(h, lse, W, ids, temperature=torch.full((32,), temperature)),
        reference, rtol=0, atol=1e-5,
    )
    # And forgetting it is not a rounding difference: the residual is
    # logit*(1 - 1/T), which here is ~0.3 nats against a 1e-5 tolerance, and grows
    # with |logit| -- far larger on a real vocabulary than on this toy one.
    assert (teacher_logprobs_from_hidden(h, lse, W, ids) - reference).abs().max() > 1e-2


def test_the_cache_replays_the_temperature_it_was_filled_at():
    """End to end through put/logprobs_at, where the witness is the tripwire."""
    temperature = 0.6
    n = 6
    W, h_flat, lse_flat = _teacher(seed=5, n=n * L, temperature=temperature)
    h, lse = h_flat.view(n, L, H), lse_flat.view(n, L)
    full = torch.log_softmax(h_flat @ W.T / temperature, dim=-1)
    wit_lp, wit_ids = torch.topk(full, K, dim=-1)

    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)
    keys = torch.arange(n, dtype=torch.long)
    cache.put(keys, "alfworld", h, lse, witness_ids=wit_ids.view(n, L, K), witness_lp=wit_lp.view(n, L, K),
              temperature=temperature)

    assert cache.check_witness(atol=1e-3) < 1e-3

    ids = torch.randint(0, VOCAB, (n, L, K), generator=torch.Generator().manual_seed(6))
    values, found = cache.logprobs_at(keys, ids)
    expected = full.gather(-1, ids.reshape(-1, K)).view(n, L, K)
    assert torch.all(found == 1)
    torch.testing.assert_close(values, expected, rtol=0, atol=1e-5)


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


L = 3  # response positions per row


def _rowwise(n, seed=0):
    """(n, L, H) hidden states and their normalisers -- the real shape a row has."""
    W, h, lse = _teacher(seed=seed, n=n * L)
    return W, h.view(n, L, H), lse.view(n, L)


def _filled_cache(n=12, task="alfworld", seed=0, base=1000):
    W, h, lse = _rowwise(n, seed=seed)
    flat = torch.log_softmax(h.reshape(-1, H) @ W.T, dim=-1)
    wit_lp, wit_ids = torch.topk(flat, K, dim=-1)
    cache = TeacherHiddenCache()
    cache.register_lm_head(task, W)
    keys = torch.arange(base, base + n, dtype=torch.long)
    cache.put(keys, task, h, lse, witness_ids=wit_ids.view(n, L, K), witness_lp=wit_lp.view(n, L, K))
    return cache, keys, W, h, lse


def _reference(h, W, ids):
    n = h.shape[0]
    flat = torch.log_softmax(h.reshape(-1, H) @ W.T, dim=-1).gather(-1, ids.reshape(-1, ids.shape[-1]))
    return flat.view(n, L, ids.shape[-1])


def test_lookup_answers_only_what_it_owns():
    cache, keys, W, h, lse = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, L, K))
    asked = keys.clone()
    asked[3] = 999_999  # a key this cache never saw
    asked[5] = -1  # a row that was never queued for scoring

    values, found = cache.logprobs_at(asked, ids)

    assert found.tolist() == [1, 1, 1, 0, 1, 0, 1, 1]
    assert torch.all(values[3] == 0) and torch.all(values[5] == 0)
    torch.testing.assert_close(values[0], _reference(h, W, ids)[0], rtol=0, atol=1e-5)


def test_a_row_answered_from_another_rows_entry_is_wrong_by_orders_of_magnitude():
    """Motivates the witness: mis-routing does not produce garbage, it produces a
    plausible log-prob. Only its distance from the teacher's own output shows it."""
    cache, keys, W, h, lse = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, L, K))
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
    ids = torch.randint(0, VOCAB, (8, L, K))
    got = exchange_teacher_logprobs(cache, keys, ids, world_size=1)
    torch.testing.assert_close(got, _reference(h, W, ids), rtol=0, atol=1e-5)


def test_an_unowned_row_raises_instead_of_training_on_zero():
    """The tally is read once per mini-batch rather than per micro-batch -- the
    read is a synchronisation and the exchange runs inside the micro-batch loop --
    but it is still read before the optimizer step, so an unresolved row cannot
    reach the weights."""
    cache, keys, *_ = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, L, K))
    asked = keys.clone()
    asked[2] = 424_242

    exchange_teacher_logprobs(cache, asked, ids, world_size=1)
    with pytest.raises(RuntimeError, match="unanswered"):
        assert_rows_were_owned_once()


def test_a_clean_exchange_leaves_nothing_for_the_check_to_raise_on():
    cache, keys, *_ = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, L, K))
    exchange_teacher_logprobs(cache, keys, ids, world_size=1)
    assert_rows_were_owned_once()          # no raise
    assert_rows_were_owned_once()          # ...and the tally was cleared


def test_the_tally_survives_across_micro_batches():
    """The whole point of deferring it: a bad row in ANY micro-batch of the
    mini-batch must still be caught by the one read at the end."""
    cache, keys, *_ = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, L, K))
    exchange_teacher_logprobs(cache, keys[:4], ids[:4], world_size=1)   # clean
    bad = keys[4:].clone()
    bad[1] = 424_242
    exchange_teacher_logprobs(cache, bad, ids[4:], world_size=1)        # not clean
    exchange_teacher_logprobs(cache, keys[:4], ids[:4], world_size=1)   # clean again

    with pytest.raises(RuntimeError, match="unanswered"):
        assert_rows_were_owned_once()


@pytest.fixture(autouse=True)
def _fresh_ownership_tally():
    """The tally is process-global and read once per mini-batch, so a test that
    leaves one behind would fail the next one."""
    from verl.workers import teacher_cache as _tc

    _tc._OWNERSHIP = None
    yield
    _tc._OWNERSHIP = None


def test_rows_never_queued_are_allowed_through():
    """hit_rate is not 1.0 -- the final turns' rows are scored by the trainer
    instead. Those carry -1 and must not trip the guard."""
    cache, keys, *_ = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, L, K))
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
        total = n * world_size
        # Each rank CACHES one half of the rows but ASKS about the other half --
        # the ownership mismatch _balance_batch creates, in its purest form.
        W, h, lse = _rowwise(total, seed=7)
        flat = torch.log_softmax(h.reshape(-1, H) @ W.T, dim=-1)
        wit_lp, wit_ids = torch.topk(flat, K, dim=-1)
        wit_lp, wit_ids = wit_lp.view(total, L, K), wit_ids.view(total, L, K)
        all_keys = torch.arange(total, dtype=torch.long)
        mine = slice(rank * n, (rank + 1) * n)

        cache = TeacherHiddenCache()
        cache.register_lm_head("alfworld", W)
        cache.put(all_keys[mine], "alfworld", h[mine], lse[mine],
                  witness_ids=wit_ids[mine], witness_lp=wit_lp[mine])
        cache.check_witness(atol=1e-3)

        other = (rank + 1) % world_size
        theirs = slice(other * n, (other + 1) * n)
        asked = all_keys[theirs].clone()
        g = torch.Generator().manual_seed(11)
        ids = torch.randint(0, VOCAB, (total, L, K), generator=g)[theirs]

        if mode == "orphan":
            asked[2] = 987_654  # nobody caches this
        elif mode == "duplicate":
            cache.put(asked[:1], "alfworld", h[theirs][:1], lse[theirs][:1])

        try:
            got = exchange_teacher_logprobs(cache, asked, ids)
            # The ownership tally is read once per mini-batch, not inside the
            # exchange -- reading it synchronises. This is that read.
            assert_rows_were_owned_once()
            err = (got - _reference(h[theirs], W, ids)).abs().max().item()
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
    cache.put(keys, "alfworld", h.unsqueeze(1), lse.unsqueeze(1))  # one position per row
    resolved = exchange_teacher_logprobs(cache, keys, s_ids.unsqueeze(1), world_size=1).squeeze(1)
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
    cache.put(torch.arange(16), "alfworld", h.unsqueeze(1), lse.unsqueeze(1))
    resolved = exchange_teacher_logprobs(cache, torch.arange(16), s_ids.unsqueeze(1), world_size=1).squeeze(1)

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


# --------------------------------------------------------------------------- #
# 6. the three bugs the first cut had
# --------------------------------------------------------------------------- #


def test_a_flattened_put_is_refused():
    """The original cut repeated the row's key across its response positions and
    let the dict keep the last write, so every position of a row was evaluated
    with the last token's hidden state -- and the witness, overwritten the same
    way, stayed self-consistent and passed. Shape is the cheapest place to stop
    that from being expressible."""
    W, h, lse = _teacher(n=6)
    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)

    with pytest.raises(ValueError, match="per-row"):
        cache.put(torch.arange(6), "alfworld", h, lse)  # (n, H) / (n,) -- flattened


def test_every_position_of_a_row_gets_its_own_hidden_state():
    """The behavioural half of the same bug: positions within a row must give
    different log-probs, because they have different hidden states."""
    cache, keys, W, h, lse = _filled_cache(n=4)
    ids = torch.randint(0, VOCAB, (4, L, K))
    values, _ = cache.logprobs_at(keys, ids)

    torch.testing.assert_close(values, _reference(h, W, ids), rtol=0, atol=1e-5)
    # ...and the row is genuinely not one position repeated.
    same_ids = ids[:, :1, :].expand(4, L, K).contiguous()
    per_pos, _ = cache.logprobs_at(keys, same_ids)
    assert (per_pos[:, 0, :] - per_pos[:, 1, :]).abs().max() > 1e-3


def test_a_duplicated_key_on_one_rank_is_refused_at_write_time():
    """DP auto-padding repeats whole rows, cache id included. Catching it where
    it is written names the cause; catching it in the exchange only reports that
    some row was answered twice."""
    cache, keys, W, h, lse = _filled_cache(n=4)
    with pytest.raises(RuntimeError, match="written twice"):
        cache.put(keys[:1], "alfworld", h[:1], lse[:1])


def test_the_witness_now_covers_every_position():
    """It has to: a cache that kept one position per row is self-consistent at
    that position. Perturbing a non-final position must still be caught."""
    cache, keys, *_ = _filled_cache(n=5)
    key = int(keys[0])
    cache._h[key][0] = cache._h[key][0] + 3.0  # position 0, not the last

    with pytest.raises(RuntimeError, match="witness"):
        cache.check_witness(atol=1e-3)


# --------------------------------------------------------------------------- #
# 7. only the trained part is kept
# --------------------------------------------------------------------------- #

PAD_L = 8  # response_length cap, well above what these rows actually use


def _padded(lengths, seed=0, task="alfworld", base=500, use_mask=True):
    """Rows whose live response positions are a prefix, the rest padding.

    ``pad_input`` zero-fills the slots the packed batch never had, so padding
    carries a zero hidden state and a zero normaliser -- the shape the real
    forward hands over.
    """
    n = len(lengths)
    W, h_flat, lse_flat = _teacher(seed=seed, n=n * PAD_L)
    h, lse = h_flat.view(n, PAD_L, H).clone(), lse_flat.view(n, PAD_L).clone()
    keep = torch.arange(PAD_L).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    h[~keep] = 0
    lse[~keep] = 0

    full = torch.log_softmax(h.reshape(-1, H) @ W.T, dim=-1)
    wit_lp, wit_ids = torch.topk(full, K, dim=-1)

    cache = TeacherHiddenCache()
    cache.register_lm_head(task, W)
    keys = torch.arange(base, base + n, dtype=torch.long)
    cache.put(
        keys, task, h, lse,
        witness_ids=wit_ids.view(n, PAD_L, K), witness_lp=wit_lp.view(n, PAD_L, K),
        live_mask=keep if use_mask else None,
    )
    return cache, keys, W, h, keep


def test_only_the_real_response_positions_are_stored():
    """response_length is a cap, not a length. Keeping the padding costs ~8x here
    and the loss never reads it."""
    lengths = [3, 1, PAD_L, 5]
    cache, keys, *_ = _padded(lengths)

    for key, want in zip(keys.tolist(), lengths):
        assert cache._h[key].shape[0] == want
        assert cache._lse[key].shape[0] == want
        assert cache._witness_ids[key].shape[0] == want

    # And the padded input is genuinely not held alive behind the entries: the
    # storage they share is the packed size, not n * PAD_L.
    held = {e.untyped_storage().data_ptr(): e.untyped_storage().nbytes() for e in cache._h.values()}
    assert sum(held.values()) == sum(lengths) * H * cache._h[int(keys[0])].element_size()


def test_an_all_padding_call_does_not_pin_the_padded_input():
    """A zero-length slice is still a view, so it would hold the whole
    (n, response_length, hidden) input alive for the step."""
    cache, keys, *_ = _padded([0, 0], seed=7)
    held = {e.untyped_storage().data_ptr(): e.untyped_storage().nbytes() for e in cache._h.values()}
    assert sum(held.values()) == 0


def test_padding_reads_back_as_zero_and_the_rest_reads_back_exact():
    lengths = [3, 1, PAD_L, 5]
    cache, keys, W, h, keep = _padded(lengths, seed=2)
    ids = torch.randint(0, VOCAB, (len(lengths), PAD_L, K), generator=torch.Generator().manual_seed(9))

    values, found = cache.logprobs_at(keys, ids)

    assert torch.all(found == 1)
    expected = torch.log_softmax(h.reshape(-1, H) @ W.T, dim=-1)
    expected = expected.gather(-1, ids.reshape(-1, K)).view(len(lengths), PAD_L, K)
    torch.testing.assert_close(values[keep], expected[keep], rtol=0, atol=1e-5)
    assert torch.all(values[~keep] == 0)


def test_the_witness_still_covers_every_stored_position():
    cache, keys, *_ = _padded([4, 2, 7], seed=3)
    assert cache.check_witness(atol=1e-3) < 1e-3

    key = int(keys[1])
    cache._h[key][0] = cache._h[key][0] + 3.0
    with pytest.raises(RuntimeError, match="witness"):
        cache.check_witness(atol=1e-3)


def test_the_mask_and_the_zero_normaliser_agree():
    """``live_mask`` is the fact; a zero lse is the inference. They must not
    disagree, or one of the two paths is storing the wrong slots."""
    lengths = [6, 2, 3]
    by_mask, keys, _, _, _ = _padded(lengths, seed=4, use_mask=True)
    by_lse, _, _, _, _ = _padded(lengths, seed=4, use_mask=False)

    for key in keys.tolist():
        assert by_mask._len[key] == by_lse._len[key] == lengths[key - 500]
        torch.testing.assert_close(by_mask._h[key], by_lse._h[key], rtol=0, atol=0)


def test_a_holey_mask_is_refused_even_though_its_count_is_a_valid_length():
    """The reason the mask is passed as a mask: a count would be reconstructed as
    a prefix of that count, which is a different set of positions and silently
    shifts everything after the hole."""
    W, h_flat, lse_flat = _teacher(n=2 * PAD_L)
    h, lse = h_flat.view(2, PAD_L, H).clone(), lse_flat.view(2, PAD_L).clone()
    mask = torch.ones(2, PAD_L, dtype=torch.bool)
    mask[0, 2] = False  # count is still PAD_L - 1, a perfectly plausible length

    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)
    with pytest.raises(RuntimeError, match="prefix"):
        cache.put(torch.arange(2), "alfworld", h, lse, live_mask=mask)


def test_a_row_whose_live_slots_are_not_a_prefix_is_refused():
    """Packing reconstructs by prefix. Responses are right-padded so that holds,
    but a hole would silently shift every position after it."""
    W, h_flat, lse_flat = _teacher(n=2 * PAD_L)
    h, lse = h_flat.view(2, PAD_L, H).clone(), lse_flat.view(2, PAD_L).clone()
    lse[0, 2] = 0.0  # a hole in the middle, not a tail

    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)
    with pytest.raises(RuntimeError, match="prefix"):
        cache.put(torch.arange(2), "alfworld", h, lse)


def test_rows_of_different_lengths_survive_the_exchange_together():
    """The packing is per row, so a batch mixing a 1-token row with a full one
    must still come back aligned."""
    lengths = [1, PAD_L, 4]
    cache, keys, W, h, keep = _padded(lengths, seed=5)
    ids = torch.randint(0, VOCAB, (3, PAD_L, K), generator=torch.Generator().manual_seed(12))

    got = exchange_teacher_logprobs(cache, keys, ids, world_size=1)

    expected = torch.log_softmax(h.reshape(-1, H) @ W.T, dim=-1)
    expected = expected.gather(-1, ids.reshape(-1, K)).view(3, PAD_L, K)
    torch.testing.assert_close(got[keep], expected[keep], rtol=0, atol=1e-5)
    assert torch.all(got[~keep] == 0)


def test_a_shorter_request_than_the_stored_row_raises():
    """Reconstruction indexes a flattened (n, resp_len) grid, so an entry longer
    than the request would spill into the next row instead of failing."""
    cache, keys, *_ = _padded([PAD_L, 2], seed=8)
    ids = torch.randint(0, VOCAB, (2, PAD_L - 1, K))
    with pytest.raises(RuntimeError, match="response_length"):
        cache.logprobs_at(keys, ids)


def test_an_all_padding_row_is_owned_and_empty():
    """Owned-but-empty and nobody-owns-it are different answers: the second has
    to raise, the first must not."""
    cache, keys, *_ = _padded([0, 3], seed=6)
    ids = torch.randint(0, VOCAB, (2, PAD_L, K))

    values, found = cache.logprobs_at(keys, ids)
    assert found.tolist() == [1, 1]
    assert torch.all(values[0] == 0)
    assert cache.check_witness(atol=1e-3) < 1e-3


# --------------------------------------------------------------------------- #
# 8. the witness is a sample; the lookup is branch-free
# --------------------------------------------------------------------------- #


def test_the_witness_covers_only_the_sampled_rows_and_still_fires():
    """The teacher's own top-k is not part of the answer under student indexing --
    nothing reads it -- so it is built for a couple of micro-batches a step and
    kept as a spot check. A mis-filed entry is a systematic routing failure, so a
    handful of rows is enough to show it."""
    n = 6
    W, h_flat, lse_flat = _teacher(seed=11, n=n * L)
    h, lse = h_flat.view(n, L, H), lse_flat.view(n, L)
    full = torch.log_softmax(h_flat @ W.T, dim=-1)
    wit_lp, wit_ids = torch.topk(full, K, dim=-1)
    sampled = torch.tensor([0, 1])  # only the first micro-batch built it

    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)
    keys = torch.arange(n, dtype=torch.long)
    cache.put(
        keys, "alfworld", h, lse,
        witness_rows=sampled,
        witness_ids=wit_ids.view(n, L, K)[sampled], witness_lp=wit_lp.view(n, L, K)[sampled],
    )

    assert sorted(cache._witness_ids) == [0, 1]
    assert cache.check_witness(atol=1e-3) < 1e-3

    # Every row is still answerable -- only the check is sampled, not the store.
    ids = torch.randint(0, VOCAB, (n, L, K), generator=torch.Generator().manual_seed(12))
    values, found = cache.logprobs_at(keys, ids)
    assert torch.all(found == 1)
    torch.testing.assert_close(values, _reference(h, W, ids), rtol=0, atol=1e-5)

    cache._h[1][0] = cache._h[1][0] + 3.0
    with pytest.raises(RuntimeError, match="witness"):
        cache.check_witness(atol=1e-3)


def test_one_call_answers_rows_from_different_teachers():
    """_balance_batch sorts by token count, not by task, so a micro-batch mixes
    teachers. Grouping its rows by task would mean a mask, a nonzero and a
    device-to-host sync inside the micro-batch loop; the ids carry each row's
    offset into the stacked projection instead."""
    tasks = ["alfworld", "search", "webshop"]
    n_per = 3
    cache = TeacherHiddenCache()
    heads, hidden, keys = {}, {}, {}
    for t_i, task in enumerate(tasks):
        W, h_flat, lse_flat = _teacher(seed=20 + t_i, n=n_per * L)
        heads[task] = W
        hidden[task] = h_flat.view(n_per, L, H)
        cache.register_lm_head(task, W)
        keys[task] = torch.arange(100 * (t_i + 1), 100 * (t_i + 1) + n_per, dtype=torch.long)
        cache.put(keys[task], task, hidden[task], lse_flat.view(n_per, L))

    # Interleave, the way a balanced micro-batch would.
    asked, owner = [], []
    for i in range(n_per):
        for task in tasks:
            asked.append(int(keys[task][i]))
            owner.append(task)
    asked = torch.tensor(asked, dtype=torch.long)
    ids = torch.randint(0, VOCAB, (len(asked), L, K), generator=torch.Generator().manual_seed(21))

    values, found = cache.logprobs_at(asked, ids)

    assert torch.all(found == 1)
    for row, task in enumerate(owner):
        want = _reference(hidden[task][row // len(tasks)].unsqueeze(0), heads[task], ids[row : row + 1])
        torch.testing.assert_close(values[row : row + 1], want, rtol=0, atol=1e-5)


def test_the_lookup_makes_no_host_round_trip():
    """The read runs inside the micro-batch loop, thousands of times a step. A
    .tolist() or an int(tensor) there is a device-to-host sync, which stalls the
    very pipeline the rollout overlap exists to keep full."""
    cache, keys, W, h, lse = _filled_cache(n=6)
    ids = torch.randint(0, VOCAB, (6, L, K))
    cache.logprobs_at(keys, ids)  # finalize once, outside the measurement

    calls = []
    originals = {name: getattr(torch.Tensor, name) for name in ("tolist", "item")}
    for name, original in originals.items():
        def traced(self, _o=original, _n=name):
            calls.append(_n)
            return _o(self)

        setattr(torch.Tensor, name, traced)
    try:
        cache.logprobs_at(keys, ids)
        assert calls == [], f"lookup synchronised via {sorted(set(calls))}"
        # The trace itself has to work, or the assertion above proves nothing.
        torch.zeros(1).item()
        assert calls == ["item"]
    finally:
        for name, original in originals.items():
            setattr(torch.Tensor, name, original)
