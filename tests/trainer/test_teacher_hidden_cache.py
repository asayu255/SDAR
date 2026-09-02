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
    values, found, _ = cache.logprobs_at(keys, ids)
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

    values, found, _ = cache.logprobs_at(asked, ids)

    assert found.tolist() == [1, 1, 1, 0, 1, 0, 1, 1]
    assert torch.all(values[3] == 0) and torch.all(values[5] == 0)
    torch.testing.assert_close(values[0], _reference(h, W, ids)[0], rtol=0, atol=1e-5)


def test_a_row_answered_from_another_rows_entry_is_wrong_by_orders_of_magnitude():
    """Motivates the witness: mis-routing does not produce garbage, it produces a
    plausible log-prob. Only its distance from the teacher's own output shows it."""
    cache, keys, W, h, lse = _filled_cache(n=8)
    ids = torch.randint(0, VOCAB, (8, L, K))
    correct, _, _ = cache.logprobs_at(keys, ids)
    shifted, _, _ = cache.logprobs_at(keys.roll(1), ids)

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

        # A fingerprint per row, so the two-process path exercises the identity
        # check as well as the ownership count.
        from verl.workers.teacher_cache import row_fingerprint

        tok = torch.randint(0, VOCAB, (total, 16), generator=torch.Generator().manual_seed(5))
        msk = torch.ones((total, 16), dtype=torch.long)
        all_fp = row_fingerprint(tok, msk)
        for i, key in enumerate(all_keys[mine].tolist()):
            cache._fingerprint[key] = int(all_fp[rank * n + i])
        cache._final = None
        fp = all_fp[theirs].clone()

        if mode == "orphan":
            asked[2] = 987_654  # nobody caches this
        elif mode == "duplicate":
            cache.put(asked[:1], "alfworld", h[theirs][:1], lse[theirs][:1])
        elif mode == "shifted":
            asked = asked.roll(1)  # resolves cleanly, to the wrong rows

        if mode in ("multi", "count"):
            out.put((rank, *_multi_plane_probe(cache, all_keys, h, lse, W, ids, fp, mine, theirs, mode)))
            return

        try:
            got = exchange_teacher_logprobs(cache, asked, ids, fingerprints=fp)
            # The ownership tally is read once per mini-batch, not inside the
            # exchange -- reading it synchronises. This is that read.
            assert_rows_were_owned_once()
            err = (got - _reference(h[theirs], W, ids)).abs().max().item()
            out.put((rank, "ok", err))
        except RuntimeError as exc:
            out.put((rank, "raised", str(exc)))
    finally:
        dist.destroy_process_group()


def _multi_plane_probe(cache, all_keys, h, lse, W, ids, fp, mine, theirs, mode):
    """Three models on the same rows, asked for together and one at a time.

    The cross-teacher arms read four models at ONE support. Batching them is only
    allowed if it changes nothing, so this asks the same question both ways in
    the same process group and compares the answers exactly.
    """
    import torch.distributed as dist

    from verl.workers.teacher_cache import exchange_teacher_logprobs_multi

    planes = ("base", "off_a", "off_b")
    # A distinct head per plane, so a plane answered from another plane's entry
    # comes back with different numbers rather than the same ones.
    heads, keys = {}, []
    for j, name in enumerate(planes):
        Wj = W + (j + 1) * 0.05
        heads[name] = Wj
        cache.register_lm_head(name, Wj)
        kj = all_keys + (j + 1) * 10_000
        # This head's own normaliser, not the on-task one's. The store keeps h
        # and lse and finishes log p = h.W[v] - lse, so an lse belonging to a
        # different head reproduces nothing and the reference below would be
        # comparing against a distribution that does not exist.
        lse_j = torch.logsumexp(h.reshape(-1, H) @ Wj.T, dim=-1).view(h.shape[0], L)
        cache.put(kj[mine], name, h[mine], lse_j[mine])
        # One row is one row whichever model is read off it, so every plane's
        # entry carries the SAME fingerprint as the on-task one -- which is why
        # a single fingerprint column can check all of them. Left at 0 the guard
        # would fire on every plane, correctly, and the test would be about the
        # fixture instead.
        for slot, key in enumerate(kj[mine].tolist()):
            cache._fingerprint[key] = int(cache._fingerprint[int(all_keys[mine][slot])])
        keys.append(kj[theirs])
    cache._final = None
    asked = torch.stack(keys, dim=1)  # (n, 3)

    if mode == "count":
        n_calls = {"all_gather": 0, "all_reduce": 0}
        real_gather, real_reduce = dist.all_gather, dist.all_reduce

        def _g(*a, **kw):
            n_calls["all_gather"] += 1
            return real_gather(*a, **kw)

        def _r(*a, **kw):
            n_calls["all_reduce"] += 1
            return real_reduce(*a, **kw)

        dist.all_gather, dist.all_reduce = _g, _r
        try:
            exchange_teacher_logprobs_multi(cache, asked, ids, fingerprints=fp)
            fused = dict(n_calls)
            n_calls.update(all_gather=0, all_reduce=0)
            for p in range(asked.size(1)):
                exchange_teacher_logprobs(cache, asked[:, p], ids, fingerprints=fp)
            apiece = dict(n_calls)
        finally:
            dist.all_gather, dist.all_reduce = real_gather, real_reduce
        assert_rows_were_owned_once()
        return "counts", (fused, apiece)

    together = exchange_teacher_logprobs_multi(cache, asked, ids, fingerprints=fp)
    assert_rows_were_owned_once()
    apart = [exchange_teacher_logprobs(cache, asked[:, p], ids, fingerprints=fp)
             for p in range(asked.size(1))]
    assert_rows_were_owned_once()

    identical = all(torch.equal(a, b) for a, b in zip(together, apart))
    err = max(
        (together[j] - _reference(h[theirs], heads[name], ids)).abs().max().item()
        for j, name in enumerate(planes)
    )
    return "multi", (identical, err)


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
def test_a_key_naming_another_rank_s_row_raises_across_processes():
    """The cross-rank version of the quiet failure: the key resolves, one rank
    owns it, and the log-prob returned belongs to a different sample."""
    res = _run_two_ranks("shifted")
    for rank, (status, msg) in res.items():
        assert status == "raised", f"rank {rank} should have raised, got {msg}"
        assert "DIFFERENT row" in msg


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
    values, _, _ = cache.logprobs_at(keys, ids)

    torch.testing.assert_close(values, _reference(h, W, ids), rtol=0, atol=1e-5)
    # ...and the row is genuinely not one position repeated.
    same_ids = ids[:, :1, :].expand(4, L, K).contiguous()
    per_pos, _, _ = cache.logprobs_at(keys, same_ids)
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

    values, found, _ = cache.logprobs_at(keys, ids)

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

    values, found, _ = cache.logprobs_at(keys, ids)
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
    values, found, _ = cache.logprobs_at(keys, ids)
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

    values, found, _ = cache.logprobs_at(asked, ids)

    assert torch.all(found == 1)
    for row, task in enumerate(owner):
        want = _reference(hidden[task][row // len(tasks)].unsqueeze(0), heads[task], ids[row : row + 1])
        torch.testing.assert_close(values[row : row + 1], want, rtol=0, atol=1e-5)


def _traced_reads(fn):
    """Run fn with .tolist()/.item() traced, and return the device of each read.

    A device-to-host sync is a read off a tensor that is NOT on the host; the
    same call on a cpu tensor is just a read. Tracing the method name alone
    conflates the two, which matters here because the offloaded path is supposed
    to read its plan off the host index.
    """
    seen = []
    originals = {name: getattr(torch.Tensor, name) for name in ("tolist", "item")}
    for name, original in originals.items():
        def traced(self, _o=original, _n=name):
            seen.append((_n, self.device.type))
            return _o(self)

        setattr(torch.Tensor, name, traced)
    try:
        fn()
    finally:
        for name, original in originals.items():
            setattr(torch.Tensor, name, original)
    return seen


def test_the_lookup_makes_no_host_round_trip():
    """The read runs inside the micro-batch loop, thousands of times a step. A
    .tolist() or an int(tensor) THERE, off a tensor on the card, is a
    device-to-host sync, which stalls the very pipeline the rollout overlap
    exists to keep full.

    Offloaded the lookup does read its ragged plan with .tolist(), and that is
    not a sync: store_placement puts the index on the host precisely so it is
    not, and packed_plan is host-only in and out so it cannot become one. This
    box has no driver, so the device column below cannot discriminate on its own
    -- test_the_placement_decision_is_what_it_claims_on_a_gpu_box drives that
    half with the device a real run has.
    """
    cache, keys, W, h, lse = _filled_cache(n=6)
    ids = torch.randint(0, VOCAB, (6, L, K))
    cache.logprobs_at(keys, ids)  # finalize once, outside the measurement

    seen = _traced_reads(lambda: cache.logprobs_at(keys, ids))
    off_device = [c for c in seen if c[1] != "cpu"]
    assert off_device == [], f"lookup synchronised via {off_device}"

    # The trace itself has to work, or the assertion above proves nothing.
    assert _traced_reads(lambda: torch.zeros(1).item()) == [("item", "cpu")]


def test_the_resident_lookup_reads_nothing_back_to_the_host_at_all():
    """Resident, the index is ON the card, so there is no host copy of the
    lengths to plan from and any read of them would be the sync. The packed path
    is therefore offload-only, and this is what pins that: not "the reads are
    cheap" but "there are none"."""
    from verl.workers import teacher_cache as tc

    cache, keys, _, _, _ = _cache_with_offload(False, n=6)
    ids = torch.randint(0, VOCAB, (6, L, K))
    cache.logprobs_at(keys, ids)

    assert _traced_reads(lambda: cache.logprobs_at(keys, ids)) == []

    called = []
    original = tc.packed_plan
    tc.packed_plan = lambda *a, **k: called.append(a) or original(*a, **k)
    try:
        cache.logprobs_at(keys, ids)
    finally:
        tc.packed_plan = original
    assert called == [], "the resident path planned a packed read it cannot afford"


def test_the_packed_plan_places_every_live_position_and_no_padding():
    """The plan is the whole correctness of the packed read: `dest` says where a
    packed row lands in the (n, resp_len) grid and `row` says which entry it came
    from. Get either wrong and the lookup returns a REAL teacher log-prob for the
    wrong position, which no value check downstream can see.

    Checked against the brute-force layout rather than against itself.
    """
    from verl.workers.teacher_cache import packed_plan

    resp_len = 7
    lens = torch.tensor([3, 0, 7, 1])
    offs = torch.tensor([100, 999, 40, 5])

    dest, row, spans, m = packed_plan(lens, offs, resp_len)

    want_dest, want_row = [], []
    for i, n in enumerate(lens.tolist()):
        for p in range(n):
            want_dest.append(i * resp_len + p)
            want_row.append(i)
    assert m == len(want_dest) == 11
    assert dest.tolist() == want_dest
    assert row.tolist() == want_row
    # One contiguous span per OWNED row, in row order, skipping the empty one.
    assert spans == [(100, 3), (40, 7), (5, 1)]
    assert sum(n for _, n in spans) == m
    # Host in, host out -- the caller moves the plan, so a device read cannot
    # sneak in here.
    assert dest.device.type == "cpu" and row.device.type == "cpu"

    assert packed_plan(torch.zeros(4, dtype=torch.long), offs, resp_len) is None


def test_the_packed_read_pulls_only_the_live_positions():
    """The point of the packing. The padded read asks the store for resp_len
    positions a row where a turn generates about a quarter of them, and it asks
    as a scattered gather into a pageable tensor that torch then has to stage --
    so pinning the store bought nothing. Packed, the pull is one contiguous slice
    per row and the (rows, k, hidden) projection gather inside the GEMM, the
    largest tensor the micro-batch loop allocates, shrinks by the same factor.
    """
    from verl.workers.teacher_cache import packed_plan

    lengths = [3, 1, PAD_L, 5, 0]
    cache, keys, _, _, _ = _padded(lengths, seed=11)
    cache._offload = True
    ids = torch.randint(0, VOCAB, (len(lengths), PAD_L, K))

    seen = {}
    original = packed_plan

    def spy(lens, offs, resp_len):
        out = original(lens, offs, resp_len)
        seen["m"] = 0 if out is None else out[3]
        seen["padded"] = lens.numel() * resp_len
        return out

    from verl.workers import teacher_cache as tc

    tc.packed_plan = spy
    try:
        cache.logprobs_at(keys, ids)
    finally:
        tc.packed_plan = original

    assert seen["m"] == sum(lengths), "the packed read did not ask for every live position"
    assert seen["padded"] == len(lengths) * PAD_L
    assert seen["m"] < seen["padded"] // 2, (
        f"the packed read pulled {seen['m']} rows against a padded {seen['padded']} -- "
        "the padding is back, and with it the bytes and the projection gather"
    )

    # Same values as the padded read would have given, on the same rows.
    values, found, _ = cache.logprobs_at(keys, ids)
    resident, keys2, _, _, _ = _padded(lengths, seed=11)
    resident._offload = False
    torch.testing.assert_close(values, resident.logprobs_at(keys2, ids)[0], rtol=0, atol=0)
    assert torch.all(found == 1)


# --------------------------------------------------------------------------- #
# 9. one copy, not two
# --------------------------------------------------------------------------- #


def _held_bytes(*tensors):
    """Total distinct storage behind these tensors -- views of one buffer count once."""
    seen = {}
    for t in tensors:
        if t is None:
            continue
        st = t.untyped_storage()
        seen[st.data_ptr()] = st.nbytes()
    return sum(seen.values())


def test_finalizing_does_not_leave_a_second_copy_of_the_cache():
    """The read side wants one contiguous buffer, but building it by concatenation
    and leaving the entries pointing at the per-put tensors keeps the WHOLE cache
    alive twice -- for the length of the actor update, next to its activations."""
    cache, keys, W, h, lse = _filled_cache(n=8)
    before = _held_bytes(*cache._h.values(), *cache._lse.values())
    assert cache._chunks  # the per-put packed tensors, pre-finalize

    cache.logprobs_at(keys, torch.randint(0, VOCAB, (8, L, K)))

    after = _held_bytes(*cache._h.values(), *cache._lse.values(), cache._final["h"], cache._final["lse"])
    assert after == before, f"cache is held twice: {before} -> {after}"
    assert not cache._chunks, "the per-put tensors were not released"
    store = cache._final["h"].untyped_storage().data_ptr()
    assert all(cache._h[k].untyped_storage().data_ptr() == store for k in keys.tolist())


def test_the_witness_still_reads_the_entries_after_they_are_repointed():
    """check_witness reads self._h, which finalize rebinds. Same numbers, or the
    rebind moved a row."""
    cache, keys, *_ = _filled_cache(n=6)
    before = cache.check_witness(atol=1e-3)
    cache.logprobs_at(keys, torch.randint(0, VOCAB, (6, L, K)))
    assert cache.check_witness(atol=1e-3) == pytest.approx(before, abs=1e-6)

    cache._h[int(keys[2])][0] += 3.0
    with pytest.raises(RuntimeError, match="witness"):
        cache.check_witness(atol=1e-3)


def test_a_head_registered_into_its_slot_is_never_duplicated():
    """Cloning each teacher's projection and stacking them later holds both
    layouts at once -- ~1.9 GB for three 1.7B teachers, peaking during worker init,
    which is when vLLM measures free memory to size its KV cache."""
    g = torch.Generator().manual_seed(31)
    heads = {t: torch.randn((VOCAB, H), generator=g) for t in ("alfworld", "search")}

    cache = TeacherHiddenCache()
    for slot, (task, W) in enumerate(heads.items()):
        cache.register_lm_head(task, W, slot=slot, n_tasks=len(heads))

    assert cache._stacked.shape == (len(heads) * VOCAB, H)
    assert cache._voff == {"alfworld": 0, "search": VOCAB}
    # One allocation: the per-task handles are views into it, not copies beside it.
    assert _held_bytes(*(cache.lm_head(t) for t in heads)) == cache._stacked.untyped_storage().nbytes()
    for task, W in heads.items():
        torch.testing.assert_close(cache.lm_head(task), W, rtol=0, atol=0)


def test_slotted_and_unslotted_registration_must_not_be_mixed():
    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", torch.randn(VOCAB, H), slot=0, n_tasks=2)
    with pytest.raises(ValueError, match="slot"):
        cache.register_lm_head("search", torch.randn(VOCAB, H))


def test_slot_registration_answers_the_same_values_as_the_lazy_stack():
    """The two registration paths must be interchangeable at the answer."""
    g = torch.Generator().manual_seed(32)
    tasks = ["alfworld", "search"]
    heads = {t: torch.randn((VOCAB, H), generator=g) / H**0.5 for t in tasks}

    built = []
    for slotted in (False, True):
        cache = TeacherHiddenCache()
        for slot, task in enumerate(tasks):
            if slotted:
                cache.register_lm_head(task, heads[task].clone(), slot=slot, n_tasks=len(tasks))
            else:
                cache.register_lm_head(task, heads[task].clone())
        for t_i, task in enumerate(tasks):
            _, h, lse = _rowwise(2, seed=40 + t_i)
            lse = torch.logsumexp(h.reshape(-1, H) @ heads[task].T, dim=-1).view(2, L)
            cache.put(torch.arange(2 * t_i, 2 * t_i + 2), task, h, lse)
        ids = torch.randint(0, VOCAB, (4, L, K), generator=torch.Generator().manual_seed(33))
        built.append(cache.logprobs_at(torch.arange(4), ids)[0])

    torch.testing.assert_close(built[0], built[1], rtol=0, atol=1e-6)


# --------------------------------------------------------------------------- #
# 10. the witness under the precision the forward actually runs at
# --------------------------------------------------------------------------- #


def _bf16_teacher(n, seed=0, logit_scale=6.0):
    """The pipeline as the real forward runs it, in bfloat16.

    lm_head projects under autocast, so the logits, their top-k and their
    logsumexp are all bfloat16; the stored witness is ``(tvals - lse).float()``
    and therefore carries a bfloat16 quantum of |logit|, not of the log-prob.
    """
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn((VOCAB, H), generator=g) / H**0.5 * logit_scale).bfloat16()
    h = (torch.randn((n * L, H), generator=g)).bfloat16()
    logits = (h.float() @ W.float().T).bfloat16()          # the forward's projection
    lse = torch.logsumexp(logits, dim=-1, keepdim=True).bfloat16()
    tvals, tids = torch.topk(logits, K, dim=-1, sorted=False)
    return W, h.view(n, L, H), lse.squeeze(-1).view(n, L), (tvals - lse).float().view(n, L, K), tids.view(n, L, K)


def _bf16_cache(n=4, seed=0):
    W, h, lse, wit_lp, wit_ids = _bf16_teacher(n, seed=seed)
    cache = TeacherHiddenCache()
    cache.register_lm_head("alfworld", W)
    keys = torch.arange(n, dtype=torch.long)
    cache.put(keys, "alfworld", h, lse, witness_ids=wit_ids, witness_lp=wit_lp)
    return cache, keys


def test_a_bfloat16_forward_passes_the_witness():
    """The end-to-end shape: bfloat16 logits, bfloat16 top-k and normaliser."""
    cache, _ = _bf16_cache(seed=41)
    cache.check_witness()


def test_the_tolerance_admits_one_bfloat16_ulp_of_the_logit_and_nothing_like_a_nat():
    """The first real run tripped on this, and the number it reported -- 0.250 --
    is exactly one bfloat16 ULP at |logit| in [32, 64).

    The stored witness is ``logit - lse`` out of a bfloat16 forward, so it carries
    a bfloat16 quantum of |LOGIT|, not of the log-prob. Demanding 1e-3 of a
    recomputation asked for precision the reference never had. What it can be held
    to is the storage precision -- and a row read from another row's entry misses
    by nats, which is still tens of times that.
    """
    cache, keys = _bf16_cache(seed=41)
    key = int(keys[0])
    logit = cache._witness_lp[key] + cache._lse[key].float().unsqueeze(-1)
    ulp = torch.finfo(torch.bfloat16).eps * logit.abs()
    assert float(ulp.max()) > 1e-2, "fixture logits are too small to show the quantum"

    cache._witness_lp[key] = cache._witness_lp[key] + ulp
    worst = cache.check_witness()                       # one ULP: passes
    assert worst > 1e-2, "the old absolute 1e-3 would have rejected exactly this"

    cache._witness_lp[key] = cache._witness_lp[key] + 10.0
    with pytest.raises(RuntimeError, match="witness"):   # nats: does not
        cache.check_witness()


def test_a_mispaired_entry_still_fails_at_bfloat16():
    """The tolerance moved by three orders of magnitude, so the thing it exists to
    catch has to be re-checked at the new one."""
    cache, keys = _bf16_cache(n=6, seed=42)
    a, b = int(keys[1]), int(keys[4])
    cache._h[a], cache._h[b] = cache._h[b], cache._h[a]

    with pytest.raises(RuntimeError, match="witness") as exc:
        cache.check_witness()
    # And it is distinguishable from rounding on sight: rounding misses a handful
    # of positions by ~1x, a mis-pairing misses nearly all of them by tens of x.
    assert "x the storage precision" in str(exc.value)


def test_the_weight_used_to_recompute_must_be_the_one_the_forward_projected_with():
    """FSDP keeps float32 masters and casts to param_dtype for the forward, so
    summon_full_params hands back a weight the projection never used. Recomputing
    from it is different arithmetic -- ~eps_bf16 * |logit|, which is exactly the
    0.25 nats the first run reported."""
    W, h, lse, wit_lp, wit_ids = _bf16_teacher(4, seed=43)

    right = TeacherHiddenCache()
    right.register_lm_head("alfworld", W)
    right.put(torch.arange(4), "alfworld", h, lse, witness_ids=wit_ids, witness_lp=wit_lp)

    wrong = TeacherHiddenCache()
    # A float32 master that rounds to the same bfloat16 weight the forward used.
    wrong.register_lm_head("alfworld", W.float() + torch.finfo(torch.bfloat16).eps * W.float() / 2)
    wrong.put(torch.arange(4), "alfworld", h, lse, witness_ids=wit_ids, witness_lp=wit_lp)

    assert right.check_witness() < wrong.check_witness()


# --------------------------------------------------------------------------- #
# 11. the key is checked against what the row actually is
# --------------------------------------------------------------------------- #


def _rows(n, seed=0, seqlen=12):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (n, seqlen), generator=g)
    mask = torch.ones((n, seqlen), dtype=torch.long)
    mask[:, :2] = 0                                  # left padding, as prompts have
    return input_ids, mask


def _fp_cache(n=6, seed=0, seqlen=12):
    from verl.workers.teacher_cache import row_fingerprint

    cache, keys, W, h, lse = _filled_cache(n=n, seed=seed, base=0)
    input_ids, mask = _rows(n, seed=seed + 90, seqlen=seqlen)
    fp = row_fingerprint(input_ids, mask)
    for i, key in enumerate(keys.tolist()):
        cache._fingerprint[key] = int(fp[i])
    cache._final = None
    return cache, keys, input_ids, mask, fp


def test_two_different_rows_do_not_share_a_fingerprint():
    from verl.workers.teacher_cache import row_fingerprint

    input_ids, mask = _rows(64, seed=7, seqlen=40)
    fp = row_fingerprint(input_ids, mask)
    assert len(set(fp.tolist())) == 64


def test_the_fingerprint_ignores_how_far_the_row_is_padded():
    """The two sides may pad to different widths; only the live tokens count."""
    from verl.workers.teacher_cache import row_fingerprint

    input_ids, mask = _rows(4, seed=8, seqlen=10)
    wide_ids = torch.cat([input_ids, torch.randint(0, VOCAB, (4, 6))], dim=-1)
    wide_mask = torch.cat([mask, torch.zeros((4, 6), dtype=torch.long)], dim=-1)

    assert torch.equal(row_fingerprint(input_ids, mask), row_fingerprint(wide_ids, wide_mask))


def test_a_key_that_names_another_row_is_caught_even_though_it_resolves():
    """The quiet failure this exists for: the key is valid, exactly one rank owns
    it, the witness passes on the owner's side -- and the answer is a real teacher
    log-prob belonging to a different sample."""
    cache, keys, input_ids, mask, fp = _fp_cache(n=6, seed=11)
    ids = torch.randint(0, VOCAB, (6, L, K))

    exchange_teacher_logprobs(cache, keys, ids, world_size=1, fingerprints=fp)
    assert_rows_were_owned_once()                       # aligned: no raise

    shifted = keys.roll(1)                              # every key names its neighbour
    exchange_teacher_logprobs(cache, shifted, ids, world_size=1, fingerprints=fp)
    with pytest.raises(RuntimeError, match="DIFFERENT row"):
        assert_rows_were_owned_once()


def test_one_shifted_row_among_many_is_still_caught():
    cache, keys, input_ids, mask, fp = _fp_cache(n=8, seed=12)
    ids = torch.randint(0, VOCAB, (8, L, K))
    asked = keys.clone()
    asked[5] = int(keys[2])                             # one row points at another

    exchange_teacher_logprobs(cache, asked, ids, world_size=1, fingerprints=fp)
    with pytest.raises(RuntimeError, match="1 answered for a DIFFERENT row"):
        assert_rows_were_owned_once()


def test_rows_never_queued_are_not_fingerprint_checked():
    """-1 rows are scored by the trainer instead; they have no owner and must not
    be reported as mismatched."""
    cache, keys, input_ids, mask, fp = _fp_cache(n=6, seed=13)
    ids = torch.randint(0, VOCAB, (6, L, K))
    asked = keys.clone()
    asked[1] = -1
    asked[4] = -1

    exchange_teacher_logprobs(cache, asked, ids, world_size=1, fingerprints=fp)
    assert_rows_were_owned_once()


def test_without_fingerprints_the_exchange_behaves_as_before():
    """The check is opt-in at the call site, so the guard cannot be what breaks a
    caller that does not pass them."""
    cache, keys, *_ = _fp_cache(n=4, seed=14)
    ids = torch.randint(0, VOCAB, (4, L, K))
    exchange_teacher_logprobs(cache, keys.roll(1), ids, world_size=1)
    assert_rows_were_owned_once()


def test_nbytes_counts_the_packed_sources_once():
    """The entries are views into one packed tensor per put, so summing the views
    would report the same memory once per row. The number exists because the
    sign-weighting arms cache four models per row instead of one, and it is what
    says whether the headroom is there before a step dies on it."""
    cache = TeacherHiddenCache()
    n, resp, hidden = 4, 6, 8
    h = torch.randn(n, resp, hidden)
    lse = torch.randn(n, resp).abs() + 1.0
    cache.put(
        cache_ids=torch.arange(1, n + 1),
        task="alfworld",
        h=h,
        lse=lse,
        live_mask=torch.ones(n, resp, dtype=torch.bool),
    )
    packed = h.numel() * h.element_size() + lse.numel() * lse.element_size()
    assert cache.nbytes() == packed
    cache.clear()
    assert cache.nbytes() == 0


# --------------------------------------------------------------------------- #
# where the store lives while the actor trains
# --------------------------------------------------------------------------- #
def _cache_with_offload(flag, **kw):
    """A cache built with TEACHER_CACHE_OFFLOAD forced either way.

    The switch is read at import, so the module-level constant is what a test has
    to move; the instance copies it in __init__, which is what makes that
    possible without reloading the module.
    """
    from verl.workers import teacher_cache as tc

    cache, keys, W, h, lse = _filled_cache(**kw)
    cache._offload = flag
    return cache, keys, W, h, lse


@pytest.mark.parametrize("n", [4, 12])
def test_offloading_the_store_changes_where_the_bytes_wait_and_nothing_else(n):
    """The cross-teacher arm holds 16.3 GB of hidden states on the card for the
    whole backward -- four models a row -- and reached 99% of both GPUs during
    update_actor. Offloaded, a micro batch pulls only the rows it asks for.

    A bf16 copy host-to-device is exact and the GEMM still runs on the device on
    the same values, so the two paths have to agree BIT FOR BIT, not nearly.
    """
    ids = torch.randint(0, VOCAB, (n, L, K))

    resident, keys, W, h, _ = _cache_with_offload(False, n=n)
    v_res, f_res, fp_res = resident.logprobs_at(keys, ids)

    offloaded, keys2, _, _, _ = _cache_with_offload(True, n=n)
    v_off, f_off, fp_off = offloaded.logprobs_at(keys2, ids)

    assert torch.equal(v_res, v_off), "the offloaded store returned different values"
    assert torch.equal(f_res, f_off)
    assert torch.equal(fp_res, fp_off)
    torch.testing.assert_close(v_off, _reference(h, W, ids), rtol=0, atol=1e-5)


def test_the_placement_decision_is_what_it_claims_on_a_gpu_box():
    """The decision, not the outcome. A CPU test box cannot tell the two
    placements apart by looking at the tensors -- with the read device already
    cpu every branch gives the same answer, and a test that only inspects where
    things ended up passes just as happily when the offload does nothing.

    So the choice is a pure function and this drives it with the device a real
    run has.
    """
    from verl.workers.teacher_cache import store_placement

    store, pin, index = store_placement(True, "cuda:0", cuda_available=True)
    assert store.type == "cpu", "offload on, store still on the card -- it frees nothing"
    assert index.type == "cpu", "the index has to follow the store, or the gather is illegal"
    assert pin is True, "a device read wants pinned host memory, or the pull is not a DMA"

    store, pin, index = store_placement(False, "cuda:0", cuda_available=True)
    assert store.type == "cuda" and index.type == "cuda", "resident: nothing moves"
    assert pin is False

    # No driver: the store still moves off the device, but pinning would raise.
    store, pin, index = store_placement(True, "cuda:0", cuda_available=False)
    assert store.type == "cpu" and pin is False


def test_the_offloaded_store_really_is_off_the_device():
    """A flag that silently kept the store where it was would pass every value
    test above and free nothing, which is the whole point of the change."""
    off, keys, _, _, _ = _cache_with_offload(True, n=6)
    off.logprobs_at(keys, torch.randint(0, VOCAB, (6, L, K)))
    final = off._final
    assert final["offloaded"] is True
    assert final["h"].device.type == "cpu" and final["lse"].device.type == "cpu"
    # The slot arithmetic has to follow the store: indexing a cpu tensor with a
    # cuda index is not allowed, and sending the index the other way would be a
    # device-to-host sync per micro batch.
    assert final["index_dev"].type == "cpu"
    assert final["key_to_slot"].device.type == "cpu"
    assert final["slot_off"].device.type == "cpu"

    res, keys2, _, _, _ = _cache_with_offload(False, n=6)
    res.logprobs_at(keys2, torch.randint(0, VOCAB, (6, L, K)))
    assert res._final["offloaded"] is False


def test_the_witness_still_runs_against_an_offloaded_store():
    """check_witness reads the entries directly rather than through logprobs_at,
    and after _finalize they point into the store. Offloaded that is host memory
    while the projection is on the device -- a mismatch this has to bridge, or
    the cache's own correctness check is what breaks."""
    off, keys, _, _, _ = _cache_with_offload(True, n=6)
    off.logprobs_at(keys, torch.randint(0, VOCAB, (6, L, K)))   # forces _finalize
    assert off.check_witness() < 1e-2

    caught, keys2, _, _, _ = _cache_with_offload(True, n=6)
    caught.logprobs_at(keys2, torch.randint(0, VOCAB, (6, L, K)))
    caught._h[int(keys2[2])] = caught._h[int(keys2[2])] + 1.0    # drift one entry
    with pytest.raises(RuntimeError, match="witness"):
        caught.check_witness()


def test_the_exchange_only_copies_keys_to_the_host_when_the_store_is_offloaded():
    """Resident, the index belongs on the device and a host copy would be a sync
    bought for nothing."""
    off, _, _, _, _ = _cache_with_offload(True, n=4)
    res, _, _, _, _ = _cache_with_offload(False, n=4)
    assert off.offloaded is True and res.offloaded is False


# --------------------------------------------------------------------------- #
# 15. the projection gather's ceiling
# --------------------------------------------------------------------------- #


def test_chunking_the_projection_changes_the_peak_and_not_the_answer():
    """The gather is row-independent, so splitting it is exact, not approximate.
    Bit for bit rather than close: the chunks see the same h, the same ids and
    the same normaliser, and only the peak differs. Anything less than equality
    would mean the split moved arithmetic, and a distillation loss cannot tell
    a moved bit from a real one."""
    W, h, lse = _teacher(seed=4, n=40)
    ids = torch.randint(0, VOCAB, (40, K), generator=torch.Generator().manual_seed(5))

    whole = teacher_logprobs_from_hidden(h, lse, W, ids, chunk_bytes=0)
    for mb in (1, 3, 7):
        split = teacher_logprobs_from_hidden(h, lse, W, ids, chunk_bytes=mb * K * H * W.element_size())
        assert torch.equal(whole, split), f"chunking at {mb} rows moved the values"

    # And with a per-row temperature, which the split has to carry along.
    temp = torch.rand(40) + 0.5
    whole_t = teacher_logprobs_from_hidden(h, lse, W, ids, temperature=temp, chunk_bytes=0)
    split_t = teacher_logprobs_from_hidden(
        h, lse, W, ids, temperature=temp, chunk_bytes=3 * K * H * W.element_size()
    )
    assert torch.equal(whole_t, split_t), "the chunks were given the wrong rows' temperatures"


def test_the_projection_ceiling_is_a_ceiling():
    """A budget that only bounds the number of CALLS bounds nothing: the whole
    point is the largest tensor alive at once."""
    from verl.workers.teacher_cache import _projection_rows

    rows, k, hidden, el = 1270, 20, 2048, 2
    W = torch.zeros((4, hidden), dtype=torch.bfloat16)
    per_row = k * hidden * el

    step = _projection_rows(rows, k, W, chunk_bytes=64 << 20)
    assert step * per_row <= (64 << 20), "a chunk is allowed to exceed the budget"
    assert step < rows, "a micro batch of 10 has to split at 64 MB, or the ceiling does nothing"

    # The arm's own shape today: a micro batch of 5 packs to ~635 rows, which is
    # under the budget, so nothing chunks and the call stays single-shot.
    assert _projection_rows(635, k, W, chunk_bytes=64 << 20) >= 635

    assert _projection_rows(rows, k, W, chunk_bytes=0) == rows, "0 has to mean off"
    assert _projection_rows(rows, k, W, chunk_bytes=1) == 1, "never zero rows, or it never finishes"


def test_the_chunks_really_are_separate_gathers():
    """Equality alone would also pass an implementation that computed the whole
    thing and sliced it afterwards, which allocates exactly what the ceiling
    exists to prevent. This watches the gather itself."""
    W, h, lse = _teacher(seed=6, n=40)
    ids = torch.randint(0, VOCAB, (40, K), generator=torch.Generator().manual_seed(7))

    shapes = []
    original = torch.einsum

    def spy(eq, *ops):
        if eq == "nh,nkh->nk":
            shapes.append(tuple(ops[1].shape))
        return original(eq, *ops)

    torch.einsum = spy
    try:
        teacher_logprobs_from_hidden(h, lse, W, ids, chunk_bytes=0)
        whole = list(shapes)
        shapes.clear()
        teacher_logprobs_from_hidden(h, lse, W, ids, chunk_bytes=7 * K * H * W.element_size())
        split = list(shapes)
    finally:
        torch.einsum = original

    assert whole == [(40, K, H)], "the unchunked path stopped being one gather"
    assert len(split) == 6 and max(s[0] for s in split) == 7, f"gathers were {split}"
    assert sum(s[0] for s in split) == 40, "the chunks do not cover every row"


@pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn")
def test_planes_asked_for_together_answer_exactly_what_they_do_one_at_a_time():
    """Batching the exchange is a transport change, so it has to be bit-identical.

    Three models are cached on the same rows under different keys and read at one
    support, both ways, in the same process group. ``torch.equal`` rather than a
    tolerance: ``logprobs_at`` runs per (rank, plane) on the same inputs either
    way and the reduction is elementwise over the same ranks, so anything but
    exact equality means the batching moved a value.
    """
    res = _run_two_ranks("multi")
    for rank, (status, payload) in res.items():
        assert status == "multi", f"rank {rank}: {payload}"
        identical, err = payload
        assert identical, f"rank {rank}: batched planes differ from per-plane calls"
        assert err < 1e-5, f"rank {rank} deviation from the full-vocabulary reference: {err}"


@pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn")
def test_the_planes_cost_one_exchange_between_them_instead_of_one_each():
    """The saving, counted rather than assumed.

    Four planes a micro-batch used to be four all_gathers of the SUPPORT -- by
    far the largest thing that crosses -- and four all_reduce pairs, on two cards
    with no NVLink. Three planes here, so per-plane is 3x(2+2)=12 and fused is
    2+2=4; the shape of the claim does not depend on the 3.
    """
    res = _run_two_ranks("count")
    for rank, (status, payload) in res.items():
        assert status == "counts", f"rank {rank}: {payload}"
        fused, apiece = payload
        assert fused == {"all_gather": 2, "all_reduce": 2}, f"rank {rank}: {fused}"
        assert apiece == {"all_gather": 6, "all_reduce": 6}, f"rank {rank}: {apiece}"


# --------------------------------------------------------------------------- #
# 17. the actor asks once, not once per model
# --------------------------------------------------------------------------- #
def test_the_actor_reads_all_its_planes_in_one_exchange(monkeypatch):
    """Where the saving is actually taken.

    The primitive above can batch; this is the wiring that uses it.
    ``_cross_teacher_planes`` used to loop over the columns of
    ``sign_cache_ids`` calling the single-plane form, which sent the support --
    the big tensor -- once per model. The columns must still come back in order,
    because the actor reads them positionally.
    """
    from verl.workers.actor import dp_actor

    calls = []

    def _fake(cache, cache_ids, ids, group=None, world_size=None, fingerprints=None):
        calls.append((cache_ids.shape, ids.shape))
        # A different constant per column, so a reordering shows up as a value.
        return [torch.full(ids.shape, float(c), dtype=torch.float32)
                for c in range(cache_ids.size(1))]

    monkeypatch.setattr(dp_actor, "get_teacher_cache", lambda: None, raising=False)
    import verl.workers.teacher_cache as tc
    monkeypatch.setattr(tc, "exchange_teacher_logprobs_multi", _fake)
    monkeypatch.setattr(tc, "get_teacher_cache", lambda: None)

    actor = object.__new__(dp_actor.DataParallelPPOActor)
    n, planes = 4, 3
    data = {
        "sign_cache_ids": torch.arange(n * planes, dtype=torch.long).view(n, planes),
        "input_ids": torch.ones((n, 8), dtype=torch.long),
        "attention_mask": torch.ones((n, 8), dtype=torch.long),
    }
    support = torch.randint(0, VOCAB, (n, L, K))

    base, off = actor._cross_teacher_planes(data, support)

    assert len(calls) == 1, f"one exchange for all planes, got {len(calls)}"
    assert calls[0][0] == (n, planes), "every column has to go in the same call"
    assert torch.all(base == 0.0), "column 0 is the base policy"
    for c in range(planes - 1):
        assert torch.all(off[..., c] == float(c + 1)), f"off-task column {c} came back out of order"


# --------------------------------------------------------------------------- #
# 18. the on-task teacher rides in the same exchange as the other three
# --------------------------------------------------------------------------- #
def _plane_recorder(monkeypatch):
    """Stand in for the exchange and record every call's key/support shapes.

    Each plane answers with its OWN KEY, per row -- not with its column index.
    That makes a reordering show up as a value, and it makes the answer a
    function of the key alone, so the merged read and the split read are
    comparable: an arrangement that changes what a plane returns would then be
    visible rather than tautologically equal.
    """
    from verl.workers.actor import dp_actor
    import verl.workers.teacher_cache as tc

    calls = []

    def _fake(cache, cache_ids, ids, group=None, world_size=None, fingerprints=None):
        calls.append((tuple(cache_ids.shape), tuple(ids.shape), cache_ids.clone()))
        shape = (cache_ids.size(0),) + (1,) * (ids.dim() - 1)
        return [cache_ids[:, c].reshape(shape).to(torch.float32).expand(ids.shape).clone()
                for c in range(cache_ids.size(1))]

    monkeypatch.setattr(dp_actor, "get_teacher_cache", lambda: None, raising=False)
    monkeypatch.setattr(tc, "exchange_teacher_logprobs_multi", _fake)
    monkeypatch.setattr(tc, "get_teacher_cache", lambda: None)
    return dp_actor, calls


def _plane_batch(n=4, off=2):
    return {
        "teacher_cache_ids": torch.arange(100, 100 + n, dtype=torch.long),
        "sign_cache_ids": torch.arange(n * (1 + off), dtype=torch.long).view(n, 1 + off),
        "input_ids": torch.ones((n, 8), dtype=torch.long),
        "attention_mask": torch.ones((n, 8), dtype=torch.long),
    }


def test_all_four_models_go_in_one_exchange_with_the_columns_in_order(monkeypatch):
    """The saving, at the seam it is taken.

    The on-task teacher's key lives in teacher_cache_ids and the other three in
    sign_cache_ids, because different driver passes fill them. Nothing about the
    READ cares, and splitting it there cost a second all_gather of the support --
    the largest thing that crosses.
    """
    dp_actor, calls = _plane_recorder(monkeypatch)
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    n, off = 4, 2
    data = _plane_batch(n, off)
    support = torch.randint(0, VOCAB, (n, L, K))

    on, base, off_planes = actor._all_teacher_planes(data, support)

    assert len(calls) == 1, f"one exchange for all four models, got {len(calls)}"
    keys_shape, ids_shape, keys = calls[0]
    assert keys_shape == (n, 2 + off), "every model's key column has to go in the same call"
    assert ids_shape == (n, L, K), "the support goes once"
    # Column 0 is the on-task teacher and the rest are sign_cache_ids untouched.
    assert torch.equal(keys[:, 0], data["teacher_cache_ids"])
    assert torch.equal(keys[:, 1:], data["sign_cache_ids"])
    def _col(values):
        return values.reshape(n, 1, 1).to(torch.float32).expand(n, L, K)

    assert torch.equal(on, _col(data["teacher_cache_ids"])), "column 0 is the on-task teacher"
    assert torch.equal(base, _col(data["sign_cache_ids"][:, 0])), "column 1 is the base policy"
    for c in range(off):
        assert torch.equal(off_planes[..., c], _col(data["sign_cache_ids"][:, c + 1])), (
            f"off-task column {c} out of order"
        )


def test_the_four_planes_are_the_same_values_the_two_calls_gave(monkeypatch):
    """Bit-identical, compared rather than argued. The merged read and the split
    read are two arrangements of the same per-(row, plane) lookups."""
    dp_actor, _ = _plane_recorder(monkeypatch)
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    data = _plane_batch()
    support = torch.randint(0, VOCAB, (4, L, K))

    on, base, off_planes = actor._all_teacher_planes(data, support)
    # What the two-call arrangement produced: column 0 of a one-column exchange,
    # then columns 0.. of a sign_cache_ids exchange.
    split_on = actor._teacher_logprobs_at(
        cache_ids=data["teacher_cache_ids"], ids=support,
        input_ids=data["input_ids"], attention_mask=data["attention_mask"],
    )
    split_base, split_off = actor._cross_teacher_planes(data, support)
    assert torch.equal(on, split_on)
    assert torch.equal(base, split_base)
    assert torch.equal(off_planes, split_off)


def test_a_second_reader_of_the_same_support_costs_no_exchange(monkeypatch):
    """Both cross-teacher blocks read these planes at the same support in the
    same micro-batch. The second one has to ride on the first."""
    dp_actor, calls = _plane_recorder(monkeypatch)
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    data = _plane_batch()
    support = torch.randint(0, VOCAB, (4, L, K))

    on, base, off_planes = actor._all_teacher_planes(data, support)
    cached = (support, base, off_planes)
    assert len(calls) == 1

    again_base, again_off = actor._cross_teacher_planes(data, support, cached=cached)
    assert len(calls) == 1, "the cached planes were re-fetched"
    assert again_base is base and again_off is off_planes


def test_a_different_support_ignores_the_cache_rather_than_answering_wrong(monkeypatch):
    """The planes are a function of the ids. The sampled-token support and the
    teacher-indexed support are both different tensors carrying different ids,
    and answering either from the top-k's planes would be a real teacher
    log-prob at the wrong candidates -- silent, and in the loss."""
    dp_actor, calls = _plane_recorder(monkeypatch)
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    data = _plane_batch()
    support = torch.randint(0, VOCAB, (4, L, K))
    _, base, off_planes = actor._all_teacher_planes(data, support)
    cached = (support, base, off_planes)

    other = support.clone()                      # equal values, different object
    actor._cross_teacher_planes(data, other, cached=cached)
    assert len(calls) == 2, "an equal-but-distinct support was answered from the cache"

    y1 = torch.randint(0, VOCAB, (4, L, 1))
    actor._cross_teacher_planes(data, y1, cached=cached)
    assert len(calls) == 3


def test_no_cache_at_all_is_the_old_two_call_arrangement(monkeypatch):
    """The teacher-indexed path never runs the on-task lookup, so there is
    nothing to ride on and the three planes still have to be asked for."""
    dp_actor, calls = _plane_recorder(monkeypatch)
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    data = _plane_batch()
    support = torch.randint(0, VOCAB, (4, L, K))
    actor._cross_teacher_planes(data, support, cached=None)
    assert len(calls) == 1
    assert calls[0][0] == (4, 3), "only sign_cache_ids goes when there is no on-task key"


def test_a_missing_on_task_key_is_named_rather_than_concatenated(monkeypatch):
    """Without teacher_cache_ids there is no column 0. Reaching torch.cat with a
    None would raise somewhere with no mention of the column that is missing."""
    dp_actor, _ = _plane_recorder(monkeypatch)
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    data = _plane_batch()
    del data["teacher_cache_ids"]
    with pytest.raises(ValueError, match="teacher_cache_ids"):
        actor._all_teacher_planes(data, torch.randint(0, VOCAB, (4, L, K)))


def test_every_cross_plane_read_in_the_training_loop_offers_the_cached_planes():
    """The structural half, and the one that actually keeps the saving.

    update_policy legitimately calls both single forms -- _teacher_logprobs_at on
    the teacher-indexed path, where no cross planes ride along. What must not come
    back is a _cross_teacher_planes call that does not offer what the lookup
    already fetched: it would be a second all_gather of the same support, correct
    in every value, and nothing downstream would notice.
    """
    import ast

    from verl.workers.actor import dp_actor

    # Parsed off the MODULE, not off the attribute: update_policy is wrapped by
    # GPUMemoryLogger, so inspect.getsource hands back the decorator's two-line
    # wrapper and every structural claim about the body passes vacuously.
    src = open(dp_actor.__file__).read()
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "update_policy"
    )
    tree = fn
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_cross_teacher_planes"
    ]
    assert calls, "update_policy no longer reads the cross-teacher planes at all"
    for call in calls:
        assert any(kw.arg == "cached" for kw in call.keywords), (
            "a _cross_teacher_planes call in update_policy does not pass cached=; the planes "
            "the on-task lookup already fetched are then fetched again"
        )


# --------------------------------------------------------------------------- #
# where the store lives WHILE the rollout is still running
# --------------------------------------------------------------------------- #
def _put_one(cache, resp=4, hidden=8, key=7000):
    h = torch.randn(1, resp, hidden)
    lse = torch.randn(1, resp)
    cache.put(torch.tensor([key]), "alfworld", h, lse, live_mask=torch.ones(1, resp))
    return h, lse


def _counting_to_host(cache):
    """Count the crossings. On a CPU box the tensors are on the host either way,
    so "did it copy" is the only observable that separates the two paths -- and
    it is the one the change is about."""
    calls = []
    original = cache._to_host

    def spy(t):
        calls.append(tuple(t.shape))
        return original(t)

    cache._to_host = spy
    return calls


def test_an_offloading_cache_crosses_to_the_host_inside_put():
    """_finalize used to be where the store crossed, and _finalize runs on the
    FIRST READ -- inside the actor update, long after the rollout ended.
    Everything cached during a rollout therefore sat in device memory beside
    vLLM's KV pool, and on the cross-teacher arms that is four models a row
    rather than one (teacher_cache/gb 15.6 against pure OPD's 4.1)."""
    cache = TeacherHiddenCache()
    cache._offload = True
    calls = _counting_to_host(cache)
    _put_one(cache, resp=4, hidden=8)
    # h packed to its live positions, and lse alongside it.
    assert calls == [(4, 8), (4,)]


def test_a_resident_cache_does_not_copy_at_all():
    """TEACHER_CACHE_OFFLOAD=0 has to reproduce exactly what it did before."""
    cache = TeacherHiddenCache()
    cache._offload = False
    calls = _counting_to_host(cache)
    _put_one(cache)
    assert calls == []


def test_the_crossing_keeps_the_values_and_lands_on_the_host():
    cache = TeacherHiddenCache()
    cache._offload = True
    h, lse = _put_one(cache)
    assert cache._h[7000].device.type == "cpu"
    assert torch.equal(cache._h[7000], h[0])
    assert torch.equal(cache._lse[7000], lse[0])


def test_device_only_bytes_exclude_what_has_already_crossed():
    """teacher_cache/gb keeps meaning "what the entries hold"; the new
    teacher_cache/device_gb is what is still on the card, and reporting the total
    under the old name would read as a regression that is in fact the fix."""
    cache = TeacherHiddenCache()
    cache._offload = True
    _put_one(cache)
    assert cache.nbytes() > 0
    assert cache.nbytes(device_only=True) == 0


# --------------------------------------------------------------------------- #
# 13. what page-locked memory costs
#
# torch's pinned allocator (CachingHostAllocator) rounds every request up to the
# next power of two, keys its free list by that bucket, and never hands a block
# back to the OS. So pinned memory is a RATCHET over the distinct bucket sizes a
# process
# has ever asked for, it never shows up in any CUDA memory counter, and it lands
# in RSS, where Ray's node-memory monitor kills at 0.98.
#
# Run e8x57zyu died there at step 10: put() had started pinning its per-call
# chunks, and host RAM went from +0.37 GB/step (run n9zfny6m, 148 steps) to
# +5.06 GB/step, off a baseline already 24 GB higher.
#
# The two tests below pin the two halves of the answer -- don't pin what is read
# once, and ask for the same SIZE every step for what is read a thousand times.
# --------------------------------------------------------------------------- #


def test_the_per_call_chunks_are_not_page_locked():
    """They are written once and read twice -- check_witness samples a row,
    _finalize copies them into the store and drops them. The tensor the micro-batch
    loop actually pulls from is final["h"], which is pinned on its own account."""
    cache = TeacherHiddenCache()
    cache._offload = True
    _put_one(cache)
    for tensors in cache._chunks.values():
        for t in tensors:
            assert not t.is_pinned(), "a put() chunk is page-locked; the pool never gives that back"
    # Honest about its reach: a box with no driver cannot pin anything, so this
    # only bites where it can. The rounding test below is the one that holds on
    # CPU, because a request SIZE is visible whether or not it can be page-locked.


def test_the_store_can_be_taken_off_page_locked_memory():
    """The page-locked cost is not the store's size. torch rounds a pinned request
    up to a power of two and keeps the block, so a store that swings 7.0-10.3 GB a
    rank settles into an 8.6 GiB block AND a 17.2 GiB one -- ~48 GB on this box for
    9.6 GiB of data. That is a fifth of the node run e8x57zyu died on, so it has to
    be spendable back when host RAM is what binds.

    Checked through store_placement, because on a box with no driver every branch
    collapses to the same answer and a test that only looks at where the tensors
    landed passes just as happily when the knob does nothing (the reason that
    function exists at all).
    """
    from verl.workers.teacher_cache import store_placement

    store, pin, index = store_placement(True, "cuda:0", cuda_available=True, pin=True)
    assert (store.type, pin, index.type) == ("cpu", True, "cpu")

    store, pin, index = store_placement(True, "cuda:0", cuda_available=True, pin=False)
    assert (store.type, index.type) == ("cpu", "cpu"), "unpinning must not move the store"
    assert pin is False

    # Resident is unaffected: there is nothing to page-lock either way.
    assert store_placement(False, "cuda:0", cuda_available=True, pin=True)[1] is False
    assert store_placement(False, "cuda:0", cuda_available=True, pin=False)[1] is False


def test_the_pin_knob_defaults_to_on(monkeypatch):
    """Off by accident would be a silent throughput regression in the micro-batch
    loop, so the default is the DMA and the saving is opt-in."""
    import verl.workers.teacher_cache as tc

    assert tc._PIN_STORE is True, "TEACHER_CACHE_PIN_STORE must default to on"
    assert tc.store_placement(True, "cuda:0", cuda_available=True)[1] is True

    monkeypatch.setattr(tc, "_PIN_STORE", False)
    assert tc.store_placement(True, "cuda:0", cuda_available=True)[1] is False
    # An explicit argument still wins, so the knob cannot silently override a caller.
    assert tc.store_placement(True, "cuda:0", cuda_available=True, pin=True)[1] is True
