"""src -> dst -> token: which teacher sends which vocabulary into whose states.

Two accumulators already exist and neither can answer this. ``SignPairCounts``
answers "how often does src agree with dst's teacher" with the vocabulary summed
out; ``TokenStateCounts`` answers "which tokens" with the SENDER summed out. The
question a transfer claim actually rests on -- are the tokens alfworld's teacher
pushes into search's states the same ones webshop's teacher pushes there -- needs
both axes at once, and that is what makes it expensive.

The cost is the design, so it is what most of these tests pin. A naive
``(dst, src, sign_on, sign_src, V)`` table is 12.3M cells at T=3; collapsing the
nine sign combinations to the three that carry an opinion FROM src, and dropping
the structurally empty ``src == dst`` diagonal, is 2.7M -- smaller than the
per-state table already shipped. Both reductions are lossless for the question,
and both are asserted below rather than left as a comment.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.sign_weights import (
        PAIR_TOKEN_CLASSES,
        SignPairTokens,
        candidate_effect,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


TASKS = ["alfworld", "search", "webshop"]
DEADZONE = 0.1


def _shift(base, delta):
    """A teacher's log-probs: the base shifted by ``delta`` nats per candidate."""
    return base + torch.as_tensor(delta, dtype=base.dtype)


def _scene(deltas_on, deltas_off, *, ids, base_logp=-2.0, bs=1, resp=1):
    """One position, k candidates, with the shifts stated directly.

    ``deltas_off`` is ``(n_off, k)``. Working in shifts rather than in
    probabilities is what makes the class assignment readable: the deadzone acts
    on exactly these numbers.
    """
    k = len(ids)
    base = torch.full((bs, resp, k), float(base_logp))
    on = _shift(base, deltas_on)
    off = torch.stack([_shift(base, d) for d in deltas_off], dim=-1)
    return {
        "support_ids": torch.tensor(ids).view(1, 1, k).expand(bs, resp, k).contiguous(),
        "on_task_logprob": on,
        "off_task_logprobs": off,
        "base_logprob": base,
        "response_mask": torch.ones(bs, resp),
    }


def _table(n_tasks=3, vocab=16, top_n=4, device="cpu"):
    return SignPairTokens(n_tasks=n_tasks, vocab_size=vocab, device=device, top_n=top_n)


def _fold(tab, scene, *, dst=0, srcs=(1, 2), effect=None):
    bs = scene["on_task_logprob"].size(0)
    tab.update(
        **scene,
        task_ids=torch.full((bs,), dst, dtype=torch.long),
        off_plane_tasks=torch.tensor(srcs, dtype=torch.long).view(1, -1).expand(bs, -1).contiguous(),
        deadzone=DEADZONE,
        effect=effect,
    )
    return tab


# --------------------------------------------------------------------------- #
# The two reductions that make it affordable
# --------------------------------------------------------------------------- #
def test_the_diagonal_is_not_allocated_at_all():
    """src == dst cannot happen: a teacher is never off-task on its own rows.

    A square T x T layout would spend a third of the array on cells that can
    only ever be zero -- 27 MB of 82 at T=3 and V=151,936.
    """
    tab = _table(n_tasks=3, vocab=16)
    assert tab.n_pairs == 3 * 2
    assert tab.n.numel() == tab.n_pairs * len(PAIR_TOKEN_CLASSES) * 16


def test_the_pair_index_is_invertible_over_every_ordered_pair():
    """Compressing the diagonal out is only safe if nothing collides."""
    for T in (2, 3, 5):
        tab = _table(n_tasks=T, vocab=4)
        seen = {}
        for dst in range(T):
            for src in range(T):
                if src == dst:
                    continue
                p = tab._pair_index(dst, src)
                assert 0 <= p < tab.n_pairs
                assert p not in seen, (p, seen.get(p), (dst, src))
                seen[p] = (dst, src)
        assert len(seen) == T * (T - 1)
        assert {(d, s) for _, (d, s) in sorted(seen.items())} == {
            (d, s) for d in range(T) for s in range(T) if d != s
        }


def test_the_three_classes_are_exactly_the_ones_carrying_an_opinion():
    """A candidate src said nothing about has no token identity worth 152k cells.

    Four candidates, one per outcome: agree, conflict, blindspot, and src silent.
    The first three land in their class and the fourth lands nowhere.
    """
    tab = _table(vocab=16)
    scene = _scene(
        #        agree  conflict  blindspot  src-silent
        deltas_on=[0.5,   0.5,      0.0,       0.5],
        deltas_off=[[0.5, -0.5,     0.5,       0.0]],
        ids=[3, 4, 5, 6],
    )
    _fold(tab, scene, srcs=(1,))
    n, _mass, _eff = tab._cpu()
    p = tab._pair_index(0, 1)
    assert int(n[p, 0, 3]) == 1   # agree
    assert int(n[p, 1, 4]) == 1   # conflict
    assert int(n[p, 2, 5]) == 1   # blindspot
    assert int(n[:, :, 6].sum()) == 0, "a silent src must not occupy a cell"
    assert int(n.sum()) == 3


def test_a_shift_inside_the_deadzone_is_silence_on_either_side():
    tab = _table(vocab=16)
    scene = _scene(
        deltas_on=[0.05, 0.5],
        deltas_off=[[0.5, 0.05]],
        ids=[3, 4],
    )
    _fold(tab, scene, srcs=(1,))
    n, _m, _e = tab._cpu()
    p = tab._pair_index(0, 1)
    # on-task silent + src speaks -> blindspot; on-task speaks + src silent -> nothing
    assert int(n[p, 2, 3]) == 1
    assert int(n[:, :, 4].sum()) == 0


# --------------------------------------------------------------------------- #
# The axes mean what they say
# --------------------------------------------------------------------------- #
def test_each_sender_is_filed_under_itself():
    """Two off-task planes disagreeing about one token must not be pooled."""
    tab = _table(vocab=16)
    scene = _scene(
        deltas_on=[0.5],
        deltas_off=[[0.5], [-0.5]],   # src 1 agrees, src 2 conflicts
        ids=[7],
    )
    _fold(tab, scene, dst=0, srcs=(1, 2))
    n, _m, _e = tab._cpu()
    assert int(n[tab._pair_index(0, 1), 0, 7]) == 1   # agree
    assert int(n[tab._pair_index(0, 2), 1, 7]) == 1   # conflict
    assert int(n[tab._pair_index(0, 1), 1, 7]) == 0
    assert int(n[tab._pair_index(0, 2), 0, 7]) == 0


def test_the_receiver_axis_follows_the_rows_task():
    tab = _table(vocab=16)
    scene = _scene(deltas_on=[0.5], deltas_off=[[0.5]], ids=[7])
    _fold(tab, scene, dst=2, srcs=(0,))
    n, _m, _e = tab._cpu()
    assert int(n[tab._pair_index(2, 0), 0, 7]) == 1
    assert int(n[tab._pair_index(0, 2), 0, 7]) == 0


def test_masked_and_untagged_rows_reach_no_cell():
    """A padding row's task is -1, and filing it under a task would invent one."""
    tab = _table(vocab=16)
    scene = _scene(deltas_on=[0.5], deltas_off=[[0.5]], ids=[7], bs=2, resp=2)
    scene["response_mask"] = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    tab.update(
        **scene,
        task_ids=torch.tensor([0, -1]),
        off_plane_tasks=torch.tensor([[1], [1]]),
        deadzone=DEADZONE,
    )
    n, _m, _e = tab._cpu()
    assert int(n.sum()) == 1, "one valid (row, position) only"


def test_a_negative_plane_task_is_dropped_without_disturbing_the_index():
    tab = _table(vocab=16)
    scene = _scene(deltas_on=[0.5], deltas_off=[[0.5], [0.5]], ids=[7])
    _fold(tab, scene, dst=1, srcs=(-1, 2))
    n, _m, _e = tab._cpu()
    assert int(n.sum()) == 1
    assert int(n[tab._pair_index(1, 2), 0, 7]) == 1


# --------------------------------------------------------------------------- #
# mass and effect
# --------------------------------------------------------------------------- #
def test_mass_is_the_receiving_teachers_probability_there():
    """A token src keeps voting on that dst's teacher was never going to say
    reaches nothing, and only the mass column can say so."""
    tab = _table(vocab=16)
    # The two candidates differ in the on-task teacher's own probability while
    # every SHIFT stays +0.5, so both are still an agreement and only the mass
    # separates them.
    on = torch.tensor([[[-0.5, -5.0]]])
    scene = _scene(deltas_on=[0.5, 0.5], deltas_off=[[0.5, 0.5]], ids=[3, 4])
    scene["on_task_logprob"] = on
    scene["base_logprob"] = on - 0.5
    scene["off_task_logprobs"] = (on + 0.5).unsqueeze(-1)
    _fold(tab, scene, srcs=(1,))
    _n, mass, _e = tab._cpu()
    p = tab._pair_index(0, 1)
    assert float(mass[p, 0, 3]) == pytest.approx(float(np.exp(-0.5)), abs=1e-6)
    assert float(mass[p, 0, 4]) == pytest.approx(float(np.exp(-5.0)), abs=1e-6)


def test_every_unanimous_sender_is_credited_with_the_whole_effect():
    """Not a share of it. The gate fires on unanimity, so the question the
    column answers is "was src one of the voices", not "how much of the vote
    was src's" -- and dividing by the number of senders would make the answer
    depend on how many tasks the run happens to have."""
    tab = _table(vocab=16)
    scene = _scene(deltas_on=[0.5], deltas_off=[[0.5], [0.5]], ids=[7])
    eff = torch.tensor([[[0.25]]], dtype=torch.float64)
    _fold(tab, scene, dst=0, srcs=(1, 2), effect=eff)
    _n, _m, e = tab._cpu()
    assert float(e[tab._pair_index(0, 1), 0, 7]) == pytest.approx(0.25)
    assert float(e[tab._pair_index(0, 2), 0, 7]) == pytest.approx(0.25)


def test_the_effect_column_is_the_shared_one_and_not_a_second_formula():
    """It comes from candidate_effect, the same function TokenStateCounts uses."""
    tab = _table(vocab=16)
    on = torch.tensor([[[-0.7, -1.6]]])
    w = torch.tensor([[[1.25, 1.0]]])
    eff = candidate_effect(mode="target", on_task_logprob=on, weight=w)
    scene = _scene(deltas_on=[0.5, 0.5], deltas_off=[[0.5, 0.5]], ids=[3, 4])
    scene["on_task_logprob"] = on
    _fold(tab, scene, srcs=(1,), effect=eff)
    _n, _m, e = tab._cpu()
    p = tab._pair_index(0, 1)
    assert float(e[p, 0, 3]) == pytest.approx(float(eff[0, 0, 0]), abs=1e-9)
    assert float(e[p, 0, 4]) == pytest.approx(float(eff[0, 0, 1]), abs=1e-9)


def test_an_unweighted_arm_still_gets_the_counts():
    """effect=None is a real configuration -- measure_only, or an observer run
    -- and it must cost the table nothing but the effect column."""
    tab = _table(vocab=16)
    scene = _scene(deltas_on=[0.5], deltas_off=[[0.5]], ids=[7])
    _fold(tab, scene, srcs=(1,), effect=None)
    n, mass, e = tab._cpu()
    assert int(n.sum()) == 1 and float(mass.sum()) > 0
    assert float(e.abs().sum()) == 0.0


# --------------------------------------------------------------------------- #
# The readings
# --------------------------------------------------------------------------- #
def _overlap_scene(ids_a, ids_b, vocab=16, reps=1):
    """Two senders naming their own token sets at one receiver."""
    tab = _table(vocab=vocab)
    for ids, src in ((ids_a, 1), (ids_b, 2)):
        scene = _scene(
            deltas_on=[0.5] * len(ids),
            deltas_off=[[0.5] * len(ids)],
            ids=list(ids),
            bs=reps,
        )
        _fold(tab, scene, dst=0, srcs=(src,))
    return tab


def test_token_overlap_is_1_when_the_senders_name_the_same_tokens():
    tab = _overlap_scene([3, 4, 5], [3, 4, 5])
    m = tab.scalar_metrics(task_names=TASKS)
    key = "sign_weight/pair/token_overlap/agree/search__and__webshop__on__alfworld"
    assert m[key] == pytest.approx(1.0)


def test_token_overlap_is_0_when_the_senders_share_nothing():
    tab = _overlap_scene([3, 4, 5], [6, 7, 8])
    m = tab.scalar_metrics(task_names=TASKS)
    key = "sign_weight/pair/token_overlap/agree/search__and__webshop__on__alfworld"
    assert m[key] == pytest.approx(0.0)


def test_token_overlap_is_weighted_not_set_valued():
    """A token seen once and a token seen a thousand times are not the same
    claim, and a set Jaccard scores them identically."""
    same_sets = _overlap_scene([3, 4], [3, 4], reps=1)
    lopsided = _table(vocab=16)
    for ids, src, reps in (([3, 4], 1, 1), ([3, 4], 2, 9)):
        scene = _scene(deltas_on=[0.5, 0.5], deltas_off=[[0.5, 0.5]], ids=ids, bs=reps)
        _fold(lopsided, scene, dst=0, srcs=(src,))
    key = "sign_weight/pair/token_overlap/agree/search__and__webshop__on__alfworld"
    assert same_sets.scalar_metrics(task_names=TASKS)[key] == pytest.approx(1.0)
    # identical SETS, 1:9 counts -> 1/9
    assert lopsided.scalar_metrics(task_names=TASKS)[key] == pytest.approx(1.0 / 9.0)


def test_the_shape_metrics_separate_a_small_stable_set_from_a_broad_one():
    narrow = _overlap_scene([3], [3], reps=8)
    broad = _table(vocab=16)
    ids = list(range(3, 11))
    scene = _scene(deltas_on=[0.5] * len(ids), deltas_off=[[0.5] * len(ids)], ids=ids)
    _fold(broad, scene, dst=0, srcs=(1,))
    head = "sign_weight/pair/token/agree/search__on__alfworld"
    n_narrow = narrow.scalar_metrics(task_names=TASKS)[f"{head}/n_distinct"]
    n_broad = broad.scalar_metrics(task_names=TASKS)[f"{head}/n_distinct"]
    assert n_narrow == 1.0 and n_broad == 8.0
    assert narrow.scalar_metrics(task_names=TASKS)[f"{head}/top4_share"] == pytest.approx(1.0)
    assert broad.scalar_metrics(task_names=TASKS)[f"{head}/top4_share"] == pytest.approx(0.5)


def test_the_rows_name_the_sender_the_receiver_and_the_class():
    tab = _table(vocab=16)
    scene = _scene(deltas_on=[0.5, 0.0], deltas_off=[[0.5, 0.5]], ids=[3, 4])
    _fold(tab, scene, dst=0, srcs=(1,), effect=torch.tensor([[[0.5, -0.25]]], dtype=torch.float64))
    rows = tab.top_tokens(task_names=TASKS)
    assert rows
    for r in rows:
        assert set(r) == {
            "table", "dst", "src", "cls", "ranked_by", "rank",
            "token_id", "count", "mass", "effect_net",
        }
        assert r["table"] == "pair_token"
        assert r["dst"] == "alfworld" and r["src"] == "search"
    assert {r["cls"] for r in rows} == {"agree", "blindspot"}
    assert {r["ranked_by"] for r in rows} == {"count", "mass", "abs_effect"}
    top = [r for r in rows if r["cls"] == "blindspot" and r["ranked_by"] == "abs_effect"][0]
    assert top["token_id"] == 4 and top["effect_net"] == pytest.approx(-0.25)


def test_an_empty_table_renders_nothing_rather_than_zeros():
    """A pair that never fired must be absent, not reported as 0 -- the two are
    different claims and only one of them is true."""
    tab = _table(vocab=16)
    assert tab.scalar_metrics(task_names=TASKS) == {}
    assert tab.top_tokens(task_names=TASKS) == []


def test_a_rendering_taken_before_more_folding_is_not_reused():
    tab = _table(vocab=16)
    scene = _scene(deltas_on=[0.5], deltas_off=[[0.5]], ids=[7])
    _fold(tab, scene, srcs=(1,))
    first = tab.top_tokens(task_names=TASKS)[0]["count"]
    _fold(tab, scene, srcs=(1,))
    assert tab.top_tokens(task_names=TASKS)[0]["count"] == first + 1
