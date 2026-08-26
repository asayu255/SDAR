"""Per-vocabulary-token diagnostics for the cross-teacher sign weighting.

Everything else the mechanism reports has the vocabulary summed out, so
``frac_agree_pos = 0.2`` cannot distinguish "the same twenty tokens every step"
from "a different thousand". These tests pin the thing that can.

Three properties carry the weight here:

* the counts are the counts -- occurrences, masked correctly, filed under the
  right task;
* ``dw`` sums to ``Z - 1``, which is what makes the ranked-by-effect list a
  statement about the objective rather than a second frequency table;
* the concentration metrics actually separate a narrow mechanism from a broad
  one, since telling those apart is the entire reason the diagnostic exists.
"""

import json
import os

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.sign_weights import (
        ACTED_STATES,
        STATE_AGREE_NEG,
        STATE_AGREE_POS,
        STATE_CONFLICT_ON_POS,
        STATE_NAMES,
        STATE_NEUTRAL_ON,
        TokenStateCounts,
        candidate_weights,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


VOCAB = 64


def _counts(n_tasks=2, top_n=5):
    return TokenStateCounts(vocab_size=VOCAB, n_tasks=n_tasks, device="cpu", top_n=top_n)


def _weight_for(state, agree=1.5, agree_neg=0.5):
    """The target-mode table, applied to a state tensor."""
    return torch.where(
        state == STATE_AGREE_POS,
        torch.full_like(state, agree, dtype=torch.float32),
        torch.where(
            state == STATE_AGREE_NEG,
            torch.full_like(state, agree_neg, dtype=torch.float32),
            torch.ones_like(state, dtype=torch.float32),
        ),
    )


# --------------------------------------------------------------------------- #
# Accumulation
# --------------------------------------------------------------------------- #
def test_counts_mass_and_effect_are_what_the_candidates_say():
    tc = _counts()
    support = torch.tensor([[[7, 8, 9], [7, 8, 9]], [[7, 20, 21], [7, 20, 21]]])
    state = torch.tensor(
        [
            [[STATE_AGREE_POS, STATE_NEUTRAL_ON, STATE_AGREE_NEG]] * 2,
            [[STATE_AGREE_POS, STATE_AGREE_POS, STATE_NEUTRAL_ON]] * 2,
        ]
    )
    logp = torch.full((2, 2, 3), -2.0)
    p = float(torch.tensor(-2.0).exp())
    # Row 1's second position is padding.
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    tc.update(
        support_ids=support,
        state=state,
        weight=_weight_for(state),
        on_task_logprob=logp,
        response_mask=mask,
        task_ids=torch.tensor([0, 1]),
    )
    n, mass, dw = tc._cpu()

    # token 7 is reinforced at all three VALID positions
    assert int(n[0, STATE_AGREE_POS, 7]) == 3
    assert float(mass[0, STATE_AGREE_POS, 7]) == pytest.approx(3 * p, rel=1e-5)
    # ...split 2 / 1 between the two tasks
    assert int(n[1, STATE_AGREE_POS, 7]) == 2
    assert int(n[2, STATE_AGREE_POS, 7]) == 1
    # token 9 is suppressed, and only task 0 ever sees it
    assert int(n[0, STATE_AGREE_NEG, 9]) == 2
    assert int(n[2, STATE_AGREE_NEG, 9]) == 0
    # 3 valid positions x 3 candidates, and not one more
    assert int(n[0].sum()) == 9

    # dw carries the direction: reinforced positive, suppressed negative,
    # untouched exactly zero.
    assert float(dw[0, 7]) == pytest.approx(3 * 0.5 * p, rel=1e-5)
    assert float(dw[0, 9]) == pytest.approx(2 * -0.5 * p, rel=1e-5)
    assert float(dw[0, 8]) == pytest.approx(0.0, abs=1e-9)


def test_the_padding_positions_contribute_nothing():
    """The mask is folded into the values rather than the indices, so this is the
    check that the values really are zeroed and not merely scattered elsewhere."""
    support = torch.tensor([[[3, 4]], [[3, 4]]])
    state = torch.full((2, 1, 2), STATE_AGREE_POS)
    logp = torch.full((2, 1, 2), -1.0)

    both = _counts()
    both.update(
        support_ids=support, state=state, weight=_weight_for(state),
        on_task_logprob=logp, response_mask=torch.ones(2, 1), task_ids=None,
    )
    one = _counts()
    one.update(
        support_ids=support, state=state, weight=_weight_for(state),
        on_task_logprob=logp, response_mask=torch.tensor([[1.0], [0.0]]), task_ids=None,
    )
    assert int(both._cpu()[0][0].sum()) == 4
    assert int(one._cpu()[0][0].sum()) == 2
    assert float(one._cpu()[2][0].sum()) == pytest.approx(float(both._cpu()[2][0].sum()) / 2, rel=1e-5)


def test_a_row_with_no_task_reaches_the_pooled_scope_only():
    """adjust_batch's padding rows and any untagged row carry task id -1. They
    are still real candidates for the pooled view, but filing them under a task
    would invent one."""
    tc = _counts()
    support = torch.tensor([[[5, 6]], [[5, 6]]])
    state = torch.full((2, 1, 2), STATE_AGREE_POS)
    tc.update(
        support_ids=support,
        state=state,
        weight=_weight_for(state),
        on_task_logprob=torch.full((2, 1, 2), -1.0),
        response_mask=torch.ones(2, 1),
        task_ids=torch.tensor([0, -1]),
    )
    n = tc._cpu()[0]
    assert int(n[0, STATE_AGREE_POS].sum()) == 4  # pooled sees both rows
    assert int(n[1, STATE_AGREE_POS].sum()) == 2  # task 0 sees its own
    assert int(n[2, STATE_AGREE_POS].sum()) == 0  # and nothing was invented


def test_the_effect_column_sums_to_the_normaliser_it_decomposes():
    """``dw`` is per-token ``(w - 1) * p``, so summing it over the vocabulary has
    to reproduce ``Z - 1`` exactly. That identity is what lets the ranked list be
    read as "this token is responsible for X of the rewrite" rather than as one
    more popularity contest."""
    torch.manual_seed(0)
    bs, resp, k = 4, 6, 5
    tc = _counts(n_tasks=1)
    support = torch.randint(0, VOCAB, (bs, resp, k))
    state = torch.randint(0, len(STATE_NAMES), (bs, resp, k))
    weight = _weight_for(state)
    logp = torch.log_softmax(torch.randn(bs, resp, k), dim=-1) + np.log(0.5)
    mask = (torch.rand(bs, resp) > 0.3).float()

    tc.update(
        support_ids=support, state=state, weight=weight,
        on_task_logprob=logp, response_mask=mask, task_ids=None,
    )
    expect = float(((weight - 1.0) * logp.exp() * mask.unsqueeze(-1)).sum())
    assert float(tc._cpu()[2][0].sum()) == pytest.approx(expect, rel=1e-4)


def test_accumulation_is_additive_across_micro_batches():
    """It is folded once per micro-batch and rendered once per call."""
    args = dict(
        support_ids=torch.tensor([[[1, 2]]]),
        state=torch.full((1, 1, 2), STATE_AGREE_POS),
        on_task_logprob=torch.full((1, 1, 2), -1.0),
        response_mask=torch.ones(1, 1),
        task_ids=torch.tensor([0]),
    )
    args["weight"] = _weight_for(args["state"])
    once, twice = _counts(), _counts()
    once.update(**args)
    twice.update(**args)
    twice.update(**args)
    assert int(twice._cpu()[0].sum()) == 2 * int(once._cpu()[0].sum())


def test_a_rendering_taken_before_more_data_is_not_reused_after_it():
    """The host copy is cached to keep the transfer to one per call; a stale one
    would report the batch that had already been read."""
    tc = _counts()
    args = dict(
        support_ids=torch.tensor([[[1, 2]]]),
        state=torch.full((1, 1, 2), STATE_AGREE_POS),
        on_task_logprob=torch.full((1, 1, 2), -1.0),
        response_mask=torch.ones(1, 1),
        task_ids=None,
    )
    args["weight"] = _weight_for(args["state"])
    tc.update(**args)
    assert int(tc._cpu()[0].sum()) == 2
    tc.update(**args)
    assert int(tc._cpu()[0].sum()) == 4
    tc.all_reduce()  # no distributed group here, but it must still drop the cache
    assert int(tc._cpu()[0].sum()) == 4


# --------------------------------------------------------------------------- #
# What the numbers say
# --------------------------------------------------------------------------- #
def _fill(tc, token_ids, state_id, n_positions, logp=-2.0):
    ids = torch.tensor(token_ids).view(1, 1, -1).expand(1, n_positions, len(token_ids))
    state = torch.full(ids.shape, state_id)
    tc.update(
        support_ids=ids,
        state=state,
        weight=_weight_for(state),
        on_task_logprob=torch.full(ids.shape, logp),
        response_mask=torch.ones(1, n_positions),
        task_ids=None,
    )


def test_concentration_separates_a_narrow_mechanism_from_a_broad_one():
    """The distinction the whole diagnostic exists for. Both runs below reinforce
    the same NUMBER of candidates, so every existing frac_* metric is identical
    between them; only these two tell them apart."""
    narrow = _counts(top_n=5)
    _fill(narrow, list(range(5)), STATE_AGREE_POS, n_positions=10)
    broad = _counts(top_n=5)
    _fill(broad, list(range(50)), STATE_AGREE_POS, n_positions=1)

    n_metric = "sign_weight/token/n_distinct/agree_pos"
    share = "sign_weight/token/top5_share/agree_pos"
    mn, mb = narrow.scalar_metrics(), broad.scalar_metrics()

    # same candidate count either way
    assert int(narrow._cpu()[0][0, STATE_AGREE_POS].sum()) == int(broad._cpu()[0][0, STATE_AGREE_POS].sum()) == 50
    assert mn[n_metric] == 5 and mb[n_metric] == 50
    assert mn[share] == pytest.approx(1.0)
    assert mb[share] == pytest.approx(5 / 50)


def test_the_two_halves_of_the_normaliser_are_reported_separately():
    """Reinforcement sits on high-probability candidates and suppression on low
    ones, so they do not cancel in Z. Reporting only the residual would hide how
    much was pushed each way to produce it."""
    tc = _counts()
    _fill(tc, [1, 2], STATE_AGREE_POS, n_positions=3)
    _fill(tc, [3], STATE_AGREE_NEG, n_positions=3)
    m = tc.scalar_metrics()
    assert m["sign_weight/token/dw_pos_sum"] > 0
    assert m["sign_weight/token/dw_neg_sum"] < 0
    p = float(torch.tensor(-2.0).exp())
    assert m["sign_weight/token/dw_pos_sum"] == pytest.approx(2 * 3 * 0.5 * p, rel=1e-5)
    assert m["sign_weight/token/dw_neg_sum"] == pytest.approx(1 * 3 * -0.5 * p, rel=1e-5)


def test_the_neutral_states_are_never_ranked():
    """They hold the overwhelming majority of candidates, so a top-N over them
    would be a list of the corpus's most frequent tokens and would say nothing
    about the mechanism."""
    tc = _counts()
    _fill(tc, [1, 2, 3], STATE_NEUTRAL_ON, n_positions=5)
    assert tc.scalar_metrics() == {} or not any(
        "neutral" in k for k in tc.scalar_metrics()
    )
    assert all(row["state"] != STATE_NAMES[STATE_NEUTRAL_ON] for row in tc.top_tokens())
    assert STATE_NEUTRAL_ON not in ACTED_STATES


# --------------------------------------------------------------------------- #
# The ranked table
# --------------------------------------------------------------------------- #
def test_the_table_names_the_tokens_and_ranks_them_by_each_question():
    tc = _counts(top_n=3)
    _fill(tc, [11], STATE_AGREE_POS, n_positions=10)   # often, small mass each
    _fill(tc, [12], STATE_AGREE_POS, n_positions=2, logp=-0.1)  # rarely, large mass
    _fill(tc, [13], STATE_AGREE_NEG, n_positions=4)

    rows = tc.top_tokens()
    by_count = [r for r in rows if r["ranked_by"] == "count" and r["state"] == "agree_pos"]
    assert [r["token_id"] for r in by_count] == [11, 12], by_count
    assert by_count[0]["count"] == 10 and by_count[1]["count"] == 2

    # ...but by effect on the target, the rare high-probability token wins.
    by_effect = [r for r in rows if r["ranked_by"] == "abs_dw"]
    assert by_effect[0]["token_id"] == 12, by_effect
    # and suppression shows up with a negative effect
    neg = [r for r in by_effect if r["token_id"] == 13]
    assert neg and neg[0]["dw"] < 0

    for r in rows:
        assert set(r) == {"scope", "ranked_by", "state", "rank", "token_id", "count", "mass", "dw"}


def test_the_table_is_scoped_per_task_as_well_as_pooled():
    tc = _counts(n_tasks=2, top_n=4)
    ids = torch.tensor([[[41, 42]], [[43, 44]]])
    state = torch.full((2, 1, 2), STATE_AGREE_POS)
    tc.update(
        support_ids=ids, state=state, weight=_weight_for(state),
        on_task_logprob=torch.full((2, 1, 2), -1.0),
        response_mask=torch.ones(2, 1), task_ids=torch.tensor([0, 1]),
    )
    rows = tc.top_tokens(task_names=["alfworld", "search"])
    scopes = {r["scope"] for r in rows}
    assert scopes == {"__pooled__", "alfworld", "search"}
    assert {r["token_id"] for r in rows if r["scope"] == "alfworld"} == {41, 42}
    assert {r["token_id"] for r in rows if r["scope"] == "search"} == {43, 44}


def test_a_token_can_top_one_ranking_and_be_absent_from_the_other():
    """The gap between the two rankings IS the finding: a token reinforced
    constantly at negligible probability is shared structure the objective never
    sees."""
    tc = _counts(top_n=1)
    _fill(tc, [21], STATE_AGREE_POS, n_positions=50, logp=-20.0)  # constant, no mass
    _fill(tc, [22], STATE_AGREE_POS, n_positions=1, logp=-0.01)   # once, all the mass
    rows = tc.top_tokens()
    top_count = [r for r in rows if r["ranked_by"] == "count"][0]
    top_effect = [r for r in rows if r["ranked_by"] == "abs_dw"][0]
    assert top_count["token_id"] == 21
    assert top_effect["token_id"] == 22


# --------------------------------------------------------------------------- #
# Against the mechanism it measures
# --------------------------------------------------------------------------- #
def test_the_states_it_files_under_are_the_ones_candidate_weights_produced():
    """Fed from candidate_weights rather than from hand-written state ids, so the
    diagnostic cannot drift from the thing it is diagnosing."""
    # one candidate, three models: on-task raises it, both off-task raise it too.
    base = torch.tensor([[[-3.0, -3.0]]])
    on = torch.tensor([[[-1.0, -5.0]]])                       # raised, then lowered
    off = torch.tensor([[[[-1.0, -1.0], [-5.0, -5.0]]]])      # agree, then agree
    weight, state = candidate_weights(
        on, off, base, mode="target",
        agree_weight=1.5, agree_neg_weight=0.5, disagree_weight=1.0, deadzone=0.1,
    )
    assert state[0, 0, 0].item() == STATE_AGREE_POS
    assert state[0, 0, 1].item() == STATE_AGREE_NEG

    tc = _counts(n_tasks=1, top_n=4)
    tc.update(
        support_ids=torch.tensor([[[31, 32]]]),
        state=state, weight=weight, on_task_logprob=on,
        response_mask=torch.ones(1, 1), task_ids=torch.tensor([0]),
    )
    n, _mass, dw = tc._cpu()
    assert int(n[0, STATE_AGREE_POS, 31]) == 1
    assert int(n[0, STATE_AGREE_NEG, 32]) == 1
    # the reinforced token is pushed up, the suppressed one down
    assert float(dw[0, 31]) > 0 and float(dw[0, 32]) < 0


def test_a_conflict_is_recorded_as_its_own_state():
    """Conflict is not weighted in target mode (the table refuses a single
    direction-blind factor), but it is exactly the disagreement a token-level
    read is for, so it must still be nameable."""
    base = torch.tensor([[[-3.0]]])
    on = torch.tensor([[[-1.0]]])                 # on-task raises
    off = torch.tensor([[[[-5.0, -5.0]]]])        # both off-task lower
    weight, state = candidate_weights(
        on, off, base, mode="target",
        agree_weight=1.5, agree_neg_weight=0.5, disagree_weight=1.0, deadzone=0.1,
    )
    assert state[0, 0, 0].item() == STATE_CONFLICT_ON_POS
    tc = _counts(n_tasks=1)
    tc.update(
        support_ids=torch.tensor([[[9]]]), state=state, weight=weight,
        on_task_logprob=on, response_mask=torch.ones(1, 1), task_ids=None,
    )
    rows = [r for r in tc.top_tokens() if r["state"] == "conflict_on_pos"]
    assert rows and rows[0]["token_id"] == 9
    # weight 1.0 in target mode -> no effect on the target, and dw says so
    assert rows[0]["dw"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def test_the_vocab_size_reader_unwraps_and_gives_up_honestly():
    from verl.workers.actor.dp_actor import model_vocab_size

    class _Cfg:
        vocab_size = 4242

    class _Inner:
        config = _Cfg()

    class _Wrapped:
        def __init__(self, inner):
            self._fsdp_wrapped_module = inner

    assert model_vocab_size(_Wrapped(_Inner())) == 4242
    # No config to read -> None, so the caller can run without the diagnostic
    # rather than size an accumulator from a guess and index out of bounds.
    assert model_vocab_size(object()) is None


def test_the_driver_writes_the_table_only_when_asked(tmp_path):
    from verl import DataProto
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    class _Fake:
        _dump_sign_token_report = OPDRayTrainer._dump_sign_token_report

    rows = [{"scope": "__pooled__", "ranked_by": "count", "state": "agree_pos",
             "rank": 0, "token_id": 7, "count": 3, "mass": 0.4, "dw": 0.2, "token": "Ġthe"}]
    out = DataProto(meta_info={"sign_token_report": rows})

    from omegaconf import OmegaConf

    fake = _Fake()
    fake.global_steps = 12
    # no directory configured -> nothing written, and no crash
    fake.config = OmegaConf.create({"trainer": {}})
    fake._dump_sign_token_report(out)

    dump = tmp_path / "tokens"
    fake.config = OmegaConf.create({"trainer": {"sign_token_dump_dir": str(dump)}})
    fake._dump_sign_token_report(out)
    path = dump / "sign_tokens_step000012.jsonl"
    assert path.exists()
    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert written == [{"step": 12, **rows[0]}]

    # an empty report writes nothing rather than an empty file
    fake.global_steps = 13
    fake._dump_sign_token_report(DataProto(meta_info={}))
    assert not (dump / "sign_tokens_step000013.jsonl").exists()


def test_a_resumed_step_overwrites_rather_than_appends(tmp_path):
    """One file per step exists so a resumed run re-writes the steps it repeats.
    Appending would leave a reader to work out which duplicate to trust."""
    from verl import DataProto
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer
    from omegaconf import OmegaConf

    class _Fake:
        _dump_sign_token_report = OPDRayTrainer._dump_sign_token_report

    fake = _Fake()
    fake.global_steps = 5
    fake.config = OmegaConf.create({"trainer": {"sign_token_dump_dir": str(tmp_path)}})
    row = {"scope": "__pooled__", "ranked_by": "count", "state": "agree_pos",
           "rank": 0, "token_id": 1, "count": 1, "mass": 0.1, "dw": 0.0, "token": "a"}
    for _ in range(3):
        fake._dump_sign_token_report(DataProto(meta_info={"sign_token_report": [row]}))
    path = os.path.join(tmp_path, "sign_tokens_step000005.jsonl")
    assert len(open(path, encoding="utf-8").read().splitlines()) == 1
