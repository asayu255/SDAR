"""Individual weighting events, with the tokens around them.

Every other accumulator in ``sign_weights`` is an aggregate, and an aggregate
cannot be read for a MECHANISM: "the weighting acts on the same forty tokens
every step" and "it acts on the connectives inside <think>" produce identical
top-N tables. Only instances separate them, which is what
:class:`SignEventSamples` dumps -- one row per sampled candidate carrying the
four models' probabilities, the weight, the effect, where in the episode it sat,
what kind of position it was, and the text being written around it.

Three things have to hold for such a dump to be worth reading rather than
suggestive, and they are what these tests are about:

* the sample is a sample -- ``top`` is the extremes and ``spread`` is not, and
  neither depends on how the mini-batch happened to be split into micro-batches;
* the annotations are right: a position inside ``<think>`` says reasoning, one
  inside ``<action>`` says env_action, and the tag tokens say tag;
* masked and padded positions never appear, since an event that was not in the
  loss is not evidence about the loss.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.sign_weights import (
        EVENT_FLOATS,
        EVENT_INTS,
        ROLE_ENV_ACTION,
        ROLE_ENV_OBS,
        ROLE_FORMAT,
        ROLE_NAMES,
        ROLE_REASONING,
        ROLE_TAG,
        ROLE_TOOL_CALL,
        STATE_AGREE_POS,
        SignEventSamples,
        token_roles,
        turn_index,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


TASKS = ["alfworld", "search", "webshop"]


# --------------------------------------------------------------------------- #
# token_roles
# --------------------------------------------------------------------------- #
# Single-token tags in most tests: what is being checked is the scan, not the
# tokenizer, and a one-token tag makes the expected output readable.
TAGS_1 = {"<think>": [90], "</think>": [91], "<action>": [92], "</action>": [93]}


def _roles(ids, tags=TAGS_1):
    return [ROLE_NAMES[int(r)] for r in token_roles(torch.tensor([ids]), tags)[0]]


def test_a_span_takes_the_role_of_the_tag_that_opened_it():
    got = _roles([90, 1, 2, 91, 5, 92, 3, 93])
    assert got == [
        "tag", "reasoning", "reasoning", "tag",
        "format",
        "tag", "env_action", "tag",
    ]


def test_everything_before_the_first_tag_is_format():
    """Nothing has been opened, which is exactly what format means."""
    assert _roles([7, 7, 90, 1]) == ["format", "format", "tag", "reasoning"]


def test_a_multi_token_tag_matches_as_a_sequence_and_claims_its_whole_span():
    """`<think>` is usually "<", "think", ">" -- three ids, not one."""
    tags = {"<think>": [40, 41, 42], "</think>": [40, 43, 42]}
    got = _roles([9, 40, 41, 42, 5, 6, 40, 43, 42, 8], tags)
    assert got == [
        "format",
        "tag", "tag", "tag",
        "reasoning", "reasoning",
        "tag", "tag", "tag",
        "format",
    ]


def test_a_prefix_of_a_tag_is_not_a_tag():
    tags = {"<think>": [40, 41, 42]}
    assert _roles([40, 41, 7, 40, 41, 42], tags) == [
        "format", "format", "format", "tag", "tag", "tag",
    ]


def test_the_search_arms_three_span_kinds_are_distinguished():
    """<search> and <answer> are the model calling out; <information> is the
    environment answering, and conflating them would put the retriever's text in
    the same bucket as the query that fetched it."""
    tags = {
        "<think>": [90], "</think>": [91],
        "<search>": [94], "</search>": [95],
        "<information>": [96], "</information>": [97],
        "<answer>": [98], "</answer>": [99],
    }
    got = _roles([90, 1, 91, 94, 2, 95, 96, 3, 97, 98, 4, 99], tags)
    assert got == [
        "tag", "reasoning", "tag",
        "tag", "tool_call", "tag",
        "tag", "env_obs", "tag",
        "tag", "tool_call", "tag",
    ]


def test_no_tags_means_no_classification_rather_than_a_wrong_one():
    assert _roles([1, 2, 3], {}) == ["format"] * 3
    assert _roles([1, 2, 3], {"<think>": []}) == ["format"] * 3


def test_unknown_tag_strings_are_ignored_not_refused():
    """The caller may hand over its whole special-token table."""
    assert _roles([90, 1], {"<think>": [90], "<|im_start|>": [50]}) == ["tag", "reasoning"]


def test_the_scan_is_per_row():
    out = token_roles(torch.tensor([[90, 1, 2], [7, 7, 7]]), TAGS_1)
    assert [int(x) for x in out[0]] == [ROLE_TAG, ROLE_REASONING, ROLE_REASONING]
    assert [int(x) for x in out[1]] == [ROLE_FORMAT] * 3


# --------------------------------------------------------------------------- #
# turn_index
# --------------------------------------------------------------------------- #
def test_a_turn_is_a_maximal_run_of_generated_tokens():
    m = torch.tensor([[1.0, 1, 0, 0, 1, 1, 1, 0, 1]])
    # The environment's reply carries the index of the turn it FOLLOWS: the
    # observation that preceded turn n+1 belongs with turn n's consequences.
    assert turn_index(m)[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2]


def test_a_single_turn_arm_is_all_turn_zero():
    assert turn_index(torch.ones(1, 5))[0].tolist() == [0] * 5


def test_a_response_that_starts_masked_does_not_go_negative():
    assert turn_index(torch.tensor([[0.0, 0, 1, 1]]))[0].tolist() == [0, 0, 0, 0]


# --------------------------------------------------------------------------- #
# SignEventSamples
# --------------------------------------------------------------------------- #
def _batch(bs=2, resp=6, k=3, n_off=2, seed=0, mask=None):
    torch.manual_seed(seed)

    def lp():
        return torch.log_softmax(torch.randn(bs, resp, k + 5), dim=-1)[..., :k]

    return {
        "support_ids": torch.randint(0, 40, (bs, resp, k)),
        "state": torch.full((bs, resp, k), STATE_AGREE_POS),
        "weight": torch.full((bs, resp, k), 1.25),
        "effect": torch.randn(bs, resp, k, dtype=torch.float64) * 0.01,
        "on_task_logprob": lp(),
        "off_task_logprobs": torch.stack([lp() for _ in range(n_off)], dim=-1),
        "base_logprob": lp(),
        "student_logprob": lp(),
        "response_mask": torch.ones(bs, resp) if mask is None else mask,
        "responses": (torch.arange(bs * resp).reshape(bs, resp) % 37),
        "norm": torch.full((bs, resp), 1.1),
        "teacher_kl": torch.rand(bs, resp) + 0.1,
        "task_ids": torch.arange(bs) % 3,
    }


def _sample(capacity=4, context=2, folds=1, **over):
    ev = SignEventSamples(capacity=capacity, context=context, device="cpu")
    for i in range(folds):
        b = _batch(seed=i, **over)
        ev.update(**b)
    return ev, b


def test_every_declared_column_is_present_and_nothing_else_is():
    ev, _ = _sample()
    rows = ev.rows(task_names=TASKS)
    assert rows
    expect = {"table", "stratum", "rank", "task", "context_ids"}
    expect |= set(EVENT_INTS) - {"task_id"}
    expect |= set(EVENT_FLOATS)
    for r in rows:
        assert set(r) == expect, set(r) ^ expect
        assert r["table"] == "event"


def test_top_is_the_largest_effects_and_spread_is_not():
    ev, b = _sample(capacity=3)
    rows = ev.rows(task_names=TASKS)
    top = [abs(r["effect"]) for r in rows if r["stratum"] == "top"]
    every = b["effect"].abs().reshape(-1).tolist()
    assert top == sorted(every, reverse=True)[: len(top)]
    assert top == sorted(top, reverse=True), "top must be ranked"
    spread = [abs(r["effect"]) for r in rows if r["stratum"] == "spread"]
    assert len(spread) == 3
    # Not a bound that can fail by luck: with 36 candidates the chance that a
    # hash-ordered 3 IS the top 3 is 1/7140.
    assert spread != top


def test_the_sample_does_not_depend_on_the_micro_batch_split():
    """Two folds of the same data must give what one fold of both would.

    The merge re-runs the same topk over the concatenation, and the spread key
    is a function of a running index rather than of the split, so both strata
    are split-invariant. Without that a reader could not compare two runs whose
    micro batch was sized differently for the GPUs they got.
    """
    whole = SignEventSamples(capacity=4, context=2, device="cpu")
    b0, b1 = _batch(seed=0), _batch(seed=1)
    for b in (b0, b1):
        whole.update(**b)

    halves = SignEventSamples(capacity=4, context=2, device="cpu")
    for b in (b0, b1):
        # One micro-batch of two rows, folded as two of one row each. The spread
        # key advances by the number of candidates seen, so this is the same
        # index stream.
        for r in range(b["support_ids"].size(0)):
            halves.update(**{
                key: (v[r : r + 1] if torch.is_tensor(v) else v) for key, v in b.items()
            })

    def key(rows):
        return [(r["stratum"], r["token_id"], round(r["effect"], 12)) for r in rows]

    assert key(whole.rows(TASKS)) == key(halves.rows(TASKS))


def test_masked_positions_never_appear():
    """An event that was not in the loss is not evidence about the loss."""
    mask = torch.zeros(2, 6)
    mask[0, 2] = 1.0
    ev, _ = _sample(capacity=8, mask=mask)
    rows = ev.rows(task_names=TASKS)
    assert rows, "the one valid position must still be sampled"
    assert {(r["position"]) for r in rows} == {2}
    assert {r["task"] for r in rows} == {"alfworld"}


def test_a_fully_masked_batch_yields_no_rows_rather_than_padding():
    ev, _ = _sample(capacity=4, mask=torch.zeros(2, 6))
    assert ev.rows(task_names=TASKS) == []


def test_the_context_window_is_the_tokens_around_the_position():
    ev, b = _sample(capacity=4, context=2)
    for r in ev.rows(task_names=TASKS):
        row = b["responses"]
        p = r["position"]
        lo, hi = p - 2, p + 2
        expect = [int(row[0 if r["task"] == "alfworld" else 1, min(max(i, 0), 5)]) for i in range(lo, hi + 1)]
        assert r["context_ids"] == expect, (p, r["context_ids"], expect)


def test_the_window_repeats_the_edge_rather_than_padding_with_a_sentinel():
    """A sentinel would be a token id a decoder has to know about; a repeated
    edge token reads as an edge and needs no agreement."""
    ev, b = _sample(capacity=64, context=3)
    at_zero = [r for r in ev.rows(task_names=TASKS) if r["position"] == 0]
    assert at_zero
    for r in at_zero:
        row = 0 if r["task"] == "alfworld" else 1
        assert r["context_ids"][:4] == [int(b["responses"][row, 0])] * 4


def test_the_row_carries_the_four_models_at_that_candidate():
    ev, b = _sample(capacity=2)
    r = ev.rows(task_names=TASKS)[0]
    bi = 0 if r["task"] == "alfworld" else 1
    p, tok = r["position"], r["token_id"]
    j = int((b["support_ids"][bi, p] == tok).nonzero()[0])
    assert r["p_on"] == pytest.approx(float(b["on_task_logprob"][bi, p, j].exp()), rel=1e-6)
    assert r["p_base"] == pytest.approx(float(b["base_logprob"][bi, p, j].exp()), rel=1e-6)
    assert r["p_student"] == pytest.approx(float(b["student_logprob"][bi, p, j].exp()), rel=1e-6)
    off = b["off_task_logprobs"][bi, p, j].exp()
    assert r["p_off_lo"] == pytest.approx(float(off.min()), rel=1e-6)
    assert r["p_off_hi"] == pytest.approx(float(off.max()), rel=1e-6)


def test_an_arm_with_no_reward_reports_nan_and_not_zero():
    """0.0 is a score. Absent is not, and the two must not be read alike."""
    ev = SignEventSamples(capacity=2, context=1, device="cpu")
    b = _batch()
    ev.update(**b, reward=None)
    assert all(np.isnan(r["reward"]) for r in ev.rows(task_names=TASKS))
    ev2 = SignEventSamples(capacity=2, context=1, device="cpu")
    ev2.update(**b, reward=torch.tensor([1.0, 0.0]))
    got = {r["task"]: r["reward"] for r in ev2.rows(task_names=TASKS)}
    assert got.get("alfworld", 1.0) == 1.0


def test_the_role_column_is_the_positions_role_not_the_candidates():
    """A role is a property of WHERE in the response the event sat; the k
    candidates at one position all share it."""
    b = _batch(bs=1, resp=6, k=3)
    b["responses"] = torch.tensor([[90, 1, 2, 91, 92, 3]])
    ev = SignEventSamples(capacity=32, context=1, device="cpu")
    ev.update(**b, roles=token_roles(b["responses"], TAGS_1))
    by_pos = {r["position"]: r["role"] for r in ev.rows(task_names=TASKS)}
    assert by_pos == {
        0: "tag", 1: "reasoning", 2: "reasoning", 3: "tag", 4: "tag", 5: "env_action",
    }


def test_without_tag_ids_the_role_column_says_format_rather_than_guessing():
    ev, _ = _sample(capacity=4)
    assert {r["role"] for r in ev.rows(task_names=TASKS)} == {"format"}


def test_an_untagged_row_reports_no_task_rather_than_task_minus_one():
    b = _batch(bs=1, resp=4, k=2)
    b["task_ids"] = torch.tensor([-1])
    ev = SignEventSamples(capacity=4, context=1, device="cpu")
    ev.update(**b)
    assert {r["task"] for r in ev.rows(task_names=TASKS)} == {None}


def test_capacity_bounds_the_dump_whatever_the_batch_size():
    ev, _ = _sample(capacity=5, folds=3, bs=4, resp=8, k=4)
    rows = ev.rows(task_names=TASKS)
    for stratum in ("top", "spread"):
        assert len([r for r in rows if r["stratum"] == stratum]) == 5


# --------------------------------------------------------------------------- #
# The wiring in the actor
# --------------------------------------------------------------------------- #
from tests.trainer.test_transfer_metrics import _update_policy_source  # noqa: E402


def test_every_reader_of_the_kl_runs_before_the_position_weight_multiplies_it():
    """The three tables all report against the UNWEIGHTED per-token KL.

    Taking the already-weighted one would make ``kl_scale`` 1 by construction
    and would credit each candidate with a cost the weighting had itself
    inflated. The ordering is the only thing enforcing that, so it is pinned:
    every ``*_stats.update`` reading ``teacher_kld`` must appear ahead of the
    line that rebinds it.
    """
    src = _update_policy_source()
    multiply = src.index("teacher_kld = teacher_kld * sign_position_weight")
    for reader in (
        "position_stats.update(",
        "token_stats.update(",
        "pair_token_stats.update(",
        "event_stats.update(",
    ):
        assert src.index(reader) < multiply, reader


def test_the_event_sampler_is_gated_on_the_config_and_not_on_the_batch():
    """A rank whose micro-batch happens to carry no sign columns must still
    build the same accumulators, or the all-reduces below it deadlock. The
    sampler itself is not all-reduced, but it is constructed beside the ones
    that are and a batch-content gate there is how the pattern gets copied."""
    src = _update_policy_source()
    line = [ln for ln in src.splitlines() if "SignEventSamples(" in ln]
    assert line, "SignEventSamples is not constructed"
    ctor = src[src.index("event_stats = ("):src.index("SignEventSamples(") + 400]
    assert "sign_enabled" not in ctor, ctor
    assert "sign_cfg_on" in ctor


def test_the_micro_batch_is_never_treated_as_a_dataproto():
    """Inside the loop ``data`` is a TensorDict, or a plain dict on the
    multi-modal path -- the DataProto is unpacked at the top of the body. A
    ``data.batch.keys()`` there raises AttributeError on the first step of every
    run, which is the kind of thing a CPU test suite cannot see and a 618 s/step
    job discovers ten minutes in."""
    src = _update_policy_source()
    # From just after the unpack -- the one line that legitimately reaches
    # through .batch is the isinstance(data, DataProto) branch that flattens it.
    body = src[src.index('responses = data["responses"]'):]
    offenders = [
        line.strip()
        for line in body.splitlines()
        if "data.batch" in line.split("#")[0]
    ]
    assert not offenders, offenders


def test_the_candidate_effect_is_computed_once_for_every_table():
    """Three tables rank by it. Three calls would let them disagree."""
    src = _update_policy_source()
    assert src.count("candidate_effect(") == 1
