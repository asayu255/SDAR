# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The pumped path has to hand the rollout loop back what the blocking one did.

Not the same tokens -- a different decode grouping gives different tokens, which
is the accepted cost of the path. The same *shape*: every row paired with its own
prompt, in the caller's order, with the same keys and the same masks. Rows
arriving from a pool come back in completion order, so an off-by-one here pairs
trajectory 7's answer with trajectory 3's prompt, and nothing downstream would
notice -- the scores would just be worse.

The other half is the guards. Every call the pool cannot serve identically has to
end up on the blocking path, silently and correctly, rather than being served
approximately.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import agent_system.multi_turn_rollout.rollout_loop as rollout_loop  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.workers.rollout.generation_output import assemble_generation_output  # noqa: E402

PAD = 0
EOS = 2
RESPONSE_LENGTH = 6
WORLD = 3


def _answer(prompt_token_ids):
    """A deterministic stand-in for the model: the answer names its prompt."""
    return [100 + prompt_token_ids[-1], EOS]


class StubWorkerGroup:
    """Three ranks that answer from _answer, one blocking call or one pool."""

    def __init__(self, refuse=None, in_session=True, rebuilds_prompt_ids=False):
        self.world_size = WORLD
        self.refuse = refuse
        self.in_session = in_session
        self.rebuilds_prompt_ids = rebuilds_prompt_ids
        self.blocking_calls = 0
        self.pump_rounds = 0
        self.resident = [dict() for _ in range(WORLD)]

    # -- the blocking path -------------------------------------------------

    def generate_sequences(self, batch):
        self.blocking_calls += 1
        if self.rebuilds_prompt_ids and "raw_prompt_ids" not in batch.non_tensor_batch:
            # What the real one does when the column is absent.
            batch.non_tensor_batch["raw_prompt_ids"] = np.array(
                [[t for t in row if t != PAD] for row in batch.batch["input_ids"].tolist()], dtype=object
            )
        prompts = batch.non_tensor_batch["raw_prompt_ids"]
        non_tensor = {k: v for k, v in batch.non_tensor_batch.items() if k != "raw_prompt_ids"}
        return assemble_generation_output(
            idx=batch.batch["input_ids"],
            attention_mask=batch.batch["attention_mask"],
            position_ids=batch.batch["position_ids"],
            response_token_ids=[_answer(list(p)) for p in prompts],
            non_tensor_batch=non_tensor,
            eos_token_id=EOS,
            pad_token_id=PAD,
            response_length=RESPONSE_LENGTH,
        )

    # -- the pumped path ---------------------------------------------------

    def rollout_pump_step(self, payloads):
        assert len(payloads) == self.world_size
        if payloads[0].get("handshake"):
            return [
                {
                    "refused": self.refuse,
                    "in_session": self.in_session,
                    "pad_token_id": PAD,
                    "response_length": RESPONSE_LENGTH,
                    "eos_token_id": EOS,
                }
                for _ in range(self.world_size)
            ]
        if payloads[0].get("stop"):
            return [{"finished": [], "failed": [], "in_flight": 0}] * self.world_size

        self.pump_rounds += 1
        replies = []
        for rank, payload in enumerate(payloads):
            for request_id, prompt_token_ids, _meta in payload["submit"]:
                self.resident[rank][request_id] = list(prompt_token_ids)
            # Answer in an order the caller did not ask for, on purpose: out of
            # a pool, completion order has nothing to do with submission order.
            finished = [(rid, _answer(p)) for rid, p in reversed(list(self.resident[rank].items()))]
            self.resident[rank] = {}
            replies.append({"finished": finished, "failed": [], "in_flight": 0})
        return replies


def _batch(rows, meta_info, extra_non_tensor=None):
    input_ids = torch.tensor(rows)
    attention_mask = (input_ids != PAD).to(torch.int64)
    position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
    non_tensor = {
        "raw_prompt_ids": np.array([[t for t in row if t != PAD] for row in rows], dtype=object),
    }
    non_tensor.update(extra_non_tensor or {})
    proto = DataProto(
        batch=TensorDict(
            {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
            batch_size=len(rows),
        ),
        non_tensor_batch=non_tensor,
    )
    proto.meta_info = meta_info
    return proto


ROWS = [[PAD, 7, 8], [5, 6, 9], [PAD, PAD, 3], [1, 2, 4], [PAD, 9, 9]]
VALIDATING = {"validate": True, "do_sample": True, "eos_token_id": EOS}


@pytest.fixture(autouse=True)
def pump_off_after_each():
    rollout_loop._PUMP_STATE.update({"client": None, "off": False})
    yield
    rollout_loop.close_pump_client()
    rollout_loop._PUMP_STATE.update({"client": None, "off": not rollout_loop._ROLLOUT_ASYNC_GENERATE})


@pytest.fixture
def pump_on(monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_ASYNC_GENERATE", True)
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_MERGE_GENERATES", False)


def test_the_pool_returns_the_same_dataproto_the_blocking_call_would_have(pump_on):
    wg = StubWorkerGroup()
    expected = wg.generate_sequences(_batch(ROWS, dict(VALIDATING)))
    got = rollout_loop._generate_sequences(wg, _batch(ROWS, dict(VALIDATING)))

    assert wg.pump_rounds > 0, "the pumped path was never taken"
    assert set(got.batch.keys()) == set(expected.batch.keys())
    for key in expected.batch.keys():
        assert torch.equal(got.batch[key], expected.batch[key]), key


def test_answers_come_back_paired_with_their_own_prompt(pump_on):
    wg = StubWorkerGroup()
    got = rollout_loop._generate_sequences(wg, _batch(ROWS, dict(VALIDATING)))
    # _answer keys off the prompt's last token, so each row's answer names the
    # prompt it belongs to even after the pool reordered the completions.
    for row, response in zip(ROWS, got.batch["responses"].tolist()):
        assert response[0] == 100 + row[-1]


def test_the_other_non_tensor_columns_survive_and_raw_prompt_ids_does_not(pump_on):
    wg = StubWorkerGroup()
    extra = {"tools_kwargs": np.array([{"a": i} for i in range(len(ROWS))], dtype=object)}
    got = rollout_loop._generate_sequences(wg, _batch(ROWS, dict(VALIDATING), extra))
    assert "raw_prompt_ids" not in got.non_tensor_batch
    assert [d["a"] for d in got.non_tensor_batch["tools_kwargs"]] == list(range(len(ROWS)))


def test_a_multimodal_call_goes_back_to_the_blocking_path(pump_on):
    wg = StubWorkerGroup()
    extra = {"multi_modal_data": np.array([{} for _ in ROWS], dtype=object)}
    rollout_loop._generate_sequences(wg, _batch(ROWS, dict(VALIDATING), extra))
    assert wg.blocking_calls == 1 and wg.pump_rounds == 0


def test_a_training_call_that_may_want_several_samples_goes_back_to_the_blocking_path(pump_on):
    wg = StubWorkerGroup()
    rollout_loop._generate_sequences(wg, _batch(ROWS, {"do_sample": True, "validate": False}))
    assert wg.blocking_calls == 1 and wg.pump_rounds == 0


def test_greedy_is_served_by_the_pool_even_outside_validation(pump_on):
    wg = StubWorkerGroup()
    rollout_loop._generate_sequences(wg, _batch(ROWS, {"do_sample": False}))
    assert wg.pump_rounds > 0 and wg.blocking_calls == 0


def test_a_call_without_raw_prompt_ids_goes_back_to_the_blocking_path(pump_on):
    # generate_sequences rebuilds them from the padded input_ids in that case;
    # the pooled path has no second copy of that and must not guess.
    wg = StubWorkerGroup(rebuilds_prompt_ids=True)
    batch = _batch(ROWS, dict(VALIDATING))
    batch.non_tensor_batch.pop("raw_prompt_ids")
    rollout_loop._generate_sequences(wg, batch)
    assert wg.blocking_calls == 1 and wg.pump_rounds == 0


def test_a_refusing_rank_puts_the_process_back_on_the_blocking_path_for_good(pump_on, capsys):
    wg = StubWorkerGroup(refuse="tensor_model_parallel_size > 1")
    for _ in range(3):
        rollout_loop._generate_sequences(wg, _batch(ROWS, dict(VALIDATING)))
    assert wg.blocking_calls == 3 and wg.pump_rounds == 0
    said = [line for line in capsys.readouterr().out.splitlines() if "staying on the blocking path" in line]
    assert len(said) == 1, "the refusal should be said once, not once per call"
    assert "tensor_model_parallel_size" in said[0]


def test_a_worker_group_with_no_pump_at_all_still_runs(pump_on, capsys):
    """A checkout without the worker-side half must not take the run down."""

    class OldWorkerGroup:
        world_size = WORLD

        def __init__(self):
            self.blocking_calls = 0

        generate_sequences = StubWorkerGroup.generate_sequences
        rebuilds_prompt_ids = False

    wg = OldWorkerGroup()
    assert not hasattr(wg, "rollout_pump_step")
    rollout_loop._generate_sequences(wg, _batch(ROWS, dict(VALIDATING)))
    assert wg.blocking_calls == 1
    assert "staying on the blocking path" in capsys.readouterr().out


def test_the_flag_off_never_touches_the_pool():
    wg = StubWorkerGroup()
    rollout_loop._generate_sequences(wg, _batch(ROWS, dict(VALIDATING)))
    assert wg.blocking_calls == 1 and wg.pump_rounds == 0


# --------------------------------------------------------------------------- #
# ROLLOUT_ASYNC_REQUIRE: a measurement must not quietly measure the other path
# --------------------------------------------------------------------------- #
def _strict(monkeypatch, on=True):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_ASYNC_REQUIRE", on)
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_ASYNC_GENERATE", True)
    rollout_loop._PUMP_STATE["client"] = None
    rollout_loop._PUMP_STATE["off"] = False


class _RefusingWG:
    world_size = 3

    def rollout_pump_step(self, payloads):
        return [{"refused": "rollout log-probs are requested", "in_session": True}] * self.world_size

    def generate_sequences(self, data):
        raise AssertionError("the blocking path must not be reached under REQUIRE")


def test_a_refusing_rank_is_an_error_when_async_is_required(monkeypatch):
    """Without this, a stock PPO config (return_rollout_log_probs=True) prints one
    line, latches off, and reports the blocking path's wall clock as the pump's."""
    _strict(monkeypatch)
    try:
        with pytest.raises(RuntimeError, match="ROLLOUT_ASYNC_REQUIRE=1 and the pool refused"):
            rollout_loop._pump_client(_RefusingWG())
    finally:
        rollout_loop._PUMP_STATE["off"] = True


def test_a_refusing_rank_only_falls_back_when_async_is_not_required(monkeypatch):
    _strict(monkeypatch, on=False)
    try:
        assert rollout_loop._pump_client(_RefusingWG()) is None
    finally:
        rollout_loop._PUMP_STATE["off"] = True


def test_a_call_the_pool_cannot_serve_is_an_error_when_async_is_required(monkeypatch):
    """A multimodal or n>1 call would otherwise slip onto the blocking path unremarked."""
    _strict(monkeypatch)
    batch = _batch(ROWS[:2], {"do_sample": True})   # neither greedy nor validation -> n is not pinned
    try:
        with pytest.raises(RuntimeError, match="cannot go through the pool.*does not pin n=1"):
            rollout_loop._generate_sequences(_RefusingWG(), batch)
    finally:
        rollout_loop._PUMP_STATE["off"] = True


def test_the_reason_names_which_thing_stopped_it():
    batch = _batch(ROWS[:2], {"do_sample": False},
                   {"multi_modal_data": np.array([{} for _ in range(2)], dtype=object)})
    assert "multi_modal_data" in rollout_loop._why_the_pump_cannot_serve(batch)

    batch = _batch(ROWS[:2], {"do_sample": False})
    assert rollout_loop._why_the_pump_cannot_serve(batch) is None
