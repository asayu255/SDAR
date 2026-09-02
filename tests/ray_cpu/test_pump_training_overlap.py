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
"""Letting the TRAINING rollout through the pumped pool, and what that unlocks.

The pool was reachable only by calls that pin n=1 in their meta_info -- greedy
and validation. A training call leaves the configured sampling params alone, so
what it asks for is the rank's own ``rollout.n``, and the driver refused rather
than risk keeping sample 0 of n. That is the right refusal when n > 1 and no
reason at all when n == 1, which is the case on these arms: ``env.rollout.n`` is
applied by repeating rows in the driver, not through SamplingParams.

What it unlocks is the reason to bother. TokenPump steps the engine on a thread
INSIDE the worker, so between two pump_step RPCs the colocated actor's call slot
is free -- the one thing in this tree that breaks the serialisation of GPU calls
that docs/gpu_profiling_report_opd.md 2.4 describes. A teacher prefetch chunk
issued while the pool is decoding therefore runs BESIDE the decode instead of
behind it, which is why the join moves from before each generation to after it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

rollout_loop = pytest.importorskip("agent_system.multi_turn_rollout.rollout_loop")


class _Batch:
    """The two things _why_the_pump_cannot_serve reads."""

    def __init__(self, meta_info, raw_prompt_ids=True, multi_modal=False):
        self.meta_info = meta_info
        self.non_tensor_batch = {}
        if raw_prompt_ids:
            self.non_tensor_batch["raw_prompt_ids"] = [[1, 2, 3]]
        if multi_modal:
            self.non_tensor_batch["multi_modal_data"] = [None]


@pytest.fixture
def pump_training(monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_PUMP_TRAINING", True)
    return rollout_loop


@pytest.fixture
def no_pump_training(monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_PUMP_TRAINING", False)
    return rollout_loop


# ------------------------------------------------------------ the servability
def test_greedy_and_validation_are_servable_whatever_the_rank_reports(no_pump_training):
    """Both override the params to n=1 themselves, so the rank's own n cannot
    make them unservable -- and they were servable before this change."""
    for meta in ({"do_sample": False}, {"validate": True}):
        assert rollout_loop._pump_pins_one_sample(meta, {"n": 8})


def test_a_training_call_is_refused_while_the_flag_is_off(no_pump_training):
    """OFF has to reproduce exactly what was there before."""
    assert not rollout_loop._pump_pins_one_sample({"do_sample": True}, {"n": 1})
    why = rollout_loop._why_the_pump_cannot_serve(_Batch({"do_sample": True}), {"n": 1})
    assert "ROLLOUT_PUMP_TRAINING is off" in why


def test_a_training_call_is_servable_when_the_rank_reports_n_of_one(pump_training):
    assert rollout_loop._pump_pins_one_sample({"do_sample": True}, {"n": 1})
    assert rollout_loop._why_the_pump_cannot_serve(_Batch({"do_sample": True}), {"n": 1}) is None


@pytest.mark.parametrize("n", [2, 8])
def test_a_rank_that_would_produce_several_sequences_is_still_refused(pump_training, n):
    """The original objection, which is correct whenever it applies: the pool
    returns one sequence per request, and quietly keeping sample 0 of n would be
    a scoring change nobody asked for."""
    assert not rollout_loop._pump_pins_one_sample({"do_sample": True}, {"n": n})
    why = rollout_loop._why_the_pump_cannot_serve(_Batch({"do_sample": True}), {"n": n})
    assert f"n={n}" in why


def test_an_unknown_n_is_declined_rather_than_guessed(pump_training):
    """A worker built before the handshake carried this key reports nothing, and
    0 reads as unknown. Guessing 1 there is exactly the scoring change the
    refusal exists to prevent."""
    assert not rollout_loop._pump_pins_one_sample({"do_sample": True}, {})
    assert not rollout_loop._pump_pins_one_sample({"do_sample": True}, None)


def test_the_other_refusals_still_apply_to_a_training_call(pump_training):
    assert "multi_modal_data" in rollout_loop._why_the_pump_cannot_serve(
        _Batch({"do_sample": True}, multi_modal=True), {"n": 1}
    )
    assert "raw_prompt_ids" in rollout_loop._why_the_pump_cannot_serve(
        _Batch({"do_sample": True}, raw_prompt_ids=False), {"n": 1}
    )


# ----------------------------------------------------------- the will-it gate
def test_will_serve_is_false_without_the_flag(no_pump_training, monkeypatch):
    monkeypatch.setattr(rollout_loop, "_pump_client", lambda wg: pytest.fail("should not be asked"))
    assert not rollout_loop._pump_will_serve(object(), _Batch({"do_sample": True}))


def test_will_serve_is_false_when_the_pool_refused(pump_training, monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_ASYNC_GENERATE", True)
    monkeypatch.setattr(rollout_loop, "_pump_client", lambda wg: None)
    assert not rollout_loop._pump_will_serve(object(), _Batch({"do_sample": True}))


def test_will_serve_reads_the_handshake_the_client_carries(pump_training, monkeypatch):
    class _Client:
        handshake_info = {"n": 1}

    monkeypatch.setattr(rollout_loop, "_ROLLOUT_ASYNC_GENERATE", True)
    monkeypatch.setattr(rollout_loop, "_pump_client", lambda wg: _Client())
    assert rollout_loop._pump_will_serve(object(), _Batch({"do_sample": True}))

    _Client.handshake_info = {"n": 4}
    assert not rollout_loop._pump_will_serve(object(), _Batch({"do_sample": True}))


# ------------------------------------------------- the join moves, in the code
def _turn_loop_source():
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "agent_system", "multi_turn_rollout", "rollout_loop.py")
    src = open(path).read()
    body = src[src.index("_overlap = _pump_will_serve("):]
    return body[: body.index("self._launch_teacher_prefetch()")]


def test_the_join_is_skipped_before_a_pumped_generation_and_paid_after_it():
    """On the blocking path the join has to come first: both land on the same
    colocated actor, so an outstanding chunk would serialise behind the
    generation anyway, on Ray's queue where the driver cannot see it. Pumped,
    joining first is what throws the window away."""
    body = _turn_loop_source()
    assert "_teacher_wait = 0.0 if _overlap else self._join_teacher_prefetch()" in body
    generate_at = body.index("_generate_sequences(actor_rollout_wg, batch_input_padded)")
    late_join = body.index("_teacher_wait = self._join_teacher_prefetch()", generate_at)
    assert late_join > generate_at


def test_the_adaptive_sizer_is_told_about_the_bigger_window():
    """The chunk now gets the glue PLUS the generation it overlaps. Sized to the
    glue alone it would leave the decode tail empty -- the space this exists for."""
    body = _turn_loop_source()
    after_generate = body[body.index("_generate_sequences(actor_rollout_wg, batch_input_padded)"):]
    assert "self.note_teacher_glue_window(" in after_generate


# ------------------------------------------------------- the three arms agree
_SCRIPTS = [
    "examples/opd_grpo_trainer/run_multitask_cross_teacher_target_qwen3.sh",
    "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_qwen3.sh",
    "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh",
    # The content-only arm is compared against the klw control above, so it is
    # inside the same "must carry the same flags" set as the rest -- the whole
    # claim it makes is that the role mask is the only difference.
    "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_content_qwen3.sh",
]


def _script(name):
    return open(os.path.join(os.path.dirname(__file__), "..", "..", name)).read()


@pytest.mark.parametrize("script", _SCRIPTS)
def test_every_cross_teacher_arm_carries_the_same_rollout_flags(script):
    """None of these is bit-identical, so an arm that carries a different set is
    not comparable with the others -- which is the failure docs/..._audit.md 0.2
    records as the costliest one this project has had."""
    src = _script(script)
    for flag in ("ROLLOUT_ASYNC_GENERATE", "ROLLOUT_PUMP_TRAINING", "ROLLOUT_PREFETCH_LOGPROB",
                 "ROLLOUT_PREFETCH_TEACHER", "ROLLOUT_PREFETCH_SIGN"):
        assert f"export {flag}=${{{flag}:-1}}" in src, flag


@pytest.mark.parametrize("script", _SCRIPTS)
def test_rollout_log_probs_are_off_or_the_pool_refuses_every_call(script):
    """vllm_rollout_spmd._pump_refuse returns "rollout log-probs are requested"
    while this is on -- the pool returns token ids, not log-probs -- so the
    handshake fails and ROLLOUT_ASYNC_GENERATE is inert for EVERY call of every
    step, validation included. On the sg1 run that is what it was. The cost of
    turning it off is the rollout-vs-actor drift check."""
    assert "actor_rollout_ref.rollout.return_rollout_log_probs=False" in _script(script)


def test_the_worker_still_refuses_the_pool_when_log_probs_are_asked_for():
    """The refusal is right; what was wrong was pinning both settings at once and
    only finding out from one log line hours in."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "verl", "workers",
                        "rollout", "vllm_rollout", "vllm_rollout_spmd.py")
    src = open(path).read()
    body = src[src.index("def _pump_refuse("):]
    body = body[: body.index("def pump_step(")]
    assert "self.return_rollout_log_probs" in body
