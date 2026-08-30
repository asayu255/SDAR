"""FSDP -> vLLM weight sync has to reach past whatever wraps the model runner.

MEASURED, 2026-08-27, vllm 0.8.5 V0 with num_scheduler_steps=4:

    AttributeError: 'MultiStepModelRunner' object has no attribute 'model'
      verl/workers/sharding_manager/fsdp_vllm.py:263 in update_params

Multi-step scheduling puts a MultiStepModelRunner in front of the real runner
and does not forward `.model`. Everything before that point succeeded -- the
engine built, the config validated, the checkpoint loaded -- so the failure
arrives minutes in and names neither multi-step nor weight sync.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from verl.utils.vllm_model_runner import unwrap_model_runner  # noqa: E402


class _Real:
    model = "the weights"


def test_an_unwrapped_runner_is_returned_as_is():
    real = _Real()
    assert unwrap_model_runner(real) is real


def test_the_multi_step_wrapper_is_unwrapped():
    """vLLM 0.8.5 names the inner one _base_model_runner."""

    class _MultiStep:
        def __init__(self, inner):
            self._base_model_runner = inner

    real = _Real()
    assert unwrap_model_runner(_MultiStep(real)) is real


def test_nested_wrappers_are_unwrapped():
    class _Wrapper:
        def __init__(self, inner):
            self._base_model_runner = inner

    real = _Real()
    assert unwrap_model_runner(_Wrapper(_Wrapper(real))) is real


def test_the_first_object_with_a_model_wins():
    """Stop at the runner that owns the weights, not at the innermost object."""

    class _Wrapper:
        model = "outer weights"

        def __init__(self, inner):
            self._base_model_runner = inner

    outer = _Wrapper(_Real())
    assert unwrap_model_runner(outer) is outer


def test_none_stays_none():
    """The manager builds with inference_engine=None on the async path."""
    assert unwrap_model_runner(None) is None


def test_an_unknown_wrapper_is_returned_so_the_error_is_the_honest_one():
    """Guessing deeper would replace one confusing AttributeError with another."""

    class _Unknown:
        pass

    unknown = _Unknown()
    assert unwrap_model_runner(unknown) is unknown


def test_a_self_referential_wrapper_terminates():
    """A cycle must fail, not hang -- this runs inside a Ray worker."""

    class _Loop:
        pass

    loop = _Loop()
    loop._base_model_runner = loop
    assert unwrap_model_runner(loop) is loop


def test_a_cycle_between_two_wrappers_terminates():
    class _Node:
        pass

    a, b = _Node(), _Node()
    a._base_model_runner = b
    b._base_model_runner = a
    assert unwrap_model_runner(a) in (a, b)
