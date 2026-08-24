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
"""The vLLM session: whether it opened, and that nesting it does not close it.

It was not, and that cost a measured 13% of the evaluation's wall clock before
anyone noticed. Both states are silent by construction: the sharding manager's
own enter/exit logging goes through ``log_gpu_memory_usage(..., logger=logger)``,
whose default level is DEBUG, on a logger pinned to WARN
(``fsdp_vllm.py``) -- so it never reaches a log file at either setting. From
outside, "the driver never asked for a session" and "the workers refused to open
one" produce the identical picture: vLLM wakes and sleeps on every turn. It took
a 2-second nvidia-smi trace to see the 21 GB unmap/remap cycle at all.

The flag is also read in a place that is easy to get wrong. It is resolved at
import time in whichever process runs the rollout loop -- the trainer's Ray
actor -- not in the launcher that exports it and not in the rollout workers.
Reading it off the workers' /proc/*/environ, which is the obvious check, answers
a different question than the one being asked.

So: the driver prints what IT resolved, and the worker prints which branch it
took and how many turns one wake covered. These tests hold both.
"""

import os
import pathlib
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

import agent_system.multi_turn_rollout.rollout_loop as rollout_loop  # noqa: E402
import verl.workers.fsdp_workers as fsdp_workers  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO / "examples" / "sft_trainer" / "eval_checkpoints.sh"


@pytest.fixture(autouse=True)
def _fresh_say(monkeypatch):
    monkeypatch.setattr(rollout_loop, "_SAID_ROLLOUT_ENV", False)


class _Worker:
    """The attributes the session hooks touch, bound to the real methods.

    ``_rank``, not ``rank``: on the real worker ``rank`` is a property over
    ``self._rank``, and reading it before distributed init raises. A print must
    never be able to break the thing it is observing, so the hooks read the
    underlying attribute defensively -- and this stub has to match, or it would
    pass while the real object raised.
    """

    def __init__(self, rank=0):
        self._rank = rank

    _say_session = fsdp_workers.ActorRolloutRefWorker._say_session


def test_a_worker_without_a_rank_yet_stays_silent_instead_of_raising():
    """begin_rollout_session used to touch nothing. Adding a print to it must
    not turn it into a call that can fail."""
    worker = _Worker.__new__(_Worker)  # no _rank at all
    worker._say_session("opened -- vLLM stays awake for this rollout")


def test_the_driver_reports_the_value_it_resolved(capsys, monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_KEEP_VLLM_AWAKE", True)
    monkeypatch.setenv("ROLLOUT_KEEP_VLLM_AWAKE", "1")
    rollout_loop._say_rollout_env()
    line = capsys.readouterr().out

    assert "[rollout-session] driver:" in line
    assert "ON" in line


def test_off_says_what_it_costs(capsys, monkeypatch):
    """A line reading 'OFF' with no consequence attached is a line nobody acts
    on. The whole point is that OFF means a wake and a sleep every turn."""
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_KEEP_VLLM_AWAKE", False)
    monkeypatch.delenv("ROLLOUT_KEEP_VLLM_AWAKE", raising=False)
    rollout_loop._say_rollout_env()
    out = capsys.readouterr().out

    assert "OFF" in out
    assert "every turn" in out
    # The raw value, not just the interpretation: unset and set-to-0 are
    # different bugs with different fixes.
    assert "<unset>" in out


def test_a_set_but_false_value_is_shown_verbatim(capsys, monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_KEEP_VLLM_AWAKE", False)
    monkeypatch.setenv("ROLLOUT_KEEP_VLLM_AWAKE", "0")
    rollout_loop._say_rollout_env()
    out = capsys.readouterr().out

    assert "'0'" in out
    assert "OFF" in out


def test_the_driver_says_it_once_per_process(capsys, monkeypatch):
    """It is called from multi_turn_loop, which runs once per validation batch
    and once per training step. A per-call line would bury the run's log."""
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_KEEP_VLLM_AWAKE", True)
    for _ in range(5):
        rollout_loop._say_rollout_env()

    assert capsys.readouterr().out.count("[rollout-session] driver:") == 1


def test_the_worker_names_the_reason_it_did_not_open(capsys):
    """Each early return in begin_rollout_session has a different cause and a
    different fix; 'no session' alone does not distinguish them."""
    worker = _Worker(rank=0)
    worker._say_session("no rollout sharding manager (SKIP_ROLLOUT_BUILD)")
    out = capsys.readouterr().out

    assert "rank 0" in out
    assert "SKIP_ROLLOUT_BUILD" in out


def test_the_worker_deduplicates(capsys):
    """These fire once per rollout -- fifty validation batches would otherwise
    print fifty identical lines."""
    worker = _Worker(rank=0)
    for _ in range(4):
        worker._say_session("opened -- vLLM stays awake for this rollout")
    worker._say_session("a different outcome")
    out = capsys.readouterr().out

    assert out.count("opened") == 1
    assert out.count("a different outcome") == 1


def test_only_rank_zero_speaks(capsys):
    """Three ranks in lockstep would triple every line for a state that is
    identical across them."""
    _Worker(rank=1)._say_session("opened -- vLLM stays awake for this rollout")

    assert capsys.readouterr().out == ""


def test_the_eval_script_forces_session_mode():
    """Evaluation is the arm that generates on every turn, so the flag it can
    least afford to inherit as off is this one."""
    text = EVAL_SCRIPT.read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

    assert "export ROLLOUT_KEEP_VLLM_AWAKE=1" in lines


def test_the_eval_script_turns_on_turn_timing():
    """Overridable, unlike the one above: turn timing changes nothing about the
    run, so a caller who wants a quiet log may switch it off."""
    text = EVAL_SCRIPT.read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

    assert 'export ROLLOUT_TURN_TIMING="${ROLLOUT_TURN_TIMING:-1}"' in lines


# --------------------------------------------------------------------------- #
# Nesting. _validate opens a session around the whole validation; multi_turn_loop
# opens one per rollout inside it -- 413 of them on this arm. With a boolean
# flag the first inner close would sleep vLLM and every batch after it would pay
# the 21 GB unmap and remap again, which is 10.4% of the evaluation's wall clock
# and would have looked exactly like the hoist working.
# --------------------------------------------------------------------------- #


class _Manager:
    """Counts what the sharding manager was actually asked to do."""

    def __init__(self):
        self.enters = 0
        self.exits = 0

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *exc):
        self.exits += 1
        return False


class _SessionWorker:
    """The real begin/end, bound onto only the state they touch."""

    def __init__(self):
        self._rank = 0
        self._is_rollout = True
        self.rollout_sharding_manager = _Manager()

    _say_session = fsdp_workers.ActorRolloutRefWorker._say_session
    begin_rollout_session = fsdp_workers.ActorRolloutRefWorker.begin_rollout_session
    end_rollout_session = fsdp_workers.ActorRolloutRefWorker.end_rollout_session


def _stub_sharding_module(monkeypatch, cls):
    """Stand in for verl.workers.sharding_manager.fsdp_vllm.

    begin_rollout_session imports it lazily -- deliberately, so a run with no
    rollout never drags in vLLM -- and this container has no vLLM to import. The
    stub keeps the lazy import and the isinstance check running for real, which
    is where the eligibility decisions actually live.
    """
    module = types.ModuleType("verl.workers.sharding_manager.fsdp_vllm")
    module.FSDPVLLMShardingManager = cls
    monkeypatch.setitem(sys.modules, "verl.workers.sharding_manager.fsdp_vllm", module)


@pytest.fixture
def worker(monkeypatch):
    _stub_sharding_module(monkeypatch, _Manager)
    return _SessionWorker()


def test_one_scope_enters_and_exits_once(worker):
    worker.begin_rollout_session()
    worker.end_rollout_session()

    assert (worker.rollout_sharding_manager.enters, worker.rollout_sharding_manager.exits) == (1, 1)


def test_an_inner_scope_does_not_wake_the_engine_again(worker):
    worker.begin_rollout_session()
    worker.begin_rollout_session()

    assert worker.rollout_sharding_manager.enters == 1


def test_an_inner_close_does_not_sleep_the_engine(worker):
    """The bug the counter exists for: _validate's session must survive the 413
    multi_turn_loop scopes nested inside it."""
    worker.begin_rollout_session()          # _validate
    for _ in range(413):                    # one per val batch
        worker.begin_rollout_session()
        worker.end_rollout_session()

    assert worker.rollout_sharding_manager.exits == 0
    assert worker._rollout_session_active is True

    worker.end_rollout_session()            # _validate closes
    assert worker.rollout_sharding_manager.exits == 1
    assert worker._rollout_session_active is False


def test_the_engine_wakes_once_for_a_whole_validation(worker):
    worker.begin_rollout_session()
    for _ in range(413):
        worker.begin_rollout_session()
        worker.end_rollout_session()
    worker.end_rollout_session()

    assert worker.rollout_sharding_manager.enters == 1


def test_an_unpaired_close_is_a_no_op(worker):
    worker.end_rollout_session()

    assert worker.rollout_sharding_manager.exits == 0


def test_closing_past_zero_does_not_go_negative(worker):
    worker.begin_rollout_session()
    worker.end_rollout_session()
    worker.end_rollout_session()

    assert worker._rollout_session_depth == 0
    assert worker.rollout_sharding_manager.exits == 1


def test_a_declined_session_leaves_nothing_to_close(worker):
    """begin returns early when there is no manager; end must not then unbalance
    a later real session."""
    worker.rollout_sharding_manager = None
    worker.begin_rollout_session()
    worker.end_rollout_session()

    assert getattr(worker, "_rollout_session_depth", 0) == 0


def test_a_non_vllm_manager_is_declined(worker, monkeypatch):
    class _Other:
        pass

    _stub_sharding_module(monkeypatch, _Other)
    worker.begin_rollout_session()

    assert worker.rollout_sharding_manager.enters == 0
    assert getattr(worker, "_rollout_session_depth", 0) == 0


# --------------------------------------------------------------------------- #
# The context manager the two call sites share.
# --------------------------------------------------------------------------- #


class _WorkerGroup:
    def __init__(self):
        self.begins = 0
        self.ends = 0

    def begin_rollout_session(self):
        self.begins += 1

    def end_rollout_session(self):
        self.ends += 1


def test_rollout_session_pairs_begin_and_end(monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_KEEP_VLLM_AWAKE", True)
    wg = _WorkerGroup()
    with rollout_loop.rollout_session(wg):
        pass

    assert (wg.begins, wg.ends) == (1, 1)


def test_rollout_session_closes_on_an_exception(monkeypatch):
    """A validation that raises mid-batch must still put vLLM back to sleep, or
    the next phase inherits 21 GB it did not budget for."""
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_KEEP_VLLM_AWAKE", True)
    wg = _WorkerGroup()
    with pytest.raises(RuntimeError):
        with rollout_loop.rollout_session(wg):
            raise RuntimeError("boom")

    assert wg.ends == 1


def test_rollout_session_is_a_no_op_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_KEEP_VLLM_AWAKE", False)
    wg = _WorkerGroup()
    with rollout_loop.rollout_session(wg):
        pass

    assert (wg.begins, wg.ends) == (0, 0)


def test_validate_wraps_its_whole_loop_in_one_session():
    """The hoist is the point. If the `with` ever moves inside the for, the 413
    inner scopes become 413 wake/sleep cycles again -- silently."""
    import inspect

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    src = inspect.getsource(RayPPOTrainer._validate)
    with_at = src.index("with rollout_session(self.actor_rollout_wg):")
    for_at = src.index("for test_data in self.val_dataloader:")
    assert with_at < for_at, "the session must open outside the batch loop"
