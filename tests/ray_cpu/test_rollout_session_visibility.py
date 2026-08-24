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
"""Whether the vLLM session opened has to be readable from the log.

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
