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

"""Two prefetches, one colocated worker group, and the order they reach the ranks.

The rollout loop drives two things that both dispatch to the SAME colocated
WorkerDict: the teacher scoring chunk, on a background thread, and the actor's
old-log-prob prefetch, on the driver thread. Both drive FSDP collectives on the
same process group, so the ranks must see them in the same order.

Nothing in Ray provides that. ``RayWorkerGroup`` sends to the ranks in a plain
Python loop with no lock, and Ray only orders the calls one caller makes to one
actor -- not the calls two threads make to several actors. Interleave the loops
and rank 0 gets teacher-then-actor while rank 1 gets actor-then-teacher; each
worker then runs its own queue in order, and the collectives mismatch.

That is what the first attempt at this arm died of, 30 minutes after the fact,
with a watchdog message naming a work sequence id and no caller.
"""

import ast
import inspect
import threading

import pytest


def _loop_source():
    import agent_system.multi_turn_rollout.rollout_loop as m

    return inspect.getsource(m)


def _prefetch_branch():
    """The body of `if self._logprob_prefetch_enabled and ...` in the turn loop."""
    tree = ast.parse(_loop_source())
    # Several branches test the flag; the one this is about is the branch that
    # actually issues the prefetch.
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if "_logprob_prefetch_enabled" not in ast.unparse(node.test):
            continue
        if "_prefetch_pending_log_probs" in ast.unparse(node.body):
            return node
    raise AssertionError("the log-prob prefetch branch is gone")


def _called_names(body):
    out = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
            if name:
                out.append(name)
    return out


# ---------------------------------------------------------------------------
# 1. the real code


def test_the_teacher_chunk_is_joined_before_the_actor_prefetch():
    names = _called_names(_prefetch_branch().body)
    assert "_join_teacher_prefetch" in names, (
        "the teacher chunk is left outstanding across a call that lands on the same "
        "colocated WorkerDict -- this is the ordering hazard, not a scheduling nicety"
    )
    assert "_prefetch_pending_log_probs" in names
    assert names.index("_join_teacher_prefetch") < names.index("_prefetch_pending_log_probs")


def test_the_join_is_not_charged_to_the_environment_step():
    """The turn record's columns sum to the turn's wall clock, and the join now
    sits inside the envstep window. It comes out, like the generation one."""
    src = _loop_source()
    assert '"envstep": _m_env - _m_decode - _teacher_wait_lp' in src
    assert '"tchwait_lp": _teacher_wait_lp' in src
    assert "_teacher_wait_lp = 0.0" in src, "it must be defined on the branch that skips the prefetch too"


def test_the_dispatch_this_protects_against_is_still_unsynchronised():
    """If RayWorkerGroup ever serialises its per-rank loop, the join becomes
    redundant. Until then, removing the join reopens the hazard -- so this test
    records WHY the join is there rather than only that it is."""
    import verl.single_controller.ray.base as base

    src = inspect.getsource(base)
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "execute_all_async"), None)
    assert fn is not None
    body = ast.unparse(fn)
    assert "for worker in self._workers" in body or "for i in range(length)" in body
    assert "Lock" not in src and "acquire()" not in src, (
        "the dispatch grew a lock; revisit whether the join is still required"
    )


# ---------------------------------------------------------------------------
# 2. the hazard itself, on a stand-in for the dispatch loop


def _dispatch(order_log, caller, ranks, gate=None):
    """One RayWorkerGroup-style send: a plain loop over the ranks, no lock."""
    for r in ranks:
        order_log[r].append(caller)
        if gate is not None:
            gate(caller, r)


def test_two_unsynchronised_dispatchers_can_reverse_the_order_across_ranks():
    ranks = [0, 1]
    log = {r: [] for r in ranks}
    reached = threading.Event()
    resume = threading.Event()

    def gate(caller, r):
        # Hand control over exactly where the real loop can be pre-empted:
        # after rank 0 has been sent to and before rank 1 has.
        if caller == "teacher" and r == 0:
            reached.set()
            resume.wait(5)

    t = threading.Thread(target=_dispatch, args=(log, "teacher", ranks), kwargs={"gate": gate})
    t.start()
    assert reached.wait(5), "the fixture never reached the interleaving point"
    _dispatch(log, "actor", ranks)
    resume.set()
    t.join(5)

    assert log[0] == ["teacher", "actor"]
    assert log[1] == ["actor", "teacher"]
    assert log[0] != log[1], "this is the mismatch the join exists to prevent"


def test_joining_first_gives_every_rank_the_same_order():
    ranks = [0, 1]
    log = {r: [] for r in ranks}
    done = threading.Event()

    def teacher():
        _dispatch(log, "teacher", ranks)
        done.set()

    t = threading.Thread(target=teacher)
    t.start()
    assert done.wait(5)      # <- the join: the driver waits for the chunk
    t.join(5)
    _dispatch(log, "actor", ranks)

    assert log[0] == log[1] == ["teacher", "actor"]


@pytest.mark.parametrize("n_ranks", [2, 3])
def test_the_order_agrees_on_every_rank_count_once_joined(n_ranks):
    ranks = list(range(n_ranks))
    log = {r: [] for r in ranks}
    for caller in ("teacher", "actor"):
        _dispatch(log, caller, ranks)
    assert len({tuple(v) for v in log.values()}) == 1
