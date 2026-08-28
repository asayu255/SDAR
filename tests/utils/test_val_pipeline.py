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
"""Running validation batches in parallel without changing what is scored.

The gap being filled is real and measured: after the retriever was batched, a
search batch is 11.87 s of generation against 1.20 s of tokenising and 1.17 s of
environment, and neither of those can be brought forward because the next
turn's prompt is the environment's answer to this one. Another batch can fill
it -- but only if running two at once cannot change a single scored row.

Three things have to hold, and each has a way of failing silently:

* **Order.** Results are accumulated into flat lists that later line up with
  data_source and task_name by position. Out-of-order retirement would score
  every row against another row's metadata, and nothing would raise.
* **Isolation.** An environment manager holds per-rollout state -- observation
  history, per-env step counters. Two batches on one manager would interleave
  into each other's history.
* **Eligibility.** alfworld draws episodes from a seeded game cycle indexed by
  position within its manager, so it cannot have a second one. A batch routed to
  a slot that does not serve its task would score different games.
"""

import threading
import time

import pytest

import verl.utils.val_pipeline as vp_module
from verl.utils.val_pipeline import Slot, run_pipelined


def _slots(count, tasks=None):
    return [Slot(f"s{i}", envs=f"envs{i}", collector=f"collector{i}", tasks=tasks) for i in range(count)]


def test_one_slot_runs_inline_with_no_threads():
    """A run that did not ask for a pipeline must behave exactly as before --
    including staying on one thread, so nothing it calls has to be reentrant."""
    threads = []

    def launch(prepared, slot):
        threads.append(threading.current_thread())
        return prepared * 2

    out = list(run_pipelined([1, 2, 3], lambda i: i, lambda p: None, launch, _slots(1)))

    assert [r for _, r in out] == [2, 4, 6]
    assert threads == [threading.main_thread()] * 3


def test_results_come_back_in_submission_order():
    """Later batches finish first here. If that leaked through, every scored row
    would be paired with another batch's metadata."""

    def launch(prepared, slot):
        time.sleep(0.05 if prepared == 0 else 0.0)
        return prepared

    out = list(run_pipelined(range(6), lambda i: i, lambda p: None, launch, _slots(3)))

    assert [p for p, _ in out] == list(range(6))
    assert [r for _, r in out] == list(range(6))


def test_prepare_runs_in_order_on_the_calling_thread():
    """prepare touches the tokeniser and the trainer's own fields. It stays
    sequential; only the rollout is allowed to be concurrent."""
    seen = []

    def prepare(item):
        seen.append((item, threading.current_thread()))
        return item

    list(run_pipelined(range(5), prepare, lambda p: None, lambda p, s: p, _slots(2)))

    assert [item for item, _ in seen] == list(range(5))
    assert {thread for _, thread in seen} == {threading.main_thread()}


def test_a_slot_serves_one_batch_at_a_time():
    """The isolation the whole design rests on: an env manager must never have
    two rollouts inside it."""
    holders = {}
    clashes = []
    lock = threading.Lock()

    def launch(prepared, slot):
        with lock:
            if slot.name in holders:
                clashes.append((slot.name, holders[slot.name], prepared))
            holders[slot.name] = prepared
        time.sleep(0.02)
        with lock:
            del holders[slot.name]
        return prepared

    list(run_pipelined(range(12), lambda i: i, lambda p: None, launch, _slots(3)))

    assert clashes == []


def test_batches_actually_overlap():
    """The control for the test above: without overlap there is nothing to
    isolate, and the pipeline would be buying nothing."""
    concurrent = []
    running = 0
    lock = threading.Lock()

    def launch(prepared, slot):
        nonlocal running
        with lock:
            running += 1
            concurrent.append(running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return prepared

    list(run_pipelined(range(6), lambda i: i, lambda p: None, launch, _slots(2)))

    assert max(concurrent) == 2


def test_a_restricted_task_only_reaches_a_slot_that_serves_it():
    """alfworld has exactly one manager. A batch of it must not land on the
    search-only slot, where it would play a different set of games."""
    ran = []
    slots = [Slot("full", "e0", "c0", tasks=None), Slot("search-only", "e1", "c1", tasks=["search"])]

    def launch(prepared, slot):
        ran.append((prepared, slot.name))
        return prepared

    items = ["alfworld", "search", "search", "alfworld", "search"]
    list(run_pipelined(items, lambda i: i, lambda p: p, launch, slots))

    for task, slot_name in ran:
        assert task == "search" or slot_name == "full", (task, slot_name)


def test_a_task_no_slot_serves_is_an_error_not_a_silent_skip():
    slots = [Slot("search-only", "e0", "c0", tasks=["search"])] * 1
    slots = [Slot("a", "e", "c", tasks=["search"]), Slot("b", "e", "c", tasks=["search"])]

    with pytest.raises(RuntimeError, match="no slot can run task"):
        list(run_pipelined(["alfworld"], lambda i: i, lambda p: p, lambda p, s: p, slots))


def test_an_exception_in_one_batch_reaches_the_caller():
    """A validation that dies must not be reported as a score."""

    def launch(prepared, slot):
        if prepared == 2:
            raise RuntimeError("rollout died")
        return prepared

    with pytest.raises(RuntimeError, match="rollout died"):
        list(run_pipelined(range(5), lambda i: i, lambda p: None, launch, _slots(2)))


def test_the_executor_is_shut_down_even_when_the_caller_stops_early():
    """The caller is a generator consumer; if it breaks out, a rollout left
    running would still be holding the worker group and the awake engine."""
    started = threading.Event()

    def launch(prepared, slot):
        started.set()
        time.sleep(0.05)
        return prepared

    generated = run_pipelined(range(20), lambda i: i, lambda p: None, launch, _slots(2))
    next(generated)
    generated.close()

    assert started.is_set()


def test_slot_accepts():
    assert Slot("a", None, None).accepts("anything") is True
    assert Slot("a", None, None, tasks=["search"]).accepts("search") is True
    assert Slot("a", None, None, tasks=["search"]).accepts("alfworld") is False


# --------------------------------------------------------------------------- #
# The trainer's side: the sequential path has to stay sequential, and the
# pipelined one has to route by task.
# --------------------------------------------------------------------------- #


def _trainer(depth, tasks=("search",)):
    """RayPPOTrainer's slot builder bound to the fields it touches."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class _T:
        _validation_slots = RayPPOTrainer._validation_slots

        def __init__(self):
            self.val_envs = "primary-envs"
            self.traj_collector = "primary-collector"
            self.config = None
            self.tokenizer = None
            self.processor = None

    return _T()


def test_depth_one_is_the_primary_manager_and_nothing_else(monkeypatch):
    """The default must not build a second environment manager -- that is 126
    more Ray actors for a run that was not asked to pipeline."""
    monkeypatch.setenv("VAL_PIPELINE_DEPTH", "1")
    slots = _trainer(1)._validation_slots()

    assert len(slots) == 1
    assert slots[0].envs == "primary-envs"
    assert slots[0].collector == "primary-collector"
    assert slots[0].accepts("alfworld") is True


def test_depth_one_is_the_default(monkeypatch):
    monkeypatch.delenv("VAL_PIPELINE_DEPTH", raising=False)
    assert len(_trainer(1)._validation_slots()) == 1


def test_the_resolved_depth_is_announced_even_at_one(monkeypatch, capsys):
    """Silence would mean both "depth is 1" and "this build has no pipeline",
    which is the pair of indistinguishable states that hid the vLLM session bug.
    """
    monkeypatch.delenv("VAL_PIPELINE_DEPTH", raising=False)
    _trainer(1)._validation_slots()
    out = capsys.readouterr().out
    assert "[val-pipeline] VAL_PIPELINE_DEPTH=1: 1 slot(s)" in out
    assert "VAL_PIPELINE_DEPTH=2" in out  # and says how to turn it on


def test_the_announcement_names_the_depth_that_was_asked_for(monkeypatch, capsys):
    monkeypatch.setenv("VAL_PIPELINE_DEPTH", "2")
    import agent_system.environments.env_manager as env_manager
    import agent_system.multi_turn_rollout.rollout_loop as rollout_loop

    monkeypatch.setattr(env_manager, "build_val_env_manager", lambda config, tasks: "envs")
    monkeypatch.setattr(rollout_loop, "TrajectoryCollector", lambda **kw: "collector")

    _trainer(2)._validation_slots()
    assert "[val-pipeline] VAL_PIPELINE_DEPTH=2: 2 slot(s)" in capsys.readouterr().out


def test_the_extra_slots_are_restricted(monkeypatch):
    """alfworld must never reach a second manager: its games are indexed by
    position within the manager it belongs to."""
    monkeypatch.setenv("VAL_PIPELINE_DEPTH", "3")
    import agent_system.environments.env_manager as env_manager
    import agent_system.multi_turn_rollout.rollout_loop as rollout_loop

    monkeypatch.setattr(env_manager, "build_val_env_manager", lambda config, tasks: f"envs{sorted(tasks)}")
    monkeypatch.setattr(rollout_loop, "TrajectoryCollector", lambda **kw: "collector")

    slots = _trainer(3)._validation_slots()

    assert len(slots) == 3
    assert slots[0].accepts("alfworld") is True
    for extra in slots[1:]:
        assert extra.accepts("alfworld") is False
        assert extra.accepts("search") is True


def test_alfworld_is_not_pipelineable():
    """The list is the safety property, so it gets its own assertion rather than
    only being read through the slots."""
    from agent_system.environments.env_manager import PIPELINEABLE_VAL_TASKS

    assert "alfworld" not in PIPELINEABLE_VAL_TASKS
    assert "search" in PIPELINEABLE_VAL_TASKS


def test_a_second_manager_is_refused_for_an_unsafe_task():
    from agent_system.environments.env_manager import build_val_env_manager

    with pytest.raises(ValueError, match="cannot have a second validation manager"):
        build_val_env_manager(config=None, tasks=["alfworld"])


def _sampler_trainer(val_per_task_batch_size, task_names, val_batch_size=126):
    """RayPPOTrainer's validation batch sampler, bound to the fields it reads."""
    from omegaconf import OmegaConf

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class _Frame:
        column_names = ["task_name"]

        def __getitem__(self, key):
            return {"task_name": task_names}[key]

    class _T:
        _validation_batch_sampler = RayPPOTrainer._validation_batch_sampler

        def __init__(self):
            self.config = OmegaConf.create(
                {
                    "env": {
                        "env_name": "multitask",
                        "multitask": {
                            "tasks": ["alfworld", "search", "webshop"],
                            "val_per_task_batch_size": val_per_task_batch_size,
                        },
                    },
                    "data": {"val_batch_size": val_batch_size},
                }
            )
            self.val_dataset = type("_D", (), {"dataframe": _Frame()})()

    return _T()


_ROWS = ["alfworld"] * 126 + ["webshop"] * 126 + ["search"] * 500


def test_uniform_sizes_keep_the_plain_loader():
    """No sampler means the loader keeps batch_size/shuffle/drop_last, which is
    the path every other arm is on."""
    assert _sampler_trainer(126, _ROWS)._validation_batch_sampler() is None


def test_a_per_task_size_builds_a_sampler():
    sampler = _sampler_trainer({"search": 252}, _ROWS)._validation_batch_sampler()
    assert sampler is not None
    assert [len(b) for b in sampler] == [126, 126, 252, 248]


def test_a_mapping_that_changes_nothing_still_keeps_the_plain_loader():
    assert _sampler_trainer({"search": 126}, _ROWS)._validation_batch_sampler() is None


def test_rows_without_a_task_column_are_a_loud_failure():
    """Grouping by task is the whole point; guessing would produce mixed batches
    and get_task_names raises on those, hundreds of batches in."""
    trainer = _sampler_trainer({"search": 252}, _ROWS)
    trainer.val_dataset = type("_D", (), {"dataframe": type("_F", (), {"column_names": ["prompt"]})()})()
    with pytest.raises(ValueError, match="no task_name column"):
        trainer._validation_batch_sampler()


def test_a_single_task_run_keeps_the_plain_loader():
    from omegaconf import OmegaConf

    trainer = _sampler_trainer({"search": 252}, _ROWS)
    trainer.config = OmegaConf.create({"env": {"env_name": "search"}, "data": {"val_batch_size": 126}})
    assert trainer._validation_batch_sampler() is None


def test_coverage_is_the_union_not_the_sum():
    """Two slots running the same seconds cover those seconds once. Summing
    would report more time covered than the run took."""
    from verl.utils.val_pipeline import _coverage

    covered, span = _coverage([(0.0, 10.0), (0.0, 10.0)])
    assert (covered, span) == (10.0, 10.0)


def test_coverage_counts_the_hole_between_two_runs():
    from verl.utils.val_pipeline import _coverage

    covered, span = _coverage([(0.0, 10.0), (15.0, 20.0)])
    assert covered == 15.0 and span == 20.0  # 5 s with nothing running


def test_coverage_merges_overlapping_and_touching_spans():
    from verl.utils.val_pipeline import _coverage

    assert _coverage([(0.0, 5.0), (3.0, 8.0), (8.0, 9.0)]) == (9.0, 9.0)


def test_coverage_of_nothing_is_nothing():
    from verl.utils.val_pipeline import _coverage

    assert _coverage([]) == (0.0, 0.0)


def test_the_pipeline_reports_where_its_wall_went(capsys):
    """The number that matters is the stretch with every slot finished and
    nothing resubmitted -- neither the per-batch table nor slots-busy sees it."""
    import time as _time

    from verl.utils.val_pipeline import Slot, run_pipelined

    slots = [Slot("a", None, None), Slot("b", None, None)]

    def launch(prepared, slot):
        _time.sleep(0.05)
        return prepared

    consumed = []
    for prepared, result in run_pipelined(
        range(4), prepare=lambda x: x, task_of=lambda p: "search", launch=launch, slots=slots
    ):
        _time.sleep(0.02)  # the caller scoring a batch
        consumed.append(result)

    assert consumed == [0, 1, 2, 3]
    out = capsys.readouterr().out
    assert "[val-pipeline] final: 4 batches over" in out
    assert "NOTHING running" in out
    assert "scoring" in out


def test_the_pipeline_reports_before_the_run_finishes(capsys, monkeypatch):
    """A summary only at the end costs the whole validation to read once."""
    import time as _time

    import verl.utils.val_pipeline as vp
    from verl.utils.val_pipeline import Slot, run_pipelined

    monkeypatch.setattr(vp, "_REPORT_EVERY", 2)
    slots = [Slot("a", None, None), Slot("b", None, None)]

    seen_early = []
    for _, _ in run_pipelined(
        range(5),
        prepare=lambda x: x,
        task_of=lambda p: "search",
        launch=lambda prepared, slot: (_time.sleep(0.01), prepared)[1],
        slots=slots,
    ):
        seen_early.append(capsys.readouterr().out)

    assert any("after 2" in chunk for chunk in seen_early)
    assert any("after 4" in chunk for chunk in seen_early)


def test_the_periodic_report_can_be_turned_off(capsys, monkeypatch):
    import verl.utils.val_pipeline as vp
    from verl.utils.val_pipeline import Slot, run_pipelined

    monkeypatch.setattr(vp, "_REPORT_EVERY", 0)
    slots = [Slot("a", None, None), Slot("b", None, None)]
    for _ in run_pipelined(
        range(4), prepare=lambda x: x, task_of=lambda p: "s", launch=lambda p, s: p, slots=slots
    ):
        pass
    out = capsys.readouterr().out
    assert "after " not in out
    assert "final:" in out


def test_the_loader_wait_is_timed_and_its_worst_case_named(capsys):
    """next(items) was the one thing on the calling thread nothing measured, and
    with num_workers>0 torch returns batches in strict worker order -- one slow
    worker stalls the loop every num_workers batches."""
    import time as _time

    from verl.utils.val_pipeline import Slot, run_pipelined

    def slow_loader():
        for i in range(4):
            _time.sleep(0.08 if i == 2 else 0.01)  # one worker is slow
            yield i

    slots = [Slot("a", None, None), Slot("b", None, None)]
    for _ in run_pipelined(
        slow_loader(), prepare=lambda x: x, task_of=lambda p: "s", launch=lambda p, s: p, slots=slots
    ):
        pass

    out = capsys.readouterr().out
    assert "dataload" in out
    worst = float(out.split("worst single wait ")[1].split("s)")[0])
    assert worst >= 0.07  # the slow one, not the average


def test_the_loader_is_not_charged_to_prepare(capsys):
    """Charging the loader's wait to prepare would name the wrong culprit."""
    import time as _time

    from verl.utils.val_pipeline import Slot, run_pipelined

    def loader():
        for i in range(3):
            _time.sleep(0.05)
            yield i

    slots = [Slot("a", None, None), Slot("b", None, None)]
    for _ in run_pipelined(
        loader(), prepare=lambda x: x, task_of=lambda p: "s", launch=lambda p, s: p, slots=slots
    ):
        pass

    out = capsys.readouterr().out
    dataload = float(out.split("dataload ")[1].split("s ")[0])
    prepare = float(out.split("prepare ")[1].split("s,")[0])
    assert dataload >= 0.14
    assert prepare < 0.05


# --------------------------------------------------------------------------- #
# A slot is released when ITS OWN batch finishes
# --------------------------------------------------------------------------- #
def test_a_finished_batch_frees_its_slot_while_an_older_one_is_still_running():
    """The head-of-line block, in the shape the real run had it.

    An alfworld batch runs 239.7 s against a search batch's ~65 s. Retiring
    oldest-first meant that while alfworld sat at the head of the queue, every
    slot behind it that had already finished stayed unusable -- so a four-slot
    pipeline ran as one for as long as alfworld took. The profiler reported it
    as TAIL_BLOCKS_READY with ready=1 and gen_inflight=1..4, those few requests
    being alfworld's late turns.
    """
    import threading

    slots = [Slot(f"s{i}", envs=None, collector=None) for i in range(3)]
    hold_first = threading.Event()
    concurrent, lock = [], threading.Lock()
    running = {"n": 0}

    def launch(item, _slot):
        with lock:
            running["n"] += 1
            concurrent.append(running["n"])
        try:
            if item == 0:
                hold_first.wait(10)      # the long batch, at the head
            else:
                time.sleep(0.05)
            return item
        finally:
            with lock:
                running["n"] -= 1

    out = []

    def consume():
        for _prepared, result in run_pipelined(range(9), lambda x: x, lambda _p: None, launch, slots):
            out.append(result)

    thread = threading.Thread(target=consume)
    thread.start()
    # Give the pipeline time to cycle the two free slots through the short
    # batches while the long one is still held.
    time.sleep(1.0)
    submitted_while_blocked = len(concurrent)
    hold_first.set()
    thread.join(20)
    assert not thread.is_alive()

    # Three slots, one of them stuck: the other two must have run repeatedly.
    assert submitted_while_blocked >= 5, (
        f"only {submitted_while_blocked} launches while the head was blocked -- "
        "the finished slots were not being recycled"
    )
    assert out == list(range(9)), out          # still submission order
    assert max(concurrent) <= len(slots)       # and never more than the slots


def test_results_come_back_in_submission_order_even_when_completion_is_reversed():
    """Out-of-order release must not become out-of-order scoring.

    The rows scored and their order are what the numbers rest on; only the
    slot's reuse moves.
    """
    slots = [Slot(f"s{i}", envs=None, collector=None) for i in range(4)]

    def launch(item, _slot):
        time.sleep(0.20 - 0.02 * item)   # earlier items finish LAST
        return item

    out = [result for _prepared, result in
           run_pipelined(range(8), lambda x: x, lambda _p: None, launch, slots)]
    assert out == list(range(8)), out


def test_a_slot_is_never_reused_before_its_own_batch_returns():
    """Two batches on one environment manager would interleave their state."""
    import threading

    slots = [Slot(f"s{i}", envs=None, collector=None) for i in range(3)]
    lock = threading.Lock()
    occupied, clashes = set(), []

    def launch(item, slot):
        with lock:
            if slot.name in occupied:
                clashes.append((slot.name, item))
            occupied.add(slot.name)
        time.sleep(0.02 + 0.01 * (item % 4))
        with lock:
            occupied.discard(slot.name)
        return item

    out = [r for _p, r in run_pipelined(range(24), lambda x: x, lambda _p: None, launch, slots)]
    assert not clashes, clashes
    assert out == list(range(24))


def test_a_task_restricted_slot_is_still_waited_for_specifically():
    """alfworld has one manager; a search slot cannot stand in for it."""
    slots = [Slot("primary", envs=None, collector=None),
             Slot("extra-1", envs=None, collector=None, tasks=["search"])]
    seen = []

    def launch(item, slot):
        seen.append((item, slot.name))
        time.sleep(0.02)
        return item

    items = ["search", "search", "alfworld", "search", "alfworld"]
    out = [r for _p, r in run_pipelined(items, lambda x: x, lambda x: x, launch, slots)]
    assert out == items
    assert all(name == "primary" for task, name in seen if task == "alfworld")


def test_an_impossible_task_still_raises_rather_than_hanging():
    slots = [Slot("extra-1", envs=None, collector=None, tasks=["search"])]
    two = [Slot("a", envs=None, collector=None, tasks=["search"]),
           Slot("b", envs=None, collector=None, tasks=["search"])]
    with pytest.raises(RuntimeError, match="no slot can run task"):
        list(run_pipelined(["search", "alfworld"], lambda x: x, lambda x: x,
                              lambda _p, _s: None, two))
    assert slots  # (the single-slot path is inline and unchanged)


def test_a_rollout_that_raises_still_raises_on_the_calling_thread():
    slots = [Slot(f"s{i}", envs=None, collector=None) for i in range(3)]

    def launch(item, _slot):
        if item == 3:
            raise ValueError("rollout blew up")
        time.sleep(0.01)
        return item

    with pytest.raises(ValueError, match="rollout blew up"):
        list(run_pipelined(range(8), lambda x: x, lambda _p: None, launch, slots))


# --------------------------------------------------------------------------- #
# The head of the INPUT queue must not hold the whole pipeline
# --------------------------------------------------------------------------- #
def test_a_restricted_batch_at_the_head_does_not_block_the_slots_behind_it():
    """The second head-of-line block, and the larger of the two.

    The extra slots serve search only -- alfworld keeps a single manager
    because its games are indexed by position within it. Pulling one item at a
    time meant a webshop or alfworld batch at the head waited for `primary`
    specifically, the iterator never advanced, and the search batches behind it
    were not even prepared: three free slots idle for the length of a 240 s
    alfworld rollout.
    """
    import threading

    slots = [Slot("primary", envs=None, collector=None),
             Slot("extra-1", envs=None, collector=None, tasks=["search"]),
             Slot("extra-2", envs=None, collector=None, tasks=["search"]),
             Slot("extra-3", envs=None, collector=None, tasks=["search"])]
    hold = threading.Event()
    started, lock = [], threading.Lock()

    def launch(task, slot):
        with lock:
            started.append((task, slot.name))
        if task == "alfworld":
            hold.wait(10)            # the long one, on primary
        else:
            time.sleep(0.03)
        return task

    items = ["alfworld", "webshop"] + ["search"] * 12
    out = []
    thread = threading.Thread(target=lambda: out.extend(
        r for _p, r in run_pipelined(items, lambda x: x, lambda x: x, launch, slots)))
    thread.start()
    time.sleep(0.6)
    searches_started = sum(1 for task, _n in started if task == "search")
    hold.set()
    thread.join(20)
    assert not thread.is_alive()

    assert searches_started >= 6, (
        f"only {searches_started} search batches started while alfworld held primary "
        f"and webshop sat at the head of the queue: {started}"
    )
    assert out == items                       # hand-back order is unchanged
    assert ("webshop", "primary") in started  # and webshop still ran, on primary


def test_restricted_batches_reach_their_slot_in_their_own_order():
    """alfworld's game cycle is positional within its manager.

    Reordering search around them is fine; reordering THEM is not.
    """
    slots = [Slot("primary", envs=None, collector=None),
             Slot("extra-1", envs=None, collector=None, tasks=["search"]),
             Slot("extra-2", envs=None, collector=None, tasks=["search"])]
    seen = []

    def launch(item, slot):
        seen.append((item, slot.name))
        time.sleep(0.01)
        return item

    items = [("alfworld", 0), ("search", 1), ("search", 2), ("alfworld", 3),
             ("search", 4), ("alfworld", 5), ("search", 6)]
    out = [r for _p, r in run_pipelined(items, lambda x: x, lambda x: x[0], launch, slots)]
    assert out == items
    alf = [item for item, _slot in seen if item[0] == "alfworld"]
    assert alf == [("alfworld", 0), ("alfworld", 3), ("alfworld", 5)], alf
    assert all(name == "primary" for item, name in seen if item[0] == "alfworld")


def test_the_lookahead_is_bounded():
    """52k rows must not all be prepared and held."""
    slots = [Slot(f"s{i}", envs=None, collector=None) for i in range(3)]
    prepared = []

    def prepare(item):
        prepared.append(item)
        return item

    gen = run_pipelined(range(500), prepare, lambda _p: None,
                        lambda _p, _s: time.sleep(0.01), slots)
    next(gen)   # one batch out
    assert len(prepared) <= 4 * len(slots), len(prepared)
    gen.close()


def test_the_lookahead_is_configurable(monkeypatch):
    import importlib
    monkeypatch.setenv("VAL_PIPELINE_LOOKAHEAD", "3")
    module = importlib.reload(vp_module)
    try:
        slots = [module.Slot(f"s{i}", envs=None, collector=None) for i in range(2)]
        prepared = []
        gen = module.run_pipelined(range(100), lambda i: prepared.append(i) or i,
                                   lambda _p: None, lambda _p, _s: time.sleep(0.01), slots)
        next(gen)
        assert len(prepared) <= 5, len(prepared)
        gen.close()
    finally:
        monkeypatch.delenv("VAL_PIPELINE_LOOKAHEAD", raising=False)
        importlib.reload(module)


def test_a_task_no_slot_serves_raises_rather_than_spinning():
    slots = [Slot("a", envs=None, collector=None, tasks=["search"]),
             Slot("b", envs=None, collector=None, tasks=["search"])]
    with pytest.raises(RuntimeError, match="no slot can run task"):
        list(run_pipelined(["search", "alfworld"], lambda x: x, lambda x: x,
                           lambda _p, _s: None, slots))
