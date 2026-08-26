# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
"""Per-task advance: one task's turn counter is its own.

The batch-wide loop makes every task wait for the slowest one at every turn
boundary, which is why preproc and envstep line up across the whole batch and
the GPU goes idle for all of them together (docs/eval_performance_summary.md
section 4 measures that at 10.6% of the wall). Advancing each task on its own
counter breaks the alignment.

What has to hold for that to be allowed:

  * ``step(tasks=[t])`` advances t and NOTHING else -- not another task's turn
    counter, not its done flags, not its last observation.
  * the row sets are disjoint, so the shared arrays need no lock.
  * ``_splice_obs`` writes only the rows it was handed.
  * every refusal in ``_per_task_groups`` names something that is not
    row-scoped, and returns None rather than corrupting it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent_system.environments.env_manager import MultiTaskEnvironmentManager  # noqa: E402
from agent_system.multi_turn_rollout import rollout_loop  # noqa: E402
from agent_system.multi_turn_rollout.rollout_loop import _per_task_groups, _splice_obs  # noqa: E402


class FakeTaskManager:
    """One task's vectorized env, counting the steps it was actually asked for."""

    def __init__(self, name, size, done_after):
        self.name = name
        self.size = size
        self.done_after = done_after
        self.steps = 0
        self.actions_seen = []

    def reset(self, task_kwargs):
        assert len(task_kwargs) == self.size
        obs = {"text": [f"{self.name}-reset-{i}" for i in range(self.size)]}
        infos = [{"i": i} for i in range(self.size)]
        return obs, infos

    def success_evaluator(self, total_infos, total_batch_list, episode_rewards, episode_lengths):
        return {"success_rate": np.asarray(episode_rewards, dtype=np.float32) > 0}

    def step(self, actions):
        assert len(actions) == self.size, f"{self.name} got {len(actions)} actions for {self.size} rows"
        self.steps += 1
        self.actions_seen.append(list(actions))
        obs = {"text": [f"{self.name}-t{self.steps}-{i}" for i in range(self.size)]}
        rewards = np.full(self.size, float(self.steps), dtype=np.float32)
        dones = np.full(self.size, self.steps >= self.done_after, dtype=bool)
        infos = [{"i": i, "is_action_valid": True} for i in range(self.size)]
        return obs, rewards, dones, infos


def _manager(sizes=(("search", 2, 2), ("alfworld", 3, 9)), max_steps=None):
    managers = {name: FakeTaskManager(name, size, done) for name, size, done in sizes}
    caps = max_steps or {name: 50 for name, _, _ in sizes}
    mgr = MultiTaskEnvironmentManager(managers, caps, config=None)
    kwargs = []
    for name, size, _ in sizes:
        kwargs.extend({"task_name": name} for _ in range(size))
    mgr.reset(kwargs)
    return mgr, managers


def _actions(mgr):
    return ["act"] * sum(len(v) for v in mgr.task_row_indices().values())


# --------------------------------------------------------------------------- #
# step(tasks=...)
# --------------------------------------------------------------------------- #
def test_task_row_indices_partitions_the_batch():
    mgr, _ = _manager()
    groups = mgr.task_row_indices()
    assert groups == {"search": [0, 1], "alfworld": [2, 3, 4]}
    flat = [row for rows in groups.values() for row in rows]
    assert sorted(flat) == list(range(5)), "the row sets must be disjoint and cover the batch"


def test_naming_one_task_steps_only_that_task():
    mgr, managers = _manager()
    mgr.step(_actions(mgr), tasks=["search"])
    assert managers["search"].steps == 1
    assert managers["alfworld"].steps == 0, "alfworld advanced on a turn that was not its own"


def test_the_other_tasks_turn_counter_does_not_move():
    mgr, _ = _manager()
    for _ in range(3):
        mgr.step(_actions(mgr), tasks=["alfworld"])
    assert mgr._task_steps["alfworld"] == 3
    assert mgr._task_steps["search"] == 0


def test_only_the_named_task_rows_carry_meaning():
    mgr, _ = _manager()
    obs, rewards, dones, infos = mgr.step(_actions(mgr), tasks=["search"])
    assert obs["text"][0] == "search-t1-0" and obs["text"][1] == "search-t1-1"
    assert obs["text"][2] is None, "alfworld's rows must be the neutral fill, not stale state"
    assert list(rewards[:2]) == [1.0, 1.0]
    assert list(rewards[2:]) == [0.0, 0.0, 0.0]
    assert infos[0] is not None and infos[2] is None


def test_a_task_that_is_not_in_the_batch_is_an_error_not_a_silent_skip():
    mgr, _ = _manager()
    with pytest.raises(ValueError, match="not in this batch"):
        mgr.step(_actions(mgr), tasks=["webshop"])


def test_no_tasks_argument_is_the_batch_wide_step():
    mgr, managers = _manager()
    mgr.step(_actions(mgr))
    assert managers["search"].steps == 1 and managers["alfworld"].steps == 1


def test_each_task_still_gets_actions_for_its_own_rows_only():
    mgr, managers = _manager()
    actions = ["a0", "a1", "a2", "a3", "a4"]
    mgr.step(actions, tasks=["alfworld"])
    assert managers["alfworld"].actions_seen == [["a2", "a3", "a4"]]


def test_a_finished_task_is_short_circuited_even_when_named():
    mgr, managers = _manager(sizes=(("search", 2, 1), ("alfworld", 3, 9)))
    mgr.step(_actions(mgr), tasks=["search"])          # done_after=1 -> now done
    mgr.step(_actions(mgr), tasks=["search"])
    assert managers["search"].steps == 1, "a done task must not be stepped again"


def test_the_per_task_step_cap_still_applies():
    mgr, managers = _manager(sizes=(("search", 2, 99), ("alfworld", 3, 99)),
                             max_steps={"search": 2, "alfworld": 50})
    for _ in range(4):
        mgr.step(_actions(mgr), tasks=["search"])
    assert managers["search"].steps == 2
    assert mgr._task_done["search"].all()


# --------------------------------------------------------------------------- #
# _splice_obs
# --------------------------------------------------------------------------- #
def test_splice_writes_only_the_rows_it_was_handed():
    obs = {"text": ["a", "b", "c", "d"], "image": None, "anchor": None}
    nxt = {"text": ["A", "B", "C", "D"], "image": None, "anchor": None}
    _splice_obs(obs, nxt, np.array([1, 3]))
    assert obs["text"] == ["a", "B", "c", "D"]


def test_splice_ignores_keys_the_step_did_not_answer():
    obs = {"text": ["a", "b"], "anchor": None}
    _splice_obs(obs, {"text": ["A", "B"], "anchor": None}, np.array([0]))
    assert obs["text"] == ["A", "b"] and obs["anchor"] is None


def test_splice_adopts_a_key_that_was_not_there_before():
    obs = {"text": ["a", "b"], "anchor": None}
    _splice_obs(obs, {"text": ["A", "B"], "anchor": ["x", "y"]}, np.array([0]))
    assert obs["anchor"] == ["x", "y"]


# --------------------------------------------------------------------------- #
# _per_task_groups refusals
# --------------------------------------------------------------------------- #
class _EnvsWithTasks:
    def __init__(self, groups):
        self._groups = groups

    def task_row_indices(self):
        return dict(self._groups)


class _EnvsWithoutTasks:
    pass


TEXT_OBS = {"text": ["a", "b"], "image": None, "anchor": None}


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_PER_TASK_ADVANCE", True)


def test_the_flag_off_stays_in_lockstep(monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_PER_TASK_ADVANCE", False)
    envs = _EnvsWithTasks({"search": [0], "alfworld": [1]})
    assert _per_task_groups(envs, TEXT_OBS, False) is None


def test_two_tasks_with_the_flag_on_are_split(flag_on):
    envs = _EnvsWithTasks({"search": [0], "alfworld": [1]})
    assert _per_task_groups(envs, TEXT_OBS, False) == {"search": [0], "alfworld": [1]}


def test_a_single_task_batch_has_nothing_to_overlap(flag_on):
    envs = _EnvsWithTasks({"alfworld": [0, 1]})
    assert _per_task_groups(envs, TEXT_OBS, False) is None


def test_an_empty_task_does_not_count_as_a_second_group(flag_on):
    envs = _EnvsWithTasks({"alfworld": [0, 1], "search": []})
    assert _per_task_groups(envs, TEXT_OBS, False) is None


def test_a_single_task_env_manager_is_refused(flag_on):
    assert _per_task_groups(_EnvsWithoutTasks(), TEXT_OBS, False) is None


def test_image_observations_are_refused(flag_on):
    """obs['image'] is one array for the batch, not a list rows can be written into."""
    envs = _EnvsWithTasks({"search": [0], "alfworld": [1]})
    image_obs = {"text": None, "image": np.zeros((2, 3)), "anchor": None}
    assert _per_task_groups(envs, image_obs, False) is None


def test_logprob_prefetch_is_refused(flag_on):
    """It appends to one shared list and issues a second GPU call during env.step."""
    envs = _EnvsWithTasks({"search": [0], "alfworld": [1]})
    assert _per_task_groups(envs, TEXT_OBS, True) is None


# --------------------------------------------------------------------------- #
# End to end: the two paths must collect the same trajectories
# --------------------------------------------------------------------------- #
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from verl import DataProto  # noqa: E402
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector  # noqa: E402

VOCAB = 64
PROMPT_LEN = 8
RESPONSE_LEN = 4


class FakeTokenizer:
    """Deterministic, and deliberately not a real tokenizer.

    Everything the loop asks of it is a pure function of the text, which is the
    property that makes the two paths comparable at all: a row's tokens must not
    depend on which other rows shared its generate call.
    """

    pad_token_id = 0

    def apply_chat_template(self, chat, add_generation_prompt=True, tokenize=False, **kwargs):
        return "|".join(str(m["content"]) for m in chat)

    def __call__(self, prompt, return_tensors=None, add_special_tokens=False):
        ids = [(ord(c) % (VOCAB - 1)) + 1 for c in str(prompt)][:PROMPT_LEN]
        ids = ids or [1]
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"][0].tolist()

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(str(int(v)) for v in row if int(v) != self.pad_token_id) for row in ids]


class FakeRolloutWG:
    """generate_sequences as a pure per-row function of the row's prompt."""

    world_size = 2

    def generate_sequences(self, data: DataProto) -> DataProto:
        prompts = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        seeds = prompts.sum(dim=-1) % (VOCAB - 1) + 1
        responses = torch.stack(
            [torch.full((RESPONSE_LEN,), int(s), dtype=torch.long) for s in seeds]
        )
        response_mask = torch.ones_like(responses)
        last = position_ids[:, -1:]
        response_position_ids = last + torch.arange(1, RESPONSE_LEN + 1)
        return DataProto.from_dict(tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=-1),
            "attention_mask": torch.cat([attention_mask, response_mask], dim=-1),
            "position_ids": torch.cat([position_ids, response_position_ids], dim=-1),
        })


def _config(max_steps):
    return OmegaConf.create({
        "env": {"max_steps": max_steps, "rollout": {"n": 0}},
        "data": {"max_prompt_length": PROMPT_LEN, "truncation": "right",
                 "apply_chat_template_kwargs": {}},
    })


def _gen_batch(sizes):
    rows = [name for name, size, _ in sizes for _ in range(size)]
    n = len(rows)
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(1, n * PROMPT_LEN + 1).reshape(n, PROMPT_LEN) % (VOCAB - 1) + 1,
            "attention_mask": torch.ones(n, PROMPT_LEN, dtype=torch.long),
            "position_ids": torch.arange(PROMPT_LEN).repeat(n, 1),
        },
        non_tensors={
            "raw_prompt": np.array([[{"role": "user", "content": f"p{i}"}] for i in range(n)], dtype=object),
            "data_source": np.array(rows, dtype=object),
            "task_name": np.array(rows, dtype=object),
            "env_kwargs": np.array([{"task_name": t} for t in rows], dtype=object),
        },
        meta_info={"eos_token_id": 2, "pad_token_id": 0, "validate": True, "do_sample": False},
    )


SIZES = (("search", 2, 2), ("alfworld", 4, 5))


def _run(per_task, monkeypatch):
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_PER_TASK_ADVANCE", per_task)
    mgr, _ = _manager(sizes=SIZES)
    collector = TrajectoryCollector(_config(max_steps=6), FakeTokenizer())
    return collector.vanilla_multi_turn_loop(_gen_batch(SIZES), FakeRolloutWG(), mgr)


def _comparable(result):
    total_batch_list, rewards, lengths, success, traj_uid, tool_callings = result
    rows = [
        [(int(r["active_masks"]), float(r["rewards"]), r["responses"].tolist()) for r in traj]
        for traj in total_batch_list
    ]
    return rows, rewards.tolist(), lengths.tolist(), tool_callings.tolist()


def test_per_task_advance_collects_the_same_trajectories(monkeypatch):
    """The two paths differ in WHEN a row is generated, never in what it holds.

    With generation a pure function of the row, the whole collection must match:
    the same turns recorded, the same rewards, the same episode lengths. A real
    engine would not reproduce the tokens bit for bit (batch shape moves the
    reduction order, as merging already showed), but the bookkeeping around it
    has no such excuse.
    """
    lockstep = _comparable(_run(False, monkeypatch))
    per_task = _comparable(_run(True, monkeypatch))
    assert per_task == lockstep


def test_the_two_paths_really_did_take_different_routes(monkeypatch):
    """Guard the test above: if per-task silently fell back, it proves nothing."""
    monkeypatch.setattr(rollout_loop, "_ROLLOUT_PER_TASK_ADVANCE", True)
    mgr, managers = _manager(sizes=SIZES)
    assert _per_task_groups(mgr, {"text": ["x"] * 6, "image": None}, False) is not None

    collector = TrajectoryCollector(_config(max_steps=6), FakeTokenizer())
    collector.vanilla_multi_turn_loop(_gen_batch(SIZES), FakeRolloutWG(), mgr)
    # search is done after 2 turns, alfworld after 5. In lockstep the manager
    # short-circuits search but the LOOP still runs 5 turns for everyone; per
    # task, search's thread simply stops.
    assert managers["search"].steps == 2
    assert managers["alfworld"].steps == 5
