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
"""``adjust_batch``'s copies must not move the yardstick GRPO measures against.

The copies exist only to reach a DP/micro-divisible row count. They carry their
original's ``uid``, so before this they were counted a second time when the
group's mean/std was formed -- and that mean/std is applied to every REAL row of
the group. Zeroing the copy's loss weight does not undo that: what was perturbed
is the statistic, not the copy's own contribution.

The second half covers the assert on the grouping itself. ``uid`` is assigned by
POSITION, so it is correct only while the row order matches the env layout. When
that breaks nothing raises on its own -- GRPO keeps normalising rows against a
baseline built from unrelated prompts -- so the check has to be explicit.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _rewards(scores, resp_len=3):
    """Token-level rewards whose row sums are ``scores`` (GRPO reads the sum)."""
    t = torch.zeros(len(scores), resp_len)
    t[:, -1] = torch.tensor(scores, dtype=torch.float32)
    return t


def _advantages(scores, uids, trajs, padding_mask=None):
    adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=_rewards(scores),
        response_mask=torch.ones(len(scores), 3),
        index=np.array(uids, dtype=object),
        traj_index=np.array(trajs, dtype=object),
        padding_mask=padding_mask,
    )
    return adv[:, 0]


# --------------------------------------------------------------------------- #
# The statistic
# --------------------------------------------------------------------------- #
def test_a_copy_shifts_the_advantage_of_every_real_row_in_its_group():
    """The failure being fixed, stated as a test so it cannot come back quietly."""
    scores = [0.0, 0.0, 0.0, 4.0]
    uids = ["g"] * 4
    trajs = [f"t{i}" for i in range(4)]

    clean = _advantages(scores, uids, trajs)
    # The '4.0' row copied onto the end, as adjust_batch would do.
    contaminated = _advantages(scores + [4.0], uids + ["g"], trajs + ["t3"])

    assert not torch.allclose(clean, contaminated[:4]), (
        "a duplicated row left the real rows' advantages unchanged; the setup no "
        "longer reproduces the contamination this module is about"
    )
    # Direction: the copied row was the good one, so it pulls the baseline up and
    # every real row is scored lower against it.
    assert (contaminated[:4] < clean).all()


def test_excluding_the_copies_restores_the_uncontaminated_advantages():
    scores = [0.0, 0.0, 0.0, 4.0]
    uids = ["g"] * 4
    trajs = [f"t{i}" for i in range(4)]

    clean = _advantages(scores, uids, trajs)
    mask = torch.tensor([False] * 4 + [True])
    fixed = _advantages(scores + [4.0], uids + ["g"], trajs + ["t3"], padding_mask=mask)

    assert torch.allclose(fixed[:4], clean, atol=1e-6)


def test_a_copy_still_receives_its_originals_advantage():
    """Excluded from the statistic, not from the output.

    The copy is a row of the batch and every row needs an advantage; it gets the
    one its original produced. (Its loss weight is 0, so the value is inert -- but
    a NaN or a KeyError here would not be.)
    """
    scores = [0.0, 2.0, 4.0]
    uids = ["g"] * 3
    trajs = ["t0", "t1", "t2"]
    mask = torch.tensor([False, False, False, True])

    out = _advantages(scores + [4.0], uids + ["g"], trajs + ["t2"], padding_mask=mask)
    assert torch.isfinite(out).all()
    assert out[3] == pytest.approx(out[2].item(), abs=1e-6)


def test_no_mask_is_exactly_the_previous_behaviour():
    """Every non-env recipe passes None; it must not change."""
    scores = [0.0, 1.0, 2.0, 3.0]
    uids = ["a", "a", "b", "b"]
    trajs = ["t0", "t1", "t2", "t3"]
    assert torch.allclose(
        _advantages(scores, uids, trajs),
        _advantages(scores, uids, trajs, padding_mask=None),
    )


def test_an_all_false_mask_changes_nothing():
    scores = [0.0, 1.0, 2.0, 3.0]
    uids = ["a", "a", "b", "b"]
    trajs = ["t0", "t1", "t2", "t3"]
    assert torch.allclose(
        _advantages(scores, uids, trajs),
        _advantages(scores, uids, trajs, padding_mask=torch.zeros(4, dtype=torch.bool)),
    )


# --------------------------------------------------------------------------- #
# adjust_batch writes the marker
# --------------------------------------------------------------------------- #
def _config(world_size=2, lp_micro=4, ppo_micro=2):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": world_size, "nnodes": 1},
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "rollout": {"log_prob_micro_batch_size_per_gpu": lp_micro},
                "ref": {"log_prob_micro_batch_size_per_gpu": lp_micro},
                "actor": {
                    "use_kl_loss": False,
                    "ppo_micro_batch_size_per_gpu": ppo_micro,
                    "ppo_mini_batch_size": 8,
                },
            },
        }
    )


def _batch(n):
    return DataProto.from_dict(
        tensors={"input_ids": torch.arange(n).view(n, 1)},
        non_tensors={"uid": np.array(["g"] * n, dtype=object)},
    )


def test_adjust_batch_marks_exactly_the_rows_it_appended():
    from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY, adjust_batch

    # divisor = lcm(4*2, 2*2) = 8; 13 rows -> 3 copies appended
    out = adjust_batch(_config(), _batch(13))
    assert len(out) == 16
    mask = out.batch[PADDING_ROW_KEY]
    assert mask.dtype is torch.bool
    assert mask[:13].sum() == 0 and mask[13:].all()


def test_an_already_divisible_batch_is_returned_untouched():
    """Absent column == no padding; the early return must stay a no-op."""
    from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY, adjust_batch

    out = adjust_batch(_config(), _batch(16))
    assert len(out) == 16
    assert out.batch.get(PADDING_ROW_KEY, None) is None


def test_the_marker_survives_a_reorder():
    """_balance_batch reorders before compute_advantage reads this.

    A row COUNT would stop meaning anything there; a column moves with its rows,
    which is the whole reason this is a column.
    """
    from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY, adjust_batch

    out = adjust_batch(_config(), _batch(13))
    order = torch.randperm(len(out))
    ids_before = out.batch["input_ids"][out.batch[PADDING_ROW_KEY]].clone()
    out.reorder(order)
    ids_after = out.batch["input_ids"][out.batch[PADDING_ROW_KEY]]
    assert sorted(ids_before.flatten().tolist()) == sorted(ids_after.flatten().tolist())


# --------------------------------------------------------------------------- #
# The grouping assert
# --------------------------------------------------------------------------- #
def _grouped(uids, tasks, trajs):
    n = len(uids)
    return DataProto.from_dict(
        tensors={"input_ids": torch.arange(n).view(n, 1)},
        non_tensors={
            "uid": np.array(uids, dtype=object),
            "task_name": np.array(tasks, dtype=object),
            "traj_uid": np.array(trajs, dtype=object),
        },
    )


def test_a_well_formed_layout_passes():
    from verl.trainer.ppo.ray_trainer import assert_grpo_groups_are_one_prompt

    # two groups of two trajectories each, one task per group
    assert_grpo_groups_are_one_prompt(
        _grouped(
            uids=["a", "a", "a", "b", "b", "b"],
            tasks=["alfworld"] * 3 + ["search"] * 3,
            trajs=["t0", "t0", "t1", "t2", "t3", "t3"],  # turns repeat traj_uid
        )
    )


def test_a_group_spanning_two_tasks_is_rejected():
    """The catastrophic form: the row order no longer matches the env layout."""
    from verl.trainer.ppo.ray_trainer import assert_grpo_groups_are_one_prompt

    with pytest.raises(AssertionError, match="spans more than one task"):
        assert_grpo_groups_are_one_prompt(
            _grouped(
                uids=["a", "a", "b", "b"],
                tasks=["alfworld", "search", "webshop", "webshop"],
                trajs=["t0", "t1", "t2", "t3"],
            )
        )


def test_uneven_group_sizes_are_rejected():
    from verl.trainer.ppo.ray_trainer import assert_grpo_groups_are_one_prompt

    with pytest.raises(AssertionError, match="different numbers of trajectories"):
        assert_grpo_groups_are_one_prompt(
            _grouped(
                uids=["a", "a", "a", "b", "b"],
                tasks=["alfworld"] * 5,
                trajs=["t0", "t1", "t2", "t3", "t4"],  # 3 vs 2
            )
        )


def test_padding_copies_do_not_trip_the_size_check():
    """A copy reuses its original's traj_uid, so the distinct count is unchanged.

    Worth pinning: the assert runs AFTER adjust_batch, so if copies counted as
    extra trajectories every padded step would abort.
    """
    from verl.trainer.ppo.ray_trainer import assert_grpo_groups_are_one_prompt

    assert_grpo_groups_are_one_prompt(
        _grouped(
            uids=["a", "a", "b", "b", "a"],           # last row is a copy
            tasks=["alfworld"] * 4 + ["alfworld"],
            trajs=["t0", "t1", "t2", "t3", "t0"],     # copy of the t0 row
        )
    )


def test_a_batch_without_the_columns_is_skipped():
    """Plain verl recipes have no task_name, and no positional convention either."""
    from verl.trainer.ppo.ray_trainer import assert_grpo_groups_are_one_prompt

    assert_grpo_groups_are_one_prompt(
        DataProto.from_dict(
            tensors={"input_ids": torch.arange(4).view(4, 1)},
            non_tensors={"uid": np.array(["a", "a", "b", "b"], dtype=object)},
        )
    )
