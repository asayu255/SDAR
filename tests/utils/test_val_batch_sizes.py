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
"""Resolving how many rows each task's validation batch holds.

The number is not free of the scoring. An environment manager is built at this
size, and alfworld seeds its worker i from the game cycle by position within the
manager -- so a different size plays different games and moves its score. search
is indifferent (every row carries its own question and ground truth), which is
why the size is worth naming per task rather than raising for everyone.

The loader and the environment managers must agree on it: a 252-row batch handed
to a manager holding 126 environments indexes past the end, silently for the
first 126 rows.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402

import agent_system.environments.env_manager as env_manager  # noqa: E402

TASKS = ["alfworld", "search", "webshop"]


def _config(val_per_task_batch_size, val_batch_size=126):
    return OmegaConf.create(
        {
            "env": {
                "env_name": "multitask",
                "multitask": {"tasks": TASKS, "val_per_task_batch_size": val_per_task_batch_size},
            },
            "data": {"val_batch_size": val_batch_size},
        }
    )


def test_one_number_still_means_every_task():
    sizes, default = env_manager._get_multitask_val_batch_sizes(_config(126), TASKS)
    assert sizes == {"alfworld": 126, "search": 126, "webshop": 126}
    assert default == 126


def test_a_mapping_resizes_only_what_it_names():
    sizes, default = env_manager._get_multitask_val_batch_sizes(_config({"search": 252}), TASKS)
    assert sizes == {"alfworld": 126, "search": 252, "webshop": 126}
    assert default == 126


def test_the_unnamed_tasks_take_the_loader_default():
    sizes, _ = env_manager._get_multitask_val_batch_sizes(_config({"search": 252}, val_batch_size=64), TASKS)
    assert sizes["alfworld"] == 64 and sizes["webshop"] == 64


def test_a_task_that_is_not_in_the_run_is_refused():
    """A typo here would silently leave the task it meant at the default."""
    with pytest.raises(ValueError, match="not in this run"):
        env_manager._get_multitask_val_batch_sizes(_config({"serach": 252}), TASKS)


@pytest.mark.parametrize("size", [0, -1])
def test_a_non_positive_size_is_refused(size):
    with pytest.raises(ValueError, match="must be positive"):
        env_manager._get_multitask_val_batch_sizes(_config({"search": size}), TASKS)


def test_the_manager_builder_takes_a_number_or_a_mapping():
    assert env_manager._size_for(126, "alfworld") == 126
    assert env_manager._size_for({"alfworld": 126, "search": 252}, "search") == 252


def test_a_mapping_missing_the_task_is_a_loud_failure():
    """Silently defaulting would build a manager of the wrong size, and the
    mismatch only shows up as rows indexing past the end of the env list."""
    with pytest.raises(KeyError):
        env_manager._size_for({"search": 252}, "alfworld")


def test_get_val_batch_sizes_is_none_for_a_single_task_run():
    config = OmegaConf.create({"env": {"env_name": "search"}, "data": {"val_batch_size": 126}})
    assert env_manager.get_val_batch_sizes(config) is None


def test_get_val_batch_sizes_reads_the_configured_tasks():
    sizes, default = env_manager.get_val_batch_sizes(_config({"search": 252}))
    assert sizes["search"] == 252 and default == 126
