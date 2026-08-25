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
"""Every method the driver calls on the worker group must still be dispatchable.

RayWorkerGroup binds its methods by scanning the worker class for the marker
that @register leaves behind, so a method that loses its decorator does not
fail to import, fail a type check, or fail at startup. It disappears, and the
run dies minutes in with

    AttributeError: 'RayWorkerGroup' object has no attribute 'generate_sequences'

after the checkpoint is loaded, vLLM is built and 126 environments are up.

It has happened: inserting a helper directly above generate_sequences moved the
decorator onto the helper. Nothing in the diff looked wrong -- the decorator was
still there, one line further down, attached to something else.

The list below is every attribute the trainer and the rollout loop reach for on
actor_rollout_wg. If a method is added there, add it here.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

import verl.workers.fsdp_workers as fsdp_workers  # noqa: E402
from verl.single_controller.base.decorator import MAGIC_ATTR  # noqa: E402

CALLED_ON_THE_WORKER_GROUP = (
    "begin_rollout_session",
    "compute_actor_topk_log_prob",
    "compute_log_prob",
    "compute_ref_log_prob",
    "end_rollout_session",
    "generate_sequences",
    "init_model",
    "load_checkpoint",
    "rollout_pump_step",
    "save_checkpoint",
    "update_actor",
    "update_actor_async",
    "wait_for_checkpoint",
)


@pytest.mark.parametrize("name", CALLED_ON_THE_WORKER_GROUP)
def test_the_method_exists_and_is_registered(name):
    method = getattr(fsdp_workers.ActorRolloutRefWorker, name, None)
    assert method is not None, f"ActorRolloutRefWorker has no {name}"
    assert hasattr(method, MAGIC_ATTR), (
        f"{name} lost its @register decorator: RayWorkerGroup will not bind it, "
        "and the run dies with AttributeError minutes after startup"
    )


def test_a_private_helper_is_not_registered():
    """The other half of the same mistake: the decorator landing on a helper.

    _record_gen_phases is called by generate_sequences on the worker itself, so
    exposing it on the group would be harmless but wrong -- and its being
    registered is exactly the symptom of the decorator having slid off.
    """
    assert not hasattr(fsdp_workers.ActorRolloutRefWorker._record_gen_phases, MAGIC_ATTR)
