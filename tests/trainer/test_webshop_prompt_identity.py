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

"""WebShop's turn-0 anchor has to identify the PROMPT, not the landing page.

The cross gate takes a GRPO group's prompt identity from its turn-0 anchor and
splits the prompts into two halves by hashing it. WebShop's format_obs keeps
only the parts of the observation that FOLLOW the instruction, and on the
landing page that is the single token "'Search'" -- identical for every
shopping goal. Every goal therefore hashed to one key on one side, the
reference could never reach min_prompts, WebShop was never a valid receiver,
and its own OPD stayed eligible for attenuation on behalf of tasks that could
not reciprocate.

These tests use the real extract_task/format_obs and the real prompt_side, so
they fail if either side of that contract regresses. They do NOT start an
environment: the observation shape is WebShop's documented
"WebShop [SEP] Instruction: [SEP] <goal> [SEP] Search".
"""

import pytest

from verl.trainer.ppo.opd_cross_gate import prompt_side


GOALS = [
    "i would like a 3 ounce bottle of citrus deodorant, and price lower than 50.00 dollars",
    "i need gluten free popcorn, and price lower than 30.00 dollars",
    "find me a wireless mouse with usb receiver, and price lower than 25.00 dollars",
    "i want a blue cotton t-shirt in size large, and price lower than 40.00 dollars",
    "looking for a stainless steel water bottle, and price lower than 20.00 dollars",
    "buy me machine wash men's dress shirts, and price lower than 60.00 dollars",
]


def _obs(goal):
    return f"WebShop [SEP] Instruction: [SEP] {goal} [SEP] Search"


@pytest.fixture
def manager():
    """The real WebshopEnvironmentManager methods, without an environment.

    extract_task and format_obs are pure functions of the observation and
    self.tasks, so they can be bound to a bare object.
    """
    from agent_system.environments.env_manager import WebshopEnvironmentManager

    m = object.__new__(WebshopEnvironmentManager)
    return m


def test_format_obs_alone_collapses_every_goal_to_one_string(manager):
    """The defect, pinned: this is WHY the anchor cannot be the formatted obs."""
    obs = [_obs(g) for g in GOALS]
    manager.tasks = manager.extract_task(obs)
    formatted = manager.format_obs(obs)
    assert len(set(formatted)) == 1, (
        "format_obs is expected to drop the goal; if this now preserves it, the "
        "anchor fix below may be redundant -- check before deleting it"
    )
    assert formatted[0] == "'Search'"


def test_the_anchor_the_reset_builds_separates_distinct_goals(manager):
    """The fix: the anchor is goal + screen, so distinct goals hash apart."""
    obs = [_obs(g) for g in GOALS]
    manager.tasks = manager.extract_task(obs)
    formatted = manager.format_obs(obs)
    # exactly the expression WebshopEnvironmentManager.reset uses
    anchors = [f"{t} [SEP] {o}" for t, o in zip(manager.tasks, formatted)]
    assert len(set(anchors)) == len(GOALS)
    keys = {prompt_side("webshop", a, seed=1)[0] for a in anchors}
    assert len(keys) == len(GOALS), f"{len(GOALS)} goals collapsed to {len(keys)} keys"


def test_the_goals_reach_both_halves(manager):
    """One key per goal is not enough: they have to land on both sides, or the
    two-sided reference has nothing to cross-fit against.

    Tested at the batch size the run actually sees (the metrics show 57-61
    prompts per task per side window), not at len(GOALS): six goals landing on
    one side is a 3% coincidence of the hash, which this test hit while the fix
    was already correct. Asserting it at n=6 would be testing the coincidence.
    """
    goals = [
        f"i would like item {i} in variant {i * 7 % 13}, and price lower than "
        f"{20 + i}.00 dollars"
        for i in range(48)
    ]
    obs = [_obs(g) for g in goals]
    manager.tasks = manager.extract_task(obs)
    formatted = manager.format_obs(obs)
    anchors = [f"{t} [SEP] {o}" for t, o in zip(manager.tasks, formatted)]
    assert len(set(anchors)) == len(goals)
    sides = [prompt_side("webshop", a, seed=1)[1] for a in anchors]
    n0, n1 = sides.count(0), sides.count(1)
    assert n0 > 0 and n1 > 0, f"all {len(goals)} goals fell on one side ({n0}/{n1})"
    # and roughly balanced -- a split that puts min_prompts out of reach on one
    # side is as useless as no split. 48 fair coins land outside 12..36 with
    # probability under 1e-4.
    assert 12 <= n0 <= 36, f"lopsided split: side0={n0} side1={n1}"


def test_the_defect_would_still_be_caught_at_this_size(manager):
    """The old anchor fails the same test, so it is the fix being measured."""
    goals = [
        f"i would like item {i} in variant {i * 7 % 13}, and price lower than "
        f"{20 + i}.00 dollars"
        for i in range(48)
    ]
    obs = [_obs(g) for g in goals]
    manager.tasks = manager.extract_task(obs)
    formatted = manager.format_obs(obs)          # the OLD anchor
    keys = {prompt_side("webshop", a, seed=1)[0] for a in formatted}
    assert len(keys) == 1, "the old anchor no longer collapses; revisit this file"


def test_the_reset_expression_is_the_one_under_test(manager):
    """Guard against the test drifting from the code it claims to cover."""
    import inspect

    from agent_system.environments.env_manager import WebshopEnvironmentManager

    src = inspect.getsource(WebshopEnvironmentManager.reset)
    assert "'anchor': [f\"{t} [SEP] {o}\" for t, o in zip(self.tasks, obs)]" in src, (
        "WebshopEnvironmentManager.reset no longer builds the anchor from the goal "
        "text; update this test together with it"
    )
    assert "'anchor': obs.copy()" not in src
