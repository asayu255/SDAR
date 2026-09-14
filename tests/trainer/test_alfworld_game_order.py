"""ALFWorld's game order must not depend on the filesystem it is read from.

collect_game_files built its list in os.walk order, and os.walk order is a property
of the filesystem, not of the data: the same 140 valid_seen games came out in three
different orders on fuji (ext4), tamago (ext4) and yamabuki (NFS). Every draw
downstream is by POSITION in that list -- TextworldBatchGymEnv.seed shuffles a copy
with RandomState(seed) and plays element 0 -- so identical seeds drew different games
on every host, and validation sets on two hosts did not even share a task-type mix.

This builds a small fake dataset under two different mount points, replays os.walk
in several different orders, and requires the same list, and the same seeded draw,
every time. It also checks the replayed orders really differ, so a pass means
something.
"""
import json
import os
import random
import sys
import tempfile
from unittest import mock

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import alfred_tw_env as mod  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def make_dataset(root):
    """Fake games: three per task type, plus one movable and one unsolvable to skip."""
    names = sorted(set(mod.TASK_TYPES.values()))
    n = 0
    for t, tt in enumerate(names):
        for k in range(3):
            d = os.path.join(root, f"{tt}-Obj{k}-None-Recep-{t}{k}", f"trial_T2019_{t}{k}")
            os.makedirs(d)
            json.dump({"task_type": tt}, open(os.path.join(d, "traj_data.json"), "w"))
            json.dump({"solvable": True}, open(os.path.join(d, "game.tw-pddl"), "w"))
            n += 1
    for bad, solvable in (("movable_thing", True), ("unsolvable", False)):
        d = os.path.join(root, f"{names[0]}-{bad}-None-X-9", "trial_T2019_99")
        os.makedirs(d)
        json.dump({"task_type": names[0]}, open(os.path.join(d, "traj_data.json"), "w"))
        json.dump({"solvable": solvable}, open(os.path.join(d, "game.tw-pddl"), "w"))
    return n


def collect(root, walk_rows):
    env = object.__new__(mod.AlfredTWEnv)
    env.config = {"dataset": {"data_path": root, "eval_id_data_path": root,
                              "eval_ood_data_path": root,
                              "num_train_games": -1, "num_eval_games": -1},
                  "env": {"task_types": sorted(mod.TASK_TYPES.keys())}}
    env.train_eval = "eval_in_distribution"
    with mock.patch.object(mod.os, "walk", side_effect=lambda *a, **k: iter(list(walk_rows))):
        env.collect_game_files()
    return [os.path.relpath(p, root) for p in env.game_files]


def first_game(lst, seed):
    """The draw that decides which game a worker plays, as TextworldBatchGymEnv.seed does."""
    g = list(lst)
    np.random.RandomState(seed).shuffle(g)
    return g[0]


with tempfile.TemporaryDirectory() as tmp:
    roots = [os.path.join(tmp, "home", "ohara", "data"), os.path.join(tmp, "opt", "home", "ohara", "data")]
    expected = None
    for root in roots:
        n_good = make_dataset(root)
        real = list(os.walk(root, topdown=False))
        orders = [real, list(reversed(real))]
        for seed in (1, 2, 3):
            perm = list(real)
            random.Random(seed).shuffle(perm)
            orders.append(perm)
        where = os.path.relpath(root, tmp)
        distinct = len({tuple(r[0] for r in o) for o in orders})
        check(distinct >= 4, f"{where}: {distinct} genuinely different os.walk orders replayed")
        lists = [collect(root, o) for o in orders]
        check(all(l == lists[0] for l in lists), f"{where}: the same game list under every walk order")
        check(lists[0] == sorted(lists[0]), f"{where}: the list is in relative-path order")
        check(len(lists[0]) == n_good,
              f"{where}: movable and unsolvable games are still skipped ({len(lists[0])} kept of {n_good + 2})")
        if expected is None:
            expected = lists[0]
        else:
            check(lists[0] == expected, "the same relative list under a different mount point")

    draws = {tuple(first_game(expected, 1001 + i) for i in range(20))}
    for root in roots:
        real = list(os.walk(root, topdown=False))
        random.Random(7).shuffle(real)
        draws.add(tuple(first_game(collect(root, real), 1001 + i) for i in range(20)))
    check(len(draws) == 1, "seeded draws for workers 0-19 are identical across mount points and walk orders")


def test_alfworld_game_order():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
