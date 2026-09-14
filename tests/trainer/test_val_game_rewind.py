"""Every validation must score the same problems, so the val envs get rewound.

The env managers are built once per process and each reset() takes the NEXT
element of the cycle: the next game for alfworld, the next goal draw for webshop.
So the second validation inside one run scored a different set than the first --
with test_freq=150 over 300 steps, @150 and @300 were not comparable, and the gap
between them mixed a policy change with a change of test set. rewind_games() puts
the cycle back where a freshly started val_only process has it, which is the state
ten repeated val-only runs of one checkpoint agreed on (identical 126-game
multiset). Training envs must refuse it: there the cycle advancing once per step is
what gives a run 3553 games instead of 15.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


# --- 1. the mechanism: re-seeding really rewinds TextWorld's game cycle -------
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/data/alfworld"))
DATA = os.path.join(os.environ["ALFWORLD_DATA"], "json_2.1.1", "valid_seen")
if not os.path.isdir(DATA):
    print("SKIP: no alfworld data on this host")
else:
    from agent_system.environments.env_package.alfworld import envs as alfenvs
    cfg_path = os.path.join(REPO, "agent_system/environments/env_package/alfworld/configs/config_tw.yaml")
    cfg = alfenvs.load_config_file(cfg_path)
    base = alfenvs.get_environment(cfg["env"]["type"])(cfg, train_eval="eval_in_distribution")
    env = base.init_env(batch_size=1)

    def play_next(seed=None):
        if seed is not None:
            env.seed(seed)
        _, info = env.reset()
        rel = info["extra.gamefile"][0].split("json_2.1.1/")[-1]
        return rel.split("/")[1] if rel.count("/") > 1 else rel

    first = play_next(1001)
    second = play_next()
    rewound = play_next(1001)
    check(first != second, f"without a rewind the next reset plays a different game ({first[:30]} -> {second[:30]})")
    check(rewound == first, "re-seeding puts the cycle back at the first game")

# --- 2. AlfworldEnvs.rewind_games seeds every worker the way __init__ did -----
from agent_system.environments.env_package.alfworld import envs as alfenvs_mod  # noqa: E402

seen = []


def fake_worker():
    return SimpleNamespace(reseed=SimpleNamespace(remote=lambda s: seen.append(s)))


envs = object.__new__(alfenvs_mod.AlfworldEnvs)
envs._seed = 1001
envs.group_n = 8
envs.num_processes = 24
envs.is_train = False
envs.workers = [fake_worker() for _ in range(envs.num_processes)]
real_get = alfenvs_mod.ray.get
alfenvs_mod.ray.get = lambda x: x
try:
    envs.rewind_games()
finally:
    alfenvs_mod.ray.get = real_get
expected = [1001 + (i // 8) for i in range(24)]
check(seen == expected, f"every worker is re-seeded with seed + i//group_n ({sorted(set(seen))} for 3 groups of 8)")

envs.is_train = True
try:
    envs.rewind_games()
    check(False, "training envs must refuse to be rewound")
except RuntimeError:
    check(True, "training envs refuse to be rewound (their cycle must keep advancing)")

# --- 3. webshop: the goal draw goes back to the beginning --------------------
from agent_system.environments.env_package.webshop import envs as webshop_envs  # noqa: E402

w = object.__new__(webshop_envs.WebshopMultiProcessEnv)
w._seed = 1001
w.is_train = False
w._workers = []          # object.__new__ skipped __init__; close() looks for this
w.env_num = 4
w.goal_idxs = range(500)
w._rng = np.random.RandomState(w._seed)
draw = lambda: tuple(w._rng.choice(w.goal_idxs, size=w.env_num, replace=False))
a, b = draw(), draw()
w.rewind_games()
c = draw()
check(a != b, "without a rewind the next validation draws different webshop goals")
check(a == c, "rewind_games puts the webshop goal draw back to the first")

w.is_train = True
try:
    w.rewind_games()
    check(False, "training webshop envs must refuse to be rewound")
except RuntimeError:
    check(True, "training webshop envs refuse to be rewound")


def test_val_game_rewind():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
