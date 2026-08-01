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
"""Measure how much memory a single WebShop worker gains per episode.

The multitask run dies of host RAM around step 24: ``perf/cpu_memory_used_gb``
climbs ~1.8GB every step and never falls back. This reproduces the growth on one
worker in a couple of minutes instead of an hour of training, and splits RSS into
anonymous and file-backed so the answer is unambiguous -- ``psutil``'s ``used``,
which is what Ray's OOM killer and that metric watch, counts only the anonymous
half, while a Lucene index being paged in shows up purely as file-backed.

    python scripts/probe_webshop_memory.py --episodes 30 --turns 8

Run it once with ``SDAR_WEBSHOP_JVM_OPTIONS=-Xmx62g`` (roughly what an uncapped
JVM picks on a 256GB host) to see the old behaviour, then again with the default
cap to see the fix.
"""

import argparse
import os
import sys

WEBSHOP_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_system", "environments", "env_package", "webshop")


def _rss_mib():
    """(total, anonymous, file-backed) resident MiB for this process."""
    fields = {}
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            key, _, value = line.partition(":")
            if key in ("Rss", "Anonymous"):
                fields[key] = int(value.split()[0]) / 1024
    total, anon = fields.get("Rss", 0.0), fields.get("Anonymous", 0.0)
    return total, anon, total - anon


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=30, help="episodes to run; one training step is one episode per worker")
    parser.add_argument("--turns", type=int, default=8, help="actions per episode")
    parser.add_argument("--every", type=int, default=5, help="print a row every N episodes")
    args = parser.parse_args()

    # Import the worker first: it sets _JAVA_OPTIONS, which only takes effect if
    # it lands before pyserini starts the JVM.
    sys.path.insert(0, os.path.dirname(os.path.dirname(WEBSHOP_ROOT)))
    from agent_system.environments.env_package.webshop.envs import WEBSHOP_JVM_OPTIONS

    os.environ.setdefault("_JAVA_OPTIONS", WEBSHOP_JVM_OPTIONS)
    print(f"_JAVA_OPTIONS={os.environ['_JAVA_OPTIONS']}")

    sys.path.append(os.path.join(WEBSHOP_ROOT, "webshop"))
    import gym  # noqa: F401  (registers WebAgentTextEnv-v0 via the import below)
    from web_agent_site.envs import WebAgentTextEnv

    env = WebAgentTextEnv(observation_mode="text", num_products=None, seed=1)
    base_total, base_anon, base_file = _rss_mib()
    print(f"after init: rss={base_total:8.1f}  anon={base_anon:8.1f}  file={base_file:8.1f} MiB")
    print(f"{'episode':>7} {'rss':>9} {'anon':>9} {'file':>9} {'d_anon/ep':>10} {'sessions':>9}")

    goals = len(env.server.goals)
    for episode in range(1, args.episodes + 1):
        env.reset(session=500 + (episode % max(goals - 500, 1)))
        for turn in range(args.turns):
            # A search then a click walks the code that allocates: Lucene query,
            # BM25 hits, template render, BeautifulSoup parse.
            actions = env.get_available_actions()
            if actions.get("has_search_bar"):
                env.step(f"search[{env.instruction_text[:60]}]")
            elif actions.get("clickables"):
                env.step(f"click[{actions['clickables'][turn % len(actions['clickables'])]}]")
            else:
                break

        if episode % args.every == 0 or episode == args.episodes:
            total, anon, filed = _rss_mib()
            per_ep = (anon - base_anon) / episode
            print(f"{episode:>7} {total:>9.1f} {anon:>9.1f} {filed:>9.1f} {per_ep:>10.2f} {len(env.server.user_sessions):>9}")

    total, anon, filed = _rss_mib()
    print(
        f"\ngrowth over {args.episodes} episodes: anon {anon - base_anon:+.1f} MiB, file {filed - base_file:+.1f} MiB\n"
        f"projected over 300 steps x 120 workers: {(anon - base_anon) / args.episodes * 300 * 120 / 1024:.1f} GiB of anonymous memory"
    )


if __name__ == "__main__":
    main()
