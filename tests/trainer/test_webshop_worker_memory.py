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
"""The two unbounded structures behind the host-RAM climb in webshop workers.

The embedded JVM (pyserini) sizes its heap off physical RAM unless told not to,
and ``SimServer.user_sessions`` keeps every episode a worker has ever run. With
120 workers alive for 300 steps neither is affordable on a 256GB host.
"""

import importlib
import os
import sys

import pytest


def _load_envs_module():
    pytest.importorskip("ray")
    pytest.importorskip("gym")
    sys.modules.pop("agent_system.environments.env_package.webshop.envs", None)
    return importlib.import_module("agent_system.environments.env_package.webshop.envs")


def test_jvm_heap_is_capped_by_default():
    envs = _load_envs_module()
    assert "-Xmx" in envs.WEBSHOP_JVM_OPTIONS


def test_jvm_options_are_overridable(monkeypatch):
    monkeypatch.setenv("SDAR_WEBSHOP_JVM_OPTIONS", "-Xmx4g")
    envs = _load_envs_module()
    assert envs.WEBSHOP_JVM_OPTIONS == "-Xmx4g"


def test_worker_sets_java_options_before_importing_pyserini(monkeypatch):
    """``_JAVA_OPTIONS`` read after the JVM has started is ignored, so the worker
    has to set it before the import that starts it."""
    envs = _load_envs_module()
    monkeypatch.delenv("_JAVA_OPTIONS", raising=False)

    class _Boom(Exception):
        pass

    monkeypatch.setattr(envs.gym, "make", lambda *a, **k: (_ for _ in ()).throw(_Boom))
    # The construction is expected to fail -- either at ``gym.make`` above, or
    # earlier at the pyserini import on a host that lacks it. Both land after the
    # line under test, which is the point.
    with pytest.raises(Exception):
        envs.WebshopWorker(seed=1, env_kwargs={})
    assert os.environ.get("_JAVA_OPTIONS") == envs.WEBSHOP_JVM_OPTIONS


def _text_env_class():
    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "agent_system", "environments", "env_package", "webshop", "webshop",
    )
    if root not in sys.path:
        sys.path.append(root)
    try:
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    except Exception as e:  # pragma: no cover - pyserini/spacy not installed
        pytest.skip(f"webshop env import unavailable: {e}")
    return WebAgentTextEnv


class _Server:
    def __init__(self):
        self.user_sessions = {"stale-1": {}, "stale-2": {}}


class _Browser:
    def __init__(self, server):
        self.server = server

    def get(self, url, session_id=None, session_int=None):
        self.server.user_sessions[session_id] = {"goal": {}, "done": False}


def _env(monkeypatch, owns_server):
    cls = _text_env_class()
    monkeypatch.setattr(cls, "observation", property(lambda self: "obs"), raising=False)
    env = object.__new__(cls)  # the real __init__ needs the product corpus
    env.server = _Server()
    env.browser = _Browser(env.server)
    env.base_url = "http://127.0.0.1:3000"
    env.session_prefix = None
    env._owns_server = owns_server
    env.observation_mode = "text"
    return env


def test_reset_drops_the_previous_episodes_sessions(monkeypatch):
    env = _env(monkeypatch, owns_server=True)
    env.reset(session=7, instruction_text="buy a thing")
    assert list(env.server.user_sessions) == ["7"]


def test_reset_leaves_a_shared_servers_sessions_alone(monkeypatch):
    env = _env(monkeypatch, owns_server=False)
    env.reset(session=7, instruction_text="buy a thing")
    assert set(env.server.user_sessions) == {"stale-1", "stale-2", "7"}


def test_mem_stats_splits_resident_memory_into_anonymous_and_file_backed():
    envs = _load_envs_module()
    worker = object.__new__(envs.WebshopWorker)
    stats = envs.WebshopWorker.mem_stats(worker)
    if not stats:  # pragma: no cover - kernel without smaps_rollup
        pytest.skip("/proc/self/smaps_rollup unavailable")
    assert stats["rss_mib"] > 0
    assert stats["anon_mib"] + stats["file_mib"] == pytest.approx(stats["rss_mib"])
