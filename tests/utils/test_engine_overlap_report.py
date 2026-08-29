"""The engine says which overlap knobs it has, instead of us guessing.

All-three-cards-busy samples read 87-90%, not 100. The missing tenth is the
engine's per-step host work sitting between kernel launches -- schedule, build
inputs, process outputs. That is not physics, and vLLM has machinery for hiding
it, but WHICH machinery depends on the version: V0's num_scheduler_steps was
removed in V1 and replaced by async scheduling. Guessing a flag name costs a
failed launch; assuming a default costs a wrong conclusion about the ceiling.
"""

import importlib
import sys
import types

import pytest


def _module(monkeypatch, *, version="0.11.0", params=("model", "async_scheduling"), raise_on_import=False):
    """Import the rollout module against a stubbed vLLM with a chosen signature."""

    def _init(self, model=None, async_scheduling=None):  # replaced below
        pass

    fake_llm = type("LLM", (), {})
    src = "def __init__(self, " + ", ".join(f"{p}=None" for p in params) + "): pass"
    namespace = {}
    exec(src, namespace)
    fake_llm.__init__ = namespace["__init__"]

    fake = types.ModuleType("vllm")
    fake.__version__ = version
    fake.LLM = fake_llm
    if raise_on_import:
        class _Boom(types.ModuleType):
            def __getattr__(self, name):
                raise RuntimeError("internals moved")

        fake = _Boom("vllm")
    monkeypatch.setitem(sys.modules, "vllm", fake)
    return fake


@pytest.fixture()
def report(monkeypatch):
    """The function under test, imported without dragging in a real vLLM."""
    mod = importlib.import_module("verl.utils.engine_overlap")
    return mod.report_engine_overlap


def test_an_available_knob_is_named_as_available(report, monkeypatch, capsys):
    _module(monkeypatch, params=("model", "async_scheduling"))
    report(engine_kwargs={})
    out = capsys.readouterr().out
    assert "async_scheduling=<default> (available)" in out, out


def test_a_knob_this_version_dropped_is_named_absent(report, monkeypatch, capsys):
    """V0's multi-step scheduler does not exist on V1, and silence would read
    like "we chose not to set it" rather than "it is gone"."""
    _module(monkeypatch, params=("model", "async_scheduling"))
    report(engine_kwargs={})
    out = capsys.readouterr().out
    assert "num_scheduler_steps=absent" in out, out


def test_a_knob_we_set_is_reported_with_its_value(report, monkeypatch, capsys):
    _module(monkeypatch, params=("model", "async_scheduling"))
    report(engine_kwargs={"async_scheduling": True})
    out = capsys.readouterr().out
    assert "async_scheduling=True (set here)" in out, out


def test_the_version_is_on_the_line(report, monkeypatch, capsys):
    """A conclusion about the ceiling is only as good as the version it held for."""
    _module(monkeypatch, version="0.8.5", params=("model",))
    report(engine_kwargs={})
    assert "vllm 0.8.5" in capsys.readouterr().out


def test_a_moved_internal_costs_the_line_and_not_the_run(report, monkeypatch, capsys):
    _module(monkeypatch, raise_on_import=True)
    report(engine_kwargs={})  # must not raise
    assert "could not report overlap knobs" in capsys.readouterr().out


def test_a_value_the_rollout_sets_itself_is_not_reported_as_a_default(report, monkeypatch, capsys):
    """enable_chunked_prefill is named in LLM(...), not in engine_kwargs.

    Printing it as "<default>" would be a lie about a value we chose, and the
    whole point of this line is to stop conclusions resting on assumed defaults.
    """
    _module(monkeypatch, params=("model", "enable_chunked_prefill"))
    report(engine_kwargs={}, explicit={"enable_chunked_prefill": False})
    assert "enable_chunked_prefill=False (set here)" in capsys.readouterr().out


def test_engine_kwargs_win_over_the_explicit_ones(report, monkeypatch, capsys):
    """They are applied last in the LLM(...) call, so they are what takes effect."""
    _module(monkeypatch, params=("model", "enable_chunked_prefill"))
    report(engine_kwargs={"enable_chunked_prefill": True}, explicit={"enable_chunked_prefill": False})
    assert "enable_chunked_prefill=True (set here)" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# which engine core is running decides whether the V0 knobs mean anything
# --------------------------------------------------------------------------- #
def _engine_from(module_name):
    inner = type("Core", (), {})
    inner.__module__ = module_name
    return type("LLM", (), {"llm_engine": inner()})()


def test_a_v1_core_is_named_v1(report, monkeypatch, capsys):
    """num_scheduler_steps and disable_async_output_proc are V0 features.

    On a V1 core they are in the signature and ignored in effect -- a flag that
    takes a value and changes nothing, which is the worst kind to reason from.
    """
    _module(monkeypatch, params=("model",))
    report(engine_kwargs={}, engine=_engine_from("vllm.v1.engine.llm_engine"))
    assert "core=v1" in capsys.readouterr().out


def test_a_v0_core_is_named_v0(report, monkeypatch, capsys):
    _module(monkeypatch, params=("model",))
    report(engine_kwargs={}, engine=_engine_from("vllm.engine.llm_engine"))
    assert "core=v0" in capsys.readouterr().out


def test_the_core_is_read_from_the_object_not_an_env_var(report, monkeypatch, capsys):
    """VLLM_USE_V1 is a request. vLLM falls back to V0 for configurations V1
    cannot serve and the variable keeps its value, so a run can ask for V1 and
    get V0 with nothing saying so."""
    _module(monkeypatch, params=("model",))
    monkeypatch.setenv("VLLM_USE_V1", "1")
    report(engine_kwargs={}, engine=_engine_from("vllm.engine.llm_engine"))
    assert "core=v0" in capsys.readouterr().out


def test_an_unrecognisable_engine_is_unknown_not_a_guess(report, monkeypatch, capsys):
    _module(monkeypatch, params=("model",))
    report(engine_kwargs={}, engine=_engine_from("somebody_elses.engine"))
    assert "core=unknown" in capsys.readouterr().out


def test_no_engine_given_is_still_a_line(report, monkeypatch, capsys):
    _module(monkeypatch, params=("model",))
    report(engine_kwargs={})
    assert "core=unknown" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# asking for a core is not getting one
# --------------------------------------------------------------------------- #
def test_a_core_that_matches_the_request_is_confirmed(monkeypatch, capsys):
    mod = importlib.import_module("verl.utils.engine_overlap")
    monkeypatch.setattr(mod, "_REQUIRE_CORE", "v0")
    mod.require_core("v0")
    assert "ROLLOUT_REQUIRE_CORE=v0: confirmed" in capsys.readouterr().out


def test_a_silent_fallback_to_the_other_core_stops_the_run(monkeypatch):
    """VLLM_USE_V1=0 is a request; vLLM falls back rather than failing.

    This rollout hands it two things V0 may not serve -- external_launcher and
    enable_sleep_mode -- and a fallback produces a working run that measures the
    core you were trying to compare against. That is the shape of the two
    failures that already cost whole runs here: a pool that refused and fell
    back to the blocking path, and a retriever that fell back to one query at a
    time. Both completed and both measured the control twice.
    """
    mod = importlib.import_module("verl.utils.engine_overlap")
    monkeypatch.setattr(mod, "_REQUIRE_CORE", "v0")
    with pytest.raises(RuntimeError, match="but vLLM built core=v1"):
        mod.require_core("v1")


def test_an_unrecognisable_core_also_stops_the_run(monkeypatch):
    """"unknown" is not "the one I asked for", and guessing is what this stops."""
    mod = importlib.import_module("verl.utils.engine_overlap")
    monkeypatch.setattr(mod, "_REQUIRE_CORE", "v0")
    with pytest.raises(RuntimeError, match="core=unknown"):
        mod.require_core("unknown")


def test_without_the_variable_nothing_is_enforced(monkeypatch, capsys):
    """Opt-in: a run that never asked for a core must not start failing."""
    mod = importlib.import_module("verl.utils.engine_overlap")
    monkeypatch.setattr(mod, "_REQUIRE_CORE", "")
    mod.require_core("v1")
    assert capsys.readouterr().out == ""


def test_the_guard_runs_even_when_the_knob_report_failed(monkeypatch, capsys):
    """A moved vLLM internal costs the diagnostic line, not the guard.

    Losing the report is cosmetic; losing the guard silently restores exactly
    the failure it exists to prevent.
    """
    mod = importlib.import_module("verl.utils.engine_overlap")
    _module(monkeypatch, raise_on_import=True)
    monkeypatch.setattr(mod, "_REQUIRE_CORE", "v0")
    with pytest.raises(RuntimeError, match="ROLLOUT_REQUIRE_CORE=v0"):
        mod.report_engine_overlap(engine_kwargs={}, engine=_engine_from("vllm.v1.engine.llm_engine"))
    assert "could not report overlap knobs" in capsys.readouterr().out
