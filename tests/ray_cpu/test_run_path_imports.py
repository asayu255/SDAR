"""Load every module a Stage-2 start loads, with the engine stubbed.

This is the test that was missing. ``vllm_rollout_spmd.py`` was ported from
another branch carrying ``from verl.utils.phase_timing import PhaseTimer``, the
module it names was not ported with it, and nothing noticed until a real run
reached ``init_workers()`` -- after the config dump, after the 90-file pool load,
minutes in. The reason nothing noticed is simple: no test imports that file,
because importing it needs vllm and CI has none.

``pytest.importorskip`` is what turns that into a permanent blind spot, so this
does the opposite. vllm and sglang are stubbed at the meta-path -- they are a GPU
engine and cannot be installed in CI at all -- and then the modules a run walks
through are imported for real. A failure is then triaged by what is missing:

* a missing ``verl.*`` module or name is THIS repository being incomplete, and
  fails the test. That is the bug this file exists for.
* a missing third-party package is the environment being thin, and skips that
  one module with the package named. A CI image that installs verl's
  requirements skips nothing.

What it does not do is run anything. It proves the program can be loaded, not
that it works -- a real smoke run needs a GPU, an engine and the Stage-1 pool.
It is the cheap half, and it is the half that would have caught this.
"""

import importlib
import importlib.machinery
import importlib.metadata
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Engines only. Anything pip can install on a CPU box is left real, so that a
# module which stopped importing for its own reasons still says so.
_STUBBED = {"vllm", "sglang", "sgl_kernel"}
_STUB_VERSIONS = {"vllm": "0.8.5", "sglang": "0.4.6"}

# The import graph of a Stage-2 start, in the order it is walked:
# main -> trainer -> init_workers -> the worker -> _build_rollout -> the engine.
_RUN_PATH = [
    "verl.trainer.main_opd_offpolicy",
    "verl.trainer.ppo.opd_offpolicy_ray_trainer",
    "verl.workers.fsdp_workers",
    "verl.workers.actor.dp_actor",
    "verl.workers.teacher_cache",
    # The exact import _build_rollout performs, and the one that broke.
    "verl.workers.rollout.vllm_rollout",
    "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
    "verl.workers.sharding_manager.fsdp_vllm",
    "verl.utils.vllm_utils",
    "verl.utils.phase_timing",
]


class _StubLoader:
    """Answer any import under a stubbed root with a MagicMock package."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _STUBBED:
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        module = MagicMock(name=spec.name)
        module.__name__ = spec.name
        module.__path__ = []       # a package, so submodules resolve
        module.__spec__ = spec
        return module

    def exec_module(self, module):
        pass


@pytest.fixture(scope="module")
def stubbed_engines():
    """Stub the engines for the duration, and put the interpreter back after.

    ``importlib.metadata.version`` is patched too: verl.third_party.vllm reads it
    to pick a compatibility shim and raises PackageNotFoundError otherwise, which
    would look like a verl failure and is not one.
    """
    loader = _StubLoader()
    sys.meta_path.insert(0, loader)
    real_version = importlib.metadata.version
    importlib.metadata.version = lambda name: _STUB_VERSIONS.get(name) or real_version(name)
    before = set(sys.modules)
    try:
        yield
    finally:
        importlib.metadata.version = real_version
        sys.meta_path.remove(loader)
        # Modules imported against the stub must not be handed to later tests.
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)


def _missing_module_of(error):
    """The module name a ModuleNotFoundError is about, or None."""
    return getattr(error, "name", None)


@pytest.mark.parametrize("module", _RUN_PATH)
def test_a_module_on_the_run_path_imports(stubbed_engines, module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        missing = _missing_module_of(error) or ""
        if missing == "verl" or missing.startswith("verl."):
            pytest.fail(
                f"{module} imports {missing}, which is not in this repository. "
                "A run dies at init_workers() with exactly this error; it is a "
                "port that took a file and left its dependency behind."
            )
        pytest.skip(f"{module} needs {missing!r}, which is not installed here")
    except ImportError as error:
        # A name that is gone from a module that does exist -- same class of
        # mistake, different spelling, and equally fatal at run time.
        if "verl" in str(error):
            pytest.fail(f"{module} could not be imported: {error}")
        pytest.skip(f"{module}: {error}")


def test_the_rollout_module_really_was_loaded(stubbed_engines):
    """The guard on the guard: if the stub ever stopped working, every case
    above would skip and this file would report success while checking nothing.
    """
    module = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_rollout_spmd")
    assert module._ROLLOUT_PHASES.phases == ("build_inputs", "engine", "assemble")


# --------------------------------------------------------------------------- #
# And the same check where it is cheapest: in the run script, before the start.
# --------------------------------------------------------------------------- #

import pathlib  # noqa: E402
import re  # noqa: E402

_OPD = pathlib.Path(__file__).resolve().parents[2] / "examples/opd_trainer"
_TRAINER_SCRIPTS = sorted(
    p for p in _OPD.glob("run_*.sh")
    if re.search(r"^\s*python3 -m verl\.trainer\.", p.read_text(), re.MULTILINE)
)


def test_there_are_run_scripts_to_check():
    assert len(_TRAINER_SCRIPTS) >= 4, f"parsed {[p.name for p in _TRAINER_SCRIPTS]}"


@pytest.mark.parametrize("script", _TRAINER_SCRIPTS, ids=lambda p: p.name)
def test_a_run_script_checks_the_worker_import_before_it_starts(script):
    """Seconds here against an hour of Ray start, a config dump and a 333 GiB
    pool load before init_workers() reaches the same import."""
    text = script.read_text()
    preflight = re.search(
        r"^python3 -c \"import (verl\.\S+(?:, verl\.\S+)*);", text, re.MULTILINE
    )
    assert preflight, (
        f"{script.name} launches a trainer with no import pre-flight; a missing "
        "module surfaces at init_workers() instead, an hour into the run"
    )
    imported = {name.strip() for name in preflight.group(1).split(",")}
    assert "verl.workers.rollout.vllm_rollout" in imported, (
        f"{script.name} pre-flights {imported} but not the rollout package -- "
        "which is the one _build_rollout imports and the one that broke"
    )
    # Before the trainer, not after it.
    assert text.index(preflight.group(0)) < text.index("python3 -m verl.trainer.")
