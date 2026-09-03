"""Every module-scope ``verl.*`` import has to resolve to a file in the tree.

This exists because of one: ``vllm_rollout_spmd.py`` was ported from another
branch carrying ``from verl.utils.phase_timing import PhaseTimer``, and the
module it names was not ported with it. Nothing here noticed, because the file
cannot be imported without vllm installed and CI has no vllm. The run did: the
import is at module scope, ``_build_rollout`` imports the package, and every
worker died at ``init_workers()`` with ModuleNotFoundError -- after the config
dump, after the 90-file pool load, minutes into a start.

The rule is deliberately narrow. Only imports that run when a file is imported
count: module scope, class bodies, and the ``if`` blocks around them. A
function-local import is how this codebase makes an optional dependency
optional (verl.utils.kernel.linear_cross_entropy behind use_fused_kernels,
verl.models.transformers.glm4v behind a model_type, the SFT trainer behind
``--arm sft`` in cache_teacher_pool.py) and does not fail unless the path is
taken. A ``try: import ... except ImportError`` at module scope is the same
statement of intent, spelled at the top of the file.

What this cannot do is check third-party imports -- verl legitimately imports
vllm, flash_attn and accelerate, which are not present in every environment. It
checks the tree against itself, which is exactly the mistake a port makes.
"""

import ast
import os
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "third_party"}


def _executes_on_import(node):
    """Statements reached by importing the file, ``try`` blocks excluded.

    Recurses into If/With/For/While and class bodies (all of which run) but not
    into functions (which do not) and not into Try (which is the guard).
    """
    for child in node.body:
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            yield child
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Try)):
            continue
        elif hasattr(child, "body"):
            yield from _executes_on_import(child)
            for attr in ("orelse", "finalbody"):
                branch = getattr(child, attr, None)
                if branch:
                    yield from _executes_on_import(ast.Module(body=branch, type_ignores=[]))


def _imported_verl_modules(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return
    for node in _executes_on_import(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif node.level:  # relative import; resolved against the package, not here
            continue
        else:
            names = [node.module] if node.module else []
        for name in names:
            if name and (name == "verl" or name.startswith("verl.")):
                yield name


def _python_files():
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".py"):
                yield pathlib.Path(dirpath) / filename


def _resolves(module):
    path = _ROOT / module.replace(".", "/")
    return path.is_dir() or path.with_suffix(".py").exists()


def test_the_scan_finds_something_to_scan():
    """Guard against the walk or the AST filter silently matching nothing."""
    seen = sum(1 for path in _python_files() for _ in _imported_verl_modules(path))
    assert seen > 100, f"only {seen} module-scope verl imports found; the scan is broken"


def test_no_module_scope_verl_import_names_a_module_that_is_not_here():
    missing = {}
    for path in _python_files():
        for module in _imported_verl_modules(path):
            if not _resolves(module):
                missing.setdefault(module, []).append(str(path.relative_to(_ROOT)))
    assert not missing, (
        "these modules are imported at module scope but do not exist in the tree, "
        "so importing the file raises ModuleNotFoundError -- which for a worker "
        f"module means the run dies at init_workers(): {missing}"
    )


@pytest.mark.parametrize("module", ["verl.utils.phase_timing"])
def test_the_module_this_test_was_written_for(module):
    """Named, so a re-port that drops it again fails with the reason attached."""
    assert _resolves(module), (
        f"{module} is gone again; vllm_rollout_spmd.py imports it at module scope "
        "and every rollout worker will die at init_model()"
    )
