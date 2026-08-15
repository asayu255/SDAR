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

"""Every name a trainer entry point reads must be bound somewhere it can see.

These entry points are the one place a NameError costs the most: their bodies run
inside a Ray actor, after Hydra composition, worker-group construction and dataset
loading, so a typo or a leftover reference surfaces minutes into a run -- and only
on a machine with GPUs, which is why no other test in this tree reaches them.

The bug this exists for: extracting ``inject_opd_grpo_config`` out of
``OPDGRPOTaskRunner.run`` moved ``teacher_paths`` into the new function but left
the diagnostic print in ``run`` still reading it, so every OPD+GRPO run died with
``NameError: name 'teacher_paths' is not defined`` after passing the expectations
check and building both datasets.

Static, so it needs neither torch nor ray: parse the file and, for each function,
check that every loaded name is bound in that function, at module level, or by
builtins. That is deliberately permissive -- it will not catch a name bound only on
one branch, or one shadowed later -- but the class of failure it does catch is
exactly the one refactors keep reintroducing.
"""

import ast
import builtins
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ENTRY_POINTS = [
    "verl/trainer/main_opd_grpo.py",
    "verl/trainer/main_opd.py",
    "verl/trainer/main_ppo.py",
    # The trainers themselves, for the same reason and by the same class of bug:
    # cherry-picking a val_only change onto a branch that already had it applied
    # the diff in reverse, taking the `val_only = ...` binding out while leaving
    # the `if val_only:` that reads it. fit() runs after _load_checkpoint(), so
    # every run on that branch loaded its models and then died on
    # ``NameError: name 'val_only' is not defined``. The whole tests/trainer tree
    # passed while it was broken -- nothing else here parses a fit() body.
    "verl/trainer/ppo/opd_grpo_ray_trainer.py",
    "verl/trainer/ppo/opd_ray_trainer.py",
    "verl/trainer/ppo/ray_trainer.py",
]


def _shallow(node):
    """Walk node's own scope, not the bodies of functions/classes nested in it."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(current))


def _all_args(fn):
    args = fn.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _bound_names(node, nested):
    """Names bound in this scope. ``nested`` includes inner comprehensions/handlers."""
    bound = set()
    for child in (ast.walk(node) if nested else _shallow(node)):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            bound.add(child.id)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in child.names)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
            # A nested def's PARAMETERS are bound inside it, and the outer walk
            # sees its body, so they have to be collected here or every helper
            # closure reports its own arguments as undefined. (This used to sit
            # in a later elif that the branch above made unreachable -- harmless
            # while only the three small main_*.py files were checked, and a
            # wall of false positives the moment a real trainer was added.)
            if nested and not isinstance(child, ast.ClassDef):
                bound |= _all_args(child)
        elif isinstance(child, ast.Lambda) and nested:
            bound |= _all_args(child)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            bound.update(child.names)
    return bound


@pytest.mark.parametrize("relpath", ENTRY_POINTS)
def test_no_undefined_names(relpath):
    path = os.path.join(REPO_ROOT, relpath)
    if not os.path.exists(path):
        pytest.skip(f"{relpath} not present on this branch")

    tree = ast.parse(open(path).read(), path)
    visible = _bound_names(tree, nested=False) | set(dir(builtins))

    undefined = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        known = visible | _bound_names(fn, nested=True) | _all_args(fn)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in known:
                undefined.append(f"{relpath}:{node.lineno}: {node.id!r} in {fn.name}()")

    assert not undefined, "undefined names:\n  " + "\n  ".join(sorted(set(undefined)))
