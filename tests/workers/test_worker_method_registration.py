"""Every worker method a trainer calls has to be bound to the worker group.

``RayWorkerGroup._bind_worker_method`` copies over only the methods that carry
``MAGIC_ATTR``, which ``@register`` sets. A method without the decorator is
invisible to the group: the call fails with ``AttributeError`` on the group, not
on the worker, and the message names the group rather than the missing
decorator.

That failure mode is quiet in the worst way. ``save_checkpoint`` was decorated
and ``load_checkpoint`` was not, so training ran, checkpointed, and completed --
and every resume died after building the models and locating the checkpoint,
which is the last place anyone looks for a missing decorator. This pins the
whole class of bug rather than the one instance.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER_FILES = ("verl/workers/fsdp_workers.py",)
TRAINER_FILES = (
    "verl/trainer/ppo/ray_trainer.py",
    "verl/trainer/ppo/opd_ray_trainer.py",
    "verl/trainer/ppo/opd_grpo_ray_trainer.py",
)
# Attribute names that are dict/list methods on a plain container, not worker
# calls -- the regex below cannot tell them apart and neither can a reader.
_NOT_WORKER_METHODS = {"get", "items", "keys", "values", "update", "append", "pop"}

_GROUP = r"(?:actor_rollout_wg|critic_wg|ref_policy_wg|rm_wg|base_wg|wg)"


def _called_on_a_worker_group():
    names = set()
    for rel in TRAINER_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        names |= set(re.findall(rf"{_GROUP}\.([a-z_][a-z0-9_]*)\(", path.read_text()))
    return names - _NOT_WORKER_METHODS


def _worker_methods():
    """``{(class, method): registered}`` over every worker class."""
    out = {}
    for rel in WORKER_FILES:
        tree = ast.parse((ROOT / rel).read_text())
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in cls.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                out[(cls.name, fn.name)] = any(
                    "register" in ast.unparse(d) for d in fn.decorator_list
                )
    return out


def test_every_method_a_trainer_calls_on_a_worker_group_is_registered():
    called = _called_on_a_worker_group()
    assert "load_checkpoint" in called, "the regex stopped seeing the trainers' calls"
    unregistered = [
        f"{cls}.{name}"
        for (cls, name), registered in _worker_methods().items()
        if name in called and not registered
    ]
    assert not unregistered, (
        "these are called on a worker group but carry no @register, so "
        "_bind_worker_method never binds them and the call raises AttributeError "
        f"on the group: {unregistered}"
    )


def test_save_and_load_checkpoint_are_registered_together_on_every_worker():
    """The pair that made this quiet. A worker that can save and cannot load
    produces runs that look healthy until the first restart."""
    methods = _worker_methods()
    classes = {cls for (cls, name) in methods if name in ("save_checkpoint", "load_checkpoint")}
    assert classes, "no worker defines a checkpoint pair"
    for cls in sorted(classes):
        for name in ("save_checkpoint", "load_checkpoint"):
            if (cls, name) in methods:
                assert methods[(cls, name)], f"{cls}.{name} is not @register-ed"


def test_the_binding_really_does_depend_on_the_decorator():
    """Not an assumption about the framework -- the condition itself."""
    src = (ROOT / "verl/single_controller/base/worker_group.py").read_text()
    body = src[src.index("def _bind_worker_method"):]
    body = body[: body.index("\n    def ", 1)] if "\n    def " in body[1:] else body
    assert "hasattr(method, MAGIC_ATTR)" in body
