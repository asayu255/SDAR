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

"""Fail-fast validation of the effective run config against an expectations file.

Motivation: experiment-defining knobs (loss type, coefficients, seeds, batch
sizes, ...) silently falling back to a default is how comparison runs get
invalidated — e.g. an OPD run intended as ``topk_kl`` executing as
``low_var_kl`` because of a script default. This module pins the *intent* in a
version-controlled YAML file and asserts, at startup and **after** any
entry-point config injection, that the effective composed config matches it.
Any mismatch aborts the run within seconds instead of surfacing after a
multi-hour training run (or never).

Expectations file format — flat dotted keys mapping to expected values::

    # examples/opd_grpo_trainer/expected_multitask_config.yaml
    algorithm.opd.kl_loss_type: low_var_kl
    actor_rollout_ref.actor.optim.lr: 1.0e-6
    data.task_balance.tasks: [alfworld, search, webshop]

Because validation runs on the composed config, it also catches mistakes this
file's author cannot see locally: Hydra overrides shadowed by entry-point
injection, typo'd ``+key=`` overrides that never take effect, and stray CLI
overrides appended via ``$@``.

Waiving a key for a probe run
-----------------------------
A measurement run sometimes has to move a pinned knob on purpose -- e.g.
``trainer.save_freq=1`` to make the checkpoint path fire every step instead of
once every two hours. Editing the expectations file for that is the worst
available option: it is the intent of the *production* run, and an edit made to
unblock a probe is an edit somebody forgets to revert, which is precisely the
silent invalidation this module exists to prevent.

So name the key instead::

    EXPECTED_CONFIG_WAIVE=trainer.save_freq bash examples/.../run.sh ++trainer.save_freq=1

Each waived key is printed on its own line at startup with the value it was
supposed to have, so a waiver is loud in the log a run gets judged from, and it
lives in one shell invocation rather than in version control. A waiver naming a
key the expectations file does not pin is itself an error -- a typo'd waiver
that quietly protects nothing belongs to the same family of mistakes as a
typo'd ``+key=`` override.
"""

import os
from typing import Any, Dict, List, Tuple

from omegaconf import OmegaConf

_MISSING = "<<MISSING>>"


def _normalize(value: Any) -> Any:
    """Normalize OmegaConf containers / numeric types for comparison."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        # Compare ints and floats on one axis (60 == 60.0).
        return float(value)
    if isinstance(value, float):
        return value
    return value


def _values_equal(got: Any, want: Any) -> bool:
    return _normalize(got) == _normalize(want)


def load_expectations(expect_file: str) -> Dict[str, Any]:
    """Load a flat dotted-key expectations YAML into an ordinary dict."""
    loaded = OmegaConf.load(expect_file)
    container = OmegaConf.to_container(loaded, resolve=True)
    assert isinstance(container, dict) and container, (
        f"expectations file {expect_file} must be a non-empty mapping of "
        "dotted config keys to expected values"
    )
    # Keys are written quoted ("a.b.c": v) so OmegaConf keeps them flat; accept
    # accidental nesting too by flattening.
    flat: Dict[str, Any] = {}

    def _flatten(prefix: str, node: Any):
        if isinstance(node, dict):
            for key, val in node.items():
                _flatten(f"{prefix}.{key}" if prefix else str(key), val)
        else:
            flat[prefix] = node

    for key, val in container.items():
        if isinstance(val, dict):
            _flatten(str(key), val)
        else:
            flat[str(key)] = val
    # Expectation values may reference the home directory ("$HOME/checkpoints/..."),
    # so one file pins the same teacher on machines whose home is not in the same
    # place. What the lock is asserting is *which* checkpoint, not where $HOME is.
    #
    # RUN_TAG_SUFFIX is the same idea for the run's own identity: the run scripts
    # append it to the wandb project and experiment names so a re-run of an arm
    # gets its own place in the charts, and the lock has to expect the name the
    # script actually passes. Substituted here rather than left to expandvars,
    # which leaves an ABSENT variable in the string verbatim -- an untagged run,
    # or a direct `python -m verl.trainer.main_opd`, would then be checked against
    # a literal "$RUN_TAG_SUFFIX" and fail. Empty is what an unset RUN_TAG means,
    # so the default is said once here instead of in every caller.
    suffix = os.environ.get("RUN_TAG_SUFFIX", "")

    def _expand(value: str) -> str:
        for spelling in ("${RUN_TAG_SUFFIX}", "$RUN_TAG_SUFFIX"):
            value = value.replace(spelling, suffix)
        return os.path.expandvars(os.path.expanduser(value))

    return {k: _expand(v) if isinstance(v, str) else v for k, v in flat.items()}


def waived_keys() -> List[str]:
    """Dotted keys the operator has explicitly excused, from EXPECTED_CONFIG_WAIVE."""
    raw = os.environ.get("EXPECTED_CONFIG_WAIVE", "")
    return [key.strip() for key in raw.replace(",", " ").split() if key.strip()]


def check_expected_config(config, expect_file: str) -> List[Tuple[str, Any, Any]]:
    """Return a list of (dotted_key, got, expected) mismatches (empty = OK)."""
    expectations = load_expectations(expect_file)
    mismatches = []
    for dotted_key, want in expectations.items():
        got = OmegaConf.select(config, dotted_key, default=_MISSING)
        if got is _MISSING or not _values_equal(got, want):
            mismatches.append((dotted_key, got, want))
    return mismatches


def enforce_expected_config(config, expect_file: str, tag: str = "expected-config") -> int:
    """Assert the composed config matches the expectations file.

    Call this AFTER the entry point's config injection so the *effective*
    values are checked. Raises AssertionError listing every mismatch; returns
    the number of validated keys on success.
    """
    mismatches = check_expected_config(config, expect_file)

    waived = waived_keys()
    if waived:
        pinned = load_expectations(expect_file)
        unknown = [key for key in waived if key not in pinned]
        assert not unknown, (
            f"[{tag}] EXPECTED_CONFIG_WAIVE names {unknown}, which "
            f"{expect_file} does not pin. A waiver that protects nothing is a "
            "typo, and silently ignoring it would defeat the point of naming keys."
        )
        for key in waived:
            got = OmegaConf.select(config, key, default=_MISSING)
            got_repr = "<missing>" if got is _MISSING else repr(got)
            print(
                f"[{tag}] WAIVED {key}: running with {got_repr}, "
                f"expectations file says {pinned[key]!r}",
                flush=True,
            )
        mismatches = [m for m in mismatches if m[0] not in set(waived)]

    if mismatches:
        lines = [
            f"[{tag}] effective config does not match {expect_file} "
            f"({len(mismatches)} mismatch(es)):"
        ]
        for dotted_key, got, want in mismatches:
            got_repr = "<missing>" if got is _MISSING else repr(got)
            lines.append(f"  - {dotted_key}: got {got_repr}, expected {want!r}")
        lines.append(
            "Fix the run script (or, if the change is intentional, update the "
            "expectations file in the same commit)."
        )
        raise AssertionError("\n".join(lines))
    n_keys = len(load_expectations(expect_file))
    checked = n_keys - len(waived)
    suffix = f", {len(waived)} waived" if waived else ""
    print(f"[{tag}] OK — {checked} expected keys match the effective config ({expect_file}){suffix}")
    return checked
