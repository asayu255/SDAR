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
    return flat


def waived_keys() -> List[str]:
    """Dotted keys the operator has explicitly excused, from EXPECTED_CONFIG_WAIVE."""
    raw = os.environ.get("EXPECTED_CONFIG_WAIVE", "")
    return [key.strip() for key in raw.replace(",", " ").split() if key.strip()]


def _waiver_covers(waiver: str, pinned_key: str) -> bool:
    """Does ``waiver`` excuse ``pinned_key``?

    Exactly, or as its parent. A mapping in the expectations file is stored
    flattened -- ``val_per_task_batch_size: {alfworld: 126, search: 252}``
    becomes two pinned keys -- but an operator waives the setting they know
    about, which is the name they would type on a command line. Without this,
    turning a pinned scalar into a pinned mapping silently invalidates every
    existing waiver for it, and the waiver check would then abort the run for
    naming a key "the file does not pin".

    The boundary is the dot: ``a.b`` covers ``a.b.c`` and not ``a.bc``.
    """
    return pinned_key == waiver or pinned_key.startswith(waiver + ".")


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
        unknown = [
            key for key in waived if not any(_waiver_covers(key, name) for name in pinned)
        ]
        assert not unknown, (
            f"[{tag}] EXPECTED_CONFIG_WAIVE names {unknown}, which "
            f"{expect_file} does not pin. A waiver that protects nothing is a "
            "typo, and silently ignoring it would defeat the point of naming keys."
        )
        for key in waived:
            # A waiver may name a parent of several pinned keys, so report every
            # one it covers rather than indexing `pinned` by the waiver itself.
            covered = [name for name in pinned if _waiver_covers(key, name)]
            for name in covered:
                got = OmegaConf.select(config, name, default=_MISSING)
                got_repr = "<missing>" if got is _MISSING else repr(got)
                print(
                    f"[{tag}] WAIVED {name}: running with {got_repr}, "
                    f"expectations file says {pinned[name]!r}",
                    flush=True,
                )
        mismatches = [
            m for m in mismatches if not any(_waiver_covers(key, m[0]) for key in waived)
        ]

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
    pinned_names = list(load_expectations(expect_file))
    # Count the pinned keys a waiver covers, not the waivers: one naming the
    # parent of a mapping excuses every entry under it, so subtracting the
    # number of waivers would report more keys checked than were checked.
    n_waived = sum(
        1 for name in pinned_names if any(_waiver_covers(key, name) for key in waived)
    )
    checked = len(pinned_names) - n_waived
    suffix = f", {n_waived} waived" if waived else ""
    print(f"[{tag}] OK — {checked} expected keys match the effective config ({expect_file}){suffix}")
    return checked
