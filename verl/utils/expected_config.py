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
"""

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
    print(f"[{tag}] OK — {n_keys} expected keys match the effective config ({expect_file})")
    return n_keys
