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

# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Dict, List, Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_MISSING = "<<MISSING>>"


# [EXPLAIN] `_normalize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _normalize(value: Any) -> Any:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Normalize OmegaConf containers / numeric types for comparison."""
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if OmegaConf.is_config(value):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value = OmegaConf.to_container(value, resolve=True)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(value, tuple):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value = list(value)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(value, list):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [_normalize(v) for v in value]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(value, dict):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {k: _normalize(v) for k, v in value.items()}
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(value, bool):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return value
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(value, int):
        # Compare ints and floats on one axis (60 == 60.0).
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return float(value)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(value, float):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return value
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return value


# [EXPLAIN] `_values_equal` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _values_equal(got: Any, want: Any) -> bool:
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _normalize(got) == _normalize(want)


# [EXPLAIN] `load_expectations` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_expectations(expect_file: str) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Load a flat dotted-key expectations YAML into an ordinary dict."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loaded = OmegaConf.load(expect_file)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    container = OmegaConf.to_container(loaded, resolve=True)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(container, dict) and container, (
        f"expectations file {expect_file} must be a non-empty mapping of "
        "dotted config keys to expected values"
    )
    # Keys are written quoted ("a.b.c": v) so OmegaConf keeps them flat; accept
    # accidental nesting too by flattening.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    flat: Dict[str, Any] = {}

    # [EXPLAIN] `_flatten` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _flatten(prefix: str, node: Any):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(node, dict):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, val in node.items():
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _flatten(f"{prefix}.{key}" if prefix else str(key), val)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            flat[prefix] = node

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in container.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(val, dict):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            _flatten(str(key), val)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            flat[str(key)] = val
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return flat


# [EXPLAIN] `check_expected_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
# [EXPLAIN] Hydra/OmegaConf の実効値を dot path で取得し、YAML 側の期待値と比較する。
# [EXPLAIN] これは default や CLI override 適用後に実行されるため、意図しない設定 drift を
# [EXPLAIN] 学習開始前に検出する実験再現性の fail-fast gate である。
def check_expected_config(config, expect_file: str) -> List[Tuple[str, Any, Any]]:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Return a list of (dotted_key, got, expected) mismatches (empty = OK)."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expectations = load_expectations(expect_file)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mismatches = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for dotted_key, want in expectations.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        got = OmegaConf.select(config, dotted_key, default=_MISSING)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if got is _MISSING or not _values_equal(got, want):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mismatches.append((dotted_key, got, want))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return mismatches


# [EXPLAIN] `enforce_expected_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def enforce_expected_config(config, expect_file: str, tag: str = "expected-config") -> int:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Assert the composed config matches the expectations file.

    Call this AFTER the entry point's config injection so the *effective*
    values are checked. Raises AssertionError listing every mismatch; returns
    the number of validated keys on success.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mismatches = check_expected_config(config, expect_file)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if mismatches:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lines = [
            f"[{tag}] effective config does not match {expect_file} "
            f"({len(mismatches)} mismatch(es)):"
        ]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for dotted_key, got, want in mismatches:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            got_repr = "<missing>" if got is _MISSING else repr(got)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            lines.append(f"  - {dotted_key}: got {got_repr}, expected {want!r}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        lines.append(
            "Fix the run script (or, if the change is intentional, update the "
            "expectations file in the same commit)."
        )
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise AssertionError("\n".join(lines))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n_keys = len(load_expectations(expect_file))
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"[{tag}] OK — {n_keys} expected keys match the effective config ({expect_file})")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return n_keys
