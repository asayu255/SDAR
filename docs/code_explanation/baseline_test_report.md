# Baseline Test Report

## Environment

- source commit: `10cacaf6fb2cd4eea971b06d3e73ada868b611f4`
- platform: Windows / Codex sandbox
- available Python: bundled artifact runtime
- temporary validation dependencies: `pytest`, `omegaconf`, `PyYAML`, `numpy`
- system Python: unavailable
- GPU validation: not attempted

## Results

| Command | Result | Classification |
|---|---|---|
| `python tests/trainer/test_expected_config.py` | `ModuleNotFoundError: ray` | blocked: dependency missing |
| `pytest -q tests/trainer/test_opd_routing.py` | module-level skip; Torch/VERL stack unavailable | skipped: dependency missing |
| `pytest -q tests/trainer/ppo/test_per_task_metrics.py` | collection error: `No module named torch` | blocked: dependency missing |
| `pytest -q tests/ray_cpu/test_async_rollout_equivalence.py` | `6 passed` | passed |
| `pytest -q tests/ray_cpu/test_rollout_speedup_mechanisms.py` | collection error: `No module named torch` | blocked: dependency missing |

## Interpretation

軽量な async rollout equivalence test は 6 件すべて成功しました。その他はテスト本体または
collection の段階で `ray` / `torch` 不足により停止しました。これは source branch の
テスト失敗とは分類しません。元コード・既存テストは変更せず、最終検証でも同じコマンドを
再実行して baseline より状態が悪化していないことを確認します。
