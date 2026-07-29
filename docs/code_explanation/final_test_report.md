# Final Test Report

## Environment

- source commit: `10cacaf6fb2cd4eea971b06d3e73ada868b611f4`
- platform: Windows / Codex sandbox
- temporary validation dependencies: `pytest`, `omegaconf`, `PyYAML`, `numpy`
- unavailable heavy dependencies: `ray`, `torch`, full VERL/GPU stack

## Baseline comparison

| Command | Baseline | Final | Comparison |
|---|---|---|---|
| `python tests/trainer/test_expected_config.py` | blocked: `ray` missing | blocked: `ray` missing | unchanged |
| `pytest -q tests/trainer/test_opd_routing.py` | dependency skip | 1 skipped | unchanged |
| `pytest -q tests/trainer/ppo/test_per_task_metrics.py` | blocked: `torch` missing | blocked: `torch` missing | unchanged |
| `pytest -q tests/ray_cpu/test_async_rollout_equivalence.py` | 6 passed | 6 passed in 0.18s | unchanged / passed |
| `pytest -q tests/ray_cpu/test_rollout_speedup_mechanisms.py` | blocked: `torch` missing | blocked: `torch` missing | unchanged |

元sourceと既存testは変更していない。実行可能な軽量testはbaseline/finalとも6件成功し、
それ以外も同じ依存関係境界で停止したため、annotationによるtest状態の悪化はない。

## Annotation validation

| Gate | Result |
|---|---|
| manifest target | 743 completed / 0 pending / 0 blocked / 0 needs-review |
| source preservation | 743/743 passed |
| visible legacy tags / generic templates | 0 / 0 |
| semantic block coverage | 10,253 blocks / 0 needs-review |
| Priority semantic review | A 8/8、B 8/8、C 11/11、D 4/4 |
| Python syntax | 425/425 passed |
| Shell syntax | 149/149 passed |
| YAML syntax | 46/46 passed |
| TOML syntax | 1/1 passed |

Python構文検査では既存source由来のinvalid escape `SyntaxWarning` が表示されるファイルがあるが、
compile failureは0である。原文保持要件によりescape文字列自体は変更していない。

## Known source/test ambiguity

`OPDRayTrainer.compute_teacher_log_probs()`は入力batchをmutationして`None`を返す一方、
`tests/trainer/test_opd_routing.py`の一部は戻り値をTensorとして期待する。依存関係不足により
そのassertionはbaseline/finalとも実行されていない。source/testは修正せず
`ambiguity_report.md`へ記録した。
