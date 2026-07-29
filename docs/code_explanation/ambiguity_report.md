# Ambiguity and Source/Test Mismatch Report

## `compute_teacher_log_probs()` の戻り値

- source: `OPDRayTrainer.compute_teacher_log_probs()` は `-> None` で、teacher
  signal を入力 `DataProto.batch` へ書き込む。
- production caller: `fit()` は戻り値を受け取らず、mutation 後の `batch` を
  `update_actor()` へ渡す。
- test: `tests/trainer/test_opd_routing.py` の routing test は戻り値を `out` に
  代入し、Tensor のように index access する。

実装と test expectation が一致していない。baseline 環境では Torch/VERL
stack 不足により test module が skip されたため、この assertion までは到達
していない。計画の禁止事項に従い source/test は修正せず、要確認として残す。

## Async rollout core

`async_rollout_core.py` と対応 test は存在し、軽量 equivalence test 6件は
baseline で成功した。一方、production rollout への完全接続は別途 call graph
で確認する必要があるため、現段階では experimental/scaffolding と分類する。
