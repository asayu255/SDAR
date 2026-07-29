<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# DeepSeek R1 Reproduction

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This recipe is under development, if you are interested, checkout the TODO list and join this project! https://github.com/volcengine/verl/issues/708 

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Reproducing Evaluation

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Eval Results of DS-R1-Distill-Qwen2.5-1.5B (k=8)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Dataset | Test Results | Reported
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
-- | -- | --
GPQA Diamond | 35.3 | 33.8
LiveCodeBench | 16.9 | 16.9
AIME 2024 | 30.4 | 28.9
CNMO 2024 (en) | 45.1 | -
CNMO 2024 (zh) | 41.0 | -

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
---

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Eval Results (DS-R1)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Dataset | Test Results (k=1) | Test Results (k=4) | Reported
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
-- | -- | -- | --
GPQA Diamond | 67.7 | 69.6 | 71.5
LiveCodeBench | 64.7 | 63.1 | 65.9
AIME 2024 | 86.7 | 79.2 | 79.8
CNMO 2024 | 75.0 | 78.5 | 78.8
