<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Digit completion

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This is an example of solving a digit completion problem. The problem is defined as below:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The prompt is a sequence of numbers with fixed difference. The agent's goal is to complete the next N numbers.
If the max number is reached, the next number should be modulo with max number.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For example,
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- prompt = [1, 2, 3]
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- N = 5
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- max_number = 6

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The response should be [4, 5, 6, 7%6, 8%6] = [4, 5, 6, 0, 1].

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Environment definition

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The core definition of the task is defined in tests/e2e/envs/digit_completion/task.py

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
It is highly recommended to take a look at it for better understanding.



<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Run experiments

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
An example of running the task is provided in `tests/e2e/run_ray_trainer.sh`.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
bash tests/e2e/run_ray_trainer.sh
```

