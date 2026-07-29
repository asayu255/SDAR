# Copyright 2024 PRIME team and/or its affiliates
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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import asyncio
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from concurrent.futures import ProcessPoolExecutor
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import partial
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Callable, Optional
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import psutil
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PreTrainedTokenizer

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.reward_score import default_compute_score


# [EXPLAIN] `single_compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loop = asyncio.get_running_loop()
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # Ensure process_completion is called properly
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        future = loop.run_in_executor(
            executor,
            partial(evaluation_func, task, completion, reference, task_extra_info)
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return await asyncio.wait_for(future, timeout=timeout)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except asyncio.TimeoutError:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[Timeout] Task timeout: {completion}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None  # Default value for timed-out rows
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None  # Default value for failed rows


# [EXPLAIN] `parallel_compute_score_async` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
async def parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if extra_info is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_info = [None] * len(tasks)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = []
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # Create tasks for all rows
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tasks_async = [
                single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=300.0)
                for c, r, t, ei in zip(completions, references, tasks, extra_info)
            ]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Exception] async gather failed: {e}")
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            terminated_count = 0
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for pid, proc in executor._processes.items():
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    p = psutil.Process(pid)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    p.terminate()
                    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                    try:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        p.wait(timeout=5)
                    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                    except psutil.TimeoutExpired:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        p.kill()
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    terminated_count += 1
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    pass
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Process results
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for result, completion, reference, task in zip(results, completions, references, tasks):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            scores.append(0.0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif isinstance(result, (int, float, bool)):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            scores.append(float(result))
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            scores.append(float(result[0]))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return scores

# [EXPLAIN] `run_reward_scoring` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loop = asyncio.new_event_loop()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    asyncio.set_event_loop(loop)
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return loop.run_until_complete(parallel_compute_score_async(
            evaluation_func, completions, references, tasks, extra_info, num_processes
        ))
    # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
    finally:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        loop.close()


# [EXPLAIN] `PrimeRewardManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class PrimeRewardManager:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.compute_score = compute_score or default_compute_score
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward_fn_key = reward_fn_key

    # [EXPLAIN] `verify` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def verify(self, data):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_ids = data.batch["prompts"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_ids = data.batch["responses"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_info = data.non_tensor_batch.get("extra_info", None)

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=64,
            )
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except asyncio.TimeoutError:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scores = [0.0 for _ in range(len(sequences_str))]
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scores = [0.0 for _ in range(len(sequences_str))]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return scores

    # [EXPLAIN] `__call__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __call__(self, data: DataProto, return_dict: bool = False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "rm_scores" in data.batch.keys():
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return data.batch["rm_scores"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        already_print_data_sources = {}

        # batched scoring
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_ids = data.batch["prompts"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_length = prompt_ids.shape[-1]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_ids = data.batch["responses"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_sources = data.non_tensor_batch["data_source"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        scores = self.verify(data)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(data)):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data_source = data_sources[i]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if data_source not in already_print_data_sources:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                already_print_data_sources[data_source] = 0

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if already_print_data_sources[data_source] < self.num_examine:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                already_print_data_sources[data_source] += 1
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(sequences_str)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if return_dict:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {"reward_tensor": reward_tensor}
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return reward_tensor
