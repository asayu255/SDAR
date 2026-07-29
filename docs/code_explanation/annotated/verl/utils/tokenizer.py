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
"""Utils for tokenization."""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import warnings

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["hf_tokenizer", "hf_processor"]


# [EXPLAIN] `set_pad_token_id` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def set_pad_token_id(tokenizer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if tokenizer.pad_token_id is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if tokenizer.pad_token is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer.pad_token = tokenizer.eos_token
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


# [EXPLAIN] `hf_tokenizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers import AutoTokenizer

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        warnings.warn("Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        kwargs["eos_token"] = "<end_of_turn>"
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        kwargs["eos_token_id"] = 107
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if correct_pad_token:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        set_pad_token_id(tokenizer)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tokenizer


# [EXPLAIN] `hf_processor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_processor(name_or_path, **kwargs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers import AutoProcessor

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        processor = None
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if processor is not None and "Processor" not in processor.__class__.__name__:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        processor = None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return processor
