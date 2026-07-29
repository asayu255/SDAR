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
"""
The base tokenizer class, required for any hybrid engine based rollout or inference with vLLM.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from abc import ABC, abstractmethod
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict, List, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["HybridEngineBaseTokenizer"]


# [EXPLAIN] `HybridEngineBaseTokenizer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class HybridEngineBaseTokenizer(ABC):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """the tokenizer property and function name should align with HF's to meet vllm requirement"""

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `vocab_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def vocab_size(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `pad_token_id` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def pad_token_id(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `eos_token_id` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def eos_token_id(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `all_special_ids` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def all_special_ids(self) -> List[int]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `all_special_tokens` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def all_special_tokens(self) -> List[str]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `encode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def encode(self, text):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers.

            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers.
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `decode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `convert_ids_to_tokens` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `get_added_vocab` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_added_vocab(self) -> Dict[str, int]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `convert_tokens_to_string` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `is_fast` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def is_fast(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False
