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
"""Copied from https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py

CharacterTokenzier for Hugging Face Transformers.

This is heavily inspired from CanineTokenizer in transformers package.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pathlib import Path
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict, List, Optional, Sequence, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


# [EXPLAIN] `CharTokenizer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class CharTokenizer(PreTrainedTokenizer):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, characters: Sequence[str], model_max_length: int, chat_template, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Character tokenizer for Hugging Face transformers.

        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        eos_token_str = "E"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sep_token_str = "S"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_token_str = "P"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        unk_token_str = "U"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.characters = characters
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_max_length = model_max_length
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        eos_token = AddedToken(eos_token_str, lstrip=False, rstrip=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sep_token = AddedToken(sep_token_str, lstrip=False, rstrip=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_token = AddedToken(pad_token_str, lstrip=False, rstrip=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        unk_token = AddedToken(unk_token_str, lstrip=False, rstrip=False)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._vocab_str_to_int = {
            sep_token_str: 0,
            eos_token_str: 1,
            pad_token_str: 2,
            unk_token_str: 3,
            **{ch: i + 4 for i, ch in enumerate(characters)},
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(
            eos_token=eos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chat_template = chat_template

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `vocab_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def vocab_size(self) -> int:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return len(self._vocab_str_to_int)

    # [EXPLAIN] `get_vocab` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_vocab(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._vocab_str_to_int

    # [EXPLAIN] `_tokenize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _tokenize(self, text: str) -> List[str]:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return list(text)

    # [EXPLAIN] `_convert_token_to_id` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _convert_token_to_id(self, token: str) -> int:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["U"])

    # [EXPLAIN] `_convert_id_to_token` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _convert_id_to_token(self, index: int) -> str:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._vocab_int_to_str[index]

    # [EXPLAIN] `convert_tokens_to_string` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def convert_tokens_to_string(self, tokens):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "".join(tokens)

    # [EXPLAIN] `build_inputs_with_special_tokens` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sep = [self.sep_token_id]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cls = [self.cls_token_id]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = cls + token_ids_0 + sep
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if token_ids_1 is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            result += token_ids_1 + sep
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return result

    # [EXPLAIN] `get_special_tokens_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if already_has_special_tokens:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = [1] + ([0] * len(token_ids_0)) + [1]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if token_ids_1 is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            result += ([0] * len(token_ids_1)) + [1]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return result

    # [EXPLAIN] `get_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_config(self) -> Dict:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
            "chat_template": self.chat_template,
        }

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `from_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def from_config(cls, config: Dict):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cfg = {}
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        cfg["model_max_length"] = config["model_max_length"]
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        cfg["chat_template"] = config["chat_template"]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cls(**cfg)

    # [EXPLAIN] `save_pretrained` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cfg = self.get_config()
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(cfg_file, "w") as f:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            json.dump(cfg, f, indent=4)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `from_pretrained` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(cfg_file) as f:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cfg = json.load(f)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cls.from_config(cfg)
