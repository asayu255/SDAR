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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import union_numpy_dict, union_tensor_dict


# [EXPLAIN] `test_union_tensor_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_union_tensor_dict():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.randn(100, 10)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data1 = TensorDict({"obs": obs, "act": torch.randn(100, 3)}, batch_size=[100])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data2 = TensorDict({"obs": obs, "next_obs": torch.randn(100, 10), "rew": torch.randn(100)}, batch_size=[100])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_with_copied_obs = TensorDict({"obs": obs.clone(), "next_obs": torch.randn(100, 10), "rew": torch.randn(100)}, batch_size=[100])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = union_tensor_dict(data1, data2)
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(AssertionError):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = union_tensor_dict(data1, data_with_copied_obs)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = np.random.random(100)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data2 = [float("nan") for _ in range(99)]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    data2.append("nan")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data2 = np.array(data2, dtype=object)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data3 = np.tile(data2, (2, 1))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    a = {"a": data, "b": data2, "c": data3}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    b = {"a": data, "b": data2, "c": data3}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    b_ = {"a": np.random.random(100)}
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    union_numpy_dict(a, b)
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(AssertionError):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        union_numpy_dict(a, b_)


# [EXPLAIN] `test_tensor_dict_constructor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_tensor_dict_constructor():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.randn(100, 10)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    act = torch.randn(100, 10, 3)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs, "act": act})

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert data.batch.batch_size == torch.Size([100])

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(AssertionError):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = DataProto.from_dict(tensors={"obs": obs, "act": act}, num_batch_dims=2)

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(AssertionError):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = DataProto.from_dict(tensors={"obs": obs, "act": act}, num_batch_dims=3)


# [EXPLAIN] `test_tensor_dict_make_iterator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_tensor_dict_make_iterator():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.randn(100, 10)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = [random.choice(["abc", "cde"]) for _ in range(100)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_iter_1 = dataset.make_iterator(mini_batch_size=10, epochs=2, seed=1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_list_1 = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data in data_iter_1:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        data_list_1.append(data)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_iter_2 = dataset.make_iterator(mini_batch_size=10, epochs=2, seed=1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_list_2 = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data in data_iter_2:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        data_list_2.append(data)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data1, data2 in zip(data_list_1, data_list_2):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(data1, DataProto)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(data2, DataProto)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = torch.all(torch.eq(data1.batch["obs"], data2.batch["obs"]))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not result.item():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(data1.batch["obs"])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(data2.batch["obs"])
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise AssertionError()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensor_result = np.all(np.equal(data1.non_tensor_batch["labels"], data2.non_tensor_batch["labels"]))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not non_tensor_result.item():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(data1.non_tensor_batch["labels"])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(data2.non_tensor_batch["labels"])


# [EXPLAIN] `test_reorder` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_reorder():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c", "d", "e", "f"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"name": "abdce"})
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    data.reorder(torch.tensor([3, 4, 2, 0, 1, 5]))

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(data.batch["obs"], torch.tensor([4, 5, 3, 1, 2, 6])))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.all(data.non_tensor_batch["labels"] == np.array(["d", "e", "c", "a", "b", "f"]))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert data.meta_info == {"name": "abdce"}


# [EXPLAIN] `test_chunk_concat` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_chunk_concat():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c", "d", "e", "f"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"name": "abdce"})

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(AssertionError):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        data.chunk(5)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_split = data.chunk(2)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(data_split) == 2
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(data_split[0].batch["obs"], torch.tensor([1, 2, 3])))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.all(data_split[0].non_tensor_batch["labels"] == np.array(["a", "b", "c"]))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert data_split[0].meta_info == {"name": "abdce"}

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(data_split[1].batch["obs"], torch.tensor([4, 5, 6])))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.all(data_split[1].non_tensor_batch["labels"] == np.array(["d", "e", "f"]))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert data_split[1].meta_info == {"name": "abdce"}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    concat_data = DataProto.concat(data_split)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(concat_data.batch["obs"], data.batch["obs"]))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.all(concat_data.non_tensor_batch["labels"] == data.non_tensor_batch["labels"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert concat_data.meta_info == data.meta_info


# [EXPLAIN] `test_pop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_pop():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.randn(100, 10)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    act = torch.randn(100, 3)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = DataProto.from_dict({"obs": obs, "act": act}, meta_info={"2": 2, "1": 1})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    poped_dataset = dataset.pop(batch_keys=["obs"], meta_info_keys=["2"])

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert poped_dataset.batch.keys() == {"obs"}
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert poped_dataset.meta_info.keys() == {"2"}

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert dataset.batch.keys() == {"act"}
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert dataset.meta_info.keys() == {"1"}


# [EXPLAIN] `test_repeat` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_repeat():
    # Create a DataProto object with some batch and non-tensor data
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    # Test interleave=True
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    repeated_data_interleave = data.repeat(repeat_times=2, interleave=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_obs_interleave = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_labels_interleave = ["a", "a", "b", "b", "c", "c"]

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(repeated_data_interleave.batch["obs"], expected_obs_interleave))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (repeated_data_interleave.non_tensor_batch["labels"] == expected_labels_interleave).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert repeated_data_interleave.meta_info == {"info": "test_info"}

    # Test interleave=False
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    repeated_data_no_interleave = data.repeat(repeat_times=2, interleave=False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_obs_no_interleave = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_labels_no_interleave = ["a", "b", "c", "a", "b", "c"]

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(repeated_data_no_interleave.batch["obs"], expected_obs_no_interleave))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (repeated_data_no_interleave.non_tensor_batch["labels"] == expected_labels_no_interleave).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert repeated_data_no_interleave.meta_info == {"info": "test_info"}


# [EXPLAIN] `test_dataproto_pad_unpad` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_dataproto_pad_unpad():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    padded_data, pad_size = pad_dataproto_to_divisor(data, size_divisor=2)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert pad_size == 1

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_labels = ["a", "b", "c", "a"]

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(padded_data.batch["obs"], expected_obs))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (padded_data.non_tensor_batch["labels"] == expected_labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert padded_data.meta_info == {"info": "test_info"}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    unpadd_data = unpad_dataproto(padded_data, pad_size=pad_size)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(unpadd_data.batch["obs"], obs))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (unpadd_data.non_tensor_batch["labels"] == labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert unpadd_data.meta_info == {"info": "test_info"}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    padded_data, pad_size = pad_dataproto_to_divisor(data, size_divisor=3)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert pad_size == 0

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_labels = ["a", "b", "c"]

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(padded_data.batch["obs"], expected_obs))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (padded_data.non_tensor_batch["labels"] == expected_labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert padded_data.meta_info == {"info": "test_info"}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    unpadd_data = unpad_dataproto(padded_data, pad_size=pad_size)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(unpadd_data.batch["obs"], obs))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (unpadd_data.non_tensor_batch["labels"] == labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert unpadd_data.meta_info == {"info": "test_info"}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    padded_data, pad_size = pad_dataproto_to_divisor(data, size_divisor=7)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert pad_size == 4

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_labels = ["a", "b", "c", "a", "b", "c", "a"]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(padded_data.batch["obs"], expected_obs))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (padded_data.non_tensor_batch["labels"] == expected_labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert padded_data.meta_info == {"info": "test_info"}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    unpadd_data = unpad_dataproto(padded_data, pad_size=pad_size)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(unpadd_data.batch["obs"], obs))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (unpadd_data.non_tensor_batch["labels"] == labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert unpadd_data.meta_info == {"info": "test_info"}


# [EXPLAIN] `test_dataproto_fold_unfold` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_dataproto_fold_unfold():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.protocol import DataProto, fold_batch_dim, unfold_batch_dim

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data1 = data.repeat(repeat_times=2, interleave=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data2 = fold_batch_dim(data1, new_batch_size=3)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(data2.batch["obs"], torch.tensor([[[1, 2], [1, 2]], [[3, 4], [3, 4]], [[5, 6], [5, 6]]]))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (data2.non_tensor_batch["labels"] == [["a", "a"], ["b", "b"], ["c", "c"]]).all()

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    data2.reorder(indices=torch.tensor([1, 2, 0]))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data3 = unfold_batch_dim(data2, batch_dims=2)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(data3.batch["obs"], torch.tensor([[3, 4], [3, 4], [5, 6], [5, 6], [1, 2], [1, 2]]))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (data3.non_tensor_batch["labels"] == ["b", "b", "c", "c", "a", "a"]).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert data3.meta_info == {"info": "test_info"}


# [EXPLAIN] `test_torch_save_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_torch_save_data_proto():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    data.save_to_disk("test_data.pt")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loaded_data = DataProto.load_from_disk("test_data.pt")

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(loaded_data.batch["obs"], data.batch["obs"]))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (loaded_data.non_tensor_batch["labels"] == data.non_tensor_batch["labels"]).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert loaded_data.meta_info == data.meta_info

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import os

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.remove("test_data.pt")


# [EXPLAIN] `test_len` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_len():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = np.array(["a", "b", "c"], dtype=object)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(data) == 3

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto(batch=None, non_tensor_batch={"labels": labels}, meta_info={"info": "test_info"})

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(data) == 3

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto(batch=None, non_tensor_batch={}, meta_info={"info": "test_info"})

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(data) == 0

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto(batch=None, non_tensor_batch=None, meta_info={"info": "test_info"})

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(data) == 0


# [EXPLAIN] `test_dataproto_index` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_dataproto_index():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_len = 100
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx_num = 10

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.randn(data_len, 10)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = [random.choice(["abc", "cde"]) for _ in range(data_len)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels_np = np.array(labels)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx_np_int = np.random.randint(0, data_len, size=(idx_num,))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_np_int = data[idx_np_int]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_np_int.batch.keys() == data.batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_np_int.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_np_int.batch["obs"].shape[0] == idx_num
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_np_int.non_tensor_batch["labels"].shape[0] == idx_num
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_np_int.batch["obs"].cpu().numpy(), obs[idx_np_int].numpy())
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_np_int.non_tensor_batch["labels"], labels_np[idx_np_int])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx_torch_int = torch.randint(0, data_len, size=(idx_num,))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_torch_int = data[idx_torch_int]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_torch_int.batch.keys() == data.batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_torch_int.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_torch_int.batch["obs"].shape[0] == idx_num
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_torch_int.non_tensor_batch["labels"].shape[0] == idx_num
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_torch_int.batch["obs"].cpu().numpy(), obs[idx_torch_int].cpu().numpy())
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_torch_int.non_tensor_batch["labels"], labels_np[idx_torch_int.cpu().numpy()])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx_list_int = [np.random.randint(0, data_len) for _ in range(idx_num)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_list_int = data[idx_list_int]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_list_int.batch.keys() == data.batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_list_int.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_list_int.batch["obs"].shape[0] == idx_num
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_list_int.non_tensor_batch["labels"].shape[0] == idx_num
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_list_int.batch["obs"].cpu().numpy(), obs[idx_list_int].cpu().numpy())
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_list_int.non_tensor_batch["labels"], labels_np[idx_list_int])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx_np_bool = np.random.randint(0, 2, size=(data_len,), dtype=bool)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_np_bool = data[idx_np_bool]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_np_bool.batch.keys() == data.batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_np_bool.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_np_bool.batch["obs"].shape[0] == idx_np_bool.sum()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_np_bool.non_tensor_batch["labels"].shape[0] == idx_np_bool.sum()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_np_bool.batch["obs"].cpu().numpy(), obs[idx_np_bool].cpu().numpy())
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_np_bool.non_tensor_batch["labels"], labels_np[idx_np_bool])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx_torch_bool = torch.randint(0, 2, size=(data_len,), dtype=torch.bool)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_torch_bool = data[idx_torch_bool]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_torch_bool.batch.keys() == data.batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_torch_bool.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_torch_bool.batch["obs"].shape[0] == idx_torch_bool.sum().item()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_torch_bool.non_tensor_batch["labels"].shape[0] == idx_torch_bool.sum().item()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_torch_bool.batch["obs"].cpu().numpy(), obs[idx_torch_bool].cpu().numpy())
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_torch_bool.non_tensor_batch["labels"], labels_np[idx_torch_bool])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx_list_bool = [np.random.randint(0, 2, dtype=bool) for _ in range(data_len)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_list_bool = data[idx_list_bool]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_list_bool.batch.keys() == data.batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_list_bool.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_list_bool.batch["obs"].shape[0] == sum(idx_list_bool)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result_list_bool.non_tensor_batch["labels"].shape[0] == sum(idx_list_bool)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_list_bool.batch["obs"].cpu().numpy(), obs[idx_list_bool].cpu().numpy())
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert np.array_equal(result_list_bool.non_tensor_batch["labels"], labels_np[idx_list_bool])


# [EXPLAIN] `test_old_vs_new_from_single_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_old_vs_new_from_single_dict():
    # [EXPLAIN] `CustomProto` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
    class CustomProto(DataProto):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Uses the new, fixed from_single_dict."""

        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] `OriginProto` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
    class OriginProto(DataProto):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Mimics the *old* from_single_dict (always returns a DataProto)."""

        # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
        @classmethod
        # [EXPLAIN] `from_single_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def from_single_dict(cls, data, meta_info=None, auto_padding=False):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensors, non_tensors = {}, {}
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k, v in data.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if torch.is_tensor(v):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    tensors[k] = v
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    non_tensors[k] = v
            # always calls DataProto.from_dict, ignoring `cls`
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return DataProto.from_dict(
                tensors=tensors,
                non_tensors=non_tensors,
                meta_info=meta_info,
                auto_padding=auto_padding,
            )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sample = {"x": torch.tensor([0])}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    orig = OriginProto.from_single_dict(sample)
    # old behavior: always DataProto, not a CustomOriginProto
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert type(orig) is DataProto
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert type(orig) is not OriginProto

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cust = CustomProto.from_single_dict(sample)
    # new behavior: respects subclass
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert type(cust) is CustomProto


# [EXPLAIN] `test_dataproto_no_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_dataproto_no_batch():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(non_tensors={"labels": labels}, meta_info={"info": "test_info"})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    selected = data.select(non_tensor_batch_keys=["labels"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (selected.non_tensor_batch["labels"] == labels).all()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pop_data = data.pop(non_tensor_batch_keys=["labels"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (pop_data.non_tensor_batch["labels"] == labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert data.non_tensor_batch == {}


# [EXPLAIN] `test_sample_level_repeat` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_sample_level_repeat():
    # Create a DataProto object with some batch and non-tensor data
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    # list
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    repeated_data_interleave = data.sample_level_repeat(repeat_times=[3, 1, 2])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_obs_interleave = torch.tensor([[1, 2], [1, 2], [1, 2], [3, 4], [5, 6], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_labels_interleave = ["a", "a", "a", "b", "c", "c"]

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(repeated_data_interleave.batch["obs"], expected_obs_interleave))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (repeated_data_interleave.non_tensor_batch["labels"] == expected_labels_interleave).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert repeated_data_interleave.meta_info == {"info": "test_info"}

    # torch.tensor
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    repeated_data_no_interleave = data.sample_level_repeat(repeat_times=torch.tensor([1, 2, 3]))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_obs_no_interleave = torch.tensor([[1, 2], [3, 4], [3, 4], [5, 6], [5, 6], [5, 6]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_labels_no_interleave = ["a", "b", "b", "c", "c", "c"]

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(repeated_data_no_interleave.batch["obs"], expected_obs_no_interleave))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (repeated_data_no_interleave.non_tensor_batch["labels"] == expected_labels_no_interleave).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert repeated_data_no_interleave.meta_info == {"info": "test_info"}


# [EXPLAIN] `test_dataproto_unfold_column_chunks` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_dataproto_unfold_column_chunks():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs2 = torch.tensor([[1, 2], [5, 6], [9, 10]])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs1": obs1, "obs2": obs2}, non_tensors={"labels": labels}, meta_info={"name": "abc"})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ret = data.unfold_column_chunks(2, split_keys=["obs1"])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_obs1 = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_obs2 = torch.tensor([[1, 2], [1, 2], [5, 6], [5, 6], [9, 10], [9, 10]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_labels = ["a", "a", "b", "b", "c", "c"]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(ret.batch["obs1"], expect_obs1))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(ret.batch["obs2"], expect_obs2))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (ret.non_tensor_batch["labels"] == expect_labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert ret.meta_info == {"name": "abc"}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs2 = torch.tensor([[1, 2], [5, 6], [9, 10]])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = [["a1", "a2"], ["b1", "b2"], ["c1", "c2"]]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs1": obs1, "obs2": obs2}, non_tensors={"labels": labels}, meta_info={"name": "abc"})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ret = data.unfold_column_chunks(2, split_keys=["obs1", "labels"])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_obs1 = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_obs2 = torch.tensor([[1, 2], [1, 2], [5, 6], [5, 6], [9, 10], [9, 10]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_labels = [["a1"], ["a2"], ["b1"], ["b2"], ["c1"], ["c2"]]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(ret.batch["obs1"], expect_obs1))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(ret.batch["obs2"], expect_obs2))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (ret.non_tensor_batch["labels"] == expect_labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert ret.meta_info == {"name": "abc"}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs1 = torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5], [6, 6], [7, 7], [8, 8]], [[9, 9], [10, 10], [11, 11], [12, 12]]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs2 = torch.tensor([[[1, 1], [2, 2]], [[5, 5], [6, 6]], [[9, 9], [10, 10]]])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b", "c"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs1": obs1, "obs2": obs2}, non_tensors={"labels": labels}, meta_info={"name": "abc"})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ret = data.unfold_column_chunks(2, split_keys=["obs1"])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_obs1 = torch.tensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]], [[7, 7], [8, 8]], [[9, 9], [10, 10]], [[11, 11], [12, 12]]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_obs2 = torch.tensor([[[1, 1], [2, 2]], [[1, 1], [2, 2]], [[5, 5], [6, 6]], [[5, 5], [6, 6]], [[9, 9], [10, 10]], [[9, 9], [10, 10]]])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_labels = ["a", "a", "b", "b", "c", "c"]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(ret.batch["obs1"], expect_obs1))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(ret.batch["obs2"], expect_obs2))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (ret.non_tensor_batch["labels"] == expect_labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert ret.meta_info == {"name": "abc"}
