# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from unittest.mock import AsyncMock, MagicMock, patch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@patch.dict(
    "sys.modules",
    {
        "verl.workers.rollout.sglang_rollout.sglang_rollout": MagicMock(SGLangRollout=MagicMock()),
    },
)
# [EXPLAIN] `TestAsyncSglangServer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestAsyncSglangServer:
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.fixture
    # [EXPLAIN] `mock_ray_actor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def mock_ray_actor(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mock_actor = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mock_actor.execute_method.remote = AsyncMock(return_value={"content": "mocked response"})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mock_actor.resume.remote = AsyncMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mock_actor.offload.remote = AsyncMock()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return mock_actor

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.fixture
    # [EXPLAIN] `server_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def server_config(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DictConfig({"rollout": {"tensor_model_parallel_size": 2}})

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.mark.asyncio
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch("verl.workers.rollout.sglang_rollout.async_sglang_server.ray.util.list_named_actors")
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch("verl.workers.rollout.async_server.AsyncServerBase._start_fastapi_server", new_callable=AsyncMock)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.mark.filterwarnings("ignore:Ray state API is no longer experimental:DeprecationWarning")
    # [EXPLAIN] `test_init_engine` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def test_init_engine(self, mock_start_fastapi_server, mock_list_actors, server_config, mock_ray_actor):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mock_list_actors.return_value = [
            {"name": "test_prefixWorkerDict_0:0", "namespace": "test"},
            {"name": "test_prefixWorkerDict_0:1", "namespace": "test"},
            {"name": "test_prefixWorkerDict_1:2", "namespace": "test"},
        ]
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSglangServer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ActualClassToInstantiate = AsyncSglangServer
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if hasattr(AsyncSglangServer, "__ray_metadata__") and hasattr(AsyncSglangServer.__ray_metadata__, "modified_class"):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ActualClassToInstantiate = AsyncSglangServer.__ray_metadata__.modified_class

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with patch("verl.workers.rollout.sglang_rollout.async_sglang_server.ray.get_actor", return_value=mock_ray_actor):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            instance = ActualClassToInstantiate(server_config, 2, 0, "test_prefix")

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            await instance.init_engine()

            # Verify instance.workers is correctly populated
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(instance.workers) == 2
