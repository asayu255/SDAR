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
from io import BytesIO
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from PIL import Image
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from qwen_vl_utils import fetch_image, fetch_video


# [EXPLAIN] `process_image` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_image(image: Union[dict, Image.Image]) -> Image.Image:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(image, Image.Image):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return image.convert("RGB")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "bytes" in image:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        image["image"] = BytesIO(image["bytes"])

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return fetch_image(image)


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


# [EXPLAIN] `process_video` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_video(
    video: dict,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not isinstance(video, dict) or "video" not in video:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    video = dict(video)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    contains_sampling_rules = "nframes" in video or "fps" in video
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not contains_sampling_rules:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if nframes is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            video["nframes"] = nframes
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif fps is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            video["fps"] = fps
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if fps_min_frames is not None:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                video["min_frames"] = fps_min_frames
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if fps_max_frames is not None:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                video["max_frames"] = fps_max_frames

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return fetch_video(video)
