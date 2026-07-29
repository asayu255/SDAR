# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import gymnasium as gym
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gymnasium import spaces
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from PIL import Image, ImageDraw, ImageFont
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from io import BytesIO


# [EXPLAIN] `NumberLineEnv` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class NumberLineEnv(gym.Env):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
A custom Gym environment simulating a number line.

Actions:
    - 0: Move left (decrease the current position by 1, if greater than 0).
    - 1: Move right (increase the current position by 1, if less than max_position).

Observation:
    - An RGB image representation of the current and goal positions on the number line.
    - The image has a shape of (300, 300, 3) and pixel values ranging from 0 to 255.

Termination:
    - The episode ends when the current position reaches the goal position.

Reward:
    - A reward of 1 is given if the current position is the same as the goal position.
    - A reward of -1 is given if the action does not move the current position closer to the goal position.
    - A reward of 0 is given otherwise.

Initialization Options:
    - max_position: The maximum value on the number line (default is 5).
      Both the start and goal positions are randomly initialized between 0 and max_position.

Special Notes:
    - The start and goal positions are re-randomized at the beginning of each episode.
    - If after 2 * max_position steps the target still is not meet, the episode is terminated and the environment is reset.
    - The environment ensures that the start and goal positions are not the same initially.
    - An observation is returned as an RGB image with the current and goal positions labeled.

"""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, max_position=5):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super(NumberLineEnv, self).__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_position = max_position

        # Define action and observation space
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.action_space = spaces.Discrete(2)  # 0: Move left, 1: Move right
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)

        # Randomize start and goal
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.start_position = random.randint(0, self.max_position)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.goal_position = random.randint(0, self.max_position)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.start_position == self.goal_position:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.goal_position = (self.goal_position + 1) % self.max_position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.position = self.start_position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.steps_made = 0

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action==-1:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._get_observation(), 0.0, False, False, {"Target": self.goal_position, "Current": self.position, "won": False}
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.steps_made += 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prev_distance = abs(self.position - self.goal_position)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action == 0 and self.position > 0:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.position -= 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif action == 1 and self.position < self.max_position:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.position += 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        current_distance = abs(self.position - self.goal_position)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        done = False
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if current_distance == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            done = True
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif current_distance > prev_distance or current_distance == prev_distance:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = -1
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = 0

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        won = reward > 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {"Target": self.goal_position, "Current": self.position, "won": won}

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observation = self._get_observation()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.start_position == self.goal_position:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.goal_position = (self.goal_position + 1) % self.max_position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        truncate = False
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not done and self.steps_made >= self.max_position * 2:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            truncate = True
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return observation, reward, done, truncate, info

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().reset(seed=seed)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        random.seed(seed)
        # Randomize start and goal
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.start_position = random.randint(0, self.max_position)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.goal_position = random.randint(0, self.max_position)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.start_position == self.goal_position:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.goal_position = (self.goal_position + 1) % self.max_position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.position = self.start_position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.steps_made = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {"Target": self.goal_position, "Current": self.position, "won": False}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_observation(), info

    # [EXPLAIN] `_get_observation` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_observation(self):
        # Draw the "Goal" and "Current" strings
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        img = Image.new('RGB', (300,300), color="white")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        draw = ImageDraw.Draw(img)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        font = ImageFont.truetype('dejavu/DejaVuSans-Bold.ttf', 36)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal_text = f"Target: {self.goal_position}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        current_text = f"Current: {self.position}"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        draw.text((30, 60), goal_text, fill='black',font=font)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        draw.text((30, 180), current_text, fill='black',font=font)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return np.array(img)

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
