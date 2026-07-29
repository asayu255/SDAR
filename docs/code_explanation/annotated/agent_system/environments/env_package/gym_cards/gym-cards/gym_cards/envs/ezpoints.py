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


# [EXPLAIN] `get_image` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_image(card_name):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    path = f"img/{card_name}.png"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cwd = os.path.dirname(__file__)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image = Image.open(os.path.join(cwd, path))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return image

# Constants for actions
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
NUMBER_ACTIONS_TEN = list(range(1, 11))
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
OPERATOR_ACTIONS = ['+', '*', '=']

# [EXPLAIN] `EZPointEnv` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class EZPointEnv(gym.Env):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A custom Gym environment for an easier version of solving the "24 Game".

    Actions:
        0: 1
        1: 2
        2: 3
        3: 4
        4: 5
        5: 6
        6: 7
        7: 8
        8: 9
        9: 10
        10: '+'
        11: '*'
        12: '='

    Termination:
        - If the formula length exceeds 5.
        - If '=' action is taken, the formula is evaluated.

    Reward:
        - 10 if the formula evaluates to the target_points.
        - 0 if an invalid action is taken.
        - 0 otherwise

    """
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, target_points=12):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.target_points = target_points
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.set_action_space()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.canvas_width, self.canvas_height = 300, 300
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)

    # [EXPLAIN] `set_action_space` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def set_action_space(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        numbers = NUMBER_ACTIONS_TEN
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.allowed_numbers = numbers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.action_space = spaces.Discrete(len(numbers) + len(OPERATOR_ACTIONS))

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
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.cards_num, self.cards = self._generate_cards()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.card_imgs = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.card_width = int(self.canvas_width / len(self.cards) * 0.9)  # Adjust as needed
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.card_height = int(self.card_width * 7/5)  # Assuming a 5:7 card ratio; adjust if different
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, card in enumerate(self.cards):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pil_img = get_image(card).resize((self.card_width, self.card_height))  # Resize the card
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.card_imgs.append(pil_img)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.formula = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.used_cards = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_observation(), info

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action==-1:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._get_observation(), 0, False, False, {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        terminated, reward, info = False, 0, {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chosen_action = self.allowed_numbers[action] if action < len(self.allowed_numbers) else OPERATOR_ACTIONS[action - len(self.allowed_numbers)]

        ## Terminate first if the formula is too long.
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(self.formula) > 5:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._terminate_step(0, 'time_limit_reached', is_truncated=True)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self._is_valid_action(chosen_action):
            ## Add a space to the formula, to make sure the formula length increases.
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._get_observation(), 0, False, False, {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif chosen_action in self.allowed_numbers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.used_cards.append(chosen_action)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if chosen_action == '=':
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._evaluate_formula()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.formula.append(chosen_action)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_observation(), reward, terminated, False, info

    # [EXPLAIN] `_generate_cards` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _generate_cards(self):
        # Generate the first card
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cards_num = [random.randint(2, 13) for _ in range(1)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        suits = ["H", "S", "D", "C"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cards_suit = [random.choice(suits) for _ in range(1)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cards = [y + self._card_num_to_str(x) for x, y in zip(cards_num, cards_suit)]
        # Set face cards to 10
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cards_num = [min(x, 10) for x in cards_num]
        # Check whether the generated card is divisable by 12
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        divisable_list = [1, 2, 3, 4, 6]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cards_num[0] in divisable_list and random.random() <= 0.5:
            # If the number is divisable by 2, 3, 4, 6, then we generate another number to 12 by multiplication with 50% chance
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            cards_num.append(12 // cards_num[0])
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            cards_num.append(12 - cards_num[0])

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        cards.append(random.choice(suits)+self._card_num_to_str(cards_num[1]))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if random.random() <= 0.5:
            # randomly swap the order of the two cards
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cards_num = cards_num[::-1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cards = cards[::-1]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cards_num, cards

    # [EXPLAIN] `_card_num_to_str` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _card_num_to_str(self, num):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        face_cards = {1: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return face_cards.get(num, str(num))



    # [EXPLAIN] `_is_valid_action` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _is_valid_action(self,action):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action not in self.allowed_numbers:
            # We don't check for operators
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return True
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_used_cards = self.used_cards + [action]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_valid = not any(new_used_cards.count(x) > self.cards_num.count(x) for x in new_used_cards)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return is_valid


    # [EXPLAIN] `_evaluate_formula` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _evaluate_formula(self):
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            formula_str = ''.join(map(str, self.formula))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = 10 if eval(formula_str) == self.target_points else 0
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception:
            # The formula is invalid
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = 0
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(self.used_cards) != 2:
                # Not all cards are used.
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reward = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        won = reward > 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": won}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_observation(), reward, True, False, info

    # [EXPLAIN] `_terminate_step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _terminate_step(self, reward, info_key, is_truncated=False):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_observation(), reward, not is_truncated, is_truncated, {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}

    # [EXPLAIN] `_get_observation` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_observation(self):
        # Create a blank white canvas
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        canvas = Image.new('RGB', (self.canvas_width, self.canvas_height), '#35654d')

        # Paste each card onto the canvas
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, pil_img in enumerate(self.card_imgs):
            # Calculate position for pasting
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            x_offset = 5+ int(i * pil_img.width * 1.1)  # adjust this multiplier (1.1) for spacing
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            y_offset = int((self.canvas_height - pil_img.height) / 2)  # center vertically
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            canvas.paste(pil_img, (x_offset, y_offset))

        # Draw formula onto the canvas
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        draw = ImageDraw.Draw(canvas)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_formula = 'Formula: '
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text = f'{" ".join(map(str, self.formula))}'
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_formula = text_formula + text
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        font = ImageFont.truetype('dejavu/DejaVuSans.ttf', 20)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        draw.text((10, self.canvas_height*0.86), text_formula, fill="white", font=font)  # adjust position and other properties as needed
        # Convert PIL image to numpy array if required
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_array = np.array(canvas)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return image_array
