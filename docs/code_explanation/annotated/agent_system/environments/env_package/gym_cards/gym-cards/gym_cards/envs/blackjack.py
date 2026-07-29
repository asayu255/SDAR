# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import gymnasium as gym
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gymnasium import spaces
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from PIL import Image, ImageDraw, ImageFont

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
image_cache = {}

# [EXPLAIN] `get_image` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_image(card_name):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if card_name in image_cache:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return image_cache[card_name]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    path = f"img/{card_name}.png"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cwd = os.path.dirname(__file__)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image = Image.open(os.path.join(cwd, path))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_cache[card_name] = image
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return image

# [EXPLAIN] `draw_card_with_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def draw_card_with_info(np_random):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    card_value = draw_card(np_random)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    suit = np_random.choice(["C", "D", "H", "S"])
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if card_value == 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        face = 'A'
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif card_value == 10:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        face = np_random.choice(["J", "Q", "K"])
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        face = str(card_value)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return card_value, face, suit

# [EXPLAIN] `draw_hand_with_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def draw_hand_with_info(np_random):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return [draw_card_with_info(np_random) for _ in range(2)]


# [EXPLAIN] `cmp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def cmp(a, b):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r = float(a > b) - float(a < b)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r = 0 if r < 0 else r
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return r


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


# [EXPLAIN] `draw_card` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def draw_card(np_random):

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return int(np_random.choice(deck))


# [EXPLAIN] `draw_hand` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def draw_hand(np_random):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return [draw_card(np_random), draw_card(np_random)]


# [EXPLAIN] `usable_ace` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def usable_ace(hand_values):  # Does this hand have a usable ace?
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return int(1 in hand_values and sum(hand_values) + 10 <= 21)


# [EXPLAIN] `sum_hand` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def sum_hand(hand):  # Return current hand total
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hand_values = [card[0] for card in hand]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if usable_ace(hand_values):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sum(hand_values) + 10
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return sum(hand_values)


# [EXPLAIN] `is_bust` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_bust(hand):  # Is this hand a bust?
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    values = [card[0] for card in hand]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return sum_hand(hand) > 21


# [EXPLAIN] `score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def score(hand):  # What is the score of this hand (0 if bust)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return 0 if is_bust(hand) else sum_hand(hand)


# [EXPLAIN] `is_natural` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_natural(hand):  # Is this hand a natural blackjack?
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return sorted(hand) == [1, 10]


# [EXPLAIN] `BlackjackEnv` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class BlackjackEnv(gym.Env):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ## Description
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards. All cards are drawn from an infinite deck
    (i.e. with replacement).

    The card values are:
    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.

    The player has the sum of cards held. The player can request
    additional cards (hit) until they decide to stop (stick) or exceed 21 (bust,
    immediate loss).

    After the player sticks, the dealer reveals their facedown card, and draws cards
    until their sum is 17 or greater. If the dealer goes bust, the player wins.

    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto [<a href="#blackjack_ref">1</a>].

    ## Action Space
    The action shape is `(1,)` in the range `{0, 1}` indicating
    whether to stick or hit.

    - 0: Stick
    - 1: Hit

    ## Observation Space
    The observation is a pixel image of the current state of the game.
    spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)

    The observation is returned as `(int(), int(), int())`.

    ## Starting State
    The starting state is initialised in the following range.

    | Observation               | Min  | Max  |
    |---------------------------|------|------|
    | Player current sum        |  4   |  12  |
    | Dealer showing card value |  2   |  11  |
    | Usable Ace                |  0   |  1   |

    ## Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:
    +1.5 (if <a href="#nat">natural</a> is True)
    +1 (if <a href="#nat">natural</a> is False)

    ## Episode End
    The episode ends if the following happens:

    - Termination:
    1. The player hits and the sum of hand exceeds 21.
    2. The player sticks.

    An ace will always be counted as usable (11) unless it busts the player.

    ## Information

    No additional information is returned.

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```

    <a id="nat"></a>`natural=False`: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

    <a id="sab"></a>`sab=False`: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

    ## References
    <a id="blackjack_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
    An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)

    ## Version History
    * v1: Fix the natural handling in Blackjack
    * v0: Initial version release
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False, is_pixel: bool = True):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.is_pixel = is_pixel
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.action_space = spaces.Discrete(2)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.is_pixel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2)))

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sab = sab

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.render_mode = render_mode
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return


    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action==-1:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._get_obs(), 0.0, False, False, {"Dealer Card": self.dealer, "Player Card": self.player, 'won': False}
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.action_space.contains(action)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action:  # hit: add a card to players hand and return
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.player.append(draw_card_with_info(self.np_random))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            won = False
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if is_bust(self.player):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                terminated = True
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reward = 0.0
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                terminated = False
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reward = 0.0
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:  # stick: play out the dealers hand, and score
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            terminated = True
            # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
            while sum_hand(self.dealer) < 17:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.dealer.append(draw_card_with_info(self.np_random))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = cmp(score(self.player), score(self.dealer))
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reward = 1.0
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reward = 1.5
            
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            won = reward > 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {"Dealer Card": self.dealer, "Player Card": self.player, 'won': won}

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_obs(), reward, terminated, False, info



    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().reset(seed=seed)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dealer = draw_hand_with_info(self.np_random)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.player = draw_hand_with_info(self.np_random)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        player_values = [card[0] for card in self.player]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, dealer_card_value, _ = (sum_hand(self.player), self.dealer[0][0], usable_ace(player_values))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        suits = ["C", "D", "H", "S"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dealer_top_card_suit = self.np_random.choice(suits)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if dealer_card_value == 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.dealer_top_card_value_str = "A"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif dealer_card_value == 10:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.dealer_top_card_value_str = str(dealer_card_value)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.render_mode == "human":
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.render()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {"Dealer Card": self.dealer, "Player Card": self.player, 'won': False}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_obs(), info

    # [EXPLAIN] `_get_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_obs(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.is_pixel:
            # Define image size and background color
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            img_size = (300, 300)  # Adjust size as needed
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            card_size = (70, 98)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            spacing = 4
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            background_color = '#35654d'

            # Create a new image with the defined size and background color
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            img = Image.new('RGB', img_size, color=background_color)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            draw = ImageDraw.Draw(img)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            font = ImageFont.truetype('dejavu/DejaVuSans-Bold.ttf', 16)
            # Load and paste dealer card image
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            draw.text((5, 5), f"Dealer", fill='white', font=font)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dealer_card = f"{self.dealer[0][2]}{self.dealer[0][1]}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dealer_card_img = get_image(dealer_card).resize(card_size)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            img.paste(dealer_card_img, (5, 25))  # Adjust position as needed
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            back_card_img = get_image("card").resize(card_size)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            img.paste(back_card_img, (78, 25))

            # Load and paste player card images
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            x_offset, y_offset = 5, 150  # Starting position for player cards
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            draw.text((5, 130), f"Player", fill='white', font=font)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for idx, (_, face, suit) in enumerate(self.player):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                card_name = f"{suit}{face}"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                card_img = get_image(card_name).resize(card_size)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                img.paste(card_img, (x_offset, y_offset))
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                x_offset += card_img.width + spacing  # Adjust spacing and position as needed
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if idx == 4:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    y_offset += card_img.height + spacing
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    x_offset = 5

            # Convert the PIL image to a NumPy array if needed
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image = np.array(img).astype(np.uint8)

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return image
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return (sum_hand(self.player), self.dealer[0][0], usable_ace(self.player))