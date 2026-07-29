# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List

# [EXPLAIN] `info_to_text_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def info_to_text_obs(env_name, info):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This function directly parse the info from the gym_cards envs
    to customized text observation.

    gym_cards envs: https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards

    please adjust text_obs accordingly as needed
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if env_name == "gym_cards/NumberLine-v0":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs = f"Target number: {info['Target']}. Current number: {info['Current']}"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif env_name == "gym_cards/EZPoints-v0":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        J, Q, and K count as 10, you can choose to increase the difficulty
        only telling the language model J, Q, K, and describe in the prompt that they count as 10
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs = f"Cards: {info['Cards']}. Numbers: {info['Numbers']}."
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        current_formula = ''.join(str(element) for element in info['Formula'])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs = text_obs + f" Current formula: {current_formula}."
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif env_name == "gym_cards/Points24-v0":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs = f"Cards: {info['Cards']}. Numbers: {info['Numbers']}."
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        current_formula = ''.join(str(element) for element in info['Formula'])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs = text_obs + f" Current formula: {current_formula}."
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif env_name == "gym_cards/Blackjack-v0":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        the info for each card is represented in (card_value, face, suit)
        e.g., (10, 'J', 'H') represents the Jack of Hearts
        and we shall only include one card for the dealer
        since we cannot see all dealer's cards
        """
        ## note that the first [0] means we only select the first card
        ## the [1] means that we are using the card value, not the number
        ## feel free to change the [1] to [0] to reduce the complexity
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dealer_card = info['Dealer Card'][0][1]
        ## the [1] means that we are using the card value, not the number
        ## feel free to change the [1] to [0] to reduce the complexity
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        player_cards = ', '.join(str(element[1]) for element in info['Player Card'])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs = f"Dealer's card: {dealer_card}. Player's cards: {player_cards}."
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("Environment not implemented.")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return text_obs
### Below are direclty copied from https://github.com/RL4VLM/RL4VLM/blob/main/VLM_PPO/a2c_ppo_acktr/rl_utils.py

# Define the function that processes the list of strings according to the specified rules
# [EXPLAIN] `text_projection` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def text_projection(text_actions: List[str], env_name):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_indices = []
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if env_name == 'gym_cards/NumberLine-v0':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_list = ["-", "+"]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif env_name == 'gym_cards/Blackjack-v0':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_list = ["stand", "hit"]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif env_name == 'gym_cards/EZPoints-v0':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif env_name == 'gym_cards/Points24-v0':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("Action list not implemented for this env!")
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for string in text_actions:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(string, str):
            # directly output a random action if the string is not a string
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output_indices.append(random.randint(0, len(action_list) - 1))
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        string = string.lower()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_index = string.find('"action":')
        # Extract everything after "action":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        string = string[action_index:]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        contained_actions = []
        # For the 'gym_cards/Points24-v0' environment, handle '10' separately
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'points' in env_name.lower() and '10' in string:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            contained_actions.append('10')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            string = string.replace('10', '')  # Remove '10' to prevent it from being counted as '1'
        # Find all actions that are contained in the string
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for action in action_list:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if action in string:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                contained_actions.append(action)
        # Remove duplicates by converting to a set and back to a list
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        contained_actions = list(set(contained_actions))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            # Only one keyword from action_list is in the string
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output_indices.append(action_list.index(contained_actions[0]))
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # The string contains none or multiple keywords, randomly select an index from action_list
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output_indices.append(random.randint(0, len(action_list) - 1))
    ## Please adjust the output dtypes accordingly
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output_indices