# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gym_cards.envs.points import Point24Env
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gym_cards.envs.ezpoints import EZPointEnv
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gym_cards.envs.blackjack import BlackjackEnv
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gym_cards.envs.numberline import NumberLineEnv
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gymnasium.envs.registration import register

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
register(
    id='gym_cards/Blackjack-v0',
    entry_point='gym_cards.envs:BlackjackEnv',
    max_episode_steps=300,
)

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
register(
    id='gym_cards/Points24-v0',
    entry_point='gym_cards.envs:Point24Env',
    max_episode_steps=300,
)

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
register(
    id='gym_cards/EZPoints-v0',
    entry_point='gym_cards.envs:EZPointEnv',
    max_episode_steps=300,
)

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
register(
    id='gym_cards/NumberLine-v0',
    entry_point='gym_cards.envs:NumberLineEnv',
    max_episode_steps=300,
)