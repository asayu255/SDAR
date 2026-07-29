# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import gym
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.environments.env_package.sokoban.sokoban.utils import NoLoggerWarnings, set_seed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.environments.env_package.sokoban.sokoban.room_utils import generate_room
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import copy

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.environments.env_package.sokoban.sokoban.base import BaseDiscreteActionEnv

# [EXPLAIN] `SokobanEnv` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    GRID_LOOKUP = {
        0: " # \t",  # wall
        1: " _ \t",  # floor
        2: " O \t",  # target
        3: " √ \t",  # box on target
        4: " X \t",  # box
        5: " P \t",  # player
        6: " S \t",  # player on target
        # Use tab separator to separate columns and \n\n to separate rows.
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ACTION_LOOKUP = {
        0: "None",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    INVALID_ACTION = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    PENALTY_FOR_INVALID = -1

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, mode, **kwargs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        BaseDiscreteActionEnv.__init__(self)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.cur_seq = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.action_sequence = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.search_depth = kwargs.pop('search_depth', 300)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.mode = mode
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array']
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        GymSokobanEnv.__init__(
            self,
            dim_room=kwargs.pop('dim_room', (6, 6)), 
            max_steps=kwargs.pop('max_steps', 100),
            num_boxes=kwargs.pop('num_boxes', 3),
            **kwargs
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._valid_actions = []


    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, seed=None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.seed = seed
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._reset_tracking_variables()
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with NoLoggerWarnings():
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with set_seed(seed):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        search_depth=self.search_depth
                    )
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except (RuntimeError, RuntimeWarning) as e:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("[SOKOBAN] Retry . . .")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return self.reset(next_seed)
            
            # self.action_sequence = self._reverse_action_sequence(action_sequence)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.player_position = np.argwhere(self.room_state == 5)[0]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.num_env_steps = self.reward_last = self.boxes_on_target = 0

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            info = {
                "won": False,
            }
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.render(self.mode), info
        

    # [EXPLAIN] `finished` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def finished(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.num_env_steps >= self.max_steps or self.success()

    # [EXPLAIN] `success` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def success(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.boxes_on_target == self.num_boxes

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action: int):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        - Step the environment with the given action.
        - Check if the action is effective (whether player moves in the env).
        """
        # assert not self.success()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action == self.INVALID_ACTION:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.render(self.mode), -0.1, False, {"action_is_effective": False, "won": False}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prev_player_position = self.player_position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, reward, done, _ = GymSokobanEnv.step(self, action, observation_mode=self.mode)
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs = self.render(self.mode)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {
            "action_is_effective": not np.array_equal(prev_player_position, self.player_position),
            "won": self.success(),
        }
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs, reward, done, info
     

    # [EXPLAIN] `render` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def render(self, mode):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array']

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mode == 'rgb_array':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            img = self.get_image(mode, scale=1) # numpy array
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return img


        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mode == 'state':
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_state = self.render(mode='state').tolist()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mode == 'list':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mode == 'tiny_rgb_array':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
    
        
    # [EXPLAIN] `copy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def copy(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_self = SokobanEnv(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_self.room_fixed = self.room_fixed.copy()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_self.room_state = self.room_state.copy()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_self.box_mapping = self.box_mapping.copy()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_self.action_sequence = self.action_sequence.copy()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_self.player_position = self.player_position.copy()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_self.reward = self.reward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_self._valid_actions = copy.deepcopy(self._valid_actions)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return new_self
    



    # def _reverse_action_sequence(self, action_sequence):
    #     def reverse_action(action):
    #         return (action % 2 + 1) % 2 + 2 * (action // 2) # 0 <-> 1, 2 <-> 3
    #     return [reverse_action(action) + 1 for action in action_sequence[::-1]] # action + 1 to match the action space
            
    # [EXPLAIN] `set_state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def set_state(self, rendered_state):
        # from the rendered state, set the room state and player position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.room_state = np.where(rendered_state == 6, 5, rendered_state)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.player_position = np.argwhere(self.room_state == 5)[0]