# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from abc import ABC, abstractmethod
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional, List, Tuple, Any, Dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from copy import deepcopy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoTokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] `BaseEnv` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class BaseEnv(ABC):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Abstract base class for all environments.
    The class needs to handle text-based input, input may be invalid
        - Environment will track the total reward for the trajectory

    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    INVALID_ACTION = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    PENALTY_FOR_INVALID = -1
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward = 0

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._actions = [] # list of all actions (including all responses from LLM)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._actions_valid = [] # list of actions that are in the correct format
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._actions_effective = [] # list of actions that are effective (actual moving in env)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `_extract_answer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _extract_answer(text: str) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Extract the answer from the text."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        match = re.search(r"<answer>(.*?)</answer>", text)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if match:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return match.group(1).strip()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None

    # [EXPLAIN] `_reset_tracking_variables` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _reset_tracking_variables(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._actions = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._actions_valid = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._actions_effective = []

    # [EXPLAIN] `get_tracking_variables` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_tracking_variables(self) -> Dict:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get statistics of valid actions."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {
            "reward": self.reward,
            "actions": self._actions,
            "actions_valid": self._actions_valid,
            "actions_effective": self._actions_effective,
        }
    
    # [EXPLAIN] `_update_tracking_variables` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _update_tracking_variables(
            self, 
            response: str,
            action: Any, 
            action_is_valid: bool,
            action_is_effective: bool,
            reward: float,
        ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        All of _actions, _actions_valid, _actions_effective are lists of the same length
            - None is used for _actions_valid and _actions_effective if the action is invalid or ineffective
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._actions.append(response)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action_is_valid:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._actions_valid.append(action)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._actions_valid.append(None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action_is_effective:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._actions_effective.append(action)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._actions_effective.append(None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.reward += reward if action_is_valid else (reward + self.PENALTY_FOR_INVALID)

    # [EXPLAIN] `_copy_tracking_variables` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _copy_tracking_variables(self, other: 'BaseEnv'):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward = other.reward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._actions = deepcopy(other._actions)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._actions_valid = deepcopy(other._actions_valid)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._actions_effective = deepcopy(other._actions_effective)



    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `formulate_output` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def formulate_output(env_feedback: str, done: bool = False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Formulate the environment feedback to as the input to the LLM
        - e.g., For Qwen, special tokens like <|im_start|>user and <|im_end|> should be added
        NOTE hard-coded now for Qwen
        """

        # obs = "\n <|im_start|>user\n" + env_feedback + "<|im_end|>\n" + "<|im_start|>assistant\n<think>"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = "\n <|im_start|>user\n" + env_feedback + "<|im_end|>\n"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not done:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            output += "<|im_start|>assistant\n<think>"
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `execute_predictions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_predictions(
        cls, 
        envs: List['BaseEnv'], 
        predictions: List[str], 
        prediction_ids: torch.Tensor,
        tokenizer: AutoTokenizer,
    ) -> List[str]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings


        TODO modify reward here, not treat penalty_for_invalid as class variable
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cur_actions, action_is_valid = cls.postprocess_predictions(envs, predictions)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_obs, dones = [], []
        
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for env, action, response, response_id, av in zip(envs, cur_actions, predictions, prediction_ids, action_is_valid):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs = ""
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "<|im_end|>" not in response:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                obs += "<|im_end|>"

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if env.finished():
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                obs += tokenizer.pad_token
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                done = True
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # thinking reward
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                thinking_reward = 0
                # n_non_pad = (response_id != tokenizer.pad_token_id).sum().item()
                # if n_non_pad > 50: # NOTE hard-coded here
                #     thinking_reward += 1
                
                
                # step in environment
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                observation, env_reward, done, extra_info = env.step(action)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                env_feedback = cls.parse_update_info_to_obs(
                    (observation, env_reward, done, extra_info), 
                    av
                )

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                obs += cls.formulate_output(env_feedback, done)
                
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                env._update_tracking_variables(
                    response=response, 
                    action=action, 
                    action_is_valid=av, 
                    action_is_effective=extra_info.get("action_is_effective", False), 
                    reward=thinking_reward + env_reward, 
                )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            next_obs.append(obs)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dones.append(done)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return next_obs, dones


    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `parse_update_info_to_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def parse_update_info_to_obs(update_info: Tuple[Any, float, bool, Dict], action_is_valid: bool) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Parse environment update information into observation string.
        
        Args:
            update_info: Tuple of (observation, reward, done, info)
            action_is_valid: Whether the action was valid
            
        Returns:
            Formatted observation string
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, seed: Optional[int] = None) -> Any:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment
            
        Returns:
            rendered environment
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `success` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def success(self) -> bool:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Check if the current environment is successful."""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `finished` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def finished(self) -> bool:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Check if the current environment is finished."""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `render` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def render(self, mode: str = 'tiny_rgb_array') -> Any:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Render the environment."""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `copy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def copy(self) -> 'BaseEnv':
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create a deep copy of the environment."""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass






# [EXPLAIN] `BaseDiscreteActionEnv` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class BaseDiscreteActionEnv(BaseEnv, ABC):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Abstract base class for environments with discrete action spaces
    This class provides common functionality for environments like FrozenLakeEnv and SokobanEnv.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    GRID_LOOKUP = {} # define the mapping from integer to string for rendering
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ACTION_LOOKUP = {} # define the mapping from integer to action string
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    INVALID_ACTION = 0 # default invalid action
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    PENALTY_FOR_INVALID = -1 # penalty for invalid action

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `parse_update_info_to_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def parse_update_info_to_obs(update_info: Tuple[Any, float, bool, Dict], action_is_valid: bool) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Parse environment update information into observation string.
        
        Args:
            update_info: Tuple of (observation, reward, done, info)
            action_is_valid: Whether the action was valid
            
        Returns:
            Observation string
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observation, reward, done, _ = update_info
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not action_is_valid:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return f"Action is invalid. You stay in the same position. The observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return f"After you take this action, the observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"


    # [EXPLAIN] `get_all_actions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_all_actions(self) -> List[int]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get list of all valid actions."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return list(range(self.ACTION_SPACE.start, self.ACTION_SPACE.start + self.ACTION_SPACE.n))
    

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, mode: str = 'tiny_rgb_array', seed: Optional[int] = None) -> Any:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Reset the environment.
        NOTE: the environment must be same for the same seed
        Args:
            mode: Mode to render the environment
            seed: Seed for the environment
            
        Returns:
            rendered environment
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `success` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def success(self) -> bool:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Check if the current environment is successful."""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `finished` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def finished(self) -> bool:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Check if the current environment is finished."""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `render` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def render(self, mode: str = 'tiny_rgb_array') -> Any:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Render the environment.
        Args:
            mode: Mode to render the environment, needs to provide:
                - 'tiny_rgb_array': a string of the environment
                - 'rgb_array': a numpy array of the environment
        Returns:
            rendered environment, maybe a string or a numpy array (image)
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `copy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def copy(self) -> 'BaseDiscreteActionEnv':
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create a deep copy of the environment."""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass