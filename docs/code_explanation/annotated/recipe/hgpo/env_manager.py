# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List, Tuple, Dict, Union, Any
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import partial
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.environments.prompts import *
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.memory import SimpleMemory, SearchMemory
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf


# [EXPLAIN] `parse_gamefile` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parse_gamefile(infos):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    gamefile = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for info in infos:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'extra.gamefile' in info:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            gamefile.append(info['extra.gamefile'])
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            gamefile.append(None)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return gamefile

# [EXPLAIN] `set_gamefile` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def set_gamefile(infos, gamefile):
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(len(infos)):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'extra.gamefile' in infos[i]:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            infos[i]['extra.gamefile'] = gamefile[i]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            infos[i]['extra.gamefile'] = None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return infos


# [EXPLAIN] `SearchEnvironmentManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SearchEnvironmentManager(EnvironmentManagerBase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    EnvironmentManager for SearchEnv.
    """
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, envs, projection_f, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.memory = SearchMemory()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(envs, projection_f, config)

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs, infos = self.envs.reset(kwargs=kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tasks = obs

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.reset(batch_size=len(obs))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return observations, infos

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, text_actions: List[str]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions, valids = self.projection_f(text_actions)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_obs, rewards, dones, infos = self.envs.step(actions)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, info in enumerate(infos):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info["is_action_valid"] = to_numpy(valids[i])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rewards = to_numpy(rewards)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dones = to_numpy(dones)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return next_observations, rewards, dones, infos

    # [EXPLAIN] `build_text_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        postprocess_text_obs: List[str] = []

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not init and self.config.env.history_length > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(text_obs)):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if init or self.config.env.history_length <= 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            postprocess_text_obs.append(obs_i)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return postprocess_text_obs


    # [EXPLAIN] `_process_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_item = total_batch_list[batch_idx][i]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if batch_item['active_masks']:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                info = total_infos[batch_idx][i]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                won_value = float(info['won'])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                success['success_rate'].append(won_value)
                
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data_source = info.get("data_source")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                success[f"{data_source}_success_rate"].append(won_value)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return  # Exit after finding the first active mask
            

# [EXPLAIN] `AlfWorldEnvironmentManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, envs, projection_f, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.memory = SimpleMemory()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(envs, projection_f, config)
    
    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs, image_obs, infos = self.envs.reset()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.reset(batch_size = len(text_obs))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tasks = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pre_text_obs = text_obs
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.extract_task(text_obs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, text_actions: List[str]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pre_text_obs = text_obs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if infos[0].get("extra.gamefile") is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, info in enumerate(infos):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info['is_action_valid'] = to_numpy(valids[i])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rewards = to_numpy(rewards)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dones = to_numpy(dones)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return next_observations, rewards, dones, infos
    
    # [EXPLAIN] `extract_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def extract_task(self, text_obs: List[str]):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for obs in text_obs:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_start = obs.find('Your task is to: ')
            
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if task_start != -1:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError("Task description not found in text observation.")
        

    # [EXPLAIN] `build_text_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        This function builds the text observation for the agent.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        postprocess_text_obs = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not init and self.config.env.history_length > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if init or self.config.env.history_length <= 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            postprocess_text_obs.append(obs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return postprocess_text_obs
            


    # [EXPLAIN] `_process_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_item = total_batch_list[batch_idx][i]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if batch_item['active_masks']:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                info = total_infos[batch_idx][i]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                won_value = float(info['won'])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                gamefile = info.get("extra.gamefile")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if gamefile:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    self._process_gamefile(gamefile, won_value, success)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return  # Exit after finding the first active mask

    # [EXPLAIN] `_process_gamefile` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _process_gamefile(self, gamefile, won_value, success):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task in tasks:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if task in gamefile:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                success[f"{task}_success_rate"].append(won_value)
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break


# [EXPLAIN] `SokobanEnvironmentManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SokobanEnvironmentManager(EnvironmentManagerBase):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, envs, projection_f, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.is_multi_modal = envs.mode == 'rgb_array'
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.memory = SimpleMemory()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(envs, projection_f, config)

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs, infos = self.envs.reset()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.is_multi_modal:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs = np.array(obs, obs[0].dtype)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.pre_text_obs = obs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.reset(batch_size = len(infos))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return observations, infos

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, text_actions: List[str]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions, valids = self.projection_f(text_actions)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, info in enumerate(infos):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info['is_action_valid'] = to_numpy(valids[i])

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.is_multi_modal:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            next_obs = np.array(next_obs, next_obs[0].dtype)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.pre_text_obs = next_obs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rewards = to_numpy(rewards)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dones = to_numpy(dones)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return next_observations, rewards, dones, infos

    # [EXPLAIN] `build_text_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        This function builds the text observation for the agent.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        postprocess_text_obs = []

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not init and self.config.env.history_length > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(infos)):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if init or self.config.env.history_length <= 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.is_multi_modal:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    obs = SOKOBAN_VISUAL_TEMPLATE
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            postprocess_text_obs.append(obs)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return postprocess_text_obs


# [EXPLAIN] `GymCardEnvironmentManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class GymCardEnvironmentManager(EnvironmentManagerBase):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, envs, projection_f, config):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(envs, projection_f, config)
    
    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, kwargs) -> Dict[str, Any]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return observations, infos

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, text_actions: List[str]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        next_observations['text'] = self.build_text_obs(infos)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        next_observations['anchor'] = next_observations['image'].copy()

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return next_observations, rewards, dones, infos


    # [EXPLAIN] `build_text_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        This function builds the text observation for the agent.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        postprocess_text_obs = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(infos)):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 'ezpoints' in self.config.env.env_name.lower():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif 'points24' in self.config.env.env_name.lower():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif 'numberline' in self.config.env.env_name.lower():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif "blackjack" in self.config.env.env_name.lower():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            postprocess_text_obs.append(obs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return postprocess_text_obs


# [EXPLAIN] `WebshopEnvironmentManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class WebshopEnvironmentManager(EnvironmentManagerBase):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, envs, projection_f, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.memory = SimpleMemory()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(envs, projection_f, config)
    
    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, kwargs) -> Dict[str, Any]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs, infos = self.envs.reset()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tasks = self.extract_task(obs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pre_text_obs = obs
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.reset(batch_size = len(infos))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return observations, infos

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, text_actions: List[str]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions, valids = self.projection_f(text_actions)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_obs = self.format_obs(next_obs)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pre_text_obs = next_obs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, info in enumerate(infos):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info['is_action_valid'] = to_numpy(valids[i])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rewards = to_numpy(rewards)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dones = to_numpy(dones)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return next_observations, rewards, dones, infos

    # [EXPLAIN] `extract_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def extract_task(self, text_obs: List[str]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tasks = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for obs in text_obs:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            parts = obs.split(" [SEP] ")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert parts[1]=='Instruction:'
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tasks.append(parts[2])
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tasks
    
    # [EXPLAIN] `format_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def format_obs(self, text_obs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        postprocess_text_obs = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(text_obs)):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                index = parts.index(self.tasks[i])
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reformatted_obs = text_obs[i]

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            postprocess_text_obs.append(reformatted_obs)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return postprocess_text_obs
    
    # [EXPLAIN] `format_avail_actions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def format_avail_actions(self, avail):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions = []

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in avail.keys():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if key not in ["has_search_bar", "clickables"]:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Unknown key in available actions: {key}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if avail["has_search_bar"]:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            actions.append("search[<your query>]")

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for txt in avail["clickables"]:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            actions.append(f"click[{txt}]")

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return actions
            
    # [EXPLAIN] `build_text_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        This function builds the text observation for the agent.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        postprocess_text_obs = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not init and self.config.env.history_length > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(text_obs)):
            
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if init or self.config.env.history_length <= 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                # if len(obs) > 13000:
                #     print(f"Warning len(obs)={len(obs)} is too long")
                #     obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                #         task_description=self.tasks[i],
                #         current_observation=text_obs[i],
                #         available_actions=reformatted_available_actions
                #     )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            postprocess_text_obs.append(obs)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return postprocess_text_obs

    # [EXPLAIN] `_process_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_item = total_batch_list[batch_idx][i]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if batch_item['active_masks']:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                info = total_infos[batch_idx][i]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                won_value = float(info['won'])
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                score_value = float(info['task_score'])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                success['success_rate'].append(won_value)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                success['webshop_task_score (not success_rate)'].append(score_value)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return


# [EXPLAIN] `AppWorldEnvironmentManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AppWorldEnvironmentManager(EnvironmentManagerBase):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, envs, projection_f, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.memory = SimpleMemory()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(envs, projection_f, config)
    
    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs, infos = self.envs.reset()
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.supervisors = [info['supervisor'] for info in infos]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.reset(batch_size = len(text_obs))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tasks = text_obs.copy()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pre_text_obs = text_obs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_text_obs = self.build_text_obs(text_obs, init=True)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, text_actions: List[str]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions, valids = self.projection_f(text_actions)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs, rewards, dones, infos = self.envs.step(actions)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.memory.store({'text_obs': text_obs, 'action': actions})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pre_text_obs = text_obs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, info in enumerate(infos):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info['is_action_valid'] = to_numpy(valids[i])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rewards = to_numpy(rewards)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dones = to_numpy(dones)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return next_observations, rewards, dones, infos
    

    # [EXPLAIN] `build_text_obs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        This function builds the text observation for the agent.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        postprocess_text_obs = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if init and self.supervisors is not None:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(len(text_obs)):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                postprocess_text_obs.append(obs)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                recent_history = self.memory[i][-self.config.env.history_length:]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                valid_history_length = len(recent_history)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                start_index = len(self.memory[i]) - valid_history_length
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                action_history = ""
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for j, record in enumerate(recent_history):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    step_number = start_index + j + 1
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    action = record["action"]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    env_obs = record["text_obs"]
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if len(action_history) > 10000:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    action_history = "... " + action_history[-10000:]

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                postprocess_text_obs.append(obs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return postprocess_text_obs

# [EXPLAIN] `make_envs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def make_envs(config):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not isinstance(config.env.rollout.n, int):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError("config.env.rollout.n should be an integer")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "search" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(search_projection)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.trainer.val_only:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = SearchEnvironmentManager(_envs, projection_f, config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "gym_cards" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.trainer.val_only:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = GymCardEnvironmentManager(_envs, projection_f, config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "alfworld" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _recipe_hgpo_dir = os.path.dirname(os.path.abspath(__file__))  # recipe/hgpo
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _alfworld_configs_dir = os.path.join(_recipe_hgpo_dir, '..', '..', 'agent_system', 'environments', 'env_package', 'alfworld', 'configs')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            alf_config_path = os.path.join(_alfworld_configs_dir, 'config_thor.yaml')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            alf_config_path = os.path.join(_alfworld_configs_dir, 'config_tw.yaml')
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.trainer.val_out:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            env_kwargs = {
                'eval_dataset': 'eval_out_of_distribution', # 'eval_in_distribution' or 'eval_out_of_distribution'
            }
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            env_kwargs = {
                'eval_dataset': 'eval_in_distribution', # 'eval_in_distribution' or 'eval_out_of_distribution'
            }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(alfworld_projection)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.trainer.val_only:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs,resources_per_worker=resources_per_worker)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
            
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs,resources_per_worker=resources_per_worker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "sokoban" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(sokoban_projection)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.trainer.val_only:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs,resources_per_worker=resources_per_worker)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = SokobanEnvironmentManager(_envs, projection_f, config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs,resources_per_worker=resources_per_worker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "webshop" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _recipe_hgpo_dir = os.path.dirname(os.path.abspath(__file__))  # recipe/hgpo
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _webshop_data_dir = os.path.join(_recipe_hgpo_dir, '..', '..', 'agent_system', 'environments', 'env_package', 'webshop', 'webshop', 'data')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.env.webshop.use_small:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            file_path = os.path.join(_webshop_data_dir, 'items_shuffle_1000.json')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attr_path = os.path.join(_webshop_data_dir, 'items_ins_v2_1000.json')
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            file_path = os.path.join(_webshop_data_dir, 'items_shuffle.json')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attr_path = os.path.join(_webshop_data_dir, 'items_ins_v2.json')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(webshop_projection)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.trainer.val_only:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs,resources_per_worker=resources_per_worker)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = WebshopEnvironmentManager(_envs, projection_f, config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs,resources_per_worker=resources_per_worker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import time
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "appworld" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(appworld_projection)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.trainer.val_only:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Environment not supported")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        exit(1)