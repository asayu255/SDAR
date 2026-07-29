# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
from typing import List, Tuple, Dict, Union, Any
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from concurrent.futures import ThreadPoolExecutor, as_completed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
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
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if len(obs) > 13000:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Warning len(obs)={len(obs)} is too long")
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

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


# [EXPLAIN] `_normalize_multitask_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _normalize_multitask_name(task_name: str) -> str:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_name = str(task_name).lower()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "alfworld" in task_name:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "alfworld"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "webshop" in task_name:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "webshop"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "search" in task_name:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "search"
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise ValueError(f"Unsupported multitask task_name: {task_name}")


# [EXPLAIN] `_plain_container` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _plain_container(value):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


# [EXPLAIN] mixed batch を task 別 manager へ分配し、reset/step 結果を元の global batch 順へ復元する。
class MultiTaskEnvironmentManager(EnvironmentManagerBase):
    # [EXPLAIN] mixed global batch を canonical task 名で分割し、task-local manager を並列呼び出しした後、
    # [EXPLAIN] observation/reward/done/info を元の global batch index へ scatter して復元する facade である。
    # [EXPLAIN] task-local index と global index を混同しないことが row/trajectory 対応を保つ条件になる。
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Route a mixed batch to existing task-specific environment managers."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, managers: Dict[str, EnvironmentManagerBase], task_max_steps: Dict[str, int], config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.managers = managers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.task_max_steps = {task: int(task_max_steps[task]) for task in managers}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._task_indices = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._task_steps = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._task_done = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._last_obs_by_task = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._last_infos_by_task = {}

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        # [EXPLAIN] kwargs の task_name から global index 群を作り、ThreadPoolExecutor で task 別 reset を並行実行する。
        # [EXPLAIN] task manager の戻りを global batch size の container へ merge し、rollout row 順を維持する。
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if kwargs is None:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError("multitask environment requires env_kwargs with task_name for every sample.")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(kwargs, np.ndarray):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kwargs = kwargs.tolist()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_to_items = defaultdict(list)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx, item_kwargs in enumerate(kwargs):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if item_kwargs is None or "task_name" not in item_kwargs:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError("Every multitask env_kwargs entry must contain task_name.")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task = _normalize_multitask_name(item_kwargs["task_name"])
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if task not in self.managers:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"No environment manager configured for task_name={task}.")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            task_to_items[task].append((idx, item_kwargs))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._task_indices = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._task_steps = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._task_done = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_obs = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_infos = {}

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_kwargs_by_task = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task, items in task_to_items.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            indices, task_kwargs = zip(*items)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._task_indices[task] = list(indices)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_kwargs_by_task[task] = list(task_kwargs)

        # Reset all task managers concurrently (env construction/reset is
        # I/O-bound), mirroring the parallel step() so the rollout doesn't
        # serialize per-task startup.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reset_results = {}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(task_kwargs_by_task) == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (task, task_kwargs), = task_kwargs_by_task.items()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reset_results[task] = self.managers[task].reset(task_kwargs)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with ThreadPoolExecutor(max_workers=len(task_kwargs_by_task)) as executor:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                futures = {
                    executor.submit(self.managers[task].reset, task_kwargs): task
                    for task, task_kwargs in task_kwargs_by_task.items()
                }
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for future in as_completed(futures):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    reset_results[futures[future]] = future.result()

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task in task_kwargs_by_task:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            indices = self._task_indices[task]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs, infos = reset_results[task]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for info in infos:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                info["task_name"] = task
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._task_steps[task] = 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._task_done[task] = np.zeros(len(indices), dtype=bool)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._last_obs_by_task[task] = obs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._last_infos_by_task[task] = infos
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_obs[task] = obs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_infos[task] = infos

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observations = self._merge_observations(task_obs, len(kwargs))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infos = self._merge_infos(task_infos, len(kwargs))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return observations, infos

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, text_actions: List[str]):
        # [EXPLAIN] action を task-local 配列へ slice し、task ごとの max_steps と done 状態を考慮して step する。
        # [EXPLAIN] 早期終了済み task は last observation を再利用して short-circuit し、他 task の進行を妨げない。
        # [EXPLAIN] 最後に reward/done/info/success を global order へ戻し、per-task metric の row mask を保持する。
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self._task_indices:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError("MultiTaskEnvironmentManager.step called before reset.")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_obs = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_rewards = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_dones = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_infos = {}

        # Tasks that still need a real environment step (others are short-circuited).
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        active_tasks = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task, indices in self._task_indices.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._task_done[task].all() or self._task_steps[task] >= self.task_max_steps[task]:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                task_obs[task] = self._last_obs_by_task[task]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                task_rewards[task] = np.zeros(len(indices), dtype=np.float32)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                task_dones[task] = np.ones(len(indices), dtype=bool)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                task_infos[task] = self._done_infos(task)
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            active_tasks[task] = [text_actions[idx] for idx in indices]

        # Step active tasks concurrently. Each manager.step is independent and
        # I/O-bound (HTTP for search, Ray/subprocess IPC for alfworld/webshop),
        # so threads overlap them and the per-turn barrier becomes ~max(task)
        # instead of sum(task). Bookkeeping below runs in the main thread to
        # avoid races on the shared state dicts.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        stepped = {}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(active_tasks) == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (task, actions), = active_tasks.items()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            stepped[task] = self.managers[task].step(actions)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif active_tasks:
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with ThreadPoolExecutor(max_workers=len(active_tasks)) as executor:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                futures = {
                    executor.submit(self.managers[task].step, actions): task
                    for task, actions in active_tasks.items()
                }
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for future in as_completed(futures):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    stepped[futures[future]] = future.result()

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task in active_tasks:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            indices = self._task_indices[task]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs, rewards, dones, infos = stepped[task]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rewards = np.asarray(rewards).reshape(-1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dones = np.asarray(dones).reshape(-1).astype(bool)

            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self._task_steps[task] += 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._task_steps[task] >= self.task_max_steps[task]:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                dones = np.ones(len(indices), dtype=bool)

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for info in infos:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                info["task_name"] = task

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._task_done[task] = np.logical_or(self._task_done[task], dones)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._last_obs_by_task[task] = obs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._last_infos_by_task[task] = infos

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_obs[task] = obs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_rewards[task] = rewards
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_dones[task] = dones
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_infos[task] = infos

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observations = self._merge_observations(task_obs, len(text_actions))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rewards = self._merge_arrays(task_rewards, len(text_actions), dtype=np.float32)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dones = self._merge_arrays(task_dones, len(text_actions), dtype=bool)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infos = self._merge_infos(task_infos, len(text_actions))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return observations, rewards, dones, infos

    # [EXPLAIN] `_done_infos` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _done_infos(self, task: str) -> List[Dict]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infos = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for info in self._last_infos_by_task[task]:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            done_info = dict(info)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            done_info["task_name"] = task
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            done_info["is_action_valid"] = to_numpy(True)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            infos.append(done_info)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return infos

    # [EXPLAIN] `_merge_observations` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _merge_observations(self, task_obs: Dict[str, Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keys = set()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for obs in task_obs.values():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            keys.update(obs.keys())

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        merged = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in keys:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            values = [None] * batch_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            has_values = False
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for task, obs in task_obs.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                obs_values = obs.get(key)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if obs_values is None:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                has_values = True
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for idx, value in zip(self._task_indices[task], obs_values):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    values[idx] = value
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            merged[key] = values if has_values else None
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return merged

    # [EXPLAIN] `_merge_infos` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _merge_infos(self, task_infos: Dict[str, List[Dict]], batch_size: int) -> List[Dict]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        merged = [None] * batch_size
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task, infos in task_infos.items():
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for idx, info in zip(self._task_indices[task], infos):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                merged[idx] = info
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return merged

    # [EXPLAIN] `_merge_arrays` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _merge_arrays(self, task_values: Dict[str, np.ndarray], batch_size: int, dtype) -> np.ndarray:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        merged = np.zeros(batch_size, dtype=dtype)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task, values in task_values.items():
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for idx, value in zip(self._task_indices[task], values):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                merged[idx] = value
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return merged

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `_slice_optional_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _slice_optional_batch(value, indices):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if value is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(value, torch.Tensor):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return value[indices]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(value, np.ndarray):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return value[indices]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [value[idx] for idx in indices]

    # [EXPLAIN] `success_evaluator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_infos = kwargs["total_infos"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_batch_list = kwargs["total_batch_list"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        episode_rewards = kwargs.get("episode_rewards")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        episode_lengths = kwargs.get("episode_lengths")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        success = defaultdict(list)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task, indices in self._task_indices.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_total_infos = [total_infos[idx] for idx in indices]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_total_batch_list = [total_batch_list[idx] for idx in indices]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_episode_rewards = self._slice_optional_batch(episode_rewards, indices)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_episode_lengths = self._slice_optional_batch(episode_lengths, indices)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_success = self.managers[task].success_evaluator(
                total_infos=task_total_infos,
                total_batch_list=task_total_batch_list,
                episode_rewards=task_episode_rewards,
                episode_lengths=task_episode_lengths,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            success["success_rate"].extend(task_success["success_rate"].tolist())
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            success[f"{task}_success_rate"].extend(task_success["success_rate"].tolist())
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, value in task_success.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if key == "success_rate":
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                success[f"{task}_{key}"].extend(value.tolist())

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {key: np.array(value) for key, value in success.items()}

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self) -> None:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for manager in self.managers.values():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            manager.close()


# [EXPLAIN] `_get_multitask_tasks` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_multitask_tasks(config) -> List[str]:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    multitask_cfg = config.env.get("multitask", {})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tasks = _plain_container(multitask_cfg.get("tasks", ["alfworld", "search", "webshop"]))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return [_normalize_multitask_name(task) for task in tasks]


# [EXPLAIN] `_get_multitask_task_max_steps` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_multitask_task_max_steps(config, tasks: List[str]) -> Dict[str, int]:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    defaults = {"alfworld": 50, "search": 4, "webshop": 15}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    multitask_cfg = config.env.get("multitask", {})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    configured = _plain_container(multitask_cfg.get("max_steps", {}))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if configured:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        defaults.update({task: int(value) for task, value in configured.items()})
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {task: defaults[task] for task in tasks}


# [EXPLAIN] `_get_multitask_per_task_batch_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_multitask_per_task_batch_size(config, tasks: List[str], is_train: bool) -> int:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_train:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_task_balance = config.data.get("task_balance", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_task_batch_size = data_task_balance.get("per_task_batch_size", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_batch_size = config.data.train_batch_size
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        multitask_cfg = config.env.get("multitask", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_task_batch_size = multitask_cfg.get("val_per_task_batch_size", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_batch_size = config.data.val_batch_size

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if per_task_batch_size is None:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if total_batch_size is None:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError("multitask val_batch_size must be set when val_per_task_batch_size is not configured.")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if int(total_batch_size) % len(tasks) != 0:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"multitask batch size {total_batch_size} is not divisible by {len(tasks)} tasks.")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_task_batch_size = int(total_batch_size) // len(tasks)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return int(per_task_batch_size)


# [EXPLAIN] `_get_multitask_task_history_length` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_multitask_task_history_length(config, task: str):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    multitask_cfg = config.env.get("multitask", {})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    configured = _plain_container(multitask_cfg.get("history_length", {}))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(configured, dict) and task in configured:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return int(configured[task])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return config.env.history_length


# [EXPLAIN] `_copy_config_for_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _copy_config_for_task(config, env_name: str, max_steps: int, task: str = None):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_config.env.env_name = env_name
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_config.env.max_steps = max_steps
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if task is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_config.env.history_length = _get_multitask_task_history_length(config, task)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return task_config


# [EXPLAIN] `_build_multitask_manager` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _build_multitask_manager(config, tasks: List[str], task_max_steps: Dict[str, int], per_task_batch_size: int, group_n: int, is_train: bool, seed: int, resources_per_worker: Dict):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    managers = {}

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for task in tasks:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task == "alfworld":
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            alf_config_path = os.path.join(os.path.dirname(__file__), "env_package/alfworld/configs/config_tw.yaml")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            env_kwargs = {"eval_dataset": config.env.alfworld.eval_dataset}
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_config = _copy_config_for_task(config, "alfworld/AlfredTWEnv", task_max_steps[task], task=task)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _envs = build_alfworld_envs(
                alf_config_path,
                seed,
                per_task_batch_size,
                group_n,
                resources_per_worker=resources_per_worker,
                is_train=is_train,
                env_kwargs=env_kwargs,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            managers[task] = AlfWorldEnvironmentManager(_envs, partial(alfworld_projection), task_config)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif task == "search":
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from agent_system.environments.env_package.search import build_search_envs, search_projection

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_config = _copy_config_for_task(config, "search", task_max_steps[task], task=task)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _envs = build_search_envs(
                seed=seed,
                env_num=per_task_batch_size,
                group_n=group_n,
                is_train=is_train,
                env_config=task_config.env,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            managers[task] = SearchEnvironmentManager(_envs, partial(search_projection), task_config)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif task == "webshop":
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.env.webshop.use_small:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                file_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_shuffle_1000.json")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                attr_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_ins_v2_1000.json")
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                file_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_shuffle.json")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                attr_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_ins_v2.json")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            env_kwargs = {
                "observation_mode": "text",
                "num_products": None,
                "human_goals": config.env.webshop.human_goals,
                "file_path": file_path,
                "attr_path": attr_path,
            }
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_config = _copy_config_for_task(config, "Webshop", task_max_steps[task], task=task)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _envs = build_webshop_envs(
                seed=seed,
                env_num=per_task_batch_size,
                group_n=group_n,
                is_train=is_train,
                env_kwargs=env_kwargs,
                resources_per_worker=resources_per_worker,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            managers[task] = WebshopEnvironmentManager(_envs, partial(webshop_projection), task_config)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unsupported multitask task: {task}")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return MultiTaskEnvironmentManager(managers=managers, task_max_steps=task_max_steps, config=config)


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
    if config.env.env_name.lower() == "multitask":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tasks = _get_multitask_tasks(config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_max_steps = _get_multitask_task_max_steps(config, tasks)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        train_per_task_batch_size = _get_multitask_per_task_batch_size(config, tasks, is_train=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_per_task_batch_size = _get_multitask_per_task_batch_size(config, tasks, is_train=False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if train_per_task_batch_size * len(tasks) != int(config.data.train_batch_size):
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(
                "multitask train batch mismatch: "
                f"{train_per_task_batch_size} * {len(tasks)} != {config.data.train_batch_size}"
            )
        # Validation supports two layouts:
        #   - per-task batches: val_batch_size == val_per_task_batch_size, the task-sorted
        #     test parquet yields one single-task batch per task (each task is evaluated
        #     in its own rollout pass)
        #   - mixed batch: val_batch_size == val_per_task_batch_size * len(tasks)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_batch_size = int(config.data.val_batch_size)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if val_batch_size not in (val_per_task_batch_size, val_per_task_batch_size * len(tasks)):
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(
                "multitask val batch mismatch: val_batch_size must be "
                f"{val_per_task_batch_size} (per-task validation batches) or "
                f"{val_per_task_batch_size * len(tasks)} (single mixed batch), got {val_batch_size}"
            )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        envs = _build_multitask_manager(
            config=config,
            tasks=tasks,
            task_max_steps=task_max_steps,
            per_task_batch_size=train_per_task_batch_size,
            group_n=group_n,
            is_train=True,
            seed=config.env.seed,
            resources_per_worker=resources_per_worker,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = _build_multitask_manager(
            config=config,
            tasks=tasks,
            task_max_steps=task_max_steps,
            per_task_batch_size=val_per_task_batch_size,
            group_n=1,
            is_train=False,
            seed=config.env.seed + 1000,
            resources_per_worker=resources_per_worker,
        )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "webshop" in tasks:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import time

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            time.sleep((train_per_task_batch_size * group_n + val_per_task_batch_size) * 0.1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "search" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(search_projection)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "gym_cards" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "alfworld" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(alfworld_projection)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
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
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(sokoban_projection)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return envs, val_envs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "webshop" in config.env.env_name.lower():
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.env.webshop.use_small:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(webshop_projection)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
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
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        projection_f = partial(appworld_projection)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
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
