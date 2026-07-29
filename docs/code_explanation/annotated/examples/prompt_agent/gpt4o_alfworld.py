# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from datetime import datetime
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.environments.env_manager import *
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from openai import OpenAI

# [EXPLAIN] `build_env` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def build_env(env_name, env_num=1):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group_n = 1
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if env_name == "alfworld":
        # Test AlfWorldEnvironmentManager
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.alfworld import alfworld_projection
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from agent_system.environments.env_package.alfworld import build_alfworld_envs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_kwargs = {
            'eval_dataset': "eval_in_distribution", # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, 'alfworld/AlfredThorEnv')
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Unsupported environment name: {env_name}")
    
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return env_manager

# [EXPLAIN] `Agent` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Agent:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, model_name="gpt-4o"):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_name = model_name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
        )
        
    # [EXPLAIN] `get_action_from_gpt` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_action_from_gpt(self, obs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user", 
                    "content": obs
                }
            ],
            temperature=0.4,
            n=1,
            stop=None
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action = response.choices[0].message.content.strip()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return action

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":

    # -------- logging ----------
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs("logs/alfworld", exist_ok=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_fp = os.path.join(
        "logs/alfworld", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_steps = 50
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    env_num = 200
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_times = 3
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    env_name = "alfworld" 

    # Keywords for 6 subtasks
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    TASKS = [
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ]

    # -------- Environment and agent setup ----------
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    env_manager = build_env(env_name, env_num)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    agent = Agent()

    # Accumulated statistics
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    overall_success_rates = []         # Overall success per round
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_success_history = defaultdict(list)  # Subtask success per round

    # ======================= Main Loop =======================
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for test_idx in range(test_times):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logging.info(f"\n========== Start test {test_idx} ==========")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        start_time = time.time()
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kwargs = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs, infos = env_manager.reset(kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_dones = [False] * env_num

        # Statistics for single round
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        overall_success_this_round = np.zeros(env_num, dtype=bool)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_success_cnt = defaultdict(int)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_total_cnt = defaultdict(int)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for step_idx in range(max_steps):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logging.info(f"Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{env_num}); SR {overall_success_this_round.mean().item()}")

            # --- Assemble actions ---
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actions = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(env_num):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if env_dones[i]:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    actions.append("None")
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    actions.append(agent.get_action_from_gpt(obs["text"][i]))

            # --- Environment stepping ---
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs, rewards, dones, infos = env_manager.step(actions)

            # --- Determine endings and successes ---
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(env_num):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if env_dones[i]:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if dones[i]:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    env_dones[i] = True
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    won = bool(infos[i].get("won", False))
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    overall_success_this_round[i] = won

                    # Parse task type
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    gamefile = infos[i].get("extra.gamefile", "")
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    matched = False
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for task in TASKS:
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if task in gamefile:
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            task_total_cnt[task] += 1
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if won:
                                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                                task_success_cnt[task] += 1
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            matched = True
                            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                            break
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if not matched:
                        # Unrecognized tasks are also counted in total
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        task_total_cnt["other"] += 1
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if won:
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            task_success_cnt["other"] += 1

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if all(env_dones):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logging.info("All environments finished early!")
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break

        # -------- Single round results --------
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        round_success_rate = overall_success_this_round.mean()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        overall_success_rates.append(round_success_rate)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logging.info(f"Test {test_idx} overall success: {round_success_rate:.4f}")

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task in TASKS + ["other"]:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if task_total_cnt.get(task, 0) > 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rate = task_success_cnt[task] / task_total_cnt[task]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                task_success_history[task].append(rate)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logging.info(
                    f"    {task:<35s}: {rate:.4f} "
                    f"({task_success_cnt[task]}/{task_total_cnt[task]})"
                )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logging.info(
            f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n"
        )

    # ======================= Final Summary =======================
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logging.info("=============== Final Summary ===============")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logging.info(
        f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {env_num * test_times}"
    )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logging.info(
        f"Overall success avg ± std: "
        f"{np.mean(overall_success_rates):.4f} ± {np.std(overall_success_rates):.4f}"
    )

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for task in TASKS + ["other"]:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_success_history.get(task):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logging.info(
                f"{task:<35s}: "
                f"{np.mean(task_success_history[task]):.4f} ± "
                f"{np.std(task_success_history[task]):.4f}"
            )
