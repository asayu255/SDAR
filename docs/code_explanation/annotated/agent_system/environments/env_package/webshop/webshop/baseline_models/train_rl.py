# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import argparse
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logger
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent import Agent, TransitionPG
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from env import WebEnv

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logging.getLogger().setLevel(logging.CRITICAL)


# [EXPLAIN] `configure_logger` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def configure_logger(log_dir, wandb):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logger.configure(log_dir, format_strs=['log'])
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    global tb
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    type_strs = ['json', 'stdout']
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if wandb: type_strs += ['wandb']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tb = logger.Logger(log_dir, [logger.make_output_format(type_str, log_dir) for type_str in type_strs])
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    global log
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log = logger.log


# [EXPLAIN] `evaluate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def evaluate(agent, env, split, nb_episodes=10):
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_score = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for method in ['greedy']:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for ep in range(nb_episodes):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log("Starting {} episode {}".format(split, ep))
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if split == 'eval':
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    score = evaluate_episode(agent, env, split, method)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif split == 'test':
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    score = evaluate_episode(agent, env, split, method, idx=ep)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log("{} episode {} ended with score {}\n\n".format(split, ep, score))
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                total_score += score
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        avg_score = total_score / nb_episodes
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return avg_score


# [EXPLAIN] `evaluate_episode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def evaluate_episode(agent, env, split, method='greedy', idx=None):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    step = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    done = False
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ob, info = env.reset(idx)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state = agent.build_state(ob, info)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    log('Obs{}: {}'.format(step, ob.encode('utf-8')))
    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
    while not done:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_acts = info['valid']
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.no_grad():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            action_str = agent.act([state], [valid_acts], method=method)[0][0]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log('Action{}: {}'.format(step, action_str))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ob, rew, done, info = env.step(action_str)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        step += 1
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log('Obs{}: {}'.format(step, ob.encode('utf-8')))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state = agent.build_state(ob, info)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    tb.logkv_mean(f'{split}Score', info['score'])
    # category = env.session['goal']['category']
    # tb.logkv_mean(f'{split}Score_{category}', rew)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if 'verbose' in info:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k, v in info['verbose'].items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if k.startswith('r'):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                tb.logkv_mean(f'{split}_' + k, v)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return info['score']


# [EXPLAIN] `agg` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def agg(envs, attr):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    res = defaultdict(int)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for env in envs:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k, v in getattr(env, attr).items():
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            res[k] += v
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return res


# [EXPLAIN] `train` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def train(agent, eval_env, test_env, envs, args):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    start = time.time()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    states, valids, transitions = [], [], []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state0 = None
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for env in envs:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ob, info = env.reset()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if state0 is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state0 = (ob, info)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        states.append(agent.build_state(ob, info))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        valids.append(info['valid'])

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for step in range(1, args.max_steps + 1):
        # get actions from policy
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_strs, action_ids, values = agent.act(states, valids, method=args.exploration_method)
        
        # log envs[0]
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.no_grad():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            action_values, _ = agent.network.rl_forward(states[:1], agent.encode_valids(valids[:1]))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions = sorted(zip(state0[1]['valid'], action_values.tolist()), key=lambda x: - x[1])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log('State  {}: {}'.format(step, state0[0].lower().encode('utf-8')))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log('Goal   {}: {}'.format(step, state0[1]['goal'].lower().encode('utf-8')))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log('Actions{}: {}'.format(step, actions))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log('>> Values{}: {}'.format(step, float(values[0])))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log('>> Action{}: {}'.format(step, action_strs[0]))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state0 = None

        # step in envs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_states, next_valids, rewards, dones = [], [], [], []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for env, action_str, action_id, state in zip(envs, action_strs, action_ids, states):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ob, reward, done, info = env.step(action_str)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if state0 is None:  # first state
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                state0 = (ob, info)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                r_att = r_opt = 0
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if 'verbose' in info:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    r_att = info['verbose'].get('r_att', 0)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    r_option = info['verbose'].get('r_option ', 0)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    r_price = info['verbose'].get('r_price', 0)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    r_type = info['verbose'].get('r_type', 0)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    w_att = info['verbose'].get('w_att', 0)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    w_option = info['verbose'].get('w_option', 0)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    w_price = info['verbose'].get('w_price', 0)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    reward_str = f'{reward/10:.2f} = ({r_att:.2f} * {w_att:.2f} + {r_option:.2f} * {w_option:.2f} + {r_price:.2f} * {w_price:.2f}) * {r_type:.2f}'
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    reward_str = str(reward)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log('Reward{}: {}, Done {}\n'.format(step, reward_str, done))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            next_state = agent.build_state(ob, info)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            next_valid = info['valid']
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            next_states, next_valids, rewards, dones = \
                next_states + [next_state], next_valids + [next_valid], rewards + [reward], dones + [done]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if done:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                tb.logkv_mean('EpisodeScore', info['score'])
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                category = env.session['goal']['category']
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                tb.logkv_mean(f'EpisodeScore_{category}', info['score'])
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if 'verbose' in info:
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for k, v in info['verbose'].items():
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if k.startswith('r'):
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            tb.logkv_mean(k, v)

        # RL update
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        transitions.append(TransitionPG(states, action_ids, rewards, values, agent.encode_valids(valids), dones))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(transitions) >= args.bptt:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _, _, last_values = agent.act(next_states, next_valids, method='softmax')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            stats = agent.update(transitions, last_values, step=step)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k, v in stats.items():
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                tb.logkv_mean(k, v)
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            del transitions[:]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.empty_cache()

        # handle done
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, env in enumerate(envs):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if dones[i]:               
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ob, info = env.reset()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if i == 0:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    state0 = (ob, info)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                next_states[i] = agent.build_state(ob, info)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                next_valids[i] = info['valid']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        states, valids = next_states, next_valids

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if step % args.eval_freq == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            evaluate(agent, eval_env, 'eval')

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if step % args.test_freq == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            evaluate(agent, test_env, 'test', 500)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if step % args.log_freq == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tb.logkv('Step', step)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tb.logkv('FPS', int((step * len(envs)) / (time.time() - start)))
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k, v in agg(envs, 'stats').items():
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                tb.logkv(k, v)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            items_clicked = agg(envs, 'items_clicked')
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tb.logkv('ItemsClicked', len(items_clicked))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tb.dumpkvs()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if step % args.ckpt_freq == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            agent.save()


# [EXPLAIN] `parse_args` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parse_args():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parser = argparse.ArgumentParser()
    # logging
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--seed', default=0, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--output_dir', default='logs')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--ckpt_freq', default=10000, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--eval_freq', default=500, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--test_freq', default=5000, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--log_freq', default=100, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--wandb', default=1, type=int)

    # rl
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--num_envs', default=4, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--step_limit', default=100, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--max_steps', default=300000, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--gamma', default=.9, type=float)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--clip', default=10, type=float)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--bptt', default=8, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--exploration_method', default='softmax', type=str, choices=['eps', 'softmax'])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--w_pg', default=1, type=float)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--w_td', default=1, type=float)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--w_il', default=0, type=float)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--w_en', default=1, type=float)

    # model
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--network', default='bert', type=str, choices=['bert', 'rnn'])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--bert_path', default="", type=str, help='which bert to load')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--embedding_dim', default=128, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--hidden_dim', default=128, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--grad_encoder', default=1, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--get_image', default=1, type=int, help='use image in models')

    # env
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--num', default=None, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--click_item_name', default=1, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--state_format', default='text_rich', type=str)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--human_goals', default=1, type=int, help='use human goals')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--num_prev_obs', default=0, type=int, help='number of previous observations')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--num_prev_actions', default=0, type=int, help='number of previous actions')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--extra_search_path', default="./data/goal_query_predict.json", type=str, help='path for extra search queries')
    

    # experimental 
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--ban_buy', default=0, type=int, help='ban buy action before selecting options')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--score_handicap', default=0, type=int, help='provide score in state')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--go_to_item', default=0, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--go_to_search', default=0, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--harsh_reward', default=0, type=int)


    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument('--debug', default=0, type=int, help='debug mode')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--f", help="a dummy argument to fool ipython", default="1")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return parser.parse_known_args()


# [EXPLAIN] `main` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args, unknown = parse_args()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if args.debug:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        args.num_envs = 2
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        args.wandb = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        args.human_goals = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        args.num = 100
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(unknown)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(args)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    configure_logger(args.output_dir, args.wandb)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    agent = Agent(args)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    train_env = WebEnv(args, split='train', id='train_')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    server = train_env.env.server
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    eval_env = WebEnv(args, split='eval', id='eval_', server=server)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_env = WebEnv(args, split='test', id='test_', server=server)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    envs = [WebEnv(args, split='train', server=server, id=f'train{i}_') for i in range(args.num_envs)]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("loaded")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    train(agent, eval_env, test_env, envs, args)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
