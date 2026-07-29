# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn as nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoTokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict, namedtuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from models.bert import BertConfigForWebshop, BertModelForWebshop
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from models.rnn import RCDQN

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
State = namedtuple('State', ('obs', 'goal', 'click', 'estimate', 'obs_str', 'goal_str', 'image_feat'))
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
TransitionPG = namedtuple('TransitionPG', ('state', 'act', 'reward', 'value', 'valid_acts', 'done'))


# [EXPLAIN] `discount_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def discount_reward(transitions, last_values, gamma):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    returns, advantages = [], []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    R = last_values.detach()  # always detached
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for t in reversed(range(len(transitions))):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, _, rewards, values, _, dones = transitions[t]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        R = torch.FloatTensor(rewards).to(device) + gamma * R * (1 - torch.FloatTensor(dones).to(device))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        baseline = values
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        adv = R - baseline
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        returns.append(R)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        advantages.append(adv)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return returns[::-1], advantages[::-1]


# [EXPLAIN] `Agent` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Agent:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, args):
        # tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side='left', max_length=512)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.tokenizer.add_tokens(['[button], [button_], [clicked button], [clicked button_]'], special_tokens=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vocab_size = len(self.tokenizer)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        embedding_dim = args.embedding_dim

        # network
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if args.network == 'rnn':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.network = RCDQN(vocab_size, embedding_dim, 
                args.hidden_dim, args.arch_encoder, args.grad_encoder, None, args.gru_embed, args.get_image, args.bert_path)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.network.rl_forward = self.network.forward
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif args.network == 'bert':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config = BertConfigForWebshop(image=args.get_image, pretrained_bert=(args.bert_path != 'scratch'))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.network = BertModelForWebshop(config)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if args.bert_path != '' and args.bert_path != 'scratch':
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.network.load_state_dict(torch.load(args.bert_path, map_location=torch.device('cpu')), strict=False)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError('Unknown network: {}'.format(args.network))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.network = self.network.to(device)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.save_path = args.output_dir
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.clip = args.clip
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.w = {'loss_pg': args.w_pg, 'loss_td': args.w_td, 'loss_il': args.w_il, 'loss_en': args.w_en}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.learning_rate)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.gamma = args.gamma

    # [EXPLAIN] `build_state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def build_state(self, ob, info):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """ Returns a state representation built from various info sources. """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_ids = self.encode(ob)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal_ids = self.encode(info['goal'])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        click = info['valid'][0].startswith('click[')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        estimate = info['estimate_score']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_str = ob.replace('\n', '[SEP]')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal_str = info['goal']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_feat = info.get('image_feat')
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return State(obs_ids, goal_ids, click, estimate, obs_str, goal_str, image_feat)


    # [EXPLAIN] `encode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def encode(self, observation, max_length=512):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """ Encode an observation """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observation = observation.lower().replace('"', '').replace("'", "").strip()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observation = observation.replace('[sep]', '[SEP]')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_ids = self.tokenizer.encode(observation, truncation=True, max_length=max_length)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return token_ids

    # [EXPLAIN] `decode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def decode(self, act):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act = self.tokenizer.decode(act, skip_special_tokens=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act = act.replace(' [ ', '[').replace(' ]', ']')
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return act
    
    # [EXPLAIN] `encode_valids` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def encode_valids(self, valids, max_length=64):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """ Encode a list of lists of strs """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [[self.encode(act, max_length=max_length) for act in valid] for valid in valids]


    # [EXPLAIN] `act` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def act(self, states, valid_acts, method, state_strs=None, eps=0.1):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """ Returns a string action from poss_acts. """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act_ids = self.encode_valids(valid_acts)

        # sample actions
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act_values, act_sizes, values = self.network.rl_forward(states, act_ids, value=True, act=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act_values = act_values.split(act_sizes)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if method == 'softmax':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            act_probs = [F.softmax(vals, dim=0) for vals in act_values]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            act_idxs = [torch.multinomial(probs, num_samples=1).item() for probs in act_probs]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif method == 'greedy':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            act_idxs = [vals.argmax(dim=0).item() for vals in act_values]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif method == 'eps': # eps exploration
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            act_idxs = [vals.argmax(dim=0).item() if random.random() > eps else random.randint(0, len(vals)-1) for vals in act_values]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        acts = [acts[idx] for acts, idx in zip(act_ids, act_idxs)]

        # decode actions
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act_strs, act_ids = [], []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for act, idx, valids in zip(acts, act_idxs, valid_acts):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if torch.is_tensor(act):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                act = act.tolist()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 102 in act:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                act = act[:act.index(102) + 1]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            act_ids.append(act)  # [101, ..., 102]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if idx is None:  # generative
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                act_str = self.decode(act)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:  # int
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                act_str = valids[idx]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            act_strs.append(act_str)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return act_strs, act_ids, values
    

    # [EXPLAIN] `update` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update(self, transitions, last_values, step=None, rewards_invdy=None):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        returns, advs = discount_reward(transitions, last_values, self.gamma)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        stats_global = defaultdict(float)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for transition, adv in zip(transitions, advs):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            stats = {}
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            log_valid, valid_sizes = self.network.rl_forward(transition.state, transition.valid_acts)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            act_values = log_valid.split(valid_sizes)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            log_a = torch.stack([values[acts.index(act)]
                                        for values, acts, act in zip(act_values, transition.valid_acts, transition.act)])

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['loss_pg'] = - (log_a * adv.detach()).mean()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['loss_td'] = adv.pow(2).mean()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['loss_il'] = - log_valid.mean()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['loss_en'] = (log_valid * log_valid.exp()).mean()
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k in stats:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                stats[k] = self.w[k] * stats[k] / len(transitions)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['loss'] = sum(stats[k] for k in stats)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['returns'] = torch.stack(returns).mean() / len(transitions)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['advs'] = torch.stack(advs).mean() / len(transitions)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['loss'].backward()

            # Compute the gradient norm
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['gradnorm_unclipped'] = sum(p.grad.norm(2).item() for p in self.network.parameters() if p.grad is not None)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            stats['gradnorm_clipped'] = sum(p.grad.norm(2).item() for p in self.network.parameters() if p.grad is not None)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k, v in stats.items():
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                stats_global[k] += v.item() if torch.is_tensor(v) else v
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            del stats
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.optimizer.step()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.optimizer.zero_grad()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return stats_global

    # [EXPLAIN] `load` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load(self):
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.network = torch.load(os.path.join(self.save_path, 'model.pt'))
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Error saving model.", e)

    # [EXPLAIN] `save` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save(self):
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.save(self.network, os.path.join(self.save_path, 'model.pt'))
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Error saving model.", e)
