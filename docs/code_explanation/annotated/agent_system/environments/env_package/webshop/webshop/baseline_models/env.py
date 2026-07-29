# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from os.path import join, dirname, abspath
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict

# connect to WebShop env
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MODEL_PATH = dirname(abspath(__file__))
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SITE_PATH = join(MODEL_PATH, '../')
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
sys.path.insert(0, SITE_PATH)

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.envs import WebAgentTextEnv
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.utils import *
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.engine.goal import get_reward


# [EXPLAIN] `WebEnv` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class WebEnv:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    ''' A wrapper of textEnv for models. Returns valid actions at each step of the game. '''

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, args, split, server=None, id=None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.env = WebAgentTextEnv(observation_mode=args.state_format, server=server,
                                   filter_goals=None, limit_goals=-1,
                                   num_products=args.num, human_goals=args.human_goals,
                                   get_image=args.get_image,
                                   num_prev_obs=args.num_prev_obs, num_prev_actions=args.num_prev_actions,
                                   session_prefix=id)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if args.num is None:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if split == 'test':
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.goal_idxs = range(500)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif split == 'eval':
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.goal_idxs = range(500, 1500)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif split == 'train':
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.goal_idxs = range(1500, len(self.env.server.goals))
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.goal_idxs = range(len(self.env.server.goals))
            
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(self.goal_idxs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.steps = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.step_limit = args.step_limit
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.stats = defaultdict(int)  # kept across episodes
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.session = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.click_item_name = args.click_item_name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.asin2name = {k.lower(): v['Title'].lower(
        ) for k, v in self.env.server.product_item_dict.items()}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name2asin = {v: k for k, v in self.asin2name.items()}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.attributes_fail = defaultdict(int)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.attributes_success = defaultdict(int)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.items_clicked = defaultdict(int)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.harsh_reward = args.harsh_reward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.go_to_item = args.go_to_item
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.go_to_search = args.go_to_search
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ban_buy = args.ban_buy
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prev_ob = self.cur_ob = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.get_image = args.get_image
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.item_rank = -1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reduce_click = 1

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if args.extra_search_path != "":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.extra_search = json.load(open(args.extra_search_path))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.extra_search = {
                k.strip("."): v for k, v in self.extra_search.items()}
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.extra_search = None

    # [EXPLAIN] `get_search_texts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_search_texts(self, atts, query, inst):
        # TODO: make it more complicated, or replace it with free-form generation
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.extra_search is not None:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if ", and price lower than" in inst:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                idx = inst.find(", and price lower than")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                inst_ = inst[:idx]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                inst_ = inst
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            texts = self.extra_search.get(inst_, []) + [inst.lower()]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            texts = [query] + \
                [f'{att} {query}' for att in atts] + [inst.lower()]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return texts

    # [EXPLAIN] `get_valid_actions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_valid_actions(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_info = self.env.get_available_actions()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if valid_info['has_search_bar']:  # only search action available
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            atts = self.session['goal']['attributes']
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            query = self.session['goal']['query']
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            inst = self.session['goal']['instruction_text']
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            texts = self.get_search_texts(atts, query, inst)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valids = [f'search[{text}]' for text in texts]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valids = []  # and text.startswith('b')]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for text in valid_info['clickables']:
                # ban buy when options not completed
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if text == 'buy now' and self.ban_buy:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    cur_options = len(self.session['options'])
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    all_options = len(
                        self.env.server.product_item_dict[self.session["asin"]]['customization_options'])
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if cur_options != all_options:
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        continue
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if text != 'search':
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.click_item_name and text in self.asin2name:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        text = 'item - ' + self.asin2name[text]
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    valids.append(f'click[{text}]')
                # do some action space reduction...
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.reduce_click and len(valids) > 20:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    valids = valids[:6] + random.sample(valids[6:], 10)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(valids) == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valids = ['finish']
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return valids

    # [EXPLAIN] `score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def score(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Calculate the score of the current state.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_acts = self.get_valid_actions()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'click[description]' not in valid_acts:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 0.0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        product = self.env.server.product_item_dict[self.session["asin"]]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal = self.session['goal']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        price = self.env.server.product_prices.get(self.session["asin"])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        options = self.session['options']
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return get_reward(product, goal, price, options)

    # [EXPLAIN] `estimate_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def estimate_score(self, atts, opts, verify=False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Calculate the score of the current state.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_acts = self.get_valid_actions()
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert 'click[description]' in valid_acts
        # estimate r_att
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc = self.step('click[description]')[0].lower()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.step('click[< prev]')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        feat = self.step('click[features]')[0].lower()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ob = self.step('click[< prev]')[0].lower()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_att = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for att in atts:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if att in desc or att in feat or att in ob:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                n_att += 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        r_att = n_att / len(atts)
        # estimate r_opt
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_opt = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for opt in opts:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for act in valid_acts:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if opt in act:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    n_opt += 1
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    break
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        r_opt = n_opt / len(opts)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        r = (n_att + n_opt + 1) / (len(atts) + len(opts) + 1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return r, r_att, r_opt

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.click_item_name and action.startswith('click[item - ') and action[13:-1] in self.name2asin:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_items = [_ for _ in self.get_valid_actions()
                           if _.startswith('click[item - ')]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if action in valid_items:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.item_rank = valid_items.index(action) + 1
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.item_rank = -1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            action = f'click[{self.name2asin[action[13:-1]]}]'

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ob, reward, done, info = self.env.step(action)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action.startswith('click[') and action[6:-1] in self.asin2name:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.items_clicked[action[6:-1]] += 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            desc = self.env.step('click[description]')[0].lower()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.env.step('click[< prev]')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            feat = self.env.step('click[features]')[0].lower()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.env.step('click[< prev]')
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            desc = feat = ''
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        r_visit = 0.0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.cur_ob, self.prev_ob = ob, self.cur_ob
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if info is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            info = {}
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.steps += 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.step_limit and self.steps >= self.step_limit:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            done = True
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if done:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info['verbose'] = self.session.get('verbose_info', {
                                               'r_att': 0.0, 'r_option': 0.0, 'r_price': 0.0, 'r_type': 0.0, 'w_att': 0.0, 'w_option': 0.0, 'w_price': 0.0})
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            verbose = info['verbose']
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            verbose['r_harsh'] = (reward == 1)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            verbose['r_exact'] = (reward == 1) and (
                self.session['goal']['asin'] == self.session['asin'])
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            verbose['r_norm'] = reward / self.steps
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            verbose['r_visit'] = r_visit
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            verbose['rank_item'] = self.item_rank
            # log reward with respect to #options
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.harsh_reward:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reward = verbose['r_harsh']
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k, v in self.session['actions'].items():
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                self.stats[f'action_{k}'] += v
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cat = self.session['goal']['category']
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.stats[f'cat_{cat}'] += 1
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for att in self.session['goal']['attributes']:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if att in info['verbose'].get('purchased_attrs', []):
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    self.attributes_success[att] += 1
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    self.attributes_fail[att] += 1

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        info.update({'valid': self.get_valid_actions(), 'goal': self.env.instruction_text,
                     'score': reward * 10, 'estimate_score': self.score(),
                     'prev_ob': self.prev_ob, 'desc': desc, 'feat': feat
                     })

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.get_image:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image_feat = self.env.get_image()
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            info['image_feat'] = image_feat

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ob, (reward + r_visit) * 10, done, info

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, idx=None):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if idx is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idx = random.sample(self.goal_idxs, k=1)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ob, info = self.env.reset(idx)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.session = self.env.server.user_sessions[self.env.session]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if info is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            info = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.cur_ob, self.prev_ob = ob, None
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        info.update({'valid': self.get_valid_actions(), 'goal': self.env.instruction_text,
                     'score': 0, 'estimate_score': self.score(),
                     'prev_ob': self.prev_ob, 'desc': '', 'feat': ''
                     })
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.steps = 0
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.go_to_search or self.go_to_item:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            name = self.session['goal']['name'].lower()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ob, _, _, info = self.step(f'search[{name}]')
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.stats['action_go_to_search'] += 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.go_to_item:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                asin = self.session['goal']['asin'].lower()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if asin in self.env.get_available_actions()['clickables']:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    ob, _, _, info = self.step(f'click[{asin}]')
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    self.stats['action_go_to_item'] += 1

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.item_rank = -1
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ob, info

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.env.close()
