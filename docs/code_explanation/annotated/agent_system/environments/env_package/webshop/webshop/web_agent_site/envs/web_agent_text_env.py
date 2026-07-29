# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import gym
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import string
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from bs4 import BeautifulSoup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from bs4.element import Comment
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from flask import Flask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    get_top_n_product_from_keywords,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    ACTION_TO_TEMPLATE,
    END_BUTTON, NEXT_PAGE, PREV_PAGE, BACK_TO_SEARCH,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.engine.goal import get_reward, get_goals
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.utils import (
    DEFAULT_FILE_PATH,
    DEFAULT_ATTR_PATH,
    FEAT_CONV,
    FEAT_IDS,
    random_idx
)

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
app = Flask(__name__)
# [EXPLAIN] `WebAgentTextEnv` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class WebAgentTextEnv(gym.Env):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Gym environment for Text mode of WebShop environment"""
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
            self,
            observation_mode='html',
            file_path=DEFAULT_FILE_PATH,
            attr_path=DEFAULT_ATTR_PATH,
            server=None,
            **kwargs
        ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Constructor for text environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        get_image
        filter_goals
        limit_goals
        num_products
        human_goals
        session
        session_prefix
        show_attrs
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super(WebAgentTextEnv, self).__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.observation_mode = observation_mode
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.kwargs = kwargs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._seed = kwargs.get('seed', 42)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        random.seed(self._seed)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.random.seed(self._seed)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.manual_seed(self._seed)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.file_path = file_path
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.attr_path = attr_path

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.base_url = 'http://127.0.0.1:3000'
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.server = SimServer(
            self._seed,
            self.base_url,
            self.file_path,
            self.attr_path,
            self.kwargs.get('filter_goals'),
            self.kwargs.get('limit_goals', -1),
            self.kwargs.get('num_products'),
            self.kwargs.get('human_goals'),
            self.kwargs.get('show_attrs', False),
        ) if server is None else server
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.browser = SimBrowser(self.server)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.session = self.kwargs.get('session')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.session_prefix = self.kwargs.get('session_prefix')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.kwargs.get('get_image', 0):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.feats = torch.load(FEAT_CONV)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ids = torch.load(FEAT_IDS)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ids = {url: idx for idx, url in enumerate(self.ids)}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prev_obs = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prev_actions = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_prev_obs = self.kwargs.get('num_prev_obs', 0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_prev_actions = self.kwargs.get('num_prev_actions', 0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.reset()

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = None
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.get_available_actions()

        # Determine action type (click, search) and argument
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_name, action_arg = parse_action(action)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action_arg is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            action_arg = action_arg.lower()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if (action_name == 'search' and 
            action_arg is not None and 
            action_arg != ''):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            status = self.browser.search(action_arg)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif (action_name == 'click' and 
              action_arg in self.text_to_clickable.keys() and 
              action_arg != 'search'):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            status = self.browser.click(action_arg, self.text_to_clickable)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            status = dict(reward=0, done=False)

        # Update observation, state with the new action
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ob = self.observation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_list = [ob]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.prev_actions.append(action)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                text_list.append(self.prev_actions[-i])
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                text_list.append(self.prev_obs[-i])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state = ' [SEP] '.join(text_list[::-1])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.prev_obs.append(ob)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if status['done']:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.reset()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return state, status['reward'], status['done'], info

    # [EXPLAIN] `get_available_actions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_available_actions(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Returns list of available actions at the current step"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        search_bar = html_obj.find(id='search_input')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        has_search_bar = True if search_bar is not None else False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buttons = html_obj.find_all(class_='btn')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        product_links  = html_obj.find_all(class_='product-link')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buying_options = html_obj.select('input[type="radio"]')

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.text_to_clickable = {
            f'{b.get_text()}'.lower(): b
            for b in buttons + product_links
        }
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for opt in buying_options:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            opt_value = opt.get('value')
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.text_to_clickable[f'{opt_value}'] = opt
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )
    
    # [EXPLAIN] `get_image` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_image(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Scrape image from page HTML and return as a list of pixel values"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html_obj = self._parse_html(self.browser.page_source)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_url = html_obj.find(id='product-image')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if image_url is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image_url = image_url['src']
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if image_url in self.ids:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                image_idx = self.ids[image_url]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                image = self.feats[image_idx]
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return image
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.zeros(512)

    # [EXPLAIN] `get_instruction_text` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_instruction_text(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get corresponding instruction text for current environment session"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html_obj = self._parse_html(self.browser.page_source)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        instruction_text = html_obj.find(id='instruction-text').h4.text
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return instruction_text

    # [EXPLAIN] `_parse_html` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _parse_html(self, html=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if html is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            html = self.state['html']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html_obj = BeautifulSoup(html, 'html.parser')
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return html_obj
    
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `observation` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def observation(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Compiles state into either the `html` or `text` observation mode"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = self.state['html']
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.observation_mode == 'html':
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return html
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.observation_mode == 'text':
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.convert_html_to_text(html, simple=True)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.observation_mode == 'text_rich':
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.convert_html_to_text(html, simple=False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.observation_mode == 'url':
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.state['url']
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )
    
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def state(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )
    
    # [EXPLAIN] `convert_html_to_text` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def convert_html_to_text(self, html, simple=False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        texts = self._parse_html(html).findAll(text=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        visible_texts = filter(tag_visible, texts)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if simple:
            # For `simple` mode, return just [SEP] separators
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            observation = ''
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for t in visible_texts:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if t == '\n': continue
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if t.parent.name == 'button':  # button
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    processed_t = f'[button] {t} [button_]'
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif t.parent.name == 'label':  # options
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if f'"{t}"' in self.state['url']:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        processed_t = f'  [clicked button] {t} [clicked button_]'
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        observation = f'You have clicked {t}.\n' + observation
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        processed_t = f'  [button] {t} [button_]'
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif t.parent.get('class') == ["product-link"]: # product asins
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if f'{t}' in self.server.user_sessions[self.session]['asins']:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        processed_t = f'\n[clicked button] {t} [clicked button_]'
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        processed_t = f'\n[button] {t} [button_]'
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else: # regular, unclickable text
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    processed_t =  str(t)
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                observation += processed_t + '\n'
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return observation
    
    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, session=None, instruction_text=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create a new session and reset environment variables"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        session_int = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if session is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.session = str(session)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(session, int):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                session_int = session
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.session = ''.join(random.choices(string.ascii_lowercase, k=10))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.session_prefix is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.session = self.session_prefix + self.session

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        init_url = f'{self.base_url}/{self.session}'
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.browser.get(init_url, session_id=self.session, session_int=session_int)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.text_to_clickable = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.instruction_text = self.get_instruction_text() if instruction_text is None else instruction_text
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs = self.observation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prev_obs = [obs]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prev_actions = []
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs, None

    # [EXPLAIN] `render` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def render(self, mode='human'):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
    

# [EXPLAIN] `tag_visible` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def tag_visible(element):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


# [EXPLAIN] `SimServer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SimServer:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Lightweight simulator of WebShop Flask application for generating HTML observations"""
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        seed,
        base_url,
        file_path,
        attr_path,
        filter_goals=None,
        limit_goals=-1,
        num_products=None,
        human_goals=0,
        show_attrs=False,
    ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Constructor for simulated server serving WebShop application
        
        Arguments:
        filter_goals (`func`) -- Select specific goal(s) for consideration based on criteria of custom function
        limit_goals (`int`) -- Limit to number of goals available
        num_products (`int`) -- Number of products to search across
        human_goals (`bool`) -- If true, load human goals; otherwise, load synthetic goals
        """
        # Load all products, goals, and search engine
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.base_url = base_url
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.all_products, self.product_item_dict, self.product_prices, _ = \
            load_products(filepath=file_path, attrpath=attr_path, num_products=num_products, human_goals=human_goals)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.search_engine = init_search_engine(num_products=num_products)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.goals = get_goals(self.all_products, self.product_prices, human_goals)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.show_attrs = show_attrs

        # Fix outcome for random shuffling of goals
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        random.seed(seed)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        random.shuffle(self.goals)

        # Apply `filter_goals` parameter if exists to select speific goal(s)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if filter_goals is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.goals = [
                goal for (i, goal) in enumerate(self.goals)
                if filter_goals(i, goal)
            ]
        
        # Imposes `limit` on goals via random selection
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if limit_goals != -1 and limit_goals < len(self.goals):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.weights = [goal['weight'] for goal in self.goals]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.cum_weights = [0] + np.cumsum(self.weights).tolist()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idxs = []
            # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
            while len(idxs) < limit_goals:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                idx = random_idx(self.cum_weights)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if idx not in idxs:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    idxs.append(idx)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.goals = [self.goals[i] for i in idxs]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f'Loaded {len(self.goals)} goals.')

        # Set extraneous housekeeping variables
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.weights = [goal['weight'] for goal in self.goals]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.cum_weights = [0] + np.cumsum(self.weights).tolist()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.user_sessions = dict()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.search_time = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.render_time = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sample_time = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.assigned_instruction_text = None  # TODO: very hacky, should remove
        
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @app.route('/', methods=['GET', 'POST'])
    # [EXPLAIN] `index` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def index(self, session_id, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Redirect to the search page with the given session ID"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = map_action_to_html(
            'start',
            session_id=session_id,
            instruction_text=kwargs['instruction_text'],
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        url = f'{self.base_url}/{session_id}'
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return html, url
    
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @app.route('/', methods=['GET', 'POST'])
    # [EXPLAIN] `search_results` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def search_results(self, session_id, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize session and return the search results page"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        session = self.user_sessions[session_id]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keywords = kwargs['keywords']  # TODO: why is this using kwargs? why not session?
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(keywords, list)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        page = 1 if 'page' not in kwargs else kwargs['page']
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        session["page"] = page
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        session["keywords"] = keywords
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        session["actions"]["search"] += 1
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        session["asin"] = None
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        session["options"] = {}

        # Perform search on keywords from items and record amount of time it takes
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        old_time = time.time()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        top_n_products = get_top_n_product_from_keywords(
            keywords,
            self.search_engine,
            self.all_products,
            self.product_item_dict,
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.search_time += time.time() - old_time
        
        # Get product list from search result asins and get list of corresponding URLs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        products = get_product_per_page(top_n_products, page)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keywords_url_string = '+'.join(keywords)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        url = (
            f'{self.base_url}/search_results/{session_id}/'
            f'{keywords_url_string}/{page}'
        )

        # Render HTML search page and record amount of time taken
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        old_time = time.time()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = map_action_to_html(
            'search',
            session_id=session_id,
            products=products,
            keywords=session["keywords"],
            page=page,
            total=len(top_n_products),
            instruction_text=session["goal"]["instruction_text"],
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.render_time += time.time() - old_time
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return html, url
    
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @app.route('/', methods=['GET', 'POST'])
    # [EXPLAIN] `item_page` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def item_page(self, session_id, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Render and return the HTML for a product item page"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        session = self.user_sessions[session_id]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        clickable_name = kwargs['clickable_name']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_to_clickable = kwargs['text_to_clickable']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        clickable = text_to_clickable[clickable_name]

        # Update session logs with information of last product asin selected
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if (clickable.get('class') is not None and
            clickable.get('class')[0] == 'product-link'):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            session["asin"] = clickable_name.upper()
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            session["actions"]["asin"] += 1
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            session["asins"].add(session["asin"])
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif clickable.get('name') is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            clickable_key = clickable['name'].lower()
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            session["options"][clickable_key] = clickable_name
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            session["actions"]["options"] += 1

        # Set fields + url of page, then render page's HTML
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        product_info = self.product_item_dict[session["asin"]]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keywords_url_string = '+'.join(session["keywords"])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        option_string = json.dumps(session['options'])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        url = (
            f'{self.base_url}/item_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/'
            f'{session["page"]}/{option_string}'
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = map_action_to_html(
            'click',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
            show_attrs=self.show_attrs,
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return html, url

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @app.route('/', methods=['GET', 'POST'])
    # [EXPLAIN] `item_sub_page` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def item_sub_page(self, session_id, **kwargs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Render and return the HTML for a product's sub page (i.e. description, features)"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        session = self.user_sessions[session_id]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        clickable_name = kwargs['clickable_name']
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k in ACTION_TO_TEMPLATE:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if clickable_name.lower() == k.lower():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                clickable_name = k
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
        
        # Set fields + url of page, then render page's HTML
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        product_info = self.product_item_dict[session["asin"]]
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        session["actions"][clickable_name] += 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keywords_url_string = '+'.join(session["keywords"])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        url = (
            f'{self.base_url}/item_sub_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/{session["page"]}/'
            f'{clickable_name}/{session["options"]}'
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = map_action_to_html(
            f'click[{clickable_name}]',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return html, url

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @app.route('/', methods=['GET', 'POST'])
    # [EXPLAIN] `done` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def done(self, session_id, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Render and return HTML for done page"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        session = self.user_sessions[session_id]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal = self.user_sessions[session_id]['goal']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        purchased_product = self.product_item_dict[session["asin"]]
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        session["actions"]["purchase"] += 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        price = self.product_prices.get(session["asin"])

        # Calculate reward for selected product and set variables for page details
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward, info = get_reward(
            purchased_product,
            goal,
            price=price,
            options=session["options"],
            verbose=True
        )

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.user_sessions[session_id]['verbose_info'] = info
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.user_sessions[session_id]['done'] = True
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.user_sessions[session_id]['reward'] = reward

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        url = (
            f'{self.base_url}/done/{session_id}/'
            f'{session["asin"]}/{session["options"]}'
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = map_action_to_html(
            f'click[{END_BUTTON}]',
            session_id=session_id,
            reward=reward,
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return html, url, reward
    
    # [EXPLAIN] `receive` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def receive(self, session_id, current_url, session_int=None, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Map action to the corresponding page"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        status = dict(reward=0.0, done=False)

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with app.app_context(), app.test_request_context():
            # Create/determine goal, instruction_text from current session
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if session_id not in self.user_sessions:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                idx = session_int if (session_int is not None and isinstance(session_int, int)) else random_idx(self.cum_weights) 
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                goal = self.goals[idx]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                instruction_text = goal['instruction_text']
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.user_sessions[session_id] = {'goal': goal, 'done': False}
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                instruction_text = \
                    self.user_sessions[session_id]['goal']['instruction_text']
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.assigned_instruction_text is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                instruction_text = self.assigned_instruction_text  # TODO: very hacky, should remove
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                self.user_sessions[session_id]['goal']['instruction_text'] = instruction_text
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            session = self.user_sessions[session_id]

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not kwargs:
                # If no action, reset the session variables
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                kwargs['instruction_text'] = instruction_text
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                html, url = self.index(session_id, **kwargs)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.user_sessions[session_id].update(
                    {
                        'keywords': None,
                        'page': None,
                        'asin': None,
                        'asins': set(),
                        'options': dict(),
                        'actions': defaultdict(int)
                    }
                )
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif 'keywords' in kwargs:
                # If search keywords are available, run a search
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                html, url = self.search_results(session_id, **kwargs)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif 'clickable_name' in kwargs:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                clickable_name = kwargs['clickable_name'].lower()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if clickable_name == END_BUTTON.lower():
                    # If "buy now" clicked, calculate reward and flag session as terminated
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    html, url, reward = self.done(session_id, **kwargs)
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    status['reward'] = reward
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    status['done'] = True
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif clickable_name == BACK_TO_SEARCH.lower():
                    # If "back to search" clicked, recursively reset the session back to search page
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    html, url, status = self.receive(session_id, current_url)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif (clickable_name == NEXT_PAGE.lower() and 
                      self.get_page_name(current_url) == 'search_results'):
                    # If "next page" clicked from search results, re-render with `page` enumerated
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] + 1,
                    )
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif (clickable_name == PREV_PAGE.lower() and 
                      self.get_page_name(current_url) == 'search_results'):
                    # If "prev page" clicked from search results, re-render with `page` denumerated
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] - 1,
                    )
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif (clickable_name == PREV_PAGE.lower() and 
                      self.get_page_name(current_url) == 'item_sub_page'):
                    # If "prev page" clicked from sub page, return to corresponding item page
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    html, url = self.item_page(session_id, **kwargs)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif (clickable_name == PREV_PAGE.lower() and 
                      self.get_page_name(current_url) == 'item_page'):
                    # If "prev page" clicked from item page, return to search results page
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    html, url = self.search_results(
                        session_id,
                        keywords=session["keywords"],
                        page=session["page"],
                        **kwargs
                    )
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif clickable_name in [k.lower() for k in ACTION_TO_TEMPLATE]:
                    # Render item_sub_page if clickable is description, features, or reviews
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    html, url = self.item_sub_page(session_id, **kwargs)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # Otherwise, render current item page
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    html, url = self.item_page(session_id, **kwargs)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return html, url, status
    
    # [EXPLAIN] `get_page_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_page_name(self, url):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Determine which page (i.e. item_page, search_results) the given URL is pointing at"""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if url is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        page_names = [
            'search_results',
            'item_page',
            'item_sub_page',
            'done'
        ]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for page_name in page_names:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if page_name in url:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return page_name
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ''  # index page


# [EXPLAIN] `SimBrowser` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SimBrowser:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Simulated browser for rendering the HTML source of WebShop environment pages"""
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, server):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.server = server
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.current_url = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.page_source = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.session_id = None

    # [EXPLAIN] `get` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get(self, url, session_id=None, session_int=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Set browser variables to corresponding link, page HTML for URL"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.session_id = url.split('/')[-1] if session_id is None else session_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.page_source, _, _ = \
            self.server.receive(self.session_id, self.current_url, session_int=session_int)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.current_url = url
    
    # [EXPLAIN] `click` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def click(self, clickable_name, text_to_clickable):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Wrapper for `receive` handler for performing click action on current page"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                clickable_name=clickable_name,
                text_to_clickable=text_to_clickable,
            )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return status
    
    # [EXPLAIN] `search` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def search(self, keywords):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Wrapper for `receive` handler for performing search action on current page"""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(keywords, str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            keywords = keywords.split(' ')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                keywords=keywords,
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return status