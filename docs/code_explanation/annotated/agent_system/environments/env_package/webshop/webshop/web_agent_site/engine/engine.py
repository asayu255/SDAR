# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
"""
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from ast import literal_eval
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from decimal import Decimal

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import cleantext
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from rank_bm25 import BM25Okapi
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from flask import render_template_string
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from rich import print
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pyserini.search.lucene import LuceneSearcher

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.utils import (
    BASE_DIR,
    DEFAULT_FILE_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_ATTR_PATH,
    HUMAN_ATTR_PATH
)

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SEARCH_RETURN_N = 50
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
PRODUCT_WINDOW = 10
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
TOP_K_ATTR = 10

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
END_BUTTON = 'Buy Now'
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
NEXT_PAGE = 'Next >'
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
PREV_PAGE = '< Prev'
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
BACK_TO_SEARCH = 'Back to Search'

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

# [EXPLAIN] `map_action_to_html` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def map_action_to_html(action, **kwargs):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    action_name, action_arg = parse_action(action)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if action_name == 'start':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            instruction_text=kwargs['instruction_text'],
        )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif action_name == 'search':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            products=kwargs['products'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            total=kwargs['total'],
            instruction_text=kwargs['instruction_text'],
        )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif action_name == 'click' and action_arg == END_BUTTON:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            reward=kwargs['reward'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            reward_info=kwargs.get('reward_info'),
            goal_attrs=kwargs.get('goal_attrs'),
            purchased_attrs=kwargs.get('purchased_attrs'),
            goal=kwargs.get('goal'),
            mturk_code=kwargs.get('mturk_code'),
            query=kwargs.get('query'),
            category=kwargs.get('category'),
            product_category=kwargs.get('product_category'),
        )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text')
        )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif action_name == 'click':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            show_attrs=kwargs['show_attrs']
        )
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError('Action name not recognized.')
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return html


# [EXPLAIN] `read_html_template` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def read_html_template(path):
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(path) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        template = f.read()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return template


# [EXPLAIN] `parse_action` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parse_action(action):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Parse action string to action name and its arguments.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pattern = re.compile(r'(.+)\[(.+)\]')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    m = re.match(pattern, action)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if m is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_name = action
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_arg = None
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_name, action_arg = m.groups()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return action_name, action_arg


# [EXPLAIN] `convert_web_app_string_to_var` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def convert_web_app_string_to_var(name, string):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if name == 'keywords':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keywords = string
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if keywords.startswith('['):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            keywords = literal_eval(keywords)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            keywords = [keywords]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        var = keywords
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif name == 'page':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        page = string
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        page = int(page)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        var = page
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError('Name of variable not recognized.')
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return var


# [EXPLAIN] `get_top_n_product_from_keywords` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins=None,
    ):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if keywords[0] == '<r>':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif keywords[0] == '<a>':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attribute = ' '.join(keywords[1:]).strip()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asins = attribute_to_asins[attribute]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        top_n_products = [p for p in all_products if p['asin'] in asins]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif keywords[0] == '<c>':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        category = keywords[1].strip()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        top_n_products = [p for p in all_products if p['category'] == category]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif keywords[0] == '<q>':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query = ' '.join(keywords[1:]).strip()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        top_n_products = [p for p in all_products if p['query'] == query]
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        keywords = ' '.join(keywords)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        docs = [search_engine.doc(hit.docid) for hit in hits]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return top_n_products


# [EXPLAIN] `get_product_per_page` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_product_per_page(top_n_products, page):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]


# [EXPLAIN] `generate_product_prices` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def generate_product_prices(all_products):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    product_prices = dict()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for product in all_products:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin = product['asin']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pricing = product['pricing']
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not pricing:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price = 100.0
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif len(pricing) == 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price = pricing[0]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price = random.uniform(*pricing[:2])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        product_prices[asin] = price
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return product_prices


# [EXPLAIN] `init_search_engine` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_search_engine(num_products=None):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if num_products == 100:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        indexes = 'indexes_100'
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif num_products == 1000:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        indexes = 'indexes_1k'
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif num_products == 100000:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        indexes = 'indexes_100k'
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif num_products is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        indexes = 'indexes'
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(f'num_products being {num_products} is not supported yet.')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    search_engine = LuceneSearcher(os.path.join(BASE_DIR, f'../search_engine/{indexes}'))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return search_engine


# [EXPLAIN] `clean_product_keys` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def clean_product_keys(products):
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for product in products:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('product_information', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('brand', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('brand_url', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('list_price', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('availability_quantity', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('availability_status', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('total_reviews', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('total_answered_questions', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('seller_id', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('seller_name', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('fulfilled_by_amazon', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('fast_track_message', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('aplus_present', None)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product.pop('small_description_old', None)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print('Keys cleaned.')
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return products


# [EXPLAIN] `load_products` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_products(filepath, attrpath, num_products=None, human_goals=True):
    # TODO: move to preprocessing step -> enforce single source of truth
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(filepath) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        products = json.load(f)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print('Products loaded.')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    products = clean_product_keys(products)
    
    # with open(DEFAULT_REVIEW_PATH) as f:
    #     reviews = json.load(f)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_reviews = dict()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_ratings = dict()
    # for r in reviews:
    #     all_reviews[r['asin']] = r['reviews']
    #     all_ratings[r['asin']] = r['average_rating']

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if human_goals:
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(HUMAN_ATTR_PATH) as f:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            human_attributes = json.load(f)
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(attrpath) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attributes = json.load(f)
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(HUMAN_ATTR_PATH) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        human_attributes = json.load(f)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print('Attributes loaded.')

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    asins = set()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_products = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attribute_to_asins = defaultdict(set)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if num_products is not None:
        # using item_shuffle.json, we assume products already shuffled
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        products = products[:num_products]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i, p in tqdm(enumerate(products), total=len(products)):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin = p['asin']
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if asin == 'nan' or len(asin) > 10:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if asin in asins:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            asins.add(asin)

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['category'] = p['category']
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['query'] = p['query']
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['product_category'] = p['product_category']

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['Title'] = p['name']
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['Description'] = p['full_description']
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        products[i]['Reviews'] = all_reviews.get(asin, [])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        products[i]['Rating'] = all_ratings.get(asin, 'N.A.')
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for r in products[i]['Reviews']:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 'score' not in r:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                r['score'] = r.pop('stars')
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 'review' not in r:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                r['body'] = ''
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                r['body'] = r.pop('review')
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['BulletPoints'] = p['small_description'] \
            if isinstance(p['small_description'], list) else [p['small_description']]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pricing = p.get('pricing')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pricing is None or not pricing:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pricing = [100.0]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price_tag = '$100.0'
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pricing = [
                float(Decimal(re.sub(r'[^\d.]', '', price)))
                for price in pricing.split('$')[1:]
            ]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(pricing) == 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                price_tag = f"${pricing[0]}"
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                price_tag = f"${pricing[0]} to ${pricing[1]}"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                pricing = pricing[:2]
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['pricing'] = pricing
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['Price'] = price_tag

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        options = dict()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        customization_options = p['customization_options']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        option_to_image = dict()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if customization_options:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for option_name, option_contents in customization_options.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if option_contents is None:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                option_name = option_name.lower()

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                option_values = []
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for option_content in option_contents:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    option_value = option_content['value'].strip().replace('/', ' | ').lower()
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    option_image = option_content.get('image', None)

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    option_values.append(option_value)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    option_to_image[option_value] = option_image
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                options[option_name] = option_values
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['options'] = options
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['option_to_image'] = option_to_image

        # without color, size, price, availability
        # if asin in attributes and 'attributes' in attributes[asin]:
        #     products[i]['Attributes'] = attributes[asin]['attributes']
        # else:
        #     products[i]['Attributes'] = ['DUMMY_ATTR']
        # products[i]['instruction_text'] = \
        #     attributes[asin].get('instruction', None)
        # products[i]['instruction_attributes'] = \
        #     attributes[asin].get('instruction_attributes', None)

        # without color, size, price, availability
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if asin in attributes and 'attributes' in attributes[asin]:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            products[i]['Attributes'] = attributes[asin]['attributes']
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            products[i]['Attributes'] = ['DUMMY_ATTR']
            
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if human_goals:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if asin in human_attributes:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                products[i]['instructions'] = human_attributes[asin]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            products[i]['instruction_text'] = \
                attributes[asin].get('instruction', None)

            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            products[i]['instruction_attributes'] = \
                attributes[asin].get('instruction_attributes', None)

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        products[i]['MainImage'] = p['images'][0]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        products[i]['query'] = p['query'].lower().strip()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        all_products.append(products[i])

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for p in all_products:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for a in p['Attributes']:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            attribute_to_asins[a].add(p['asin'])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    product_item_dict = {p['asin']: p for p in all_products}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    product_prices = generate_product_prices(all_products)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return all_products, product_item_dict, product_prices, attribute_to_asins
