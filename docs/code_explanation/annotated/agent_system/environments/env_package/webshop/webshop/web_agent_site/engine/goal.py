# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Functions for specifying goals and reward calculations.
"""
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import itertools
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import spacy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from rich import print
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from thefuzz import fuzz
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.engine.normalize import normalize_color

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
nlp = spacy.load("en_core_web_sm")

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
PRICE_RANGE = [10.0 * i for i in range(1, 100)]

# [EXPLAIN] `get_goals` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_goals(all_products, product_prices, human_goals=True):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if human_goals:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return get_human_goals(all_products, product_prices)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return get_synthetic_goals(all_products, product_prices)
    
# [EXPLAIN] `get_human_goals` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_human_goals(all_products, product_prices):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    goals = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cnt_atts = defaultdict(int)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cnt = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for item in all_products:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin = item['asin']
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'instructions' not in item: continue
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for product in item['instructions']:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attributes = product['instruction_attributes']
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(attributes) == 0: 
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                cnt += 1
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if product_prices is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                price = product_prices[asin]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                price_range = [p for p in PRICE_RANGE if p > price][:4]
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if len(price_range) >= 2:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    _, price_upper = sorted(random.sample(price_range, 2))
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    price_text = \
                        f', and price lower than {price_upper:.2f} dollars'
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    price_upper = 1000000
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    price_text = ''
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                price_upper = 1000000

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            goals.append({
                'asin': asin,
                'category': item['category'],
                'query': item['query'],
                'name': item['name'],
                'product_category': item['product_category'],
                'instruction_text': product['instruction'].strip('.') + price_text,
                'attributes': attributes,
                'price_upper': price_upper,
                'goal_options': product['instruction_options'],
            })
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for att in attributes:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                cnt_atts[att] += 1
            # goals += product_goals
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for goal in goals:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        goal['weight'] = 1
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(cnt, 'skipped')
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return goals


# [EXPLAIN] `get_synthetic_goals` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_synthetic_goals(all_products, product_prices):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    goals = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cnt_atts = defaultdict(int)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for product in all_products:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ('instruction_text' not in product or 
            product['instruction_text'] is None):
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        product_goals = []        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin = product['asin']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attributes = product['instruction_attributes']
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(attributes) > 0

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if product_prices is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price = product_prices[asin]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price_range = [p for p in PRICE_RANGE if p > price][:4]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(price_range) >= 2:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                _, price_upper = sorted(random.sample(price_range, 2))
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                price_text = \
                    f', and price lower than {price_upper:.2f} dollars'
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                price_upper = 1000000
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                price_text = ''
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price_upper = 1000000
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price_text = ''

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        instruction_text = product['instruction_text']

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        options = product['options']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        option_names = sorted(options)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        combinations = list(itertools.product(
            *(options[option_name] for option_name in option_names)
        ))
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for combination in combinations:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            goal_options = dict()
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i, o in enumerate(combination):
#                option_text.append(f'{option_names[i]}: {o}')
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                goal_options[option_names[i]] = o
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            option_text = ', and '.join([
                f'{k}: {v}' for k, v in goal_options.items()
            ])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            option_text = ' with ' + option_text if option_text else ''
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            product_goals.append({
                'asin': asin,
                'category': product['category'],
                'query': product['query'],
                'name': product['name'],
                'product_category': product['product_category'],
                'instruction_text': f'{instruction_text}{option_text}{price_text}',
                'attributes': attributes,
                'price_upper': price_upper,
                'goal_options': goal_options,
                'name': product['Title'],
            })
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for att in attributes:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                cnt_atts[att] += 1
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        goals += product_goals
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for goal in goals:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        goal['weight'] = sum(1. / cnt_atts[att] for att in goal['attributes']) / len(goal['attributes'])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return goals


# [EXPLAIN] `get_type_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_type_reward(purchased_product, goal):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Determines the type reward - captures whether chosen product is in the same category"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_match = purchased_product['query'] == goal['query']

    # Check number of unique categories that match, ignoring order
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    purchased_product_category = [x.strip() for x in purchased_product['product_category'].split('›')]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    goal_product_category = [x.strip() for x in goal['product_category'].split('›')]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    category_match = len(set(purchased_product_category) & set(goal_product_category)) >= 2

    # Determine whether types align based on product name similarity
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    purchased_type = purchased_product['name']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    desired_type = goal['name']

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    purchased_type_parse = nlp(purchased_type)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    desired_type_parse = nlp(desired_type)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    purchased_type_parse = [t.text.lower() for t in purchased_type_parse if t.pos_ in ('PNOUN', 'NOUN', 'PROPN')]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    desired_type_parse = [t.text.lower() for t in desired_type_parse if t.pos_ in ('PNOUN', 'NOUN', 'PROPN')]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n_intersect_type = len(
        set(purchased_type_parse) & set(desired_type_parse)
    )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(desired_type_parse) == 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        title_score = 0.2
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        title_score = n_intersect_type / len(desired_type_parse)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r_type = 1.0

    # Adjust r_type score based on query, category title matching/scores
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    match = query_match or category_match or title_score > 0.2
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not match:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        r_type = 0.5

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if title_score < 0.1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        r_type = 0.1
    
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if title_score == 0.0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        r_type = 0.0

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dict(
        r_type=r_type,
        query_match=query_match,
        category_match=category_match,
        title_score=title_score,
    )


# [EXPLAIN] `get_attribute_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_attribute_reward(purchased_product, goal):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Determines whether purchased products shares same attributes as goal"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    purchased_attrs = purchased_product['Attributes']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    goal_attrs = goal['attributes']

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_attr_matches = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for g_attr in goal_attrs:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        matched = False
        # Check whether goal attribute found in purchased product attribute list
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for p_attr in purchased_attrs:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            score = fuzz.token_set_ratio(p_attr, g_attr)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if score > 85:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                num_attr_matches += 1
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                matched = True
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
        # If not in purchased attrs, check Title, Bullet Points (Features), Desc
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if (
            not matched and
            (
                g_attr in purchased_product['Title'].lower() or
                g_attr in ' '.join(purchased_product['BulletPoints']).lower() or
                g_attr in purchased_product['Description'].lower()
            )
        ):
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            num_attr_matches += 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            matched = True
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r_attr = num_attr_matches / len(goal_attrs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return r_attr, num_attr_matches


# [EXPLAIN] `get_option_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_option_reward(purchased_options, goal_options):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Calculate reward for purchased product's options w.r.t. goal options"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    purchased_options = [normalize_color(o) for o in purchased_options]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    goal_options = [normalize_color(o) for o in goal_options]

    # Perform fuzzy matching of each purchased option against each goal option
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_option_matches = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for g_option in goal_options:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for p_option in purchased_options:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            score = fuzz.token_set_ratio(p_option, g_option)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if score > 85:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                num_option_matches += 1
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
    
    # Calculate option reward as fraction of goal options hit
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r_option = num_option_matches / len(goal_options) if len(goal_options) > 0 else None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return r_option, num_option_matches


# [EXPLAIN] `get_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_reward(purchased_product, goal, price, options, **kwargs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Get cumulative reward score for purchased product and goal"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r_type_dict = get_type_reward(purchased_product, goal)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r_price = (
        price <= goal['price_upper']
    ) if goal['price_upper'] > 0 else None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r_att, num_attr_matches = get_attribute_reward(purchased_product, goal)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r_option, num_option_matches = get_option_reward(
        list(options.values()),
        goal['goal_options'].items()
        if isinstance(goal['goal_options'], dict)
        else goal['goal_options']
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_reward = (
        (num_attr_matches + num_option_matches + r_price) \
            / (len(goal['attributes']) + len(goal['goal_options']) + 1)
    )

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    total_reward *= r_type_dict['r_type']

    # If verbose flag enabled, store score sub-components into dictionary
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if kwargs.get('verbose', False):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info =  {
            'r_type': r_type_dict['r_type'],
            'r_att': r_att,
            'w_att': len(goal['attributes']) / (len(goal['attributes']) + len(goal['goal_options']) + 1),
            'query_match': r_type_dict['query_match'],
            'category_match': r_type_dict['category_match'],
            'title_score': r_type_dict['title_score'],
        }
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if r_option is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            info['r_option'] = r_option
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info['w_option'] = len(goal['goal_options']) / (len(goal['attributes']) + len(goal['goal_options']) + 1)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if r_price is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            info['r_price'] = r_price
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info['w_price'] = 1 / (len(goal['attributes']) + len(goal['goal_options']) + 1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return total_reward, info
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return total_reward
