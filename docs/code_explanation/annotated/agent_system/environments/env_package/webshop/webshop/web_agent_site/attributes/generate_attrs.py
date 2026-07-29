# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import yaml
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pathlib import Path
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sklearn.feature_extraction.text import TfidfVectorizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sklearn.feature_extraction import text as sk_text
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pandas as pd
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from rich import print

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ITEMS_PATH = './data/ITEMS_mar1.json'
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
REVIEWS_PATH = './data/reviews.json'
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ATTR_DIR = './data/attributes'

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
random.seed(0)


# [EXPLAIN] `get_stop_words` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_stop_words():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    extra_stop_words = set([str(i) for i in range(1000)])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    stop_words = sk_text.ENGLISH_STOP_WORDS.union(extra_stop_words)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return stop_words


# [EXPLAIN] `load_products` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_products(num=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Loads products from the `items.json` file and combine them with reviews
    through `asin`.
    Return: dict[asin, product]
    """
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(ITEMS_PATH) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_products = json.load(f)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if num is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            random.shuffle(all_products)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            all_products = all_products[:num]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        products = dict()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asins = set()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for p in all_products:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            asin = p['asin']
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if asin in asins:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            asins.add(asin)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            products[asin] = p

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(REVIEWS_PATH) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reviews = json.load(f)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reviews = {r['asin']: r for r in reviews}

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for asin, p in products.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if asin in reviews:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            p['review'] = reviews[asin]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            p['review'] = None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return products


# [EXPLAIN] `get_top_attrs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_top_attrs(attributes, k):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attr_to_asins = defaultdict(list)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for asin, attr_scores in attributes.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        top_attr_scoress = attr_scores[:k]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for attr, score in top_attr_scoress:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            attr_to_asins[attr].append(asin)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total = len([asin for asin, _ in attributes.items()])
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    top_attrs = [
        (attr, len(asins) / total)
        for attr, asins in attr_to_asins.items()
    ]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    top_attrs = sorted(top_attrs, key=lambda x: -x[1])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    top_attrs = [f'{attr} | {score:.4f}' for attr, score in top_attrs]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return top_attrs


# [EXPLAIN] `get_corpus` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_corpus(
        products,
        keys=('name', 'small_description'),
        category_type='category'
    ):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    keys: `name`, `small_description`, `review`
    category_type: `category`, `query`
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_products = list(products.values())
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    asins_by_cat = defaultdict(set)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    corpus_by_cat = defaultdict(list)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for p in all_products:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        category = p[category_type]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin = p['asin']
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if asin in asins_by_cat[category]:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        asins_by_cat[category].add(asin)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in keys:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if key == 'review':
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rs = p['review']['reviews']
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if r is not None:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    text_ = ' '.join([r['review'].lower() for r in rs])
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    text_ = ''
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                text_ = p[key].lower()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            text.append(text_)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text = ' '.join(text)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        corpus_by_cat[category].append((asin, text))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return corpus_by_cat


# [EXPLAIN] `generate_ngram_attrs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def generate_ngram_attrs(corpus_by_cat, ngram_range, k, attrs):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vectorizer = TfidfVectorizer(
        stop_words=get_stop_words(),
        ngram_range=ngram_range,
        max_features=1000,
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    top_attrs_by_cat = dict()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for category, corpus in tqdm(corpus_by_cat.items(),
                                 total=len(corpus_by_cat)):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asins = [_[0] for _ in corpus]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        texts = [_[1] for _ in corpus]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vec = vectorizer.fit_transform(texts).todense()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        df = pd.DataFrame(vec, columns=vectorizer.get_feature_names_out())

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attrs_by_cat = dict()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for asin, (row_name, row) in zip(asins, df.iterrows()):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attr_scores = sorted(
                list(zip(row.index, row)),
                key=lambda x: -x[1]
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attrs_by_cat[asin] = attr_scores
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attrs[asin] = attr_scores
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        top_attrs_by_cat[category.lower()] = get_top_attrs(attrs_by_cat, k=k)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(top_attrs_by_cat.keys())
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return top_attrs_by_cat


# [EXPLAIN] `generate_attrs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def generate_attrs(corpus_by_cat, k, save_name):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attrs = dict()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for n in range(1, 3):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ngram_range = (n, n)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        top_attrs_by_cat = \
            generate_ngram_attrs(corpus_by_cat, ngram_range, k, attrs)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if save_name is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            save_path = Path(ATTR_DIR) / f'{save_name}_{n}-gram.yaml'
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with open(save_path, 'w') as f:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                yaml.dump(top_attrs_by_cat, f, default_flow_style=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f'Saved: {save_path}')

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    save_path = Path(ATTR_DIR) / f'{save_name}_attrs_unfiltered.json'
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(save_path, 'w') as f:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        json.dump(attrs, f)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f'Saved: {save_path}')


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == '__main__':
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    python -m web_agent_site.attributes.generate_attrs

    Inspect in notebooks/attributes.ipynb.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    products = load_products(num=40000)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    corpus_by_cat_broad = get_corpus(products, category_type='category')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    generate_attrs(corpus_by_cat_broad, k=5, save_name='broad')

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    corpus_by_cat_narrow = get_corpus(products, category_type='query')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    generate_attrs(corpus_by_cat_narrow, k=5, save_name='narrow')
