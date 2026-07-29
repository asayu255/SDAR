# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
sys.path.insert(0, '../')

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.utils import DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from web_agent_site.engine.engine import load_products

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
all_products, *_ = load_products(filepath=DEFAULT_FILE_PATH, attrpath=DEFAULT_ATTR_PATH)


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
docs = []
# [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
for p in tqdm(all_products, total=len(all_products)):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    option_texts = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    options = p.get('options', {})
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for option_name, option_contents in options.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        option_contents_text = ', '.join(option_contents)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        option_texts.append(f'{option_name}: {option_contents_text}')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    option_text = ', and '.join(option_texts)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    doc = dict()
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    doc['id'] = p['asin']
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    doc['contents'] = ' '.join([
        p['Title'],
        p['Description'],
        p['BulletPoints'][0],
        option_text,
    ]).lower()
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    doc['product'] = p
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    docs.append(doc)


# [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
with open('./resources_100/documents.jsonl', 'w+') as f:
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for doc in docs[:100]:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        f.write(json.dumps(doc) + '\n')

# [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
with open('./resources/documents.jsonl', 'w+') as f:
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for doc in docs:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        f.write(json.dumps(doc) + '\n')

# [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
with open('./resources_1k/documents.jsonl', 'w+') as f:
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for doc in docs[:1000]:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        f.write(json.dumps(doc) + '\n')

# [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
with open('./resources_100k/documents.jsonl', 'w+') as f:
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for doc in docs[:100000]:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        f.write(json.dumps(doc) + '\n')
