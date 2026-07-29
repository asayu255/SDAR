# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import yaml
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pathlib import Path
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from rich import print

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ATTR_DIR = './data/attributes'

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ATTR_PATHS = [
    'narrow_2-gram.yaml',
    'narrow_1-gram.yaml',
    'broad_2-gram.yaml',
    'broad_1-gram.yaml',
]
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ATTR_PATHS = [Path(ATTR_DIR) / af for af in ATTR_PATHS]


# [EXPLAIN] `annotate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def annotate(attr_path):
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(attr_path) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attrs_by_cat = yaml.safe_load(f)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    unique_attrs = set()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_attrs = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _, attrs in attrs_by_cat.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attrs = [a.split('|')[0].strip() for a in attrs]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        unique_attrs.update(attrs)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        all_attrs += attrs
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f'Total unique attributes: {len(unique_attrs)}')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total = len(all_attrs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_left = len(all_attrs)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    annotated_attrs_by_cat = dict()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for category, attrs in attrs_by_cat.items():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(
            f'Category: [ {category} ] | '
            f'Number of attributes: {len(attrs)}\n'
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        annotated_attrs = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, attr in enumerate(attrs):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attr, score = attr.split(' | ')
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(
                f'{"[" + str(i) + "]":<5} '
                f'[bold green]{attr:<30}[/bold green] | '
                f'[red]{category}[/red] | '
                f'{score}'
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tags = input(
                'Annotate [1: ITEM, 2: PROP, 3: USE, '
                '⎵: next example, q: next category] > '
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print('\n')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tags = tags.strip()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            annotated_attrs.append(f'{attr} | {score} | {tags}')
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if 'q' in tags:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
        
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        num_left -= len(attrs)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f'{num_left} / {total} total attributes left.')

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ans = input('Starting the next category... [y/n] > ')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ans == 'n':
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break

# [EXPLAIN] `main` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main():
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for attr_path in ATTR_PATHS:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        annotate(attr_path)

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == '__main__':
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    python -m web_agent_site.attributes.annotate
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
