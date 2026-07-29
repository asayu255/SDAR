# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Tuple

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
COLOR_SET = [
    'alabaster', 'apricot', 'aqua', 'ash', 'asphalt', 'azure',
    'banana', 'beige', 'black', 'blue', 'blush', 'bordeaux', 'bronze',
    'brown', 'burgundy', 'camel', 'camo', 'caramel', 'champagne',
    'charcoal', 'cheetah', 'chestnut', 'chocolate', 'christmas', 'coffee',
    'cognac', 'copper', 'coral', 'cranberry', 'cream', 'crystal', 'dark',
    'denim', 'eggplant', 'elephant', 'espresso', 'fuchsia', 'gold', 'granite',
    'grape', 'graphite', 'grass', 'gray', 'green', 'grey', 'heather', 'indigo',
    'ivory', 'ivy', 'khaki', 'lavender', 'lemon', 'leopard', 'light', 'lilac',
    'lime', 'magenta', 'maroon', 'mauve', 'merlot', 'midnight', 'mint', 'mocha',
    'multicolor', 'mushroom', 'mustard', 'natural', 'navy', 'nude', 'olive',
    'orange', 'peach', 'pewter', 'pink',    'plum', 'purple', 'rainbow', 'red',
    'rose', 'royal', 'rust', 'sand', 'sapphire', 'seashell', 'silver', 'skull',
    'slate', 'steel', 'stone', 'stonewash', 'sunflower', 'tan', 'taupe', 'teal',
    'tiger', 'turquoise', 'violet', 'walnut', 'wheat', 'white', 'wine', 'yellow',
]

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SIZE_SET = [
    'xx-large', '3x-large', '4x-large', '5x-large', 'x-large', 'x-small',
    'medium', 'large', 'small',
    'queen', 'twin', 'full', 'king', 'one size',
    'pack',
]

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SIZE_PATTERNS = [
    re.compile(r'(.*)neck(.*)sleeve'),
    re.compile(r'(.*) women \| (.*) men'),
    re.compile(r'(.*)w x(.*)l'),
    re.compile(r'(.*)w by (.*)l'),
    re.compile(r'(.*)w x(.*)h'),
    re.compile(r'(.*)wide'),
    re.compile(r'(.*)x-wide'),
    re.compile(r'(.*)narrow'),
    re.compile(r'(.*)petite'),
    re.compile(r'(.*)inch'),
    re.compile(r'(.*)plus'),
    re.compile(r'(.*)mm'),
    re.compile(r'women(.*)'),
    re.compile(r'(.*)x(.*)'),
    re.compile(r'(.*)ft'),
    re.compile(r'(.*)feet'),
    re.compile(r'(.*)meter'),
    re.compile(r'(.*)yards'),
    re.compile(r'(.*)\*(.*)'),
    re.compile(r'(.*)\-(.*)'),
    re.compile(r'(\d+)"$'),
    re.compile(r'(\d+)f$'),
    re.compile(r'(\d+)m$'),
    re.compile(r'(\d+)cm$'),
    re.compile(r'(\d+)g$'),
]
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SIZE_PATTERNS = [re.compile(s) for s in SIZE_SET] + SIZE_PATTERNS

# [EXPLAIN] `normalize_color` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def normalize_color(color_string: str) -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Extracts the first color found if exists"""
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for norm_color in COLOR_SET:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if norm_color in color_string:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return norm_color
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return color_string

# [EXPLAIN] `normalize_color_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def normalize_color_size(product_prices: dict) -> Tuple[dict, dict]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Get mappings of all colors, sizes to corresponding values in COLOR_SET, SIZE_PATTERNS"""
    
    # Get all colors, sizes from list of all products
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_colors, all_sizes = set(), set()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for (_, color, size), _ in product_prices.items():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        all_colors.add(color.lower())
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        all_sizes.add(size.lower())
    
    # Create mapping of each original color value to corresponding set value
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    color_mapping = {'N.A.': 'not_matched'} 
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for c in all_colors:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        matched = False
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for base in COLOR_SET:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if base in c:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                color_mapping[c] = base
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                matched = True
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not matched:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            color_mapping[c] = 'not_matched'

    # Create mapping of each original size value to corresponding set value
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    size_mapping = {'N.A.': 'not_matched'}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for s in all_sizes:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        matched = False
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for pattern in SIZE_PATTERNS:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            m = re.search(pattern, s)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if m is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                matched = True
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                size_mapping[s] = pattern.pattern
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not matched:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if s.replace('.', '', 1).isdigit():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                size_mapping[s] = 'numeric_size'
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                matched= True
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not matched:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            size_mapping[s] = 'not_matched'
    
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return color_mapping, size_mapping
    
