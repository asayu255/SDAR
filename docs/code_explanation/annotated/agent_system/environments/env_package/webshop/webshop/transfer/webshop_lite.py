# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from flask import render_template_string, Flask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from predict_help import Page

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
app=Flask(__name__)
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
app.debug=True

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SESSION_ID = "ABC"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
TEMPLATE_DIR = "../web_agent_site/templates/"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
KEYWORDS = ["placeholder (not needed)"] # To Do: Does this matter?
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
QUERY = ""
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
product_map = {}

# [EXPLAIN] `read_html_template` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def read_html_template(path):
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(path) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        template = f.read()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return template

# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@app.route('/', methods=['GET', 'POST'])
# [EXPLAIN] `index` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def index(session_id, **kwargs):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Hello world")

# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@app.route('/', methods=['GET', 'POST'])
# [EXPLAIN] `search_results` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def search_results(data):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    path = os.path.join(TEMPLATE_DIR, 'results_page.html')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    html = render_template_string(
        read_html_template(path=path),
        session_id=SESSION_ID,
        products=data,
        keywords=KEYWORDS,
        page=1,
        total=len(data),
        instruction_text=QUERY,
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return html

# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@app.route('/', methods=['GET', 'POST'])
# [EXPLAIN] `item_page` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def item_page(session_id, asin, keywords, page, options):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    path = os.path.join(TEMPLATE_DIR, 'item_page.html')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    html = render_template_string(
        read_html_template(path=path),
        session_id=session_id,
        product_info=product_map[asin],
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=QUERY
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return html

# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@app.route('/', methods=['GET', 'POST'])
# [EXPLAIN] `item_sub_page` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def item_sub_page(session_id, asin, keywords, page, sub_page, options):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    path = os.path.join(TEMPLATE_DIR, sub_page.value.lower() + "_page.html")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    html = render_template_string(
        read_html_template(path),
        session_id=session_id, 
        product_info=product_map[asin],
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=QUERY
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return html

# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@app.route('/', methods=['GET', 'POST'])
# [EXPLAIN] `done` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def done(asin, options, session_id, **kwargs):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    path = os.path.join(TEMPLATE_DIR, 'done_page.html')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    html = render_template_string(
        read_html_template(path),
        session_id=session_id,
        reward=1,
        asin=asin,
        options=product_map[asin]["options"],
        reward_info=kwargs.get('reward_info'),
        goal_attrs=kwargs.get('goal_attrs'),
        purchased_attrs=kwargs.get('purchased_attrs'),
        goal=kwargs.get('goal'),
        mturk_code=kwargs.get('mturk_code'),
        query=kwargs.get('query'),
        category=kwargs.get('category'),
        product_category=kwargs.get('product_category'),
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return html
    
# Project Dictionary Information onto Fake Amazon
# [EXPLAIN] `dict_to_fake_html` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dict_to_fake_html(data, page_type, asin=None, sub_page_type=None, options=None, prod_map={}, query=""):
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    global QUERY, product_map
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    QUERY = query
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    product_map = prod_map
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with app.app_context(), app.test_request_context():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if page_type == Page.RESULTS:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return search_results(data)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if page_type == Page.ITEM_PAGE:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return item_page(SESSION_ID, asin, KEYWORDS, 1, options)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if page_type == Page.SUB_PAGE:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if sub_page_type is not None:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return item_sub_page(SESSION_ID, asin, KEYWORDS, 1, sub_page_type, options)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise Exception("Sub page of type", sub_page_type, "unrecognized")