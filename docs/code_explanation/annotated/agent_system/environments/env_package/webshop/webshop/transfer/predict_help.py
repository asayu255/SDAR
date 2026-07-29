# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from bs4 import BeautifulSoup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from bs4.element import Comment
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from enum import Enum
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re, time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from urllib.parse import urlencode

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json, requests, torch

# [EXPLAIN] `Page` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Page(Enum):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    DESC = "description"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    FEATURES = "features"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ITEM_PAGE = "item_page"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    RESULTS = "results"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    REVIEWS = "reviews"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    SEARCH = "search"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    SUB_PAGE = "item_sub_page"

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
HEADER_ = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36'
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
DEBUG_HTML = "temp.html"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
NUM_PROD_LIMIT = 10

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
WEBSHOP_URL = "http://3.83.245.205:3000"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
WEBSHOP_SESSION = "abc"


# [EXPLAIN] `parse_results_ebay` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parse_results_ebay(query, page_num=None, verbose=True):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_string = '+'.join(query.split())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    page_num = 1 if page_num is None else page_num
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    url = f'https://www.ebay.com/sch/i.html?_nkw={query_string}&_pgn={page_num}'
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Search Results URL: {url}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    soup = BeautifulSoup(webpage.text, 'html.parser')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    products = soup.select('.s-item__wrapper.clearfix')

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for item in products[:NUM_PROD_LIMIT]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        title = item.select_one('.s-item__title').text.strip()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "shop on ebay" in title.lower():
            # Skip "Shop on ebay" product title
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        link = item.select_one('.s-item__link')['href']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin = link.split("?")[0][len("https://www.ebay.com/itm/"):]

        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price = item.select_one('.s-item__price').text
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "to" in price:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                prices = price.split(" to ")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                price = [p.strip("$") for p in prices]
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price = None
        
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        results.append({
            "asin": asin,
            "Title": title,
            "Price": price
        })
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Scraped {len(results)} products")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return results


# [EXPLAIN] `parse_item_page_ebay` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parse_item_page_ebay(asin, verbose=True):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    product_dict = {}
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["asin"] = asin
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    url = f"https://www.ebay.com/itm/{asin}"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Item Page URL: {url}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    begin = time.time()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    end = time.time()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Item page scraping took {end-begin} seconds")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    soup = BeautifulSoup(webpage.content, "html.parser")

    # Title
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        product_dict["Title"] = soup.find('h1', {'class': 'x-item-title__mainTitle'}).text.strip()
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        product_dict["Title"] = "N/A"

    # Price: Get price string, extract decimal numbers from string
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        price_str = soup.find('div', {'class': 'mainPrice'}).text
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prices = re.findall('\d*\.?\d+', price_str)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        product_dict["Price"] = prices[0]
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        product_dict["Price"] = "N/A"

     # Main Image
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        img_div = soup.find('div', {'id': 'mainImgHldr'})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        img_link = img_div.find('img', {'id': 'icImg'})["src"]
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        product_dict["MainImage"] = img_link
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        product_dict["MainImage"] = ""
    
    # Rating
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rating = soup.find('span', {'class': 'reviews-star-rating'})["title"].split()[0]
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rating = None
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["Rating"] = rating

    # Options
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    options, options_to_images = {}, {} # TODO: options_to_images possible?
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        option_blocks = soup.findAll('select', {'class': 'msku-sel'})
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for block in option_blocks:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            name = block["name"].strip().strip(":")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            option_tags = block.findAll("option")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            opt_list = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for option_tag in option_tags:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "select" not in option_tag.text.lower():
                    # Do not include "- select -" (aka `not selected`) choice
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    opt_list.append(option_tag.text)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            options[name] = opt_list
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        options = {}
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["options"], product_dict["option_to_image"] = options, options_to_images

    # Description
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    desc = None
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # Ebay descriptions are shown in `iframe`s
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc_link = soup.find('iframe', {'id': 'desc_ifr'})["src"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc_webpage = requests.get(desc_link, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc_soup = BeautifulSoup(desc_webpage.content, "html.parser")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc = ' '.join(desc_soup.text.split())
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc = "N/A"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["Description"] = desc

    # Features
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    features = None
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        features = soup.find('div', {'class': 'x-about-this-item'}).text
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        features = "N/A"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["BulletPoints"] = features

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return product_dict
    

# [EXPLAIN] `parse_results_ws` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parse_results_ws(query, page_num=None, verbose=True):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_string = '+'.join(query.split())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    page_num = 1 if page_num is None else page_num
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    url = (
        f'{WEBSHOP_URL}/search_results/{WEBSHOP_SESSION}/'
        f'{query_string}/{page_num}'
    )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Search Results URL: {url}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    soup = BeautifulSoup(webpage.content, 'html.parser')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    products = soup.findAll('div', {'class': 'list-group-item'})

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for product in products:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin = product.find('a', {'class': 'product-link'})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        title = product.find('h4', {'class': 'product-title'})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        price = product.find('h5', {'class': 'product-price'})

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "\n" in title:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            title = title.text.split("\n")[0].strip()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            title = title.text.strip().strip("\n")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "to" in price.text:
            # Parse if price presented as range
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prices = price.text.split(" to ")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price = [float(p.strip().strip("\n$")) for p in prices]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            price = float(price.text.strip().strip("\n$"))

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        results.append({
            "asin": asin.text,
            "Title": title,
            "Price": price
        })

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Scraped {len(results)} products")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return results


# [EXPLAIN] `parse_item_page_ws` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parse_item_page_ws(asin, query, page_num, options, verbose=True):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    product_dict = {}
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["asin"] = asin

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_string = '+'.join(query.split())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    options_string = json.dumps(options)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    url = (
        f'{WEBSHOP_URL}/item_page/{WEBSHOP_SESSION}/'
        f'{asin}/{query_string}/{page_num}/{options_string}'
    )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Item Page URL: {url}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    soup = BeautifulSoup(webpage.content, 'html.parser')

    # Title, Price, Rating, and MainImage
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    product_dict["Title"] = soup.find('h2').text
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    h4_headers = soup.findAll("h4")
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for header in h4_headers:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text = header.text
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "Price" in text:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            product_dict["Price"] = text.split(":")[1].strip().strip("$")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "Rating" in text:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            product_dict["Rating"] = text.split(":")[1].strip()
    
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    product_dict["MainImage"] = soup.find('img')['src']

    # Options
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    options, options_to_image = {}, {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    option_blocks = soup.findAll("div", {'class': 'radio-toolbar'})
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for block in option_blocks:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        name = block.find("input")["name"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        labels = block.findAll("label")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs = block.findAll("input")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        opt_list = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for label, input in zip(labels, inputs):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            opt = label.text
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            opt_img_path = input["onclick"].split("href=")[1].strip('\';')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            opt_img_url = f'{WEBSHOP_URL}{opt_img_path}'

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            opt_list.append(opt)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            options_to_image[opt] = opt_img_url
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        options[name] = opt_list
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["options"] = options
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["option_to_image"] = options_to_image

    # Description
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    url = (
        f'{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/'
        f'{asin}/{query_string}/{page_num}/Description/{options_string}'
    )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Item Description URL: {url}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    soup = BeautifulSoup(webpage.content, 'html.parser')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    product_dict["Description"] = soup.find(name="p", attrs={'class': 'product-info'}).text.strip()

    # Features
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    url = (
        f'{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/'
        f'{asin}/{query_string}/{page_num}/Features/{options_string}'
    )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Item Features URL: {url}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    soup = BeautifulSoup(webpage.content, 'html.parser')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bullets = soup.find(name="ul").findAll(name="li")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    product_dict["BulletPoints"] = '\n'.join([b.text.strip() for b in bullets])

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return product_dict


# Query -> Search Result ASINs
# [EXPLAIN] `parse_results_amz` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parse_results_amz(query, page_num=None, verbose=True):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    url = 'https://www.amazon.com/s?k=' + query.replace(" ", "+")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if page_num is not None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        url += "&page=" + str(page_num)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Search Results URL: {url}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    soup = BeautifulSoup(webpage.content, 'html.parser')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    products = soup.findAll('div', {'data-component-type': 's-search-result'})
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if products is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        temp = open(DEBUG_HTML, "w")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        temp.write(str(soup))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        temp.close()
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise Exception("Couldn't find search results page, outputted html for inspection")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results = []

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for product in products[:NUM_PROD_LIMIT]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin = product['data-asin']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        title = product.find("h2", {'class': "a-size-mini"})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        price_div = product.find("div", {'class': 's-price-instructions-style'})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        price = price_div.find("span", {'class': 'a-offscreen'})

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = {
            'asin': asin,
            'Title': title.text.strip(),
            'Price': price.text.strip().strip("$")
        }
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        results.append(result)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Scraped", len(results), "products")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return results


# Scrape information of each product
# [EXPLAIN] `parse_item_page_amz` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parse_item_page_amz(asin, verbose=True):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    product_dict = {}
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["asin"] = asin

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    url = f"https://www.amazon.com/dp/{asin}"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Item Page URL:", url)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    begin = time.time()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    end = time.time()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if verbose:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Item page scraping took {end-begin} seconds")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    soup = BeautifulSoup(webpage.content, "html.parser")

    # Title
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        title = soup.find("span", attrs={"id": 'productTitle'})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        title = title.string.strip().replace(',', '')
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        title = "N/A"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["Title"] = title
 
    # Price
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parent_price_span = soup.find(name="span", class_="apexPriceToPay")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        price_span = parent_price_span.find(name="span", class_="a-offscreen")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        price = float(price_span.getText().replace("$", ""))
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        price = "N/A"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["Price"] = price

    # Rating
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rating = soup.find(name="span", attrs={"id": "acrPopover"})
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if rating is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rating = "N/A"
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rating = rating.text
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rating = "N/A"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    product_dict["Rating"] = rating.strip("\n").strip()
 
    # Features
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        features = soup.find(name="div", attrs={"id": "feature-bullets"}).text
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        features = "N/A"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["BulletPoints"] = features
    
    # Description
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc_body = soup.find(name="div", attrs={"id": "productDescription_feature_div"})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc_div = desc_body.find(name="div", attrs={"id": "productDescription"})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc_ps = desc_div.findAll(name="p")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc = " ".join([p.text for p in desc_ps])
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        desc = "N/A"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    product_dict["Description"] = desc.strip()

    # Main Image
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        imgtag = soup.find("img", {"id":"landingImage"})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        imageurl = dict(imgtag.attrs)["src"]
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        imageurl = ""
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["MainImage"] = imageurl

    # Options
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    options, options_to_image = {}, {}
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        option_body = soup.find(name='div', attrs={"id": "softlinesTwister_feature_div"})
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if option_body is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            option_body = soup.find(name='div', attrs={"id": "twister_feature_div"})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        option_blocks = option_body.findAll(name='ul')
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for block in option_blocks:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            name = json.loads(block["data-a-button-group"])["name"]
            # Options
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            opt_list = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for li in block.findAll("li"):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                img = li.find(name="img")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if img is not None:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    opt = img["alt"].strip()
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    opt_img = img["src"]
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if len(opt) > 0:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        options_to_image[opt] = opt_img
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    opt = li.text.strip()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if len(opt) > 0:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    opt_list.append(opt)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            options[name.replace("_name", "").replace("twister_", "")] = opt_list
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        options = {}
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_dict["options"], product_dict["option_to_image"] = options, options_to_image
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return product_dict


# Get text observation from html
# TODO[john-b-yang]: Similar to web_agent_site/envs/...text_env.py func def, merge?
# [EXPLAIN] `convert_html_to_text` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def convert_html_to_text(html, simple=False, clicked_options=None, visited_asins=None):
    # [EXPLAIN] `tag_visible` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def tag_visible(element):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (
            element.parent.name not in ignore and not isinstance(element, Comment)
        )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    html_obj = BeautifulSoup(html, 'html.parser')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    texts = html_obj.findAll(text=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    visible_texts = filter(tag_visible, texts)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if simple:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        observation = ''
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for t in visible_texts:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if t == '\n': continue
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if t.parent.name == 'button':  # button
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                processed_t = f'[button] {t} [button]'
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif t.parent.name == 'label':  # options
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if f'{t}' in clicked_options:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    processed_t = f'  [clicked button] {t} [clicked button]'
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    observation = f'You have clicked {t}.\n' + observation
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    processed_t = f'  [button] {t} [button]'
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif t.parent.get('class') == ["product-link"]: # asins
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if f'{t}' in visited_asins:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    processed_t = f'\n[clicked button] {t} [clicked button]'
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    processed_t = f'\n[button] {t} [button]'
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else: # regular, unclickable text
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                processed_t =  str(t)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            observation += processed_t + '\n'
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return observation


# Get action from dict of values retrieved from html
# [EXPLAIN] `convert_dict_to_actions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def convert_dict_to_actions(page_type, products=None, asin=None, page_num=None) -> dict:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    info = {"valid": []}
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if page_type == Page.RESULTS:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        info["valid"] = ['click[back to search]']
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if products is None or page_num is None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(page_num)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(products)
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise Exception('Provide `products`, `page_num` to get `results` valid actions')
        # Decide whether to add `next >` as clickable based on # of search results
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(products) > 10:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info["valid"].append('click[next >]')
        # Add `< prev` as clickable if not first page of search results
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if page_num > 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info["valid"].append('click[< prev]')
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for product in products:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info["valid"].append("click[item - " + product["Title"] + "]")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if page_type == Page.ITEM_PAGE:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if products is None or asin is None:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise Exception('Provide `products` and `asin` to get `item_page` valid actions')
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        info["valid"] = ['click[back to search]', 'click[< prev]', 'click[description]',\
            'click[features]', 'click[buy now]'] # To do: reviews
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "options" in products[asin]:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, values in products[asin]["options"].items():
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for value in values:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    info["valid"].append("click[" + value + "]")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if page_type == Page.SUB_PAGE:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        info["valid"] = ['click[back to search]', 'click[< prev]']
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    info['image_feat'] = torch.zeros(512)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return info