# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import gradio as gr
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json, time, torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import BartTokenizer, BartForConditionalGeneration, AutoModel, AutoTokenizer

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from webshop_lite import dict_to_fake_html
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from predict_help import (
    Page, convert_dict_to_actions, convert_html_to_text,
    parse_results_amz, parse_item_page_amz,
    parse_results_ws, parse_item_page_ws,
    parse_results_ebay, parse_item_page_ebay,
    WEBSHOP_URL, WEBSHOP_SESSION
)

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ENVIRONMENTS = ['amazon', 'webshop', 'ebay']

# IL+RL: 'webshop/il-rl-choice-bert-image_1'
# IL: 'webshop/il-choice-bert-image_0'
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
BERT_MODEL_PATH = 'webshop/il-choice-bert-image_0'

# load IL models
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
bart_model = BartForConditionalGeneration.from_pretrained('webshop/il_search_bart')

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side='left')
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
bert_tokenizer.add_tokens(['[button]', '[button_]', '[clicked button]', '[clicked button_]'], special_tokens=True)
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
bert_model = AutoModel.from_pretrained(BERT_MODEL_PATH, trust_remote_code=True)

# [EXPLAIN] `process_str` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_str(s):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    s = s.lower().replace('"', '').replace("'", "").strip()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    s = s.replace('[sep]', '[SEP]')
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return s


# [EXPLAIN] `process_goal` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_goal(state):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state = state.lower().replace('"', '').replace("'", "")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state = state.replace('\n[button] search [button_]', '').strip()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ', and price lower than' in state:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state = state.split(', and price lower than')[0]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return state


# [EXPLAIN] `data_collator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def data_collator(batch):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, labels, images = [], [], [], [], [], [], []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for sample in batch:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        state_input_ids.append(sample['state_input_ids'])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        state_attention_mask.append(sample['state_attention_mask'])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        action_input_ids.extend(sample['action_input_ids'])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        action_attention_mask.extend(sample['action_attention_mask'])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        sizes.append(sample['sizes'])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        labels.append(sample['labels'])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        images.append(sample['images'])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_state_len = max(sum(x) for x in state_attention_mask)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_action_len = max(sum(x) for x in action_attention_mask)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {
        'state_input_ids': torch.tensor(state_input_ids)[:, :max_state_len],
        'state_attention_mask': torch.tensor(state_attention_mask)[:, :max_state_len],
        'action_input_ids': torch.tensor(action_input_ids)[:, :max_action_len],
        'action_attention_mask': torch.tensor(action_attention_mask)[:, :max_action_len],
        'sizes': torch.tensor(sizes),
        'images': torch.tensor(images),
        'labels': torch.tensor(labels),
    }


# [EXPLAIN] `bart_predict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def bart_predict(input):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = bart_tokenizer(input)['input_ids']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = bart_model.generate(input_ids, max_length=512, num_return_sequences=5, num_beams=5)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return bart_tokenizer.batch_decode(output.tolist(), skip_special_tokens=True)[0]


# [EXPLAIN] `bert_predict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def bert_predict(obs, info, softmax=True):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    valid_acts = info['valid']
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert valid_acts[0].startswith('click[')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state_encodings = bert_tokenizer(process_str(obs), max_length=512, truncation=True, padding='max_length')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    action_encodings = bert_tokenizer(list(map(process_str, valid_acts)), max_length=512, truncation=True,  padding='max_length')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = {
        'state_input_ids': state_encodings['input_ids'],
        'state_attention_mask': state_encodings['attention_mask'],
        'action_input_ids': action_encodings['input_ids'],
        'action_attention_mask': action_encodings['attention_mask'],
        'sizes': len(valid_acts),
        'images': info['image_feat'].tolist(),
        'labels': 0
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = data_collator([batch])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = bert_model(**batch)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if softmax:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        idx = torch.multinomial(torch.nn.functional.softmax(outputs.logits[0], dim=0), 1)[0].item()
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        idx = outputs.logits[0].argmax(0).item()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return valid_acts[idx]

# [EXPLAIN] `get_return_value` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_return_value(env, asin, options, search_terms, page_num, product):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    asin_url = None

    # Determine product URL + options based on environment
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if env == 'webshop':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_str = "+".join(search_terms.split())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        options_str = json.dumps(options)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin_url = (
            f'{WEBSHOP_URL}/item_page/{WEBSHOP_SESSION}/'
            f'{asin}/{query_str}/{page_num}/{options_str}'
        )
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        asin_url = f"https://www.ebay.com/itm/{asin}" if env == 'ebay' else \
            f"https://www.amazon.com/dp/{asin}"
    
    # Extract relevant fields for product
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    product_reduced = {k: v for k, v in product.items() if k in ["asin", "Title", "Description", "BulletPoints"]}
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_reduced["Description"] = product_reduced["Description"][:100] + "..."
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    product_reduced["Features"] = product_reduced.pop("BulletPoints")
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    product_reduced["Features"] = product_reduced["Features"][:100] + "..."

    # Create HTML to show link to product
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    html = """<!DOCTYPE html><html><head><title>Chosen Product</title></head><body>"""
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    html += f"""Product Image:<img src="{product["MainImage"]}" height="50px" /><br>""" if len(product["MainImage"]) > 0 else ""
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    html += f"""Link to Product:
        <a href="{asin_url}" style="color:blue;text-decoration:underline;" target="_blank">{asin_url}</a>
        </body></html>"""

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return product_reduced, options if len(options) > 0 else "None Selected", html
        

# [EXPLAIN] `predict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def predict(obs, info):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Given WebShop environment observation and info, predict an action.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    valid_acts = info['valid']
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if valid_acts[0].startswith('click['):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return bert_predict(obs, info)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "search[" + bart_predict(process_goal(obs)) + "]"

# [EXPLAIN] `run_episode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_episode(goal, env, verbose=True):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Interact with amazon to find a product given input goal.
    Input: text goal
    Output: a url of found item on amazon.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    env = env.lower()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if env not in ENVIRONMENTS:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[ERROR] Environment {env} not recognized")
        
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = "Amazon Shopping Game\nInstruction:" + goal + "\n[button] search [button]"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    info = {'valid': ['search[stuff]'], 'image_feat': torch.zeros(512)}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    product_map = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    title_to_asin_map = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    search_results_cache = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    visited_asins, clicked_options = set(), set()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sub_page_type, page_type, page_num = None, None, None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    search_terms, prod_title, asin = None, None, None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    options = {}
    
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(100):
        # Run prediction
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action = predict(obs, info)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if verbose:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("====")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(action)
        
        # Previous Page Type, Action -> Next Page Type
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        action_content = action[action.find("[")+1:action.find("]")]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prev_page_type = page_type
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action.startswith('search['):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            page_type = Page.RESULTS
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            search_terms = action_content
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            page_num = 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif action.startswith('click['):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if action.startswith('click[item -'):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                prod_title = action_content[len("item -"):].strip()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                found = False
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for key in title_to_asin_map:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if prod_title == key:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        asin = title_to_asin_map[key]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        page_type = Page.ITEM_PAGE
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        visited_asins.add(asin)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        found = True
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        break
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not found:
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise Exception("Product to click not found")
                    
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif any(x.value in action for x in [Page.DESC, Page.FEATURES, Page.REVIEWS]):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                page_type = Page.SUB_PAGE
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                sub_page_type = Page(action_content.lower())
                
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif action == 'click[< prev]':
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if sub_page_type is not None:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    page_type, sub_page_type = Page.ITEM_PAGE, None
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif prev_page_type == Page.ITEM_PAGE:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    page_type = Page.RESULTS
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    options, clicked_options = {}, set()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif prev_page_type == Page.RESULTS and page_num > 1:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    page_type = Page.RESULTS
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    page_num -= 1
                    
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif action == 'click[next >]':
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                page_type = Page.RESULTS
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                page_num += 1
                
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif action.lower() == 'click[back to search]':
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                page_type = Page.SEARCH
                
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif action == 'click[buy now]':
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return get_return_value(env, asin, options, search_terms, page_num, product_map[asin])
            
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif prev_page_type == Page.ITEM_PAGE:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                found = False
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for opt_name, opt_values in product_map[asin]["options"].items():
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if action_content in opt_values:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        options[opt_name] = action_content
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        page_type = Page.ITEM_PAGE
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        clicked_options.add(action_content)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        found = True
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        break
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not found:
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise Exception("Unrecognized action: " + action)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise Exception("Unrecognized action:" + action)
        
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if verbose:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Parsing {page_type.value} page...")
        
        # URL -> Real HTML -> Dict of Info
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if page_type == Page.RESULTS:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if search_terms in search_results_cache:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data = search_results_cache[search_terms]
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if verbose:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Loading cached results page for \"{search_terms}\"")
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                begin = time.time()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if env == 'amazon':
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    data = parse_results_amz(search_terms, page_num, verbose)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if env == 'webshop':
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    data = parse_results_ws(search_terms, page_num, verbose)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if env == 'ebay':
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    data = parse_results_ebay(search_terms, page_num, verbose)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                end = time.time()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if verbose:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Parsing search results took {end-begin} seconds")

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                search_results_cache[search_terms] = data
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for d in data:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    title_to_asin_map[d['Title']] = d['asin']
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif page_type == Page.ITEM_PAGE or page_type == Page.SUB_PAGE:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if asin in product_map:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if verbose:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print("Loading cached item page for", asin)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data = product_map[asin]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                begin = time.time()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if env == 'amazon':
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    data = parse_item_page_amz(asin, verbose)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if env == 'webshop':
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    data = parse_item_page_ws(asin, search_terms, page_num, options, verbose)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if env == 'ebay':
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    data = parse_item_page_ebay(asin, verbose)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                end = time.time()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if verbose:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print("Parsing item page took", end-begin, "seconds")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                product_map[asin] = data
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif page_type == Page.SEARCH:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if verbose:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("Executing search")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs = "Amazon Shopping Game\nInstruction:" + goal + "\n[button] search [button]"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            info = {'valid': ['search[stuff]'], 'image_feat': torch.zeros(512)}
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise Exception("Page of type `", page_type, "` not found")

        # Dict of Info -> Fake HTML -> Text Observation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        begin = time.time()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        html_str = dict_to_fake_html(data, page_type, asin, sub_page_type, options, product_map, goal)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs = convert_html_to_text(html_str, simple=False, clicked_options=clicked_options, visited_asins=visited_asins)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        end = time.time()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if verbose:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("[Page Info -> WebShop HTML -> Observation] took", end-begin, "seconds")

        # Dict of Info -> Valid Action State (Info)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        begin = time.time()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prod_arg = product_map if page_type == Page.ITEM_PAGE else data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = convert_dict_to_actions(page_type, prod_arg, asin, page_num)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        end = time.time()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if verbose:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Extracting available actions took", end-begin, "seconds")
        
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if i == 50:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return get_return_value(env, asin, options, search_terms, page_num, product_map[asin])

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
gr.Interface(
    fn=run_episode,
    inputs=[
        gr.inputs.Textbox(lines=7, label="Input Text"),
        gr.inputs.Radio(['Amazon', 'eBay'], type="value", default="Amazon", label='Environment')
    ],
    outputs=[
        gr.outputs.JSON(label="Selected Product"),
        gr.outputs.JSON(label="Selected Options"),
        gr.outputs.HTML()
    ],
    examples=[
        ["I want to find a gold floor lamp with a glass shade and a nickel finish that i can use for my living room, and price lower than 270.00 dollars", "Amazon"],
        ["I need some cute heart-shaped glittery cupcake picks as a gift to bring to a baby shower", "Amazon"],
        ["I want to buy ballet shoes which have rubber sole in grey suede color and a size of 6", "Amazon"],
        ["I would like a 7 piece king comforter set decorated with flowers and is machine washable", "Amazon"],
        ["I'm trying to find white bluetooth speakers that are not only water resistant but also come with stereo sound", "eBay"],
        ["find me the soy free 3.5 ounce 4-pack of dang thai rice chips, and make sure they are the aged cheddar flavor.  i also need the ones in the resealable bags", "eBay"],
        ["I am looking for a milk chocolate of 1 pound size in a single pack for valentine day", "eBay"],
        ["I'm looking for a mini pc intel core desktop computer which supports with windows 11", "eBay"]
    ],
    title="WebShop",
    article="<p style='padding-top:15px;text-align:center;'>To learn more about this project, check out the <a href='https://webshop-pnlp.github.io/' target='_blank'>project page</a>!</p>",
    description="<p style='text-align:center;'>Sim-to-real transfer of agent trained on WebShop to search a desired product on Amazon from any natural language query!</p>",
).launch(inline=False)
