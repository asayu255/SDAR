# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from datasets import Dataset, DatasetDict, load_from_disk
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import (BartForConditionalGeneration, BartTokenizer, Trainer,
                          TrainingArguments)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.models.bart.modeling_bart import shift_tokens_right

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
BOS_TOKEN_ID = 0
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
PAD_TOKEN_ID = 1
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
EOS_TOKEN_ID = 2
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
UNK_TOKEN_ID = 3

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
PATH = "./data/goal_query_map.json"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
HUMAN_GOAL_PATH = './data/human_goals.json'
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
GOAL_PATH = "./data/items_human_ins.json"


# [EXPLAIN] `process_str` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_str(s):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    s = s.lower().replace('"', '').replace("'", "").strip()
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

# [EXPLAIN] `get_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_data(split):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = json.load(open(PATH))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    goals, searches = [], []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for goal, search_list in data.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal = process_goal(goal)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for search in search_list:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            search = process_str(search)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            goals.append(goal)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            searches.append(search)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = len(goals)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    human_goals = json.load(open(HUMAN_GOAL_PATH, 'r'))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    goal_range = range(len(human_goals))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if split == 'train':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal_range = range(500, len(human_goals))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif split == 'validation':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal_range = range(500, 1500)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif split == 'test':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal_range = range(0, 500)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif split == "all":  # all human instructions, but without groundtruth search queries
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_data = json.load(open(GOAL_PATH))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_goals = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_goals_processed = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for ins_list in all_data.values():
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for ins in ins_list:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ins = ins['instruction']
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                all_goals.append(ins)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                all_goals_processed.append(process_str(ins))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return all_goals_processed, all_goals
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    goals_, searches_ = [], []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for goal, search in zip(goals, searches):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if goal in human_goals and human_goals.index(goal) in goal_range:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            goals_.append(goal)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            searches_.append(search)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return goals_, searches_


# [EXPLAIN] `get_dataset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_dataset(name, flip=False, variant=None, size=None):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fname = name + "-flip" if flip else name
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fpath = os.path.join(os.path.dirname(__file__), fname)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    d = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splits = ["train", "validation", "test"]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if name == "web_search":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        splits = ["train", "validation", "test", "all"]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for split in splits:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input, output = get_data(split) if name != "nl2bash" else get_data(
            split, variant=variant)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        l = len(input) if size is None else int(len(input) * size)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("{} size: {}".format(split, l))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if flip:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input, output = output, input
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input, output = input[:l], output[:l]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        d[split] = process_dataset(input, output)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    d = DatasetDict(d)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return d


# [EXPLAIN] `process_dataset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_dataset(input, output, max_len=256):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_encodings = tokenizer(input, padding='max_length',
                                max_length=max_len, truncation=True, return_tensors='pt')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_encodings = tokenizer(
        output, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = output_encodings['input_ids']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    decoder_input_ids = shift_tokens_right(labels, PAD_TOKEN_ID, EOS_TOKEN_ID)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels[labels[:, :] == PAD_TOKEN_ID] = -100
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = Dataset.from_dict({
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,
    })
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dataset.set_format(type='torch', columns=[
                       'input_ids', 'labels', 'decoder_input_ids', 'attention_mask'])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dataset


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = get_dataset("web_search", flip=False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    train_dataset = dataset['train']
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(train_dataset[0])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    model.resize_token_embeddings(len(tokenizer))
    # model = BartForConditionalGeneration.from_pretrained('./models/qdmr-high-level-base/checkpoint-10000')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    training_args = TrainingArguments(
        output_dir='./ckpts/web_search',
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        evaluation_strategy="steps",
        logging_dir='./logs',
        logging_steps=50,
        eval_steps=20,
        save_steps=200
        # eval_accumulation_steps=1
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset["validation"],
        compute_metrics=None,
    )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    trainer.train()
