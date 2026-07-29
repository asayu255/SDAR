# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import BartForConditionalGeneration

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from train_search import get_data, get_dataset, tokenizer

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = BartForConditionalGeneration.from_pretrained(
        './ckpts/web_search/checkpoint-800')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    model.eval()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = model.to('cuda')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = get_dataset("web_search")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataloader = torch.utils.data.DataLoader(dataset["all"], batch_size=32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _, all_goals = get_data("all")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_dec = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for batch in tqdm(dataloader):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = model.generate(
            input_ids=batch["input_ids"].to('cuda'),
            attention_mask=batch["attention_mask"].to('cuda'),
            num_beams=10, num_return_sequences=10,
            max_length=512, early_stopping=True
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dec = tokenizer.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(dec) % 10 == 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(dec) // 10):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            all_dec.append(dec[i*10:(i+1)*10])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(all_goals) == len(all_dec)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    d = {goal: dec for goal, dec in zip(all_goals, all_dec)}
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open('./data/goal_query_predict.json', 'w') as f:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        json.dump(d, f)
