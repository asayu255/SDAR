# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import argparse
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from copy import deepcopy

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pandas as pd

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.hdfs_io import copy, makedirs


# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
TASKS = ("alfworld", "search", "webshop")
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
EMPTY_PROMPT = [{"role": "user", "content": ""}]


# [EXPLAIN] `_sample_dataframe` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _sample_dataframe(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    replace = len(df) < size
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if replace:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.warning("Sampling %s rows with replacement from %s available rows.", size, len(df))
    # Pick a uniform random subset (same budget/distribution as the single-task
    # RandomSampler), then restore the original row order so ordering is decided
    # solely by the dataloader's TaskBalancedSampler at train time. This avoids a
    # double shuffle (prep-time + dataloader) and mirrors the single-task path,
    # which shuffles exactly once.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sampled = df.sample(n=size, replace=replace, random_state=seed)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return sampled.sort_index().reset_index(drop=True)


# [EXPLAIN] `_dummy_task_dataframe` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _dummy_task_dataframe(task_name: str, split: str, size: int) -> pd.DataFrame:
    # [EXPLAIN] AlfWorld/WebShop は parquet に実 episode を固定せず、task_name と dummy prompt を持つ row だけを作る。
    # [EXPLAIN] game/task の内容は reset 時に environment 内部 schedule から決まり、row 数は prompt batch 構成を規定する。
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rows = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for idx in range(size):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        rows.append(
            {
                "data_source": task_name,
                "prompt": deepcopy(EMPTY_PROMPT),
                "ability": "agent",
                "reward_model": None,
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "task_name": task_name,
                },
                "metadata": None,
                "env_kwargs": {"task_name": task_name},
                "task_name": task_name,
            }
        )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return pd.DataFrame(rows)


# [EXPLAIN] `_as_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _as_dict(value):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return deepcopy(value) if isinstance(value, dict) else {}


# [EXPLAIN] `_process_search_row` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _process_search_row(row: pd.Series, split: str, idx: int) -> dict:
    # [EXPLAIN] Search は dummy ではなく parquet の question、ground truth、data source を保持し、
    # [EXPLAIN] rollout 時に `env_kwargs` として HTTP retriever を使う Search environment へ渡す。
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    item = row.to_dict()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    env_kwargs = _as_dict(item.get("env_kwargs"))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    extra_info = _as_dict(item.get("extra_info"))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    question = env_kwargs.get("question", item.get("question", ""))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_source = env_kwargs.get("data_source", item.get("data_source", "unknown"))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth = env_kwargs.get("ground_truth")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ground_truth is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_model = item.get("reward_model")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(reward_model, dict):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ground_truth = reward_model.get("ground_truth")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ground_truth is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ground_truth = item.get("golden_answers", [])

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    env_kwargs.update(
        {
            "ground_truth": ground_truth,
            "question": question,
            "data_source": data_source,
            "task_name": "search",
        }
    )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    extra_info.update({"split": split, "index": idx, "task_name": "search"})

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    item["data_source"] = data_source
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    item["extra_info"] = extra_info
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    item["env_kwargs"] = env_kwargs
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    item["task_name"] = "search"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return item


# [EXPLAIN] `_search_dataframe` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _search_dataframe(search_dir: str, split: str, size: int, seed: int) -> pd.DataFrame:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    path = os.path.join(os.path.expanduser(search_dir), f"{split}.parquet")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not os.path.exists(path):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise FileNotFoundError(
            f"Search parquet not found: {path}. Run examples.data_preprocess.preprocess_search_r1_dataset first."
        )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full = pd.read_parquet(path)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if split == "test":
        # Match the single-task search baseline: run_search_qwen3.sh validates with
        # val_dataloader(shuffle=False), which consumes the first val_batch_size rows of
        # test.parquet. Take the leading `size` rows deterministically so the multitask
        # search validation is computed over the same fixed population (fair metric comparison).
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(full) < size:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"search test parquet has {len(full)} rows but {size} were requested.")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        selected = full.head(size).reset_index(drop=True)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # Train: uniform random subset without replacement (same budget/distribution as the
        # single-task RandomSampler). Ordering is left to the dataloader's TaskBalancedSampler.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        selected = _sample_dataframe(full, size=size, seed=seed)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rows = [_process_search_row(row, split=split, idx=idx) for idx, (_, row) in enumerate(selected.iterrows())]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return pd.DataFrame(rows)


# [EXPLAIN] `_build_split` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _build_split(search_dir: str, split: str, per_task_size_by_task: dict[str, int], seed: int) -> pd.DataFrame:
    # [EXPLAIN] task ごとに必要 row 数を独立に作成・sample した後で concat し、
    # [EXPLAIN] 後段 TaskBalancedSampler が参照する contiguous task layout を形成する。
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    frames = [
        _dummy_task_dataframe("alfworld", split, per_task_size_by_task["alfworld"]),
        _search_dataframe(search_dir, split, per_task_size_by_task["search"], seed),
        _dummy_task_dataframe("webshop", split, per_task_size_by_task["webshop"]),
    ]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    df = pd.concat(frames, ignore_index=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    counts = df["task_name"].value_counts().to_dict()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected = {task: per_task_size_by_task[task] for task in TASKS}
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if counts != expected:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise RuntimeError(f"Unexpected multitask counts for {split}: got {counts}, expected {expected}")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return df


# [EXPLAIN] `main` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_dir = os.path.expanduser(args.local_dir)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(local_dir, exist_ok=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    search_train_size = args.total_training_steps * args.per_task_batch_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    env_train_size = args.env_train_per_task_size or args.per_task_batch_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    split_sizes = {
        "train": {
            "alfworld": env_train_size,
            "search": search_train_size,
            "webshop": env_train_size,
        },
        "test": {task: args.val_per_task_size for task in TASKS},
    }

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for split, per_task_size_by_task in split_sizes.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        df = _build_split(args.search_dir, split, per_task_size_by_task, seed=args.seed)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_path = os.path.join(local_dir, f"{split}.parquet")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        df.to_parquet(output_path, index=False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.info("Saved %s rows to %s with task counts %s", len(df), output_path, df["task_name"].value_counts().to_dict())

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        search_env_kwargs = df[df["task_name"] == "search"]["env_kwargs"].iloc[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        missing = {"ground_truth", "question", "data_source"} - set(search_env_kwargs)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if missing:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError(f"Search env_kwargs is missing keys: {sorted(missing)}")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if args.hdfs_dir:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        makedirs(args.hdfs_dir)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        copy(src=local_dir, dst=args.hdfs_dir)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.info("Copied multitask data to HDFS: %s", args.hdfs_dir)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parser = argparse.ArgumentParser(description="Build balanced SDAR multitask parquet files.")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--search_dir", default="~/data/searchR1_processed_direct")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--local_dir", default="~/data/verl-agent/sdar_multitask")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--hdfs_dir", default=None)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--total_training_steps", default=150, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--per_task_batch_size", default=16, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--env_train_per_task_size", default=None, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--val_per_task_size", default=128, type=int)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--seed", default=1, type=int)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = parser.parse_args()

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
