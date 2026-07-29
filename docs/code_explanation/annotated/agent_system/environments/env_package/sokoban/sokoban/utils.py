# env_utils.py
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from datetime import datetime, timedelta
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from zoneinfo import ZoneInfo
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from contextlib import contextmanager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] `permanent_seed` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def permanent_seed(seed: int) -> None:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    random.seed(seed)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    np.random.seed(seed)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.manual_seed(seed)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if torch.cuda.is_available():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.manual_seed_all(seed)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@contextmanager
# [EXPLAIN] `set_seed` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def set_seed(seed):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    random_state = random.getstate()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    np_random_state = np.random.get_state()

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        random.seed(seed)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.random.seed(seed)
        # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
        yield
    # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
    finally:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        random.setstate(random_state)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.random.set_state(np_random_state)


# [EXPLAIN] `setup_logging` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def setup_logging(output_dir):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(output_dir, exist_ok=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_file = os.path.join(output_dir, 'training.log')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logging.Formatter.converter = lambda *args: (datetime.now(UTC) - timedelta(hours=2)).timetuple()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return logging.getLogger()


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@contextmanager
# [EXPLAIN] `NoLoggerWarnings` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def NoLoggerWarnings():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from gym import logger
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logger.set_level(logger.ERROR)
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
        yield
    # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
    finally:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.set_level(logger.INFO)


# [EXPLAIN] `get_train_val_env` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_train_val_env(env_class, config: dict):

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    val_env = None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if config.env.name == 'frozenlake':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env = env_class(size=config.env.size, p=config.env.p)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif config.env.name == 'bandit':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env = env_class(n_arms=config.env.n_arms)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif config.env.name == 'two_armed_bandit':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lo_name, hi_name = config.env.low_risk_name, config.env.high_risk_name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lo_val_name = config.env.low_risk_name if config.env.low_risk_val_name is None else config.env.low_risk_val_name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hi_val_name = config.env.high_risk_name if config.env.high_risk_val_name is None else config.env.high_risk_val_name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env = env_class(low_risk_name=lo_name, high_risk_name=hi_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_env = env_class(low_risk_name=lo_val_name, high_risk_name=hi_val_name)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO] val_env low_risk_name: {val_env.low_risk_name}, high_risk_name: {val_env.high_risk_name}")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if val_env.low_risk_name is None or val_env.high_risk_name is None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("[WARNING] val_env arm are None, falling back to not create val_env")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            val_env = None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif config.env.name == 'sokoban':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env = env_class(dim_room=(config.env.dim_x, config.env.dim_y), num_boxes=config.env.num_boxes, max_steps=config.env.max_steps, search_depth=config.env.search_depth)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif config.env.name == 'countdown':
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env = env_class(parquet_path=config.env.train_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_env = env_class(parquet_path=config.env.val_path)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Environment {config.env.name} not supported")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return env, val_env