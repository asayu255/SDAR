# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
A unified tracking interface that supports logging data to different backend
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import dataclasses
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from enum import Enum
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import partial
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pathlib import Path
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Dict, List, Union


# [EXPLAIN] `Tracking` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Tracking:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "clearml"]

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = "console", config=None):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(default_backend, str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            default_backend = [default_backend]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for backend in default_backend:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if backend == "tracking":
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                import warnings

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert backend in self.supported_backend, f"{backend} is not supported"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.logger = {}

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "tracking" in default_backend or "wandb" in default_backend:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import wandb

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            wandb.init(project=project_name, name=experiment_name, config=config)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.logger["wandb"] = wandb

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "mlflow" in default_backend:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import os

            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import mlflow

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if MLFLOW_TRACKING_URI:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            experiment = mlflow.set_experiment(project_name)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "swanlab" in default_backend:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import os

            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import swanlab

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if SWANLAB_API_KEY:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config is None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                config = {}  # make sure config is not None, otherwise **config will raise error
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.logger["swanlab"] = swanlab

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "vemlp_wandb" in default_backend:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import os

            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import volcengine_ml_platform
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from volcengine_ml_platform import wandb as vemlp_wandb

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.logger["vemlp_wandb"] = vemlp_wandb

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "tensorboard" in default_backend:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.logger["tensorboard"] = _TensorboardAdapter()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "console" in default_backend:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.utils.logger.aggregate_logger import LocalLogger

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.console_logger = LocalLogger(print_to_console=True)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.logger["console"] = self.console_logger

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "clearml" in default_backend:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

    # [EXPLAIN] `log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log(self, data, step, backend=None):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for default_backend, logger_instance in self.logger.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if backend is None or default_backend in backend:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger_instance.log(data=data, step=step)

    # [EXPLAIN] `__del__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __del__(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "wandb" in self.logger:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.logger["wandb"].finish(exit_code=0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "swanlab" in self.logger:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.logger["swanlab"].finish()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "vemlp_wandb" in self.logger:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.logger["vemlp_wandb"].finish(exit_code=0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "tensorboard" in self.logger:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.logger["tensorboard"].finish()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "clearnml" in self.logger:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.logger["clearnml"].finish()


# [EXPLAIN] `ClearMLLogger` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ClearMLLogger:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, project_name: str, experiment_name: str, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.project_name = project_name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.experiment_name = experiment_name

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import clearml

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._task.connect_configuration(config, name="Hyperparameters")

    # [EXPLAIN] `_get_logger` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_logger(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._task.get_logger()

    # [EXPLAIN] `log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log(self, data, step):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import numpy as np
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import pandas as pd

        # logs = self._rewrite_logs(data)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logger = self._get_logger()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k, v in data.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            title, series = k.split("/", 1)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(v, (int, float, np.floating, np.integer)):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif isinstance(v, pd.DataFrame):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.warning(f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This invocation of ClearML logger\'s function is incorrect so this attribute was dropped. ')

    # [EXPLAIN] `finish` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def finish(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._task.mark_completed()


# [EXPLAIN] `_TensorboardAdapter` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class _TensorboardAdapter:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import os

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.utils.tensorboard import SummaryWriter

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        os.makedirs(tensorboard_dir, exist_ok=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.writer = SummaryWriter(tensorboard_dir)

    # [EXPLAIN] `log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log(self, data, step):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in data:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.writer.add_scalar(key, data[key], step)

    # [EXPLAIN] `finish` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def finish(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.writer.close()


# [EXPLAIN] `_MlflowLoggingAdapter` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class _MlflowLoggingAdapter:
    # [EXPLAIN] `log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log(self, data, step):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import mlflow

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        mlflow.log_metrics(metrics=results, step=step)


# [EXPLAIN] `_compute_mlflow_params_from_objects` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if params is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {}

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


# [EXPLAIN] `_transform_params_to_json_serializable` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dataclasses.is_dataclass(x):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return _transform(dataclasses.asdict(x))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(x, dict):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {k: _transform(v) for k, v in x.items()}
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(x, list):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if convert_list_to_dict:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return [_transform(v) for v in x]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(x, Path):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return str(x)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(x, Enum):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return x.value

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return x


# [EXPLAIN] `_flatten_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import pandas as pd

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(ans, dict)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ans


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclasses.dataclass
# [EXPLAIN] `ValidationGenerationsLogger` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ValidationGenerationsLogger:
    # [EXPLAIN] `log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log(self, loggers, samples, step):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "wandb" in loggers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.log_generations_to_wandb(samples, step)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "swanlab" in loggers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.log_generations_to_swanlab(samples, step)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "mlflow" in loggers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.log_generations_to_mlflow(samples, step)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "clearml" in loggers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.log_generation_to_clearml(samples, step)

    # [EXPLAIN] `log_generations_to_wandb` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log_generations_to_wandb(self, samples, step):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Log samples to wandb as a table"""
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import wandb

        # Create column names for all samples
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], [])

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        row_data = []
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        row_data.append(step)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for sample in samples:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            row_data.extend(sample)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        new_table.add_data(*row_data)

        # Update reference and log
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        wandb.log({"val/generations": new_table}, step=step)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.validation_table = new_table

    # [EXPLAIN] `log_generations_to_swanlab` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log_generations_to_swanlab(self, samples, step):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Log samples to swanlab as text"""
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import swanlab

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        swanlab_text_list = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, sample in enumerate(samples):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        swanlab.log({"val/generations": swanlab_text_list}, step=step)

    # [EXPLAIN] `log_generations_to_mlflow` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log_generations_to_mlflow(self, samples, step):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import json
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import tempfile

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import mlflow

        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with tempfile.TemporaryDirectory() as tmp_dir:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                row_data = []
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for sample in samples:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    row_data.append(data)
                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with open(validation_gen_step_file, "w") as file:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    json.dump(row_data, file)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                mlflow.log_artifact(validation_gen_step_file)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    # [EXPLAIN] `log_generation_to_clearml` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log_generation_to_clearml(self, samples, step):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Log validation generation to clearml as table"""

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import clearml
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import pandas as pd

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task: clearml.Task | None = clearml.Task.current_task()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logger = task.get_logger()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )
