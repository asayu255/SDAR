# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from setuptools import setup

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
setup(
    name="gym-cards",
    version="0.0.1",
    packages=['gym_cards'],
    install_requires=["gymnasium", "numpy", "Pillow"]
)