# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import argparse
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from huggingface_hub import hf_hub_download

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index", help="Hugging Face repository ID")
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
parser.add_argument("--local_dir", type=str, required=True, help="Local directory to save files")

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
args = parser.parse_args()

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
repo_id = "PeterJinGo/wiki-18-e5-index"
# [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
for file in ["part_aa", "part_ab"]:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    hf_hub_download(
        repo_id=repo_id,
        filename=file,  # e.g., "e5_Flat.index"
        repo_type="dataset",
        local_dir=args.local_dir,
    )

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
repo_id = "PeterJinGo/wiki-18-corpus"
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
hf_hub_download(
    repo_id=repo_id,
    filename="wiki-18.jsonl.gz",
    repo_type="dataset",
    local_dir=args.local_dir,
)
