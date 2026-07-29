"""自然文注釈マップの読み書き・render・品質判定を共有する。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from common import ROOT, decode_text, load_manifest, read_source_bytes


ANNOTATION_MAP = ROOT / "docs" / "code_explanation" / "annotation_map.jsonl"

GENERIC_EXACT = {
    "この JSON 要素は実行時設定またはデータ構造の一部を保持する。",
    "後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。",
    "必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。",
    "実行時の設定または状態を評価し、成立する経路だけを選択する。",
    "このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。",
    "この設定項目の値を、entry point または worker が読み取る構成として定義する。",
    "この段落は実装の意図、利用条件または検証上の注意を説明する。",
    "計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。",
    "この論理行で現在の処理ブロックに必要な状態または制御を定義する。",
    "実行設定または後続 command が参照する環境値・引数を定義する。",
    "batch、micro-batch、task または token の要素を順に処理する。",
    "後続処理が依存する shape、dtype、設定または分散条件を検証する。",
    "直前の条件が成立しない場合の代替経路を実行する。",
    "実験起動、環境準備または検証 command の一段階を実行する。",
    "直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。",
    "不正な設定・shape・task または実行状態を例外として明示する。",
    "context manager で resource、autocast、no_grad または session の寿命を限定する。",
    "現在の分岐または反復の制御を明示する。",
    "発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。",
    "失敗し得る処理を開始し、後続の例外処理へ制御を接続する。",
    "後続処理で参照する状態の寿命またはスコープを明示的に調整する。",
    "終了条件を満たすまで rollout または状態更新を反復する。",
    "現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。",
    "成功・失敗にかかわらず resource 解放または状態復元を実行する。",
    "この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。",
    "以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。",
    "この cell の原文と実行・説明上の役割を対応付ける。",
    "mixed batch を task 別 manager へ分配し、reset/step 結果を元の global batch 順へ復元する。",
    "task ごとの index 集合から同数を選び、distributed batch 内のtask 構成を均衡させる sampler を定義する。",
    "task_name で分割した row を各 teacher worker group へ送り、得られた teacher signal を元の global row 順で入力 batch に書き戻す。",
    "rollout 生成から teacher forward、actor 更新、検証、checkpoint までの学習 phase を順序付ける trainer loop である。",
    "student actor の micro-batch forward、teacher 分布との KL、gradient accumulation、optimizer step を実行する。",
}

GENERIC_PATTERNS = (
    re.compile(r"^`[^`]+` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。$"),
    re.compile(r"^`[^`]+` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。$"),
    re.compile(r"^この(?:節|項目|値|要素|処理|行|論理行|module|class|function)\b"),
)


def is_generic_comment(text: str) -> bool:
    normalized = " ".join(text.split())
    return normalized in GENERIC_EXACT or any(
        pattern.search(normalized) for pattern in GENERIC_PATTERNS
    )


def annotation_type_for(lines: Iterable[str]) -> str:
    text = " ".join(lines).lower()
    if any(word in text for word in ("shape", "dtype", "tensor", "logit", "mask")):
        return "tensor_shape"
    if any(word in text for word in ("allgather", "reducescatter", "allreduce", "rank", "fsdp", "worker group")):
        return "distributed"
    if any(word in text for word in ("blocking", "同期", "semaphore", "future", "ray.get", "barrier")):
        return "synchronization"
    if any(word in text for word in ("config", "設定", "hydra", "omegaconf", "環境変数")):
        return "configuration"
    if any(word in text for word in ("例外", "失敗", "error", "raise", "fallback")):
        return "error_condition"
    if any(word in text for word in ("実験的", "experimental", "scaffolding")):
        return "experimental"
    if any(word in text for word in ("断定でき", "可能性", "推測", "要確認")):
        return "uncertainty"
    if any(word in text for word in ("gpu", "cpu", "driver", "remote", "実行", "phase")):
        return "execution_context"
    return "semantic"


def load_annotation_map() -> list[dict]:
    if not ANNOTATION_MAP.exists():
        return []
    rows = []
    for line in ANNOTATION_MAP.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def save_annotation_map(rows: list[dict]) -> None:
    rows.sort(
        key=lambda row: (
            row["original_path"],
            row["source_start_line"],
            row.get("sequence", 0),
        )
    )
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    ANNOTATION_MAP.write_text(text, encoding="utf-8", newline="\n")


def rows_by_path(rows: list[dict] | None = None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows if rows is not None else load_annotation_map():
        grouped[row["original_path"]].append(row)
    for path_rows in grouped.values():
        path_rows.sort(
            key=lambda row: (row["source_start_line"], row.get("sequence", 0))
        )
    return grouped


def comment_style_for(language: str) -> str:
    if language in {"javascript", "typescript", "c", "cpp", "java", "go", "rust", "protobuf", "json", "jsonc"}:
        return "slash"
    if language in {"markdown", "rst", "text"}:
        return "html"
    return "hash"


def render_comment(row: dict, newline: str) -> str:
    indent = row.get("indent", "")
    style = row.get("comment_style", "hash")
    rendered = []
    for text in row["annotation_lines"]:
        if style == "html":
            rendered.append(f"{indent}<!-- {text} -->{newline}")
        elif style == "slash":
            rendered.append(f"{indent}// {text}{newline}")
        else:
            rendered.append(f"{indent}# {text}{newline}")
    return "".join(rendered)


def render_notebook(source: bytes, original_path: str) -> bytes:
    notebook = json.loads(source.decode("utf-8"))
    output = [f"# {Path(original_path).name} セル別参照", ""]
    for index, cell in enumerate(notebook.get("cells", []), 1):
        cell_type = cell.get("cell_type", "unknown")
        output.extend([f"## Cell {index}: {cell_type}", ""])
        source_text = "".join(cell.get("source", []))
        if cell_type == "markdown":
            output.extend([source_text, ""])
        else:
            output.extend(["```python", source_text.rstrip("\n"), "```", ""])
    return ("\n".join(output).rstrip() + "\n").encode("utf-8")


def render_path(manifest_row: dict, path_rows: list[dict]) -> bytes:
    source = read_source_bytes(manifest_row["original_path"])
    if manifest_row["language"] == "notebook":
        return render_notebook(source, manifest_row["original_path"])

    text, encoding = decode_text(source)
    if text is None or encoding is None:
        raise ValueError(f"cannot decode {manifest_row['original_path']}")
    source_lines = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"
    by_line: dict[int, list[dict]] = defaultdict(list)
    for row in path_rows:
        by_line[row["source_start_line"]].append(row)

    output: list[str] = []
    for line_number, source_line in enumerate(source_lines, 1):
        for row in by_line.get(line_number, []):
            output.append(render_comment(row, newline))
        output.append(source_line)
    for row in by_line.get(len(source_lines) + 1, []):
        output.append(render_comment(row, newline))
    return "".join(output).encode(encoding)


def completed_manifest_rows() -> list[dict]:
    return [row for row in load_manifest() if row["status"] == "completed"]
