"""manifest の対象ファイルへ `[EXPLAIN]` 行だけを挿入してミラーする。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation_rules import make_annotated_text
from common import ROOT, decode_text, load_manifest, read_source_bytes, save_manifest


def annotate_notebook(source: bytes, original_path: str) -> str:
    notebook = json.loads(source.decode("utf-8"))
    output = [
        f"# {Path(original_path).name} セル別解説",
        "",
        "<!-- [EXPLAIN] Notebook JSON は変更せず、cell 順序を保った解説を作成する。 -->",
        "",
    ]
    for index, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type", "unknown")
        output.extend(
            [
                f"## Cell {index + 1}: {cell_type}",
                "",
                "<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->",
                "",
            ]
        )
        source_text = "".join(cell.get("source", []))
        if cell_type == "markdown":
            output.extend([source_text, ""])
        else:
            output.extend(["```python", source_text.rstrip("\n"), "```", ""])
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--relevance", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    selected_paths = set(args.path)
    selected_relevance = set(args.relevance)
    manifest = load_manifest()
    changed = 0
    for row in manifest:
        if row["status"] not in {"pending", "needs_review"}:
            continue
        selected = args.all
        selected |= row["original_path"] in selected_paths
        selected |= row["execution_relevance"] in selected_relevance
        if not selected:
            continue

        source = read_source_bytes(row["original_path"])
        target = ROOT / row["annotated_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if row["language"] == "notebook":
            annotated = annotate_notebook(source, row["original_path"])
            target.write_text(annotated, encoding="utf-8", newline="\n")
            row["source_preservation"] = "not_applicable_notebook"
            inserted = len(json.loads(source.decode("utf-8")).get("cells", []))
        else:
            text, encoding = decode_text(source)
            if text is None or encoding is None:
                row["status"] = "blocked"
                row["skip_reason"] = "source_decode_failed"
                continue
            annotated, inserted = make_annotated_text(
                row["language"], text, row["original_path"]
            )
            target.write_text(annotated, encoding=encoding, newline="")
        row["status"] = "in_progress"
        row["notes"] += f"; generated_explain_lines={inserted}"
        changed += 1

    save_manifest(manifest)
    print(f"generated={changed}")


if __name__ == "__main__":
    main()
