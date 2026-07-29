"""旧 `[EXPLAIN]` コメントから自然文注釈マップを抽出する。

このscriptは旧mirrorを読むだけで、annotated treeは変更しない。汎用テンプレートは
mapへ入れず、固有説明だけを保持候補として移行する。
"""

from __future__ import annotations

import re
from pathlib import Path

from annotation_map_utils import (
    annotation_type_for,
    comment_style_for,
    completed_manifest_rows,
    is_generic_comment,
    save_annotation_map,
)
from common import ROOT, decode_text, read_source_bytes


TAG_RE = re.compile(r"\[EXPLAIN\](?:\[([^\]]+)\])?\s*")


def clean_legacy_line(line: str) -> tuple[str, str, str] | None:
    match = TAG_RE.search(line)
    if not match:
        return None
    qualifier = match.group(1)
    indent = line[: len(line) - len(line.lstrip())]
    content = line[match.end() :].strip()
    if content.endswith("-->"):
        content = content[:-3].rstrip()
    if qualifier == "推測":
        content = f"コードだけから設計意図までは断定できないが、{content}"
    elif qualifier == "非実効":
        content = f"この経路は現在のPure OPD実行では非実効であり、{content}"
    elif qualifier == "実験的":
        content = f"この経路は実験的であり、{content}"
    elif qualifier == "要確認":
        content = f"実装上の不確実性として、{content}"
    return indent, content, qualifier or ""


def main() -> None:
    rows: list[dict] = []
    legacy_lines = 0
    deleted_generic_lines = 0
    retained_lines = 0
    retained_blocks = 0

    for manifest_row in completed_manifest_rows():
        if manifest_row["language"] == "notebook":
            continue
        target = ROOT / manifest_row["annotated_path"]
        source = read_source_bytes(manifest_row["original_path"])
        source_text, _ = decode_text(source)
        annotated_text, _ = decode_text(target.read_bytes())
        if source_text is None or annotated_text is None:
            continue
        source_lines = source_text.splitlines()
        consumed_source = 0
        pending: list[tuple[str, str, str]] = []
        sequence = 0

        def flush() -> None:
            nonlocal pending, retained_lines, retained_blocks, sequence
            if not pending:
                return
            kept = [item for item in pending if item[1] and not is_generic_comment(item[1])]
            if kept:
                annotation_lines = [item[1] for item in kept]
                sequence += 1
                rows.append(
                    {
                        "original_path": manifest_row["original_path"],
                        "source_start_line": min(consumed_source + 1, len(source_lines) + 1),
                        "source_end_line": min(consumed_source + 1, len(source_lines)),
                        "annotation_lines": annotation_lines,
                        "annotation_type": annotation_type_for(annotation_lines),
                        "review_status": "reviewed",
                        "legacy_disposition": "keep",
                        "indent": kept[0][0],
                        "comment_style": comment_style_for(manifest_row["language"]),
                        "sequence": sequence,
                    }
                )
                retained_lines += len(kept)
                retained_blocks += 1
            pending = []

        reconstructed: list[str] = []
        for line in annotated_text.splitlines():
            cleaned = clean_legacy_line(line)
            if cleaned is not None:
                legacy_lines += 1
                if not cleaned[1] or is_generic_comment(cleaned[1]):
                    deleted_generic_lines += 1
                pending.append(cleaned)
                continue
            flush()
            reconstructed.append(line)
            consumed_source += 1
        flush()

        if reconstructed != source_lines:
            raise RuntimeError(
                f"legacy mirror does not reconstruct source: {manifest_row['original_path']}"
            )

    save_annotation_map(rows)
    print(f"legacy_explain_lines={legacy_lines}")
    print(f"retained_blocks={retained_blocks}")
    print(f"retained_lines={retained_lines}")
    print(f"deleted_generic_lines={deleted_generic_lines}")


if __name__ == "__main__":
    main()
