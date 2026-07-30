"""日本語解説付きPythonからコメントだけを現行sourceの注釈mapへ移植する。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import re
import tokenize
from collections import defaultdict
from pathlib import Path

from annotation_map_utils import (
    annotation_type_for,
    load_annotation_map,
    save_annotation_map,
)
from common import ROOT


JAPANESE = re.compile(r"[ぁ-んァ-ヶ一-龠々〆〤]")


def normalized_code(line: str) -> str:
    """inline commentを除き、空白差を無視できる1行anchorを作る。"""

    comment_col = None
    try:
        for token in tokenize.generate_tokens(io.StringIO(line).readline):
            if token.type == tokenize.COMMENT:
                comment_col = token.start[1]
                break
    except (tokenize.TokenError, IndentationError):
        pass
    if comment_col is not None:
        line = line[:comment_col]
    return " ".join(line.strip().split())


def source_anchors(lines: list[str]) -> dict[str, list[int]]:
    anchors: dict[str, list[int]] = defaultdict(list)
    for line_number, line in enumerate(lines, 1):
        key = normalized_code(line)
        if key:
            anchors[key].append(line_number)
    return anchors


def nearest_anchor(
    explained_lines: list[str],
    row: int,
    source_lines: list[str],
    anchors: dict[str, list[int]],
    inline_prefix: str | None = None,
) -> int | None:
    expected = max(1, round(row / max(1, len(explained_lines)) * len(source_lines)))
    candidates: list[tuple[int, int]] = []
    if inline_prefix:
        key = " ".join(inline_prefix.strip().split())
        candidates.extend((abs(line - expected), line) for line in anchors.get(key, []))

    # 古い実装行に付いたコメントでも、直後または直前の不変なcode行へ寄せられる。
    for distance in range(0, 80):
        for candidate_row in (row + distance, row - distance):
            if not 1 <= candidate_row <= len(explained_lines):
                continue
            key = normalized_code(explained_lines[candidate_row - 1])
            if not key:
                continue
            candidates.extend(
                (distance * 1000 + abs(line - expected), line)
                for line in anchors.get(key, [])
            )
        if candidates:
            break
    return min(candidates)[1] if candidates else None


def comment_groups(text: str) -> list[dict]:
    lines = text.splitlines()
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    comments = [token for token in tokens if token.type == tokenize.COMMENT]
    groups: list[dict] = []
    current: dict | None = None
    for token in comments:
        row, col = token.start
        prefix = lines[row - 1][:col]
        full_line = not prefix.strip()
        clean = token.string.lstrip("#").strip()
        if full_line and current and current["full_line"] and row == current["end_row"] + 1:
            current["end_row"] = row
            current["lines"].append(clean)
            continue
        current = {
            "row": row,
            "end_row": row,
            "full_line": full_line,
            "inline_prefix": None if full_line else prefix,
            "lines": [clean],
        }
        groups.append(current)
    return [
        group for group in groups
        if JAPANESE.search(" ".join(group["lines"]))
    ]


def node_docstrings(tree: ast.AST) -> dict[str, tuple[int, str]]:
    output: dict[str, tuple[int, str]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        if isinstance(node, ast.Module):
            name = "<module>"
            line = node.body[0].lineno if node.body else 1
            next_prefix = ""
        else:
            node_name = getattr(node, "name", type(node).__name__)
            name = f"{prefix}.{node_name}" if prefix else node_name
            line = int(getattr(node, "lineno", 1))
            next_prefix = name
        doc = ast.get_docstring(node, clean=False)
        if doc:
            output[name] = (line, doc)
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, next_prefix)

    visit(tree, "")
    return output


def japanese_doc_lines(doc: str) -> list[str]:
    lines = []
    for line in doc.splitlines():
        clean = line.strip()
        if clean and JAPANESE.search(clean):
            lines.append(clean)
    return lines


def import_file(original_path: str, explained_path: Path, priority: str | None) -> list[dict]:
    source_text = (ROOT / original_path).read_text(encoding="utf-8")
    explained_text = explained_path.read_text(encoding="utf-8-sig")
    source_lines = source_text.splitlines()
    explained_lines = explained_text.splitlines()
    anchors = source_anchors(source_lines)
    source_comment_texts = {
        token.string.lstrip("#").strip()
        for token in tokenize.generate_tokens(io.StringIO(source_text).readline)
        if token.type == tokenize.COMMENT
    }
    sha256 = hashlib.sha256(explained_path.read_bytes()).hexdigest()
    rows: list[dict] = []
    skipped = 0

    for group in comment_groups(explained_text):
        anchor = nearest_anchor(
            explained_lines,
            group["end_row"] + (1 if group["full_line"] else 0),
            source_lines,
            anchors,
            group["inline_prefix"],
        )
        if anchor is None:
            skipped += 1
            continue
        annotation_lines = [
            line for line in group["lines"]
            if line and line not in source_comment_texts
        ]
        if not annotation_lines:
            continue
        rows.append({
            "original_path": original_path,
            "source_start_line": anchor,
            "source_end_line": anchor,
            "annotation_lines": annotation_lines,
            "annotation_type": annotation_type_for(annotation_lines),
            "review_status": "reviewed",
            "legacy_disposition": "attachment_import",
            "indent": source_lines[anchor - 1][:len(source_lines[anchor - 1]) - len(source_lines[anchor - 1].lstrip())],
            "comment_style": "hash",
            "sequence": len(rows),
            "curated_id": f"attachment-{Path(original_path).stem}-comment-{len(rows) + 1}",
            "priority": priority,
            "attachment_sha256": sha256,
        })

    source_docs = node_docstrings(ast.parse(source_text, filename=original_path))
    explained_docs = node_docstrings(ast.parse(explained_text, filename=str(explained_path)))
    for name, (_, doc) in explained_docs.items():
        annotation_lines = japanese_doc_lines(doc)
        if not annotation_lines or name not in source_docs:
            continue
        anchor = source_docs[name][0]
        rows.append({
            "original_path": original_path,
            "source_start_line": anchor,
            "source_end_line": anchor,
            "annotation_lines": annotation_lines,
            "annotation_type": annotation_type_for(annotation_lines),
            "review_status": "reviewed",
            "legacy_disposition": "attachment_import",
            "indent": source_lines[anchor - 1][:len(source_lines[anchor - 1]) - len(source_lines[anchor - 1].lstrip())],
            "comment_style": "hash",
            "sequence": len(rows),
            "curated_id": f"attachment-{Path(original_path).stem}-doc-{len(rows) + 1}",
            "priority": priority,
            "attachment_sha256": sha256,
        })

    # 同じanchor・同じ文面の重複だけを除去する。
    unique = {}
    for row in rows:
        key = (row["source_start_line"], tuple(row["annotation_lines"]))
        unique.setdefault(key, row)
    result = list(unique.values())
    print(
        f"{original_path}: imported={len(result)} skipped_unanchored={skipped} "
        f"attachment={explained_path.name}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("ORIGINAL_PATH", "EXPLAINED_PATH", "PRIORITY"),
        required=True,
        help="PRIORITYはA-Dまたは-",
    )
    args = parser.parse_args()

    imported: list[dict] = []
    replaced_paths = set()
    for original_path, explained_path, priority in args.pair:
        replaced_paths.add(original_path)
        imported.extend(
            import_file(
                original_path,
                Path(explained_path),
                None if priority == "-" else priority,
            )
        )

    rows = [
        row for row in load_annotation_map()
        if row["original_path"] not in replaced_paths
    ]
    rows.extend(imported)
    save_annotation_map(rows)
    print(f"replaced_paths={len(replaced_paths)} imported_total={len(imported)}")


if __name__ == "__main__":
    main()
