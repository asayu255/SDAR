"""`[EXPLAIN]` 行を除去した注釈版が source と一致するか検証する。"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, decode_text, load_manifest, read_source_bytes, save_manifest


def normalize_newlines(text: str) -> str:
    return text.replace("\n", "\n").replace("\n", "\n")


def strip_explain_lines(text: str) -> str:
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if "[EXPLAIN]" not in line
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", default=[])
    args = parser.parse_args()
    selected = set(args.path)
    manifest = load_manifest()
    failures: list[str] = []
    checked = 0
    for row in manifest:
        if row["status"] not in {"in_progress", "completed", "needs_review"}:
            continue
        if selected and row["original_path"] not in selected:
            continue
        if row["language"] == "notebook":
            row["source_preservation"] = "not_applicable_notebook"
            continue
        target = ROOT / row["annotated_path"]
        if not target.exists():
            row["source_preservation"] = "missing"
            failures.append(row["original_path"])
            continue
        source_text, _ = decode_text(read_source_bytes(row["original_path"]))
        annotated_text, _ = decode_text(target.read_bytes())
        if source_text is None or annotated_text is None:
            row["source_preservation"] = "decode_failed"
            failures.append(row["original_path"])
            continue
        restored = strip_explain_lines(annotated_text)
        if normalize_newlines(restored) == normalize_newlines(source_text):
            row["source_preservation"] = "passed"
        else:
            row["source_preservation"] = "failed"
            failures.append(row["original_path"])
        checked += 1
    save_manifest(manifest)
    print(f"checked={checked} failures={len(failures)}")
    for path in failures:
        print(f"FAIL {path}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
