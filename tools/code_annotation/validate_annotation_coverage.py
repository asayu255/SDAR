"""論理行の直前に `[EXPLAIN]` がある割合を検証する。"""

from __future__ import annotations

import argparse

from annotation_rules import eligible_lines
from common import ROOT, decode_text, load_manifest, read_source_bytes, save_manifest


def coverage_for(source: str, annotated: str, language: str) -> tuple[int, int]:
    eligible = eligible_lines(language, source)
    if not eligible:
        return 0, 0
    source_number = 0
    covered = 0
    explain_since_source = False
    for line in annotated.splitlines():
        if "[EXPLAIN]" in line:
            explain_since_source = True
            continue
        source_number += 1
        if source_number in eligible and explain_since_source:
            covered += 1
        explain_since_source = False
    return covered, len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", default=[])
    args = parser.parse_args()
    selected = set(args.path)
    manifest = load_manifest()
    failures = []
    checked = 0
    for row in manifest:
        if row["status"] not in {"in_progress", "completed", "needs_review"}:
            continue
        if selected and row["original_path"] not in selected:
            continue
        target = ROOT / row["annotated_path"]
        if not target.exists():
            failures.append(row["original_path"])
            continue
        if row["language"] == "notebook":
            row["annotation_coverage"] = 1.0
            continue
        source, _ = decode_text(read_source_bytes(row["original_path"]))
        annotated, _ = decode_text(target.read_bytes())
        if source is None or annotated is None:
            failures.append(row["original_path"])
            continue
        covered, total = coverage_for(source, annotated, row["language"])
        ratio = 1.0 if total == 0 else covered / total
        row["annotation_coverage"] = round(ratio, 6)
        if ratio != 1.0:
            failures.append(row["original_path"])
        checked += 1
    save_manifest(manifest)
    print(f"checked={checked} failures={len(failures)}")
    for path in failures:
        print(f"FAIL {path}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
