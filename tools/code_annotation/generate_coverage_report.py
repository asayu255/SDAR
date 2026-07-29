"""manifest から再開用 coverage report を生成する。"""

from __future__ import annotations

from collections import Counter

from common import DOC_ROOT, load_manifest


def main() -> None:
    rows = load_manifest()
    statuses = Counter(row["status"] for row in rows)
    targets = [row for row in rows if row["status"] != "skipped"]
    completed = [row for row in targets if row["status"] == "completed"]
    source_lines = sum(row["source_line_count"] for row in targets)
    completed_lines = sum(row["source_line_count"] for row in completed)
    preservation_passed = sum(
        row["source_preservation"] in {"passed", "not_applicable_notebook"}
        for row in completed
    )
    full_coverage = sum(row["annotation_coverage"] == 1.0 for row in completed)
    text = f"""# Annotation Coverage Report

- total inventoried files: {len(rows)}
- target files: {len(targets)}
- completed: {statuses.get("completed", 0)}
- in progress: {statuses.get("in_progress", 0)}
- pending: {statuses.get("pending", 0)}
- skipped: {statuses.get("skipped", 0)}
- blocked: {statuses.get("blocked", 0)}
- needs review: {statuses.get("needs_review", 0)}
- target source lines: {source_lines}
- completed source lines: {completed_lines}
- completed source preservation passed: {preservation_passed}/{len(completed)}
- completed annotation coverage 100%: {full_coverage}/{len(completed)}

この数値は `manifest.json` の機械判定から生成しています。`pending`、`blocked`、
`needs_review` が残る場合は完全完了ではありません。
"""
    (DOC_ROOT / "coverage_report.md").write_text(
        text.rstrip() + "\n", encoding="utf-8", newline="\n"
    )
    print(text)


if __name__ == "__main__":
    main()
