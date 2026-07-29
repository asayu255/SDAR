"""固定 source commit の全 tracked file を inventory する。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import (
    DOC_ROOT,
    LOCK_NAMES,
    ROOT,
    TEXT_EXTENSIONS,
    annotated_path_for,
    annotation_mode_for,
    branch_delta,
    count_lines,
    decode_text,
    execution_relevance,
    file_type_for,
    is_generated,
    is_vendor,
    language_for,
    production_status,
    read_source_bytes,
    sha256,
    source_commit,
    tracked_paths,
)


FIELDS = [
    "source_commit",
    "original_path",
    "annotated_path",
    "language",
    "file_type",
    "source_size_bytes",
    "source_line_count",
    "source_nonempty_line_count",
    "source_sha256",
    "changed_vs_master",
    "git_status_vs_master",
    "execution_relevance",
    "production_status",
    "annotation_mode",
    "status",
    "skip_reason",
    "annotation_coverage",
    "source_preservation",
    "syntax_validation",
    "baseline_test_relevance",
    "notes",
    "last_commit",
]


def classify(path: str, changed: bool, text: str | None) -> tuple[str, str]:
    suffix = Path(path).suffix.lower()
    name = Path(path).name
    if text is None:
        return "skipped", "binary_or_non_utf8"
    if name in LOCK_NAMES or is_generated(path):
        return "skipped", "generated_or_lock_file"
    if path.startswith(".claude/") or path.startswith(".vscode/"):
        return "skipped", "local_tool_configuration"
    if is_vendor(path, changed):
        return "skipped", "unchanged_vendor_copy"
    if suffix not in TEXT_EXTENSIONS and name not in {"Dockerfile", "Makefile"}:
        return "skipped", "unsupported_or_non_source_text"
    return "pending", ""


def main() -> None:
    commit = source_commit()
    delta = branch_delta()
    rows: list[dict] = []
    for path in tracked_paths(commit):
        data = read_source_bytes(path)
        text, encoding = decode_text(data)
        language = language_for(path)
        changed = path in delta
        status, skip_reason = classify(path, changed, text)
        line_count, nonempty_count = count_lines(text or "")
        row = {
            "source_commit": commit,
            "original_path": path,
            "annotated_path": (
                annotated_path_for(path, language) if status == "pending" else ""
            ),
            "language": language,
            "file_type": file_type_for(path, language),
            "source_size_bytes": len(data),
            "source_line_count": line_count,
            "source_nonempty_line_count": nonempty_count,
            "source_sha256": sha256(data),
            "changed_vs_master": changed,
            "git_status_vs_master": delta.get(path, ""),
            "execution_relevance": execution_relevance(path, changed),
            "production_status": production_status(path),
            "annotation_mode": annotation_mode_for(language),
            "status": status,
            "skip_reason": skip_reason,
            "annotation_coverage": 0.0 if status == "pending" else None,
            "source_preservation": "not_checked" if status == "pending" else "skipped",
            "syntax_validation": "not_checked" if status == "pending" else "skipped",
            "baseline_test_relevance": (
                "direct" if path.startswith("tests/") else "indirect"
            ),
            "notes": f"source_encoding={encoding or 'binary'}",
            "last_commit": "",
        }
        rows.append(row)

    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (DOC_ROOT / "manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    pending = sum(row["status"] == "pending" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    print(
        f"source_commit={commit} total={len(rows)} "
        f"pending={pending} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
