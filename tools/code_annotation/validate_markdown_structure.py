"""Markdown注釈がfence内部やfront matterの前へ入っていないことを検証する。"""

from __future__ import annotations

from common import ROOT, load_manifest


def main() -> None:
    failures: list[str] = []
    checked = 0
    for row in load_manifest():
        if row["status"] != "completed" or row["language"] not in {"markdown", "rst"}:
            continue
        checked += 1
        source_lines = (ROOT / row["original_path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        annotated_lines = (ROOT / row["annotated_path"]).read_text(
            encoding="utf-8"
        ).splitlines()

        if (
            source_lines
            and source_lines[0].strip() == "---"
            and (not annotated_lines or annotated_lines[0].strip() != "---")
        ):
            failures.append(f"{row['annotated_path']}: front matter moved")

        in_fence = False
        fence_marker = ""
        for line_number, line in enumerate(annotated_lines, 1):
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                    fence_marker = ""
                continue
            if in_fence and "[EXPLAIN]" in line:
                failures.append(
                    f"{row['annotated_path']}:{line_number}: EXPLAIN inside fence"
                )

    for failure in failures:
        print(failure)
    print(f"markdown_checked={checked} failures={len(failures)}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
