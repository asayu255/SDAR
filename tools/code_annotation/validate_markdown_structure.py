"""Markdown/RST注釈mapがfenceやfront matter内部を指していないことを検証する。"""

from __future__ import annotations

from collections import defaultdict

from annotation_map_utils import load_annotation_map
from common import ROOT, load_manifest


def main() -> None:
    annotations = defaultdict(list)
    for row in load_annotation_map():
        annotations[row["original_path"]].append(row)

    failures: list[str] = []
    checked = 0
    for row in load_manifest():
        if row["status"] != "completed" or row["language"] not in {"markdown", "rst"}:
            continue
        checked += 1
        source_lines = (ROOT / row["original_path"]).read_text(
            encoding="utf-8"
        ).splitlines()

        fence_lines: set[int] = set()
        in_fence = False
        fence_marker = ""
        front_matter_lines: set[int] = set()
        in_front_matter = bool(source_lines and source_lines[0].strip() == "---")
        for line_number, line in enumerate(source_lines, 1):
            stripped = line.strip()
            if in_front_matter:
                front_matter_lines.add(line_number)
                if line_number > 1 and stripped == "---":
                    in_front_matter = False
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                    fence_lines.add(line_number)
                elif marker == fence_marker:
                    fence_lines.add(line_number)
                    in_fence = False
                    fence_marker = ""
                continue
            if in_fence:
                fence_lines.add(line_number)

        for annotation in annotations[row["original_path"]]:
            line_number = int(annotation["source_start_line"])
            if line_number in fence_lines:
                failures.append(
                    f"{row['original_path']}:{line_number}: annotation inside fence"
                )
            if line_number in front_matter_lines:
                failures.append(
                    f"{row['original_path']}:{line_number}: annotation inside front matter"
                )

    for failure in failures:
        print(failure)
    print(f"markdown_checked={checked} failures={len(failures)}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
