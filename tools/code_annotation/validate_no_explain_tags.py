"""最終mirrorに識別用 `[EXPLAIN]` tagが残っていないことを検証する。"""

from __future__ import annotations

from annotation_map_utils import completed_manifest_rows
from common import ROOT


def main() -> None:
    failures: list[str] = []
    for row in completed_manifest_rows():
        path = ROOT / row["annotated_path"]
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if "[EXPLAIN]" in line:
                failures.append(f"{row['annotated_path']}:{line_number}")
    for failure in failures:
        print(failure)
    print(f"explain_tag_count={len(failures)}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
