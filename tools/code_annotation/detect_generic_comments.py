"""annotation_mapに禁止された汎用テンプレートがないことを検証する。"""

from __future__ import annotations

from annotation_map_utils import is_generic_comment, load_annotation_map


def main() -> None:
    failures: list[str] = []
    for row in load_annotation_map():
        for text in row["annotation_lines"]:
            if is_generic_comment(text):
                failures.append(
                    f"{row['original_path']}:{row['source_start_line']}: {text}"
                )
    for failure in failures:
        print(failure)
    print(f"generic_comment_count={len(failures)}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
