"""master と固定 source commit の branch delta manifest を作成する。"""

from __future__ import annotations

import json

from common import DOC_ROOT, branch_delta, load_manifest, source_commit


def main() -> None:
    delta = branch_delta()
    by_path = {row["original_path"]: row for row in load_manifest()}
    rows = []
    for path, git_status in sorted(delta.items()):
        manifest_row = by_path.get(path)
        rows.append(
            {
                "source_commit": source_commit(),
                "original_path": path,
                "git_status_vs_master": git_status,
                "target_status": (
                    manifest_row["status"] if manifest_row else "source_only"
                ),
                "execution_relevance": (
                    manifest_row["execution_relevance"]
                    if manifest_row
                    else "unrelated"
                ),
                "annotated_path": (
                    manifest_row["annotated_path"] if manifest_row else ""
                ),
            }
        )
    (DOC_ROOT / "branch_delta_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"branch_delta_files={len(rows)}")


if __name__ == "__main__":
    main()
