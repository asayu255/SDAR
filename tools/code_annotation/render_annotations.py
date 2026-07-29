"""source commitの元fileとannotation_mapから自然文mirrorを再生成する。"""

from __future__ import annotations

import argparse

from annotation_map_utils import (
    completed_manifest_rows,
    render_path,
    rows_by_path,
)
from common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    selected = set(args.path)
    grouped = rows_by_path()
    rendered = 0

    for manifest_row in completed_manifest_rows():
        original_path = manifest_row["original_path"]
        if not args.all and original_path not in selected:
            continue
        target = ROOT / manifest_row["annotated_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(render_path(manifest_row, grouped.get(original_path, [])))
        rendered += 1
    print(f"rendered={rendered}")


if __name__ == "__main__":
    main()
