"""manifest、block coverage、semantic reviewから品質reportを生成する。"""

from __future__ import annotations

import json
from collections import Counter

from annotation_map_utils import load_annotation_map
from common import DOC_ROOT, load_manifest


def _jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    manifest = load_manifest()
    targets = [row for row in manifest if row["status"] != "skipped"]
    annotations = load_annotation_map()
    blocks = _jsonl(DOC_ROOT / "block_coverage.jsonl")
    reviews = json.loads((DOC_ROOT / "semantic_review.json").read_text(encoding="utf-8"))
    statuses = Counter(row["status"] for row in manifest)
    block_statuses = Counter(row["coverage_status"] for row in blocks)
    annotation_types = Counter(row["annotation_type"] for row in annotations)

    text = f"""# Annotation Coverage Report

- total inventoried files: {len(manifest)}
- target files: {len(targets)}
- completed: {statuses.get("completed", 0)}
- skipped: {statuses.get("skipped", 0)}
- pending / blocked / needs review: {statuses.get("pending", 0)} / {statuses.get("blocked", 0)} / {statuses.get("needs_review", 0)}
- source preservation passed: {sum(row["source_preservation"] in {"passed", "not_applicable_notebook"} for row in targets)}/{len(targets)}
- semantic annotation blocks: {len(annotations)}
- analyzed source blocks: {len(blocks)}
- explained: {block_statuses.get("explained", 0)}
- covered by parent comment: {block_statuses.get("covered_by_parent_comment", 0)}
- self explanatory: {block_statuses.get("self_explanatory", 0)}
- needs review: {block_statuses.get("needs_review", 0)}

## Priority review

| Priority | reviewed | required |
| --- | ---: | ---: |
"""
    for priority in ("A", "B", "C", "D"):
        rows = [row for row in reviews if row["priority"] == priority]
        reviewed = sum(row["review_status"] != "needs_review" for row in rows)
        text += f"| {priority} | {reviewed} | {len(rows)} |\n"

    text += "\n## Annotation types\n\n"
    for name, count in sorted(annotation_types.items()):
        text += f"- {name}: {count}\n"
    text += (
        "\n行ごとの強制coverageは使用していません。意味blockを"
        "`explained`、`covered_by_parent_comment`、`self_explanatory`、"
        "`needs_review`へ分類し、Priority A〜Dは別途全pathのsemantic reviewを必須にしています。\n"
    )
    (DOC_ROOT / "coverage_report.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
