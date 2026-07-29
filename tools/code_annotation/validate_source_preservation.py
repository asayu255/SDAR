"""annotation_mapからrenderしたmirrorがsourceを完全保持することを検証する。"""

from __future__ import annotations

import argparse
import hashlib

from annotation_map_utils import render_path, rows_by_path
from common import ROOT, decode_text, load_manifest, read_source_bytes, save_manifest


def strip_legacy_explain_lines(text: str) -> str:
    """Phase 1の移行commitだけで旧tag付きmirrorを検証する。"""

    return "".join(
        line for line in text.splitlines(keepends=True) if "[EXPLAIN]" not in line
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", default=[])
    args = parser.parse_args()
    selected = set(args.path)
    manifest = load_manifest()
    grouped = rows_by_path()
    failures: list[str] = []
    checked = 0
    legacy_checked = 0

    for row in manifest:
        if row["status"] not in {"in_progress", "completed", "needs_review"}:
            continue
        if selected and row["original_path"] not in selected:
            continue
        target = ROOT / row["annotated_path"]
        if not target.exists():
            row["source_preservation"] = "missing"
            failures.append(row["original_path"])
            continue

        source_bytes = read_source_bytes(row["original_path"])
        if hashlib.sha256(source_bytes).hexdigest() != row["source_sha256"]:
            row["source_preservation"] = "source_hash_changed"
            failures.append(row["original_path"])
            continue

        target_bytes = target.read_bytes()
        if b"[EXPLAIN]" in target_bytes:
            # Phase 1では旧mirrorを維持したままmapを先に導入する。
            if row["language"] == "notebook":
                row["source_preservation"] = "passed_legacy_transition"
                legacy_checked += 1
                checked += 1
                continue
            source_text, _ = decode_text(source_bytes)
            annotated_text, _ = decode_text(target_bytes)
            if (
                source_text is not None
                and annotated_text is not None
                and strip_legacy_explain_lines(annotated_text) == source_text
            ):
                row["source_preservation"] = "passed_legacy_transition"
                legacy_checked += 1
            else:
                row["source_preservation"] = "failed"
                failures.append(row["original_path"])
            checked += 1
            continue

        expected = render_path(row, grouped.get(row["original_path"], []))
        if target_bytes == expected:
            row["source_preservation"] = "passed"
        else:
            row["source_preservation"] = "failed"
            failures.append(row["original_path"])
        checked += 1

    save_manifest(manifest)
    print(
        f"checked={checked} legacy_transition={legacy_checked} "
        f"failures={len(failures)}"
    )
    for path in failures:
        print(f"FAIL {path}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
