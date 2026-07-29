"""注釈版 Python を compile() し、構文を検証する。"""

from __future__ import annotations

from common import ROOT, load_manifest, save_manifest


def main() -> None:
    manifest = load_manifest()
    failures = []
    checked = 0
    for row in manifest:
        if row["language"] != "python" or row["status"] not in {
            "in_progress",
            "completed",
            "needs_review",
        }:
            continue
        path = ROOT / row["annotated_path"]
        try:
            compile(path.read_bytes(), str(path), "exec")
            row["syntax_validation"] = "passed"
        except Exception as exc:  # noqa: BLE001 - report every syntax/decode failure
            row["syntax_validation"] = f"failed:{type(exc).__name__}:{exc}"
            failures.append(row["original_path"])
        checked += 1
    save_manifest(manifest)
    print(f"python_checked={checked} failures={len(failures)}")
    for path in failures:
        print(f"FAIL {path}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
