"""検証結果を集約し、条件を満たす manifest row だけ completed にする。"""

from __future__ import annotations

from common import load_manifest, save_manifest


SYNTAX_REQUIRED = {"python", "shell", "yaml", "toml"}


def main() -> None:
    manifest = load_manifest()
    completed = 0
    for row in manifest:
        if row["status"] not in {"in_progress", "needs_review"}:
            continue
        preservation_ok = row["source_preservation"] in {
            "passed",
            "not_applicable_notebook",
        }
        coverage_ok = row["annotation_coverage"] == 1.0
        if row["language"] in SYNTAX_REQUIRED:
            syntax_ok = row["syntax_validation"] == "passed"
        else:
            if row["syntax_validation"] == "not_checked":
                row["syntax_validation"] = "not_applicable"
            syntax_ok = row["syntax_validation"] == "not_applicable"
        if preservation_ok and coverage_ok and syntax_ok:
            row["status"] = "completed"
            completed += 1
    save_manifest(manifest)
    print(f"newly_completed={completed}")


if __name__ == "__main__":
    main()
