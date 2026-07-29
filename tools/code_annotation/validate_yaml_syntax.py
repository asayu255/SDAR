"""PyYAML または OmegaConf があれば注釈版 YAML の構文を検証する。"""

from __future__ import annotations

from common import ROOT, load_manifest, save_manifest


def main() -> None:
    manifest = load_manifest()
    candidates = [
        row
        for row in manifest
        if row["language"] == "yaml"
        and row["status"] in {"in_progress", "completed", "needs_review"}
    ]
    loader = None
    try:
        import yaml

        loader = lambda text: yaml.safe_load(text)
    except ImportError:
        try:
            from omegaconf import OmegaConf

            loader = lambda text: OmegaConf.create(text)
        except ImportError:
            pass
    if loader is None:
        for row in candidates:
            row["syntax_validation"] = "blocked:yaml_parser_not_found"
        save_manifest(manifest)
        print(
            f"yaml_checked=0 blocked={len(candidates)} "
            "reason=yaml_parser_not_found"
        )
        return
    failures = []
    for row in candidates:
        try:
            loader((ROOT / row["annotated_path"]).read_text(encoding="utf-8"))
            row["syntax_validation"] = "passed"
        except Exception as exc:  # noqa: BLE001 - parser-specific exception
            row["syntax_validation"] = f"failed:{type(exc).__name__}:{exc}"
            failures.append(row["original_path"])
    save_manifest(manifest)
    print(f"yaml_checked={len(candidates)} failures={len(failures)}")
    for path in failures:
        print(f"FAIL {path}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
