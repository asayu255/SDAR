"""利用可能な bash で注釈版 shell script の `bash -n` を実行する。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from common import ROOT, load_manifest, save_manifest


def main() -> None:
    manifest = load_manifest()
    candidates_bash = [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    ]
    bash = next((str(path) for path in candidates_bash if path.exists()), None)
    if bash is None:
        bash = shutil.which("bash")
    candidates = [
        row
        for row in manifest
        if row["language"] == "shell"
        and row["status"] in {"in_progress", "completed", "needs_review"}
    ]
    if not bash:
        for row in candidates:
            row["syntax_validation"] = "blocked:bash_not_found"
        save_manifest(manifest)
        print(f"shell_checked=0 blocked={len(candidates)} reason=bash_not_found")
        return
    failures = []
    for row in candidates:
        path = ROOT / row["annotated_path"]
        result = subprocess.run(
            [bash, "-n", str(path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            row["syntax_validation"] = f"failed:{result.stderr.strip()}"
            failures.append(row["original_path"])
        else:
            row["syntax_validation"] = "passed"
    save_manifest(manifest)
    print(f"shell_checked={len(candidates)} failures={len(failures)}")
    for path in failures:
        print(f"FAIL {path}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
