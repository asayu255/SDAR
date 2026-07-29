"""SDAR 注釈ミラー用の共通定義と Git 補助関数。"""

from __future__ import annotations

import hashlib
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = ROOT / "docs" / "code_explanation"
ANNOTATED_ROOT = DOC_ROOT / "annotated"
SOURCE_COMMIT_FILE = DOC_ROOT / "SOURCE_COMMIT"

SAFE_DIRECTORY = ROOT.as_posix()

CRITICAL_PATHS = {
    "examples/opd_trainer/run_multitask_qwen3.sh",
    "examples/opd_trainer/expected_multitask_config.yaml",
    "verl/trainer/main_opd.py",
    "verl/trainer/ppo/opd_ray_trainer.py",
    "verl/trainer/ppo/core_algos.py",
    "verl/workers/actor/dp_actor.py",
    "verl/workers/fsdp_workers.py",
    "tests/trainer/test_opd_routing.py",
    "tests/trainer/test_expected_config.py",
}

TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".sh",
    ".bash",
    ".zsh",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".md",
    ".rst",
    ".txt",
    ".json",
    ".jsonc",
    ".ipynb",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".go",
    ".rs",
    ".sql",
    ".proto",
}

LOCK_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "uv.lock",
    "Cargo.lock",
}

GENERATED_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    "build",
    "dist",
    "_build",
    "htmlcov",
    "node_modules",
}

VENDOR_PREFIXES = (
    "agent_system/environments/env_package/alfworld/alfworld/",
)


def git(*args: str, check: bool = True) -> str:
    """Git をリポジトリ所有者差異の影響を受けず UTF-8 で実行する。"""

    cmd = ["git", "-c", f"safe.directory={SAFE_DIRECTORY}", *args]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode:
        raise RuntimeError(
            f"git command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def source_commit() -> str:
    return SOURCE_COMMIT_FILE.read_text(encoding="utf-8").strip()


def tracked_paths(commit: str | None = None) -> list[str]:
    commit = commit or source_commit()
    raw = git("ls-tree", "-r", "-z", "--name-only", commit)
    return [path for path in raw.split("\0") if path]


def branch_delta() -> dict[str, str]:
    """master との merge-base から source までの path -> status を返す。"""

    commit = source_commit()
    merge_base = git("merge-base", "origin/master", commit).strip()
    rows = git("diff", "--name-status", "-z", merge_base, commit).split("\0")
    result: dict[str, str] = {}
    i = 0
    while i < len(rows) and rows[i]:
        status = rows[i]
        if status.startswith(("R", "C")):
            old_path, new_path = rows[i + 1], rows[i + 2]
            result[old_path] = f"{status}:source"
            result[new_path] = status
            i += 3
        else:
            result[rows[i + 1]] = status
            i += 2
    return result


def read_source_bytes(path: str) -> bytes:
    data = (ROOT / path).read_bytes()
    return data


def decode_text(data: bytes) -> tuple[str | None, str | None]:
    if b"\x00" in data:
        return None, None
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    return None, None


def language_for(path: str) -> str:
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    if name in {"Dockerfile", "Makefile"} or suffix == ".mk":
        return "make"
    return {
        ".py": "python",
        ".pyi": "python",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "config",
        ".conf": "config",
        ".md": "markdown",
        ".rst": "rst",
        ".txt": "text",
        ".json": "json",
        ".jsonc": "jsonc",
        ".ipynb": "notebook",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".sql": "sql",
        ".proto": "protobuf",
    }.get(suffix, "unknown")


def is_generated(path: str) -> bool:
    parts = set(Path(path).parts)
    name = Path(path).name
    return bool(parts & GENERATED_PARTS) or name in LOCK_NAMES or name.endswith(".min.js")


def is_vendor(path: str, changed: bool) -> bool:
    if changed:
        return False
    return path.startswith(VENDOR_PREFIXES) or "/third_party/" in path


def annotated_path_for(path: str, language: str) -> str:
    source = Path(path)
    if language == "json":
        name = f"{source.stem}.annotated.jsonc"
        source = source.with_name(name)
    elif language == "notebook":
        name = f"{source.stem}.annotated.md"
        source = source.with_name(name)
    return (Path("docs") / "code_explanation" / "annotated" / source).as_posix()


def execution_relevance(path: str, changed: bool) -> str:
    if path in CRITICAL_PATHS:
        return "pure_opd_critical"
    if "async_rollout" in path:
        return "experimental"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("docs/") or path.lower().endswith((".md", ".rst")):
        return "documentation"
    if changed and (
        path.startswith("agent_system/")
        or path.startswith("verl/")
        or path.startswith("examples/")
    ):
        return "pure_opd_transitive"
    if path.startswith(("verl/", "agent_system/", "examples/", "recipe/")):
        return "shared_training"
    if path.startswith((".github/", "docker/", "scripts/")):
        return "optional_optimization"
    return "unrelated"


def production_status(path: str) -> str:
    if "async_rollout" in path:
        return "experimental"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("docs/") or path.lower().endswith((".md", ".rst")):
        return "documentation"
    return "production"


def file_type_for(path: str, language: str) -> str:
    if path.startswith(".github/workflows/"):
        return "workflow"
    if path.startswith("tests/"):
        return "test"
    if language in {"yaml", "toml", "ini", "config", "json", "jsonc"}:
        return "config"
    if language in {"markdown", "rst", "text"}:
        return "documentation"
    if language == "notebook":
        return "notebook"
    return "source"


def annotation_mode_for(language: str) -> str:
    if language == "json":
        return "jsonc"
    if language == "notebook":
        return "notebook_markdown"
    return "source_mirror"


def count_lines(text: str) -> tuple[int, int]:
    lines = text.splitlines()
    return len(lines), sum(bool(line.strip()) for line in lines)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest() -> list[dict]:
    return json.loads((DOC_ROOT / "manifest.json").read_text(encoding="utf-8"))


def save_manifest(rows: Iterable[dict]) -> None:
    payload = list(rows)
    (DOC_ROOT / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if payload:
        with (DOC_ROOT / "manifest.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(payload[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(payload)


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()
