"""生成物の改行を LF・末尾1改行へ機械的に正規化する。"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def normalize(path: Path) -> None:
    data = path.read_bytes()
    has_bom = data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    stripped = text.rstrip("\n")
    if stripped.endswith("\\n"):
        stripped = stripped[:-2].rstrip("\n")
    encoded = (stripped + "\n").encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if has_bom else b"") + encoded)


def main() -> None:
    paths = [
        ROOT / "AGENTS.md",
        *(ROOT / "docs" / "code_explanation").glob("*"),
        *(ROOT / "tools" / "code_annotation").glob("*.py"),
    ]
    normalized = 0
    for path in paths:
        if path.is_file():
            normalize(path)
            normalized += 1
    print(f"normalized={normalized}")


if __name__ == "__main__":
    main()
