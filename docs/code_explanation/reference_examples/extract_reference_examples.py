#!/usr/bin/env python3
"""Extract the exact uploaded trainer files from gzip+Base64 payloads."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "payload"
REPO = HERE.parents[2]

FILES = {
    "ray_trainer.py": {
        "parts": [
            "ray_trainer.py.gz.b64.000",
            "ray_trainer.py.gz.b64.001",
            "ray_trainer.py.gz.b64.002",
            "ray_trainer.py.gz.b64.003",
        ],
        "sha256": "1b6e3d19cf543dff46d59fe7b0a736c51de132cb6a0cfa5e716c528a958c09d8",
        "install": "docs/code_explanation/annotated/verl/trainer/ppo/ray_trainer.py",
    },
    "main_ppo.py": {
        "parts": ["main_ppo.py.gz.b64"],
        "sha256": "68f5c3f0820d963eded83d53573b2ac12313e6c45a20fbce718ca7a47c56031d",
        "install": "docs/code_explanation/annotated/verl/trainer/main_ppo.py",
    },
    "skillsd_ray_trainer.py": {
        "parts": ["skillsd_ray_trainer.py.gz.b64"],
        "sha256": "ca9a6a951d705614e4d67cbeafa7d1cf4befe9579574a327a9517137fee454c9",
        "install": "docs/code_explanation/annotated/verl/trainer/ppo/skillsd_ray_trainer.py",
    },
}


def decode(parts: list[str]) -> bytes:
    encoded = "".join((PAYLOAD / part).read_text(encoding="ascii").strip() for part in parts)
    return gzip.decompress(base64.b64decode(encoded))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install",
        action="store_true",
        help="write directly to the annotated mirror instead of materialized/",
    )
    args = parser.parse_args()

    materialized = HERE / "materialized"
    for name, spec in FILES.items():
        content = decode(spec["parts"])
        actual = hashlib.sha256(content).hexdigest()
        if actual != spec["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {name}: {actual}")

        target = REPO / spec["install"] if args.install else materialized / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print(f"wrote {target.relative_to(REPO)} ({len(content)} bytes, sha256={actual})")


if __name__ == "__main__":
    main()
