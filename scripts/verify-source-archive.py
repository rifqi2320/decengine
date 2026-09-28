#!/usr/bin/env python3
"""Reject build products, caches, model weights, and secrets in a source ZIP."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import PurePosixPath

FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "target",
    "build",
    "dist",
}
FORBIDDEN_SUFFIXES = {
    ".dylib",
    ".so",
    ".dll",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".gguf",
    ".zip",
}
REQUIRED = {
    "decengine/README.md",
    "decengine/Cargo.toml",
    "decengine/pyproject.toml",
    "decengine/include/decengine.h",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()
    errors: list[str] = []
    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        for raw_name in names:
            path = PurePosixPath(raw_name)
            if any(part in FORBIDDEN_PARTS for part in path.parts):
                errors.append(f"forbidden path: {raw_name}")
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append(f"forbidden artifact: {raw_name}")
            if path.name in {".env", ".DS_Store"}:
                errors.append(f"forbidden local file: {raw_name}")
        missing = REQUIRED - names
        errors.extend(f"missing required source: {name}" for name in sorted(missing))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"source archive verified: {len(names)} entries")


if __name__ == "__main__":
    main()
