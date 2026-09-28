#!/usr/bin/env python3
"""Fast dependency and packaging invariant checks with no third-party dependencies."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATES = ROOT / "crates"


def fail(message: str) -> None:
    print(f"boundary error: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    for path in CRATES.rglob("*.rs"):
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        if "mlx_rs" in source and not relative.startswith("crates/decengine-engine-mlx/"):
            fail(f"mlx_rs leaked into {relative}")
        if "reqwest" in source and not relative.startswith("crates/decengine-models/"):
            fail(f"network client leaked outside explicit model installation: {relative}")
        if re.search(r"https?://", source) and not relative.startswith(
            "crates/decengine-models/"
        ):
            fail(f"URL found in inference/runtime source: {relative}")

    manifests = sorted((ROOT / "models" / "manifests").glob("*.json"))
    ids = {json.loads(path.read_text(encoding="utf-8"))["id"] for path in manifests}
    expected = {"Qwen/Qwen3-Embedding-0.6B", "microsoft/harrier-oss-v1-0.6b"}
    if ids != expected:
        fail(f"model allowlist changed: {ids!r}")

    header = (ROOT / "include" / "decengine.h").read_text(encoding="utf-8")
    native = (ROOT / "python" / "decengine" / "_native.py").read_text(encoding="utf-8")
    symbols = {
        "de_abi_version",
        "de_version",
        "de_engine_create",
        "de_engine_decide_json",
        "de_engine_destroy",
        "de_string_free",
        "de_last_error",
    }
    for symbol in symbols:
        if symbol not in header or symbol not in native:
            fail(f"ABI symbol {symbol} is missing from header or Python CFFI declarations")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    forbidden_build_hooks = ("maturin", "setuptools-rust", "setuptools_rust", "pyo3")
    for hook in forbidden_build_hooks:
        if hook in pyproject.lower():
            fail(f"Python package unexpectedly includes native build hook {hook}")

    print("boundary checks passed")


if __name__ == "__main__":
    main()
