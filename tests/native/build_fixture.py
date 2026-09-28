#!/usr/bin/env python3
"""Build the C ABI fixture used to test CFFI without linking Rust into Python."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path


def build(output_dir: Path) -> Path:
    compiler = shutil.which("cc") or shutil.which("clang")
    if compiler is None:
        raise RuntimeError("a C compiler is required for the CFFI acceptance fixture")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("native_fixture.c")
    system = platform.system()
    if system == "Darwin":
        output = output_dir / "libdecengine.dylib"
        command = [compiler, "-std=c11", "-dynamiclib", str(source), "-o", str(output)]
    elif system == "Windows":
        output = output_dir / "decengine.dll"
        command = [compiler, "-std=c11", "-shared", str(source), "-o", str(output)]
    else:
        output = output_dir / "libdecengine.so"
        command = [
            compiler,
            "-std=c11",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(output),
        ]
    subprocess.run(command, check=True)
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build/test-native"))
    args = parser.parse_args()
    print(build(args.output_dir))


if __name__ == "__main__":
    main()
