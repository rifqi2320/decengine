"""CFFI ABI-mode loader. This module never invokes a compiler."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from cffi import FFI  # type: ignore[import-untyped]

from .exceptions import (
    ContextTooLong,
    DecengineError,
    EngineError,
    InvalidManifest,
    InvalidRequest,
    ModelLoadFailed,
    ModelNotFound,
    NativeLibraryNotFound,
    OutOfMemory,
    UnsupportedModel,
)

ABI_VERSION = 1

ffi = FFI()
ffi.cdef(
    """
    typedef unsigned int uint32_t;
    typedef struct de_engine de_engine_t;
    typedef int de_status_t;

    uint32_t de_abi_version(void);
    const char *de_version(void);
    de_status_t de_engine_create(
        const char *model_id,
        const char *options_json,
        de_engine_t **out_engine
    );
    de_status_t de_engine_decide_json(
        de_engine_t *engine,
        const char *request_json,
        char **out_response_json
    );
    void de_engine_destroy(de_engine_t *engine);
    void de_string_free(char *value);
    const char *de_last_error(void);
    """
)

_STATUS_ERRORS: dict[int, type[DecengineError]] = {
    1: InvalidRequest,
    2: ModelNotFound,
    3: UnsupportedModel,
    4: ModelLoadFailed,
    5: ContextTooLong,
    6: EngineError,
    7: OutOfMemory,
    8: InvalidManifest,
    255: EngineError,
}


def resolve_library(explicit: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    if configured := os.environ.get("DECENGINE_LIB_PATH"):
        candidates.append(Path(configured).expanduser())

    configured_home = os.environ.get("DECENGINE_HOME")
    home = Path(configured_home) if configured_home else Path.home() / ".decengine"
    metadata = home / "state" / "native-library"
    if metadata.is_file():
        try:
            configured_path = metadata.read_text(encoding="utf-8").strip()
        except OSError:
            configured_path = ""
        if configured_path:
            candidates.append(Path(configured_path).expanduser())

    package_root = Path(__file__).resolve().parent
    if sys.platform == "darwin":
        names = ("libdecengine.dylib",)
    elif sys.platform == "win32":
        names = ("decengine.dll", "libdecengine.dll")
    else:
        names = ("libdecengine.so",)
    for name in names:
        candidates.extend(
            [
                package_root / name,
                Path("/usr/local/lib") / name,
                Path("/opt/homebrew/lib") / name,
            ]
        )

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved
    locations = "\n".join(f"  - {path}" for path in checked) or "  (no locations)"
    raise NativeLibraryNotFound(
        "decengine native library was not found. Install the native release or set "
        f"DECENGINE_LIB_PATH. Checked:\n{locations}"
    )


class NativeLibrary:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = resolve_library(path)
        try:
            self.lib: Any = ffi.dlopen(str(self.path))
        except OSError as error:
            raise NativeLibraryNotFound(f"cannot load {self.path}: {error}") from error
        actual = int(self.lib.de_abi_version())
        if actual != ABI_VERSION:
            raise NativeLibraryNotFound(
                f"native ABI version {actual} is incompatible with Python ABI {ABI_VERSION}"
            )

    @property
    def version(self) -> str:
        pointer = self.lib.de_version()
        if pointer == ffi.NULL:
            raise EngineError("native library returned a null version")
        value: bytes = ffi.string(pointer)
        return value.decode("utf-8")

    def error_message(self) -> str:
        pointer = self.lib.de_last_error()
        if pointer == ffi.NULL:
            return "native operation failed without a diagnostic"
        value: bytes = ffi.string(pointer)
        return value.decode("utf-8", errors="replace")

    def check(self, status: int) -> None:
        if status == 0:
            return
        error_type = _STATUS_ERRORS.get(status, EngineError)
        raise error_type(self.error_message())
