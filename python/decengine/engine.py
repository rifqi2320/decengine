"""High-level in-process Engine implementation."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from contextlib import suppress
from types import TracebackType
from typing import Any

from ._native import NativeLibrary, ffi
from .exceptions import EngineError, InvalidQuestion, InvalidRequest
from .questions import Question
from .results import DecisionResponse


class Engine:
    """A model-bound native decision engine.

    The native library is loaded in-process through CFFI ABI mode. Calls are serialized
    per Engine instance because v1 does not promise parallel GPU evaluation.
    """

    def __init__(
        self,
        model: str,
        *,
        native_library: str | os.PathLike[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise InvalidRequest("model must be a non-blank string")
        if options is not None and not isinstance(options, Mapping):
            raise InvalidRequest("options must be a mapping")
        self.model = model
        self._native = NativeLibrary(native_library)
        self._lock = threading.RLock()
        self._handle: Any = ffi.NULL
        self._closed = False

        try:
            options_json = json.dumps(dict(options or {}), allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise InvalidRequest(f"options are not JSON serializable: {error}") from error
        output = ffi.new("de_engine_t **")
        status = int(
            self._native.lib.de_engine_create(
                model.encode("utf-8"), options_json.encode("utf-8"), output
            )
        )
        self._native.check(status)
        if output[0] == ffi.NULL:
            raise InvalidRequest("native engine creation succeeded with a null handle")
        self._handle = output[0]

    @property
    def closed(self) -> bool:
        return self._closed

    def decide(self, *, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        if not isinstance(questions, Mapping) or not questions:
            raise InvalidQuestion("questions must be a non-empty mapping")
        wire_questions: dict[str, Any] = {}
        for name, question in questions.items():
            if not isinstance(name, str) or not name.strip():
                raise InvalidQuestion("question names must be non-blank strings")
            if not hasattr(question, "to_wire"):
                raise InvalidQuestion(f"question {name!r} is not Choice, Noul, or Score")
            wire_questions[name] = question.to_wire()
        request = {"model": self.model, "state": state, "questions": wire_questions}
        try:
            encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise InvalidRequest(f"state is not JSON serializable: {error}") from error

        with self._lock:
            self._require_open()
            output = ffi.new("char **")
            status = int(
                self._native.lib.de_engine_decide_json(self._handle, encoded, output)
            )
            self._native.check(status)
            if output[0] == ffi.NULL:
                raise InvalidRequest("native decision succeeded with a null response")
            try:
                raw = ffi.string(output[0]).decode("utf-8")
            finally:
                self._native.lib.de_string_free(output[0])
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EngineError(f"native response is not valid JSON: {error}") from error
        return DecisionResponse.from_wire(value)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._handle != ffi.NULL:
                self._native.lib.de_engine_destroy(self._handle)
                self._handle = ffi.NULL
            self._closed = True

    def _require_open(self) -> None:
        if self._closed or self._handle == ffi.NULL:
            raise InvalidRequest("Engine is closed")

    def __enter__(self) -> Engine:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def native_version(native_library: str | os.PathLike[str] | None = None) -> str:
    return NativeLibrary(native_library).version
