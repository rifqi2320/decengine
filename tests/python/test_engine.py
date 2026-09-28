from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

from _support import REPOSITORY, native_fixture

from decengine import (
    Choice,
    ChoiceResult,
    Engine,
    EngineError,
    InvalidQuestion,
    InvalidRequest,
    ModelNotFound,
    NativeLibraryNotFound,
    Noul,
    NoulResult,
    Question,
    Score,
    ScoreResult,
    native_version,
)
from decengine._native import resolve_library


def questions() -> dict[str, Question]:
    return {
        "route": Choice(
            "Which team should handle this?",
            {
                "billing": "payments charges invoices refunds",
                "technical": "bugs outages integrations",
                "sales": "pricing purchasing contracts",
            },
        ),
        "urgent": Noul("Does this require attention today?"),
        "severity": Score(
            "How severe is this?",
            ["minor inconvenience", "work blocked", "critical business impact"],
        ),
    }


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = native_fixture()

    def test_native_version_uses_abi_mode_library(self) -> None:
        self.assertEqual(native_version(self.library), "test-native-1")

    def test_decide_returns_typed_results(self) -> None:
        with Engine("Qwen/Qwen3-Embedding-0.6B", native_library=self.library) as engine:
            response = engine.decide(
                state={"subject": "Refund charged twice", "deadline": "today"},
                questions=questions(),
            )
            self.assertIsInstance(response["route"], ChoiceResult)
            self.assertEqual(response["route"].selected, "billing")  # type: ignore[union-attr]
            self.assertIsInstance(response["urgent"], NoulResult)
            self.assertTrue(bool(response["urgent"]))
            self.assertIsInstance(response["severity"], ScoreResult)
            self.assertAlmostEqual(response["severity"].value, 1.6)  # type: ignore[union-attr]
            self.assertEqual(response.usage.candidate_cache_misses, 3)

            cached = engine.decide(
                state={"subject": "another refund"}, questions=questions()
            )
            self.assertEqual(cached.usage.candidate_cache_hits, 3)

        self.assertTrue(engine.closed)

    def test_close_is_idempotent_and_prevents_use(self) -> None:
        engine = Engine("Qwen/Qwen3-Embedding-0.6B", native_library=self.library)
        engine.close()
        engine.close()
        with self.assertRaisesRegex(InvalidRequest, "closed"):
            engine.decide(state="ticket", questions=questions())

    def test_status_maps_to_typed_exception(self) -> None:
        with self.assertRaisesRegex(ModelNotFound, "not installed"):
            Engine("missing/model", native_library=self.library)

    def test_state_must_be_json_serializable(self) -> None:
        with (
            Engine("Qwen/Qwen3-Embedding-0.6B", native_library=self.library) as engine,
            self.assertRaisesRegex(InvalidRequest, "not JSON serializable"),
        ):
            engine.decide(state={1, 2, 3}, questions=questions())

    def test_nan_is_rejected(self) -> None:
        with (
            Engine("Qwen/Qwen3-Embedding-0.6B", native_library=self.library) as engine,
            self.assertRaises(InvalidRequest),
        ):
            engine.decide(state={"value": math.nan}, questions=questions())

    def test_empty_questions_are_rejected_before_ffi(self) -> None:
        with (
            Engine("Qwen/Qwen3-Embedding-0.6B", native_library=self.library) as engine,
            self.assertRaises(InvalidQuestion),
        ):
            engine.decide(state="ticket", questions={})

    def test_native_error_is_mapped(self) -> None:
        with (
            Engine("Qwen/Qwen3-Embedding-0.6B", native_library=self.library) as engine,
            self.assertRaisesRegex(EngineError, "forced native fixture"),
        ):
            engine.decide(state="force_native_error", questions=questions())

    def test_environment_library_resolution(self) -> None:
        with patch.dict(os.environ, {"DECENGINE_LIB_PATH": str(self.library)}):
            self.assertEqual(resolve_library(), self.library)

    def test_missing_explicit_library_reports_checked_path(self) -> None:
        missing = REPOSITORY / "does-not-exist" / "libdecengine.so"
        with (
            patch.dict(
                os.environ,
                {
                    "DECENGINE_LIB_PATH": str(missing),
                    "DECENGINE_HOME": str(REPOSITORY / "absent"),
                },
                clear=False,
            ),
            self.assertRaisesRegex(NativeLibraryNotFound, "DECENGINE_LIB_PATH"),
        ):
            resolve_library(missing)


if __name__ == "__main__":
    unittest.main()
