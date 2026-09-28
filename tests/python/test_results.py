from __future__ import annotations

import unittest

from decengine import ChoiceResult, EngineError, NoulResult, ScoreResult
from decengine.results import DecisionResponse


def response_with(results: object, usage: object | None = None) -> dict[str, object]:
    return {
        "id": "dec_test",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "created": 1,
        "results": results,
        "usage": usage
        if usage is not None
        else {
            "input_characters": 10,
            "questions": 1,
            "candidates": 2,
            "candidate_cache_hits": 0,
            "candidate_cache_misses": 1,
        },
    }


class ResultTests(unittest.TestCase):
    def test_parses_every_result_variant(self) -> None:
        response = DecisionResponse.from_wire(
            response_with(
                {
                    "route": {
                        "type": "choice",
                        "selected": "billing",
                        "probabilities": {"billing": 0.8, "sales": 0.2},
                        "confidence": 0.6,
                    },
                    "urgent": {"type": "noul", "value": 0.4, "confidence": 0.5},
                    "severity": {
                        "type": "score",
                        "value": 2.5,
                        "distribution": {"low": 0.25, "high": 0.75},
                        "legend": {"low": "minor", "high": "critical"},
                        "confidence": 0.7,
                    },
                }
            )
        )
        self.assertIsInstance(response["route"], ChoiceResult)
        self.assertIsInstance(response["urgent"], NoulResult)
        self.assertFalse(bool(response["urgent"]))
        self.assertIsInstance(response["severity"], ScoreResult)

    def test_rejects_malformed_response_envelopes(self) -> None:
        cases: list[tuple[object, str]] = [
            ([], "must be an object"),
            ({}, "results must be an object"),
            ({"results": {}}, "usage must be an object"),
            (response_with({}, {}), "malformed native response"),
        ]
        for value, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(EngineError, message):
                DecisionResponse.from_wire(value)

    def test_rejects_invalid_result_shapes(self) -> None:
        cases: list[tuple[object, str]] = [
            (1, "must be an object"),
            ({"type": "unknown"}, "unknown type"),
            (
                {
                    "type": "choice",
                    "selected": "billing",
                    "probabilities": {},
                    "confidence": 0.5,
                },
                "non-empty object",
            ),
            (
                {
                    "type": "choice",
                    "selected": "billing",
                    "probabilities": {"billing": "high"},
                    "confidence": 0.5,
                },
                "not numeric",
            ),
            (
                {
                    "type": "choice",
                    "selected": "billing",
                    "probabilities": {"billing": 1.1},
                    "confidence": 0.5,
                },
                "outside",
            ),
            (
                {
                    "type": "choice",
                    "selected": "billing",
                    "probabilities": {"billing": 0.8, "sales": 0.8},
                    "confidence": 0.5,
                },
                "does not sum",
            ),
            (
                {
                    "type": "choice",
                    "selected": "sales",
                    "probabilities": {"billing": 1.0},
                    "confidence": 0.5,
                },
                "selected unknown",
            ),
            (
                {
                    "type": "choice",
                    "probabilities": {"billing": 1.0},
                    "confidence": 0.5,
                },
                "malformed native result",
            ),
            (
                {
                    "type": "score",
                    "value": 1.0,
                    "distribution": {"low": 1.0},
                    "legend": [],
                    "confidence": 0.5,
                },
                "legend must be an object",
            ),
            (
                {
                    "type": "score",
                    "value": float("inf"),
                    "distribution": {"low": 1.0},
                    "legend": {"low": "minor"},
                    "confidence": 0.5,
                },
                "must be finite and numeric",
            ),
        ]
        for result, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(EngineError, message):
                DecisionResponse.from_wire(response_with({"result": result}))


if __name__ == "__main__":
    unittest.main()
