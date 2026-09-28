from __future__ import annotations

import unittest

from decengine import Choice, InvalidQuestion, Noul, Score, ScoreLevel


class QuestionTests(unittest.TestCase):
    def test_choice_copies_mutable_mapping(self) -> None:
        options = {"billing": "refunds", "sales": "pricing"}
        choice = Choice("Which team?", options)
        options["billing"] = "changed"
        self.assertEqual(choice.options["billing"], "refunds")

    def test_choice_requires_two_options(self) -> None:
        with self.assertRaisesRegex(InvalidQuestion, "2..=128"):
            Choice("Which team?", {"billing": "refunds"})

    def test_choice_requires_mapping_and_nonblank_values(self) -> None:
        with self.assertRaisesRegex(InvalidQuestion, "must be a mapping"):
            Choice("Which team?", ["billing", "sales"])  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidQuestion, "criterion"):
            Choice("Which team?", {"billing": "refunds", "sales": " "})

    def test_noul_rejects_blank_prompt(self) -> None:
        with self.assertRaises(InvalidQuestion):
            Noul("  ")

    def test_score_accepts_string_sequence(self) -> None:
        score = Score("Severity?", ["minor", "blocked", "critical"])
        self.assertEqual([level.value for level in score.levels], [0.0, 1.0, 2.0])
        self.assertEqual(score.to_wire()["levels"][2]["criterion"], "critical")

    def test_score_accepts_explicit_values(self) -> None:
        score = Score(
            "Severity?",
            [
                ScoreLevel("low", "minor", 1.0),
                ScoreLevel("high", "critical", 5.0),
            ],
        )
        self.assertEqual(score.levels[-1].value, 5.0)

    def test_score_accepts_mapping(self) -> None:
        score = Score("Severity?", {"low": "minor", "high": "critical"})
        self.assertEqual([level.label for level in score.levels], ["low", "high"])

    def test_score_rejects_non_increasing_values(self) -> None:
        with self.assertRaisesRegex(InvalidQuestion, "strictly increasing"):
            Score(
                "Severity?",
                [
                    ScoreLevel("high", "critical", 2.0),
                    ScoreLevel("low", "minor", 1.0),
                ],
            )

    def test_score_rejects_invalid_level_collections(self) -> None:
        with self.assertRaisesRegex(InvalidQuestion, "sequence or mapping"):
            Score("Severity?", 3)  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidQuestion, "strings or ScoreLevel"):
            Score("Severity?", ["minor", object()])  # type: ignore[list-item]
        with self.assertRaisesRegex(InvalidQuestion, "2..=128"):
            Score("Severity?", ["only one"])
        with self.assertRaisesRegex(InvalidQuestion, "labels must be unique"):
            Score(
                "Severity?",
                [
                    ScoreLevel("same", "minor", 0.0),
                    ScoreLevel("same", "critical", 1.0),
                ],
            )

    def test_score_level_requires_numeric_finite_value(self) -> None:
        with self.assertRaisesRegex(InvalidQuestion, "numeric"):
            ScoreLevel("low", "minor", "zero")  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidQuestion, "finite"):
            ScoreLevel("high", "critical", float("inf"))


if __name__ == "__main__":
    unittest.main()
