"""Smoke tests for the varied fixture contract; run with unittest."""
import json
import tempfile
import unittest
from pathlib import Path

import validate


class VariedFixtureTests(unittest.TestCase):
    def test_seed_is_valid_and_varied(self):
        path = Path(__file__).with_name("seed.jsonl")
        self.assertEqual(validate.validate(path), 8)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({next(iter(row["request"]["questions"].values()))["type"] for row in rows},
                         {"choice", "noul", "score"})
        self.assertGreaterEqual(len({row["domain"] for row in rows}), 6)
        self.assertEqual(len({tuple(row["request"]["questions"]) for row in rows}), 8)
        self.assertGreaterEqual(len({len(next(iter(row["request"]["questions"].values())).get("options", {}))
                                     for row in rows}), 2)

    def test_rejects_choice_target_echoed_in_context(self):
        row = json.loads(Path(__file__).with_name("seed.jsonl").read_text(encoding="utf-8").splitlines()[0])
        row["request"]["state"]["context"] += " We should verify_remote now."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "possible leakage"):
                validate.validate(path)

    def test_rejects_score_target_value_mismatch(self):
        row = json.loads(Path(__file__).with_name("seed.jsonl").read_text(encoding="utf-8").splitlines()[2])
        row["reference"]["target"]["value"] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "match exactly"):
                validate.validate(path)


if __name__ == "__main__":
    unittest.main()
