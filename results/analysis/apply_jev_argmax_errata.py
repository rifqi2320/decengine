#!/usr/bin/env python3
"""Create a corrected judgment derivative from archived JEV impact probabilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/harrier-runs/20260923T084828012314Z"
SOURCE = RUN / "judgments.jsonl"
FIXTURE = ROOT / "benchmarks/cases/decision-cases-100.jsonl"
ARCHIVE = ROOT / "results/comparison-runs/20260923T082226631235Z/cases.jsonl"
CORRECTED = RUN / "judgments-corrected.jsonl"
ERRATA = RUN / "judgments-errata.json"
TARGET_IDS = {"case-095", "case-097", "case-099"}
ENGINE_MAP = {"harrier": "harrier", "qwen": "qwen_mlx", "jev": "jev"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path, id_key: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row[id_key])
        if case_id in rows:
            raise ValueError(f"{path}:{line_no}: duplicate {case_id}")
        rows[case_id] = row
    return rows


def main() -> int:
    judgment_rows = read_jsonl(SOURCE, "case_id")
    fixtures = read_jsonl(FIXTURE, "id")
    archive = read_jsonl(ARCHIVE, "case_id")
    if len(judgment_rows) != 100 or set(judgment_rows) != set(fixtures) or set(fixtures) != set(archive):
        raise ValueError("judgments, fixture, and archive must contain the same 100 cases")

    changes = []
    for case_id in sorted(TARGET_IDS):
        row = judgment_rows[case_id]
        fixture = fixtures[case_id]
        archived_answer = archive[case_id]["jev"]["raw_response"]["body"]["answers"]["impact"]
        probabilities = archived_answer.get("probabilities")
        levels = fixture["request"]["questions"]["impact"]["levels"]
        if not isinstance(probabilities, dict):
            raise ValueError(f"{case_id}: raw JEV impact probabilities missing")
        # Deterministic argmax tie policy: first level in fixture order among
        # equal maximum probabilities. The three audited cases all have a unique max.
        indexed = [(index, float(probabilities.get(str(index), float("nan"))))
                   for index in range(len(levels))]
        if any(value != value for _, value in indexed):
            raise ValueError(f"{case_id}: probabilities do not cover every fixture level")
        maximum = max(value for _, value in indexed)
        winner_indices = [index for index, value in indexed if value == maximum]
        prediction = str(levels[winner_indices[0]]["label"])
        reference = fixture["reference"]["impact"]
        is_correct = prediction == reference

        before = {
            "prediction": row["predictions"]["jev"]["impact"],
            "exact_match": row["exact_matches"]["jev"]["impact"],
            "total_exact_matches": row.get("total_exact_matches", row.get("exact_match_totals"))["jev"],
            "winner": row.get("winner", row.get("winners")),
        }
        row["predictions"]["jev"]["impact"] = prediction
        row["exact_matches"]["jev"]["impact"] = is_correct
        total_field = "total_exact_matches" if "total_exact_matches" in row else "exact_match_totals"
        row[total_field]["jev"] = sum(
            bool(value) for value in row["exact_matches"]["jev"].values()
        )
        recomputed_winners = []
        totals = {
            source_engine: sum(bool(value) for value in row["exact_matches"][source_engine].values())
            for source_engine in ENGINE_MAP
        }
        max_total = max(totals.values())
        recomputed_winners = [engine for engine, score in totals.items() if score == max_total]
        winner_field = "winner" if "winner" in row else "winners"
        row[winner_field] = recomputed_winners
        after = {
            "prediction": prediction,
            "exact_match": is_correct,
            "total_exact_matches": row[total_field]["jev"],
            "winner": recomputed_winners,
        }
        changed = {
            key: {"before": before[key], "after": after[key]}
            for key in before if before[key] != after[key]
        }
        changes.append({
            "case_id": case_id,
            "raw_jev_probabilities_by_level_index": probabilities,
            "fixture_levels": [level["label"] for level in levels],
            "argmax_level_indices": winner_indices,
            "argmax_probability": maximum,
            "reference_impact": reference,
            "changed_fields": changed,
            "recomputed_winners": recomputed_winners,
        })

    # This erratum is deliberately narrow: exactly the three audited JEV values.
    if set(judgment_rows) != set(fixtures):
        raise ValueError("case ID set unexpectedly changed")
    output_text = "".join(json.dumps(judgment_rows[case_id], ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False) + "\n"
                         for case_id in [f"case-{index:03d}" for index in range(1, 101)])
    CORRECTED.write_text(output_text, encoding="utf-8")
    errata = {
        "schema_version": 1,
        "method": "Use archived raw JEV impact probabilities; map probability index to same-index fixture score level.",
        "tie_policy": "If max probabilities tie, choose the first level in fixture order; no ties occur in the affected cases.",
        "not_subjective_review": True,
        "source_judgments": str(SOURCE),
        "corrected_judgments": str(CORRECTED),
        "fixture": str(FIXTURE),
        "original_archive": str(ARCHIVE),
        "source_judgments_sha256": sha256(SOURCE),
        "fixture_sha256": sha256(FIXTURE),
        "original_archive_sha256": sha256(ARCHIVE),
        "corrected_judgments_sha256": sha256(CORRECTED),
        "corrected_case_ids": sorted(TARGET_IDS),
        "changes": changes,
    }
    ERRATA.write_text(json.dumps(errata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"corrected": str(CORRECTED), "errata": str(ERRATA),
                      "case_ids": sorted(TARGET_IDS),
                      "changed_fields": {item["case_id"]: sorted(item["changed_fields"])
                                         for item in changes}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
