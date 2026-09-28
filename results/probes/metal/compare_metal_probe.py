#!/usr/bin/env python3
"""Strictly compare Metal probe predictions to archived CPU probe outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/probes/comparisons/20260923T104208Z"
SPECS = (
    ("Qwen+Metal probe", "qwen", "Qwen+Metal probe/predictions-all300.jsonl",
     "probe-qwen-predictions-all300.jsonl"),
    ("Harrier+Metal probe", "harrier", "Harrier+Metal probe/predictions-all300.jsonl",
     "probe-harrier-predictions-all300.jsonl"),
)
TOLERANCE = 1e-4


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    failures = []
    for system, model, metal_file, cpu_file in SPECS:
        metal_path = Path(__file__).parent / metal_file
        cpu_path = RUN / cpu_file
        assert metal_path.is_file(), f"missing {system} output: {metal_path}"
        actual, expected = rows(metal_path), rows(cpu_path)
        assert len(actual) == len(expected) == 300, f"{system}: expected 300 rows"
        worst = 0.0
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a["id"] != e["id"]:
                failures.append(f"{system} row {i}: id differs {a['id']} != {e['id']}")
            for task in ("owner", "urgent", "impact"):
                if a[task]["label"] != e[task]["label"]:
                    failures.append(f"{system} {a['id']} {task}: label differs")
                ap, ep = a[task]["probabilities"], e[task]["probabilities"]
                if ap.keys() != ep.keys():
                    failures.append(f"{system} {a['id']} {task}: class keys differ")
                    continue
                error = max(abs(ap[k] - ep[k]) for k in ap)
                worst = max(worst, error)
                if error > TOLERANCE:
                    failures.append(f"{system} {a['id']} {task}: max probability error {error:.8g} > {TOLERANCE}")
        print(f"{system}: {len(actual)} cases; max probability abs error={worst:.8g}; labels exact")
    if failures:
        raise SystemExit("\n".join(failures[:30]))


if __name__ == "__main__":
    main()
