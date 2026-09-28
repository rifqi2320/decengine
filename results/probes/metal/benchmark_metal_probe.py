#!/usr/bin/env python3
"""Summarize synchronized, post-load/post-warmup timings for isolated 300-case runs."""
import json
import math
import re
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(p * len(values)) - 1)]


def main():
    root = Path(__file__).parent
    specs = ("Qwen+Metal probe", "Harrier+Metal probe")
    for system in specs:
        directory = root / system
        timings = [json.loads(line) for line in (directory / "predictions-all300.jsonl.timings.jsonl").read_text().splitlines() if line]
        if len(timings) != 300 or len({row["id"] for row in timings}) != 300:
            raise SystemExit(f"{system}: expected timings for 300 unique cases, got {len(timings)}")
        summary = {"system": system, "case_count": len(timings), "scope": "300 unique cases after model load and one warmup per task; synchronized GPU stages", "units": "milliseconds", "stages": {}}
        for name, key in (("end_to_end", "case_wall_ns"), ("tokenization", "tokenization_ns"), ("embedding", "embedding_ns"), ("classifier", "classifier_ns")):
            values = [row[key] / 1_000_000 for row in timings]
            summary["stages"][name] = {"mean": sum(values) / len(values), "p50": percentile(values, .50), "p95": percentile(values, .95)}
        resource_text = (directory / "time-resource.txt").read_text()
        elapsed = re.search(r"([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys", resource_text)
        rss = re.search(r"\n\s*(\d+)\s+maximum resident set size", resource_text)
        footprint = re.search(r"\n\s*(\d+)\s+peak memory footprint", resource_text)
        if not (elapsed and rss and footprint):
            raise SystemExit(f"{system}: could not parse /usr/bin/time -l resource report")
        summary["process_resources"] = {
            "measurement": "whole CLI invocation including model load and warmup; process memory on macOS unified memory; not a GPU-only allocation counter",
            "elapsed_seconds_including_load_warmup": float(elapsed.group(1)),
            "user_seconds": float(elapsed.group(2)),
            "system_seconds": float(elapsed.group(3)),
            "maximum_resident_set_bytes": int(rss.group(1)),
            "peak_memory_footprint_bytes": int(footprint.group(1)),
        }
        (directory / "timing-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{system}: " + ", ".join(f"{stage} mean/p50/p95={s['mean']:.3f}/{s['p50']:.3f}/{s['p95']:.3f} ms" for stage, s in summary["stages"].items()))


if __name__ == "__main__":
    main()
