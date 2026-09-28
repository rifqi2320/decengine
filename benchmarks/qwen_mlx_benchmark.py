#!/usr/bin/env python3
"""Measure one state against five typed decisions through the production MLX FFI."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

from decengine import Choice, Engine, Noul, Score

MODEL = "Qwen/Qwen3-Embedding-0.6B"
STATE = {
    "from": "user@example.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
QUESTIONS = {
    "department": Choice(
        "Which department should handle this email?",
        {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    ),
    "urgency": Score(
        "How urgent is this request?",
        ["not urgent", "soon", "critical deadline or blocking issue"],
    ),
    "refund": Noul("Does the customer ask for money back?"),
    "cancellation_risk": Noul("Does the customer state an intention to cancel their plan?"),
    "response_priority": Choice(
        "What response priority should this email receive?",
        {
            "standard": "normal queue response within standard service level",
            "expedited": "respond sooner because the request is important",
            "immediate": "respond immediately because a critical issue is blocking the customer",
        },
    ),
}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    model_store = os.environ["DECENGINE_MODEL_STORE"]
    started = time.perf_counter_ns()
    with Engine(MODEL, options={"engine": "mlx", "model_store": model_store}) as engine:
        load_ms = (time.perf_counter_ns() - started) / 1e6
        for _ in range(10):
            response = engine.decide(state=STATE, questions=QUESTIONS)
        samples_ms = []
        for _ in range(100):
            started = time.perf_counter_ns()
            response = engine.decide(state=STATE, questions=QUESTIONS)
            samples_ms.append((time.perf_counter_ns() - started) / 1e6)
    report = {
        "backend": "mlx-rs / Metal",
        "model": MODEL,
        "state_count": 1,
        "question_count": len(QUESTIONS),
        "load_ms": load_ms,
        "warmup_iterations": 10,
        "timed_iterations": len(samples_ms),
        "latency_ms": {
            "min": min(samples_ms),
            "mean": statistics.mean(samples_ms),
            "p50": percentile(samples_ms, 0.5),
            "p95": percentile(samples_ms, 0.95),
            "max": max(samples_ms),
            "per_question_mean": statistics.mean(samples_ms) / len(QUESTIONS),
        },
        "result_ids": list(response.results),
        "samples_ms": samples_ms,
    }
    output = Path(os.environ.get("DECENGINE_BENCHMARK_OUTPUT", "qwen-mlx-benchmark.json"))
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
