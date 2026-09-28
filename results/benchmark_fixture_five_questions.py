import json
import statistics
import time
from pathlib import Path

from decengine import Choice, Engine, Noul, Score

state = {
    "from": "user@example.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
questions = {
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

def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)

started = time.perf_counter_ns()
with Engine("Qwen/Qwen3-Embedding-0.6B") as engine:
    load_ms = (time.perf_counter_ns() - started) / 1e6
    for _ in range(10):
        response = engine.decide(state=state, questions=questions)
    samples_ms = []
    for _ in range(100):
        started = time.perf_counter_ns()
        response = engine.decide(state=state, questions=questions)
        samples_ms.append((time.perf_counter_ns() - started) / 1e6)

report = {
    "backend": "test-native C fixture via Python CFFI",
    "model_requested": "Qwen/Qwen3-Embedding-0.6B",
    "state_count": 1,
    "input_question_count": len(questions),
    "fixture_reported_question_count": response.usage.questions,
    "load_ms": load_ms,
    "warmup_iterations": 10,
    "timed_iterations": len(samples_ms),
    "latency_ms": {
        "min": min(samples_ms),
        "mean": statistics.mean(samples_ms),
        "p50": percentile(samples_ms, 0.50),
        "p95": percentile(samples_ms, 0.95),
        "max": max(samples_ms),
    },
    "returned_result_ids": list(response.results),
    "samples_ms": samples_ms,
}
Path("results/fixture-five-question-benchmark.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
