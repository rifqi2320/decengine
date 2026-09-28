#!/usr/bin/env python3
"""Black-box UAT from the perspective of an application developer."""

from __future__ import annotations

from decengine import Choice, ChoiceResult, Engine, Noul, NoulResult, Score, ScoreResult


def main() -> None:
    ticket = {
        "id": "T-1042",
        "subject": "Refund charged twice",
        "body": "The duplicate payment blocks month-end close. Please help today.",
    }
    with Engine("Qwen/Qwen3-Embedding-0.6B") as engine:
        result = engine.decide(
            state=ticket,
            questions={
                "route": Choice(
                    "Which team should handle this?",
                    {
                        "billing": "payments, charges, invoices, and refunds",
                        "technical": "bugs, outages, and integrations",
                        "sales": "pricing, purchasing, and contracts",
                    },
                ),
                "urgent": Noul("Does this require attention today?"),
                "severity": Score(
                    "How severe is this?",
                    ["minor inconvenience", "work blocked", "critical business impact"],
                ),
            },
        )

        route = result["route"]
        urgent = result["urgent"]
        severity = result["severity"]
        assert isinstance(route, ChoiceResult)
        assert isinstance(urgent, NoulResult)
        assert isinstance(severity, ScoreResult)
        assert route.selected == "billing"
        assert urgent.value >= 0.5
        assert severity.value >= 1.0
        assert result.model == "Qwen/Qwen3-Embedding-0.6B"
        assert result.usage.questions == 3

        repeated = engine.decide(
            state={**ticket, "id": "T-1043"},
            questions={
                "route": Choice(
                    "Which team should handle this?",
                    {
                        "billing": "payments, charges, invoices, and refunds",
                        "technical": "bugs, outages, and integrations",
                        "sales": "pricing, purchasing, and contracts",
                    },
                ),
                "urgent": Noul("Does this require attention today?"),
                "severity": Score(
                    "How severe is this?",
                    ["minor inconvenience", "work blocked", "critical business impact"],
                ),
            },
        )
        assert repeated.usage.candidate_cache_hits == 3

    print("UAT PASS: typed routing and repeated-schema cache behavior")


if __name__ == "__main__":
    main()
