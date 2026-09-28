"""Small application using decengine's typed results for workflow branching."""

from decengine import Choice, ChoiceResult, Engine, Noul, NoulResult

ticket = {
    "subject": "Integration webhook is failing",
    "body": "Production notifications stopped this morning.",
}

with Engine("Qwen/Qwen3-Embedding-0.6B") as engine:
    decision = engine.decide(
        state=ticket,
        questions={
            "route": Choice(
                "Which team should handle this?",
                {
                    "billing": "payments and refunds",
                    "technical": "bugs, outages, and integrations",
                    "sales": "pricing and contracts",
                },
            ),
            "urgent": Noul("Does this require attention today?"),
        },
    )

route = decision["route"]
urgent = decision["urgent"]
assert isinstance(route, ChoiceResult)
assert isinstance(urgent, NoulResult)
print({"queue": route.selected, "expedite": bool(urgent)})
