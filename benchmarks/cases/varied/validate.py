#!/usr/bin/env python3
"""Validate varied-question JSONL authoring fixtures (stdlib only)."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT = Path(__file__).with_name("seed.jsonl")
FAMILY_RE = re.compile(r"^qf-[a-f0-9]{8}$")
CASE_RE = re.compile(r"^vq-[0-9]{3,}$")


def fail(path: Path, line: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line}: {message}")


def validate_record(record: Any, path: Path, line: int, seen: set[str]) -> None:
    if not isinstance(record, dict):
        raise fail(path, line, "record must be an object")
    required = {"id", "domain", "question_family_id", "request", "reference"}
    if set(record) != required:
        raise fail(path, line, f"top-level keys must be {sorted(required)}")
    case_id = record["id"]
    if not isinstance(case_id, str) or not CASE_RE.fullmatch(case_id) or case_id in seen:
        raise fail(path, line, "id must be unique and match vq-NNN")
    seen.add(case_id)
    if not isinstance(record["domain"], str) or not record["domain"].strip():
        raise fail(path, line, "domain must be non-empty text")
    family = record["question_family_id"]
    if not isinstance(family, str) or not FAMILY_RE.fullmatch(family):
        raise fail(path, line, "question_family_id must be opaque qf- plus 8 lowercase hex characters")

    request = record["request"]
    if not isinstance(request, dict) or set(request) != {"model", "state", "questions"}:
        raise fail(path, line, "request must contain exactly model, state, questions")
    if not isinstance(request["model"], str) or not isinstance(request["state"], dict):
        raise fail(path, line, "request model must be text and state must be an object")
    questions = request["questions"]
    if not isinstance(questions, dict) or len(questions) != 1:
        raise fail(path, line, "author one question per example")
    name, question = next(iter(questions.items()))
    if not isinstance(name, str) or not name.strip() or not isinstance(question, dict):
        raise fail(path, line, "question name and object are required")
    kind = question.get("type")
    if not isinstance(question.get("prompt"), str) or len(question["prompt"].strip()) < 12:
        raise fail(path, line, "question prompt must be substantive text")

    reference = record["reference"]
    if not isinstance(reference, dict) or set(reference) != {"target", "rationale"}:
        raise fail(path, line, "reference must contain exactly target and rationale")
    target, rationale = reference["target"], reference["rationale"]
    if not isinstance(target, dict) or not isinstance(rationale, str) or len(rationale.strip()) < 12:
        raise fail(path, line, "target object and substantive rationale required")
    # Candidate keys/labels necessarily occur in the wire question; leakage checks
    # target only case context and prompt, not the legitimate answer candidate set.
    narrative_text = json.dumps({"state": request["state"], "prompt": question["prompt"]},
                                ensure_ascii=False).casefold()

    if kind == "choice":
        if set(question) != {"type", "prompt", "options"}:
            raise fail(path, line, "choice question needs type, prompt, options")
        options = question["options"]
        if not isinstance(options, dict) or not 2 <= len(options) <= 16:
            raise fail(path, line, "choice requires 2..16 candidate options")
        if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or len(v.strip()) < 2
               for k, v in options.items()):
            raise fail(path, line, "candidate keys/descriptions must be nonblank strings")
        selected = target.get("selected")
        if set(target) != {"selected"} or selected not in options:
            raise fail(path, line, "choice target must select an available candidate key")
        for token in (str(selected), options[selected]):
            if token.casefold() in narrative_text:
                raise fail(path, line, "selected key/description appears in state or prompt (possible leakage)")
    elif kind == "noul":
        if set(question) != {"type", "prompt"} or set(target) != {"value"} or type(target["value"]) is not bool:
            raise fail(path, line, "noul target must be a boolean value")
    elif kind == "score":
        if set(question) != {"type", "prompt", "levels"}:
            raise fail(path, line, "score question needs type, prompt, levels")
        levels = question["levels"]
        if not isinstance(levels, list) or not 2 <= len(levels) <= 16:
            raise fail(path, line, "score requires 2..16 rubric levels")
        values = []
        for level in levels:
            if not isinstance(level, dict) or set(level) != {"label", "criterion", "value"}:
                raise fail(path, line, "each score level needs label, criterion, value")
            if (not isinstance(level["label"], str) or not level["label"].strip() or
                    not isinstance(level["criterion"], str) or not level["criterion"].strip() or
                    isinstance(level["value"], bool) or not isinstance(level["value"], (int, float)) or
                    not math.isfinite(level["value"])):
                raise fail(path, line, "score level fields must be nonblank, finite values")
            values.append(level["value"])
        if any(a >= b for a, b in zip(values, values[1:])):
            raise fail(path, line, "score values must be strictly increasing")
        if set(target) != {"label", "value"}:
            raise fail(path, line, "score target needs label and value")
        matches = [level for level in levels if level["label"] == target["label"]]
        if len(matches) != 1 or matches[0]["value"] != target["value"]:
            raise fail(path, line, "score target label/value must match exactly one rubric level")
        if str(target["label"]).casefold() in narrative_text:
            raise fail(path, line, "target score label appears in state or prompt (possible leakage)")
    else:
        raise fail(path, line, "type must be choice, noul, or score")


def validate(path: Path) -> int:
    seen: set[str] = set()
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                raise fail(path, line, "blank lines are not allowed")
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise fail(path, line, f"invalid JSON: {exc.msg}") from exc
            validate_record(record, path, line, seen)
            count += 1
    if count == 0:
        raise ValueError(f"{path}: fixture must contain at least one record")
    return count


if __name__ == "__main__":
    fixture = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if len(sys.argv) > 2:
        raise SystemExit("usage: validate.py [fixture.jsonl]")
    try:
        total = validate(fixture)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"valid: {total} varied-question records ({fixture})")
