#!/usr/bin/env python3
"""Dependency-free black-box client for the compatibility endpoints."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=2) as response:
        assert response.status == 200
        return json.load(response)


def wait_ready(base_url: str) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            if get_json(f"{base_url}/healthz")["status"] == "ok":
                return
        except (OSError, urllib.error.URLError, KeyError):
            time.sleep(0.1)
    raise RuntimeError("decengine server did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18787")
    args = parser.parse_args()
    wait_ready(args.base_url)

    health = get_json(f"{args.base_url}/healthz")
    assert health["engine"] == "fixture"
    models = get_json(f"{args.base_url}/v1/models")
    assert len(models["data"]) == 2

    fixture = Path(__file__).parents[1] / "protocol" / "fixtures" / "systemone-request.json"
    request = urllib.request.Request(
        f"{args.base_url}/v1/systemone",
        data=fixture.read_bytes(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 200
        decision = json.load(response)
    assert decision["results"]["route"]["type"] == "choice"
    assert decision["results"]["urgent"]["type"] == "noul"
    assert decision["results"]["severity"]["type"] == "score"
    print("HTTP UAT PASS: compatibility client deserialized Choice, Noul, and Score")


if __name__ == "__main__":
    main()
