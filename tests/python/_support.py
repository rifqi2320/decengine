from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))
sys.path.insert(0, str(REPOSITORY / "tests" / "native"))

from build_fixture import build  # noqa: E402


@lru_cache(maxsize=1)
def native_fixture() -> Path:
    return build(REPOSITORY / "build" / "test-native")
