"""Minimal JSON loading utilities for registry orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def load_json_file(path: Path | str) -> JsonObject:
    """Load a JSON object from disk."""

    with Path(path).open(encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return loaded
