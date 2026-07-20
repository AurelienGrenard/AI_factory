"""Low-level writers shared by model and product database recipes."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from tools.registry.common.grids import add_numeric_ids
from tools.registry.common.paths import registry_database_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_TIER = "validation"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def without_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def with_numeric_ids(parameters: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return add_numeric_ids(list(parameters))


def database_path(
    kind: str,
    section: str,
    database_id: str,
    suffix: str,
    *,
    registry_tier: str = DEFAULT_REGISTRY_TIER,
) -> Path:
    return registry_database_path(
        PROJECT_ROOT,
        registry_tier,
        kind,
        section,
        database_id,
        suffix,
    )


def registry_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def display_name(identifier: str) -> str:
    return identifier.replace("_", " ").title()
