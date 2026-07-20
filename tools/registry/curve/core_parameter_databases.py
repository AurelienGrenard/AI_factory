"""Writers for reproducible market-curve databases."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tools.registry.common.database_writing import (
    DEFAULT_REGISTRY_TIER,
    database_path,
    display_name,
    registry_relative,
    with_numeric_ids,
    write_json,
    write_yaml,
)


def write_curve_database(
    *,
    database_id: str,
    curve_family: str,
    parameters: Iterable[dict[str, Any]],
    title: str,
    construction: dict[str, Any],
    parameter_docs: dict[str, str],
    equations: dict[str, str],
    registry_tier: str = DEFAULT_REGISTRY_TIER,
) -> Path:
    rows = with_numeric_ids(parameters)
    json_path = database_path(
        "curves", "data", database_id, "json", registry_tier=registry_tier
    )
    yaml_path = database_path(
        "curves", "specifications", database_id, "yaml", registry_tier=registry_tier
    )
    generator_path = database_path(
        "curves", "generators", database_id, "py", registry_tier=registry_tier
    )
    write_json(
        json_path,
        {
            "format": "ai_factory.curves.v1",
            "database_id": database_id,
            "curve_family": display_name(curve_family),
            "specification": registry_relative(yaml_path),
            "generation_script": registry_relative(generator_path),
            "row_count": len(rows),
            "curves": rows,
        },
    )
    write_yaml(
        yaml_path,
        {
            "title": title,
            "format": "ai_factory.curves.v1",
            "database_id": database_id,
            "curve_family": display_name(curve_family),
            "json_path": registry_relative(json_path),
            "generation_script": registry_relative(generator_path),
            "parameters": parameter_docs,
            "equations": equations,
            "construction": {"row_count": len(rows), **construction},
        },
    )
    return json_path
