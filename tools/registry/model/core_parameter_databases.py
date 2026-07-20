"""Writers for reproducible model-parameter databases."""

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


def _model_construction_doc(
    construction: dict[str, Any],
    *,
    row_count: int,
) -> dict[str, Any]:
    documented: dict[str, Any] = {"row_count": row_count}
    if "method" in construction:
        documented["method"] = construction["method"]
    if "rule" in construction:
        documented["rule"] = construction["rule"]

    values = {
        key: value
        for key, value in construction.items()
        if key not in {"method", "rule", "row_count", "columns"}
    }
    if values:
        documented["values"] = values
    return documented


def write_model_database(
    *,
    database_id: str,
    model_family: str,
    parameters: Iterable[dict[str, Any]],
    title: str,
    construction: dict[str, Any],
    parameter_docs: dict[str, str],
    dynamics: dict[str, Any],
    registry_tier: str = DEFAULT_REGISTRY_TIER,
) -> Path:
    rows = with_numeric_ids(parameters)
    json_file = database_path(
        "models", "data", database_id, "json", registry_tier=registry_tier
    )
    spec_file = database_path(
        "models", "specifications", database_id, "yaml", registry_tier=registry_tier
    )
    generator_file = database_path(
        "models", "generators", database_id, "py", registry_tier=registry_tier
    )
    write_json(
        json_file,
        {
            "format": "ai_factory.models.v1",
            "database_id": database_id,
            "model_family": display_name(model_family),
            "specification": registry_relative(spec_file),
            "generation_script": registry_relative(generator_file),
            "row_count": len(rows),
            "models": rows,
        },
    )
    write_yaml(
        spec_file,
        {
            "title": title,
            "format": "ai_factory.models.v1",
            "database_id": database_id,
            "model_family": display_name(model_family),
            "json_path": registry_relative(json_file),
            "generation_script": registry_relative(generator_file),
            "parameters": parameter_docs,
            "dynamics": dynamics,
            "construction": _model_construction_doc(
                construction,
                row_count=len(rows),
            ),
        },
    )
    return json_file
