"""Writers for reproducible product-parameter databases."""

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


def write_product_database(
    *,
    database_id: str,
    product_family: str,
    payoff: dict[str, Any],
    parameters: Iterable[dict[str, Any]],
    title: str,
    construction: dict[str, Any],
    parameter_docs: dict[str, str],
    registry_tier: str = DEFAULT_REGISTRY_TIER,
) -> Path:
    rows = with_numeric_ids(parameters)
    json_file = database_path(
        "products", "data", database_id, "json", registry_tier=registry_tier
    )
    spec_file = database_path(
        "products", "specifications", database_id, "yaml", registry_tier=registry_tier
    )
    generator_file = database_path(
        "products", "generators", database_id, "py", registry_tier=registry_tier
    )
    write_json(
        json_file,
        {
            "format": "ai_factory.products.v1",
            "database_id": database_id,
            "product_family": display_name(product_family),
            "specification": registry_relative(spec_file),
            "generation_script": registry_relative(generator_file),
            "row_count": len(rows),
            "products": rows,
        },
    )
    write_yaml(
        spec_file,
        {
            "title": title,
            "format": "ai_factory.products.v1",
            "database_id": database_id,
            "product_family": display_name(product_family),
            "json_path": registry_relative(json_file),
            "generation_script": registry_relative(generator_file),
            "parameters": parameter_docs,
            "payoff": payoff,
            "construction": {
                "row_count": len(rows),
                **{
                    key: value
                    for key, value in construction.items()
                    if key not in {"row_count", "columns"}
                },
            },
        },
    )
    return json_file
