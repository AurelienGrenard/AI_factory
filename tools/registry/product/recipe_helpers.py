"""Parameter-grid helpers used by product database recipes."""

from __future__ import annotations

from typing import Any

from tools.registry.common.database_writing import without_none
from tools.registry.product.core_parameter_databases import write_product_database


def write_product_recipe_database(
    *,
    database_id: str,
    product_family: str,
    title: str,
    payoff_key: str,
    expression: str,
    scaling_rule: str,
    parameters: list[dict[str, Any]],
    construction: dict[str, Any],
    parameter_docs: dict[str, str],
    registry_tier: str = "validation",
) -> None:
    write_product_database(
        database_id=database_id,
        product_family=product_family,
        title=f"{title} {database_id}",
        payoff={
            "expression": expression,
            "scaling_rule": scaling_rule,
        },
        parameters=parameters,
        construction=without_none(construction),
        parameter_docs=parameter_docs,
        registry_tier=registry_tier,
    )
