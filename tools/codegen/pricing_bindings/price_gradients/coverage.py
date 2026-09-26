"""Inventory Markovian pricing bindings and their published derivative orders.

This reports launcher coverage, not numerical qualification or support for
every coordinate at the listed order. The pricing manifest remains the source
of truth for which model/product pairs exist.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def markovian_coverage(pricing_bindings, gradient_bindings, models):
    """Return one deterministic row per existing Markovian pricing binding."""
    gradients = {}
    for specification in gradient_bindings:
        binding = specification.pricing
        key = (binding.model, binding.curve, binding.product, binding.engine)
        if key in gradients:
            raise ValueError(f"Duplicate price-gradient binding: {key}")
        gradients[key] = specification

    rows = []
    for binding in pricing_bindings:
        model = models[binding.model]
        if binding.asset_class != "fixed_income" and model.family != "markovian":
            continue
        key = (binding.model, binding.curve, binding.product, binding.engine)
        gradient = gradients.pop(key, None)
        orders = gradient.supported_orders if gradient is not None else ()
        rows.append({
            "model": binding.model,
            "curve": binding.curve,
            "product": binding.product,
            "engine": binding.engine,
            "pricing_binding": binding.unit_path,
            "gradient_binding": gradient.unit_path if gradient is not None else None,
            "supported_orders": list(orders),
            "coverage": (
                "full_hessian" if "mixed_second" in orders
                else "first_and_diagonal_second" if "diagonal_second" in orders
                else "first_only" if "first" in orders
                else "missing_binding"
            ),
        })

    if gradients:
        raise ValueError(
            "Price-gradient bindings without a Markovian pricing binding: "
            + repr(sorted(gradients, key=str))
        )
    return rows


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from capability_manifest import (  # pylint: disable=import-outside-toplevel
        MODEL_BY_NAME,
        PRICE_GRADIENT_BINDING_SPECS,
        PRODUCT_BINDING_SPECS,
    )

    rows = markovian_coverage(
        PRODUCT_BINDING_SPECS,
        PRICE_GRADIENT_BINDING_SPECS,
        MODEL_BY_NAME,
    )
    summary = {
        label: sum(row["coverage"] == label for row in rows)
        for label in (
            "missing_binding", "first_only",
            "first_and_diagonal_second", "full_hessian"
        )
    }
    print(json.dumps({"summary": summary, "bindings": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
