"""Persistent Premia references for fitted Hull-White and G2++ options."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from validation.model.fixed_income.reference_pipeline import (
    PRODUCT_KINDS,
    REGIME_ROW_COUNTS,
    detailed_pricer,
    is_call_side,
    persist_generated_reference,
    run_reference_cli,
    validate_cached_reference,
    validate_product_kind,
)
from validation.reference_price_dataset import (
    ReferenceDatasetValidation,
    ReferencePrice,
)


_PRODUCT_KINDS = PRODUCT_KINDS


# =============================================================================
# Model-specific reference mapping
# =============================================================================


_SPECS = frozenset(
    {
        ("hull_white", "nelson_siegel"),
        ("hull_white", "svensson"),
        ("g2_plus_plus", "nelson_siegel"),
        ("g2_plus_plus", "svensson"),
    }
)


def _spec(model_name: str, curve_name: str) -> tuple[str, str]:
    """Validate and return one supported fitted model/curve pair."""

    pair = (model_name, curve_name)
    if pair not in _SPECS:
        raise ValueError(
            f"Unsupported fitted-rate pair '{model_name}/{curve_name}'."
        )
    return pair


def _premia_pricer(
    model_name: str,
    product_kind: str,
    row_priced: int,
) -> dict[str, Any]:
    """Describe the Premia formula selected by model and option side."""

    side = "call" if is_call_side(product_kind) else "put"
    if model_name == "hull_white":
        identifier = f"premia_cf_hull_white_1d_zb_{side}"
        method = (
            "CF_HullWhite1d_ZBCallEuro"
            if side == "call"
            else "CF_HullWhite1d_ZBPutEuro"
        )
    else:
        identifier = f"premia_cf_hull_white_2d_zb_{side}"
        method = "CF_ZBCallEuroHW2D" if side == "call" else "CF_ZBPutEuroHW2D"
    return detailed_pricer(identifier, "Premia", "19", method, row_priced)


def reference_pricers(
    model_name: str,
    curve_name: str,
    product_kind: str,
) -> dict[str, Any]:
    """Describe callable engines and detail only the Premia method used."""

    _spec(model_name, curve_name)
    validate_product_kind(product_kind)
    return {
        regime: {
            "row_count": count,
            "premia": _premia_pricer(model_name, product_kind, count),
            "quantlib_specialized": {"status": "available"},
            "quantlib_monte_carlo": {"status": "not_available"},
        }
        for regime, count in REGIME_ROW_COUNTS
    }


# =============================================================================
# Common fixed-income reference interface
# =============================================================================


def generate_reference_dataset(
    price_dataset_path: str | Path,
    reference_dataset_path: str | Path,
    model_name: str,
    curve_name: str,
    product_kind: str,
) -> ReferenceDatasetValidation:
    """Run the selected Premia formula and persist its 1000 aligned prices."""

    _spec(model_name, curve_name)
    used_pricer = _premia_pricer(model_name, product_kind, 1)["id"]
    from validation.premia.model.fixed_income.fitted_rate_option import (
        reference_prices_from_premia_fitted_rate_option,
    )

    started = time.perf_counter()
    generated = reference_prices_from_premia_fitted_rate_option(
        price_dataset_path,
        model_name,
        curve_name,
        product_kind,
    )
    wall_seconds = time.perf_counter() - started
    rows = tuple(
        ReferencePrice(
            row.row_id,
            row.model_id,
            row.product_id,
            row.price,
            row.standard_error,
            used_pricer,
            row.curve_id,
        )
        for row in generated
    )
    return persist_generated_reference(
        price_dataset_path,
        reference_dataset_path,
        rows,
        reference_pricers(model_name, curve_name, product_kind),
        wall_seconds,
    )


def validate_dataset(
    price_dataset_path: str | Path,
    reference_dataset_path: str | Path,
) -> ReferenceDatasetValidation:
    """Validate one fitted Gaussian-rate cache without external engines."""

    return validate_cached_reference(price_dataset_path, reference_dataset_path)


def run_product_validation_cli(
    model_name: str,
    curve_name: str,
    product_kind: str,
) -> int:
    """Generate explicitly or compare the immutable reference by default."""

    _spec(model_name, curve_name)
    validate_product_kind(product_kind)
    return run_reference_cli(
        f"Validate one {model_name}/{curve_name} {product_kind} dataset.",
        f"{model_name}/{curve_name}",
        lambda source, destination: generate_reference_dataset(
            source,
            destination,
            model_name,
            curve_name,
            product_kind,
        ),
    )


__all__ = (
    "_PRODUCT_KINDS",
    "_SPECS",
    "generate_reference_dataset",
    "reference_pricers",
    "run_product_validation_cli",
    "validate_dataset",
)
