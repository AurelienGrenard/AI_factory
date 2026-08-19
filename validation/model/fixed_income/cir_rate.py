"""Generate and validate persistent QuantLib references for CIR rate options."""

from __future__ import annotations

from pathlib import Path
import time

from validation.model.fixed_income.reference_pipeline import (
    PRODUCT_KINDS,
    REGIME_ROW_COUNTS,
    detailed_pricer,
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


def reference_pricers(product_kind: str, backend_version: str) -> dict:
    """Record callable engines separately from the selected reliable reference."""

    validate_product_kind(product_kind)
    sections = {}
    for regime, count in REGIME_ROW_COUNTS:
        method = (
            "CoxIngersollRoss.discountBondOption with "
            "OTM-tail parity stabilization"
        )
        quantlib = detailed_pricer(
            "quantlib_cir_discount_bond_option",
            "QuantLib",
            backend_version,
            method,
            count,
        )
        sections[regime] = {
            "row_count": count,
            "premia": {"status": "available but not reliable"},
            "quantlib_specialized": dict(quantlib),
            "quantlib_monte_carlo": {"status": "not_available"},
        }
    return sections


# =============================================================================
# Common fixed-income reference interface
# =============================================================================


def generate_reference_dataset(
    price_dataset_path: str | Path,
    reference_dataset_path: str | Path,
    product_kind: str,
) -> ReferenceDatasetValidation:
    """Persist 1000 QuantLib prices and compare both ordered regimes."""

    import QuantLib as ql

    from validation.quantlib.model.fixed_income.cir.reference import quantlib_model
    from validation.quantlib.rate_option import (
        reference_prices_from_quantlib_rate_option,
    )

    started = time.perf_counter()
    quantlib_rows = reference_prices_from_quantlib_rate_option(
        price_dataset_path, quantlib_model, product_kind, False
    )
    quantlib_seconds = time.perf_counter() - started
    rows = tuple(
        ReferencePrice(
            row.row_id,
            row.model_id,
            row.product_id,
            row.price,
            row.standard_error,
            "quantlib_cir_discount_bond_option",
        )
        for row in quantlib_rows
    )
    return persist_generated_reference(
        price_dataset_path,
        reference_dataset_path,
        rows,
        reference_pricers(product_kind, ql.__version__),
        quantlib_seconds,
    )


def validate_dataset(
    price_dataset_path: str | Path,
    reference_dataset_path: str | Path,
) -> ReferenceDatasetValidation:
    """Validate CIR prices from the cache without rerunning external engines."""

    return validate_cached_reference(price_dataset_path, reference_dataset_path)


def run_product_validation_cli(product_kind: str) -> int:
    """Generate explicitly or compare the immutable reference by default."""

    validate_product_kind(product_kind)
    return run_reference_cli(
        f"Validate one CIR {product_kind} dataset.",
        "CIR",
        lambda source, destination: generate_reference_dataset(
            source, destination, product_kind
        ),
    )


__all__ = (
    "_PRODUCT_KINDS",
    "generate_reference_dataset",
    "reference_pricers",
    "run_product_validation_cli",
    "validate_dataset",
)
