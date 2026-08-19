"""Persistent references for standalone Gaussian short-rate options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

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


@dataclass(frozen=True)
class GaussianRateModelSpec:
    """Reference hierarchy for one standalone Gaussian-rate model."""

    premia_model: str | None
    quantlib_pricing_method: str | None


_MODELS: Mapping[str, GaussianRateModelSpec] = {
    "vasicek": GaussianRateModelSpec("vasicek", None),
    "ornstein_uhlenbeck": GaussianRateModelSpec(
        "ornstein_uhlenbeck",
        None,
    ),
    "g2": GaussianRateModelSpec(None, "G2.discountBondOption"),
}


def _spec(model_name: str) -> GaussianRateModelSpec:
    """Return the small model-specific part of the shared hierarchy."""

    try:
        return _MODELS[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported Gaussian-rate model '{model_name}'.") from error


def _premia_pricing_method(
    model_name: str,
    product_kind: str,
) -> tuple[str, str]:
    """Name the exact Premia formula and any required OU mapping."""

    side = "call" if is_call_side(product_kind) else "put"
    method = (
        "CF_Vasicek1d_ZBCallEuro"
        if side == "call"
        else "CF_Vasicek1d_ZBPutEuro"
    )
    identifier = f"premia_cf_vasicek_zb_{side}"
    if model_name == "ornstein_uhlenbeck":
        identifier += "_centered_ou_mapping"
        method += " (centered OU mapping)"
    return identifier, method


def reference_pricers(
    model_name: str,
    product_kind: str,
    used_backend_version: str,
) -> dict[str, Any]:
    """Describe the ordered hierarchy and detail only the selected engine."""

    spec = _spec(model_name)
    validate_product_kind(product_kind)
    sections: dict[str, Any] = {}
    for regime, count in REGIME_ROW_COUNTS:
        if spec.premia_model is not None:
            identifier, method = _premia_pricing_method(model_name, product_kind)
            premia = detailed_pricer(
                identifier, "Premia", used_backend_version, method, count
            )
            quantlib = {"status": "available"}
        else:
            premia = {"status": "not_available"}
            if spec.quantlib_pricing_method is None:
                raise ValueError(f"{model_name}: QuantLib method is not declared.")
            quantlib = detailed_pricer(
                "quantlib_g2_discount_bond_option",
                "QuantLib",
                used_backend_version,
                spec.quantlib_pricing_method,
                count,
            )
        sections[regime] = {
            "row_count": count,
            "premia": premia,
            "quantlib_specialized": quantlib,
            "quantlib_monte_carlo": {"status": "not_available"},
        }
    return sections


# =============================================================================
# Common fixed-income reference interface
# =============================================================================


def generate_reference_dataset(
    price_dataset_path: str | Path,
    reference_dataset_path: str | Path,
    model_name: str,
    product_kind: str,
) -> ReferenceDatasetValidation:
    """Run the historically selected independent engine and persist 1000 rows."""

    spec = _spec(model_name)
    validate_product_kind(product_kind)
    started = time.perf_counter()
    if spec.premia_model is not None:
        from validation.premia.model.fixed_income.rate_option import (
            reference_prices_from_premia_rate_option,
        )

        generated = reference_prices_from_premia_rate_option(
            price_dataset_path, spec.premia_model, product_kind
        )
        identifier, _ = _premia_pricing_method(model_name, product_kind)
        rows = tuple(
            ReferencePrice(
                row.row_id,
                row.model_id,
                row.product_id,
                row.price,
                row.standard_error,
                identifier,
            )
            for row in generated
        )
        backend_version = "19"
    else:
        import QuantLib as ql

        from validation.quantlib.model.fixed_income.g2.reference import (
            quantlib_model,
        )
        from validation.quantlib.rate_option import (
            reference_prices_from_quantlib_rate_option,
        )

        generated = reference_prices_from_quantlib_rate_option(
            price_dataset_path, quantlib_model, product_kind, False
        )
        rows = tuple(
            ReferencePrice(
                row.row_id,
                row.model_id,
                row.product_id,
                row.price,
                row.standard_error,
                "quantlib_g2_discount_bond_option",
            )
            for row in generated
        )
        backend_version = ql.__version__
    wall_seconds = time.perf_counter() - started
    return persist_generated_reference(
        price_dataset_path,
        reference_dataset_path,
        rows,
        reference_pricers(model_name, product_kind, backend_version),
        wall_seconds,
    )


def validate_dataset(
    price_dataset_path: str | Path,
    reference_dataset_path: str | Path,
) -> ReferenceDatasetValidation:
    """Validate one standalone Gaussian-rate cache without external engines."""

    return validate_cached_reference(price_dataset_path, reference_dataset_path)


def run_product_validation_cli(model_name: str, product_kind: str) -> int:
    """Generate explicitly or compare the immutable reference by default."""

    _spec(model_name)
    validate_product_kind(product_kind)
    return run_reference_cli(
        f"Validate one {model_name} {product_kind} dataset.",
        model_name,
        lambda source, destination: generate_reference_dataset(
            source,
            destination,
            model_name,
            product_kind,
        ),
    )


__all__ = (
    "_MODELS",
    "_PRODUCT_KINDS",
    "generate_reference_dataset",
    "reference_pricers",
    "run_product_validation_cli",
    "validate_dataset",
)
