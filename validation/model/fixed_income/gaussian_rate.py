"""Uniform validation policy for analytical Gaussian short-rate options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from validation.dataset_validation import (
    ValidationPolicy,
    build_dataset_validation_report,
    quantlib_row_exception,
    run_dataset_validation_cli,
    unavailable_engine,
)
from validation.hierarchy import (
    BackendBatchResult,
    ValidationEngine,
    isolate_backend_exceptions,
)
from validation.premia.model.fixed_income.rate_option import (
    validation_batch_from_premia_rate_option,
)
from validation.quantlib.model.fixed_income.g2.reference import (
    quantlib_model as g2_quantlib_model,
)
from validation.quantlib.model.fixed_income.ornstein_uhlenbeck.reference import (
    quantlib_model as ornstein_uhlenbeck_quantlib_model,
)
from validation.quantlib.model.fixed_income.vasicek.reference import (
    quantlib_model as vasicek_quantlib_model,
)
from validation.quantlib.price_validation import ValidationRegime
from validation.quantlib.rate_option import validation_from_quantlib_rate_option
from validation.reporting import DatasetValidationReport


_PRODUCT_KINDS = frozenset(
    {
        "caplet",
        "floorlet",
        "zero_coupon_bond_call",
        "zero_coupon_bond_put",
    }
)


@dataclass(frozen=True)
class GaussianRateModelSpec:
    """Independent reference mapping for one standalone Gaussian-rate model."""

    model_name: str
    quantlib_model: Any
    quantlib_pricing_method: str
    premia_model: str | None


_MODELS: Mapping[str, GaussianRateModelSpec] = {
    "vasicek": GaussianRateModelSpec(
        "vasicek",
        vasicek_quantlib_model,
        "Vasicek.discountBondOption",
        "vasicek",
    ),
    "ornstein_uhlenbeck": GaussianRateModelSpec(
        "ornstein_uhlenbeck",
        ornstein_uhlenbeck_quantlib_model,
        "Vasicek.discountBondOption (centered OU mapping)",
        "ornstein_uhlenbeck",
    ),
    "g2": GaussianRateModelSpec(
        "g2", g2_quantlib_model, "G2.discountBondOption", None
    ),
}


def _premia_pricing_method(product_kind: str) -> str:
    """Return the exact Premia method after the caplet bond-option identity."""

    if product_kind in {"zero_coupon_bond_call", "floorlet"}:
        return "CF_Vasicek1d_ZBCallEuro"
    return "CF_Vasicek1d_ZBPutEuro"


def _spec(model_name: str) -> GaussianRateModelSpec:
    try:
        return _MODELS[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported Gaussian-rate model '{model_name}'.") from error


def _engine_plan(
    model_name: str,
    product_kind: str,
) -> tuple[ValidationEngine, ...]:
    if product_kind not in _PRODUCT_KINDS:
        raise ValueError(f"Unsupported rate product '{product_kind}'.")
    spec = _spec(model_name)

    if spec.premia_model is not None:
        def premia(
            price_dataset_path: Path,
            regime: ValidationRegime,
            row_ids: tuple[str, ...],
        ) -> BackendBatchResult:
            return validation_batch_from_premia_rate_option(
                price_dataset_path,
                spec.premia_model,
                product_kind,
                regime=regime,
                row_ids=row_ids,
            )

        premia_engine = ValidationEngine(
            "Premia",
            "specialized pricer",
            premia,
            pricing_method=_premia_pricing_method(product_kind),
        )
    else:
        premia_engine = unavailable_engine(
            "Premia",
            "specialized pricer",
            "Premia Hull-White 2D requires an external fitted curve and does "
            "not expose both standalone G2 initial factors",
        )

    def quantlib_specialized(
        price_dataset_path: Path,
        regime: ValidationRegime,
        row_ids: tuple[str, ...],
    ) -> BackendBatchResult:
        return isolate_backend_exceptions(
            row_ids,
            lambda selected: validation_from_quantlib_rate_option(
                price_dataset_path,
                spec.quantlib_model,
                product_kind,
                False,
                regime=regime,
                row_ids=selected,
            ),
            quantlib_row_exception,
        )

    return (
        premia_engine,
        ValidationEngine(
            "QuantLib",
            "specialized pricer",
            quantlib_specialized,
            pricing_method=spec.quantlib_pricing_method,
        ),
        unavailable_engine(
            "QuantLib",
            "Monte Carlo",
            "an exact specialized QuantLib engine is available",
        ),
    )


def validate_dataset(
    price_dataset_path: str | Path,
    model_name: str,
    product_kind: str,
) -> DatasetValidationReport:
    """Validate one complete Gaussian-rate price dataset."""

    return build_dataset_validation_report(
        price_dataset_path,
        lambda: _engine_plan(model_name, product_kind),
        ValidationPolicy(),
    )


def run_product_validation_cli(model_name: str, product_kind: str) -> int:
    """Expose the identical two-path CLI for every thin product module."""

    return run_dataset_validation_cli(
        lambda path: validate_dataset(path, model_name, product_kind),
        f"Validate one {model_name} {product_kind} dataset.",
    )


__all__ = (
    "_MODELS",
    "_PRODUCT_KINDS",
    "_engine_plan",
    "run_product_validation_cli",
    "validate_dataset",
)
