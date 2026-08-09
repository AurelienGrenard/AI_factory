"""Uniform validation policy for fitted Hull-White and G2++ options."""

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
from validation.premia.model.fixed_income.fitted_rate_option import (
    validation_batch_from_premia_fitted_rate_option,
)
from validation.quantlib.model.fixed_income.g2_plus_plus.nelson_siegel.reference import (
    quantlib_model as g2_plus_plus_nelson_siegel_quantlib_model,
)
from validation.quantlib.model.fixed_income.g2_plus_plus.svensson.reference import (
    quantlib_model as g2_plus_plus_svensson_quantlib_model,
)
from validation.quantlib.model.fixed_income.hull_white.nelson_siegel.reference import (
    quantlib_model as hull_white_nelson_siegel_quantlib_model,
)
from validation.quantlib.model.fixed_income.hull_white.svensson.reference import (
    quantlib_model as hull_white_svensson_quantlib_model,
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
class FittedGaussianRateSpec:
    """Independent engines for one model and initial-curve family."""

    model_name: str
    curve_name: str
    quantlib_model: Any


_SPECS: Mapping[tuple[str, str], FittedGaussianRateSpec] = {
    ("hull_white", "nelson_siegel"): FittedGaussianRateSpec(
        "hull_white", "nelson_siegel", hull_white_nelson_siegel_quantlib_model
    ),
    ("hull_white", "svensson"): FittedGaussianRateSpec(
        "hull_white", "svensson", hull_white_svensson_quantlib_model
    ),
    ("g2_plus_plus", "nelson_siegel"): FittedGaussianRateSpec(
        "g2_plus_plus",
        "nelson_siegel",
        g2_plus_plus_nelson_siegel_quantlib_model,
    ),
    ("g2_plus_plus", "svensson"): FittedGaussianRateSpec(
        "g2_plus_plus", "svensson", g2_plus_plus_svensson_quantlib_model
    ),
}


def _spec(model_name: str, curve_name: str) -> FittedGaussianRateSpec:
    try:
        return _SPECS[(model_name, curve_name)]
    except KeyError as error:
        raise ValueError(
            f"Unsupported fitted-rate pair '{model_name}/{curve_name}'."
        ) from error


def _engine_plan(
    model_name: str,
    curve_name: str,
    product_kind: str,
) -> tuple[ValidationEngine, ...]:
    if product_kind not in _PRODUCT_KINDS:
        raise ValueError(f"Unsupported rate product '{product_kind}'.")
    spec = _spec(model_name, curve_name)

    def premia(
        price_dataset_path: Path,
        regime: ValidationRegime,
        row_ids: tuple[str, ...],
    ) -> BackendBatchResult:
        return validation_batch_from_premia_fitted_rate_option(
            price_dataset_path,
            spec.model_name,
            spec.curve_name,
            product_kind,
            regime=regime,
            row_ids=row_ids,
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
                True,
                regime=regime,
                row_ids=selected,
            ),
            quantlib_row_exception,
        )

    return (
        ValidationEngine("Premia", "specialized pricer", premia),
        ValidationEngine(
            "QuantLib", "specialized pricer", quantlib_specialized
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
    curve_name: str,
    product_kind: str,
) -> DatasetValidationReport:
    """Validate one complete fitted Gaussian-rate price dataset."""

    return build_dataset_validation_report(
        price_dataset_path,
        lambda: _engine_plan(model_name, curve_name, product_kind),
        ValidationPolicy(),
    )


def run_product_validation_cli(
    model_name: str, curve_name: str, product_kind: str
) -> int:
    """Expose the identical two-path CLI for every thin product module."""

    return run_dataset_validation_cli(
        lambda path: validate_dataset(
            path, model_name, curve_name, product_kind
        ),
        f"Validate one {model_name}/{curve_name} {product_kind} dataset.",
    )


__all__ = (
    "_PRODUCT_KINDS",
    "_SPECS",
    "_engine_plan",
    "run_product_validation_cli",
    "validate_dataset",
)
