"""Premia validation for Hull-White and G2++ fitted-curve claims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from validation.hierarchy import BackendBatchResult, BackendException
from validation.premia.bridge import PremiaInput, price_rows, price_rows_partial
from validation.premia.price_validation import (
    PremiaPriceComparison,
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
    load_parameter_rows,
    load_price_validation_input,
    parameter_number,
    select_validation_regime,
    select_validation_row_ids,
    summarize_premia_comparisons,
)


_PRODUCTS = {
    "zero_coupon_bond_call": "zero_coupon_call",
    "zero_coupon_bond_put": "zero_coupon_put",
    "caplet": "zero_coupon_put",
    "floorlet": "zero_coupon_call",
}
_MODEL_PARAMETERS = {
    "hull_white": ("mean_reversion", "volatility"),
    "g2_plus_plus": (
        "mean_reversion_x",
        "volatility_x",
        "mean_reversion_y",
        "volatility_y",
        "correlation",
    ),
}
_CURVE_PARAMETERS = {
    "nelson_siegel": ("beta0", "beta1", "beta2", "tau"),
    "svensson": ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2"),
}


@dataclass(frozen=True)
class PremiaFittedRateOptionReference:
    """One source identity and its scaled fitted-rate Premia price."""

    row_id: str
    model_id: str
    curve_id: str
    product_id: str
    price: float
    standard_error: float


def _numbers(
    parameters: Mapping[str, object],
    names: tuple[str, ...],
    context: str,
) -> tuple[float, ...]:
    return tuple(parameter_number(parameters, name, context) for name in names)


def _prepared_inputs(
    price_dataset_path: str | Path,
    model_name: str,
    curve_name: str,
    product_kind: str,
    regime: ValidationRegime,
    row_ids: Sequence[str] | None,
):
    if model_name not in _MODEL_PARAMETERS:
        raise ValueError(f"Unsupported Premia fitted-rate model '{model_name}'.")
    if curve_name not in _CURVE_PARAMETERS:
        raise ValueError(f"Unsupported Premia fitted curve '{curve_name}'.")
    if product_kind not in _PRODUCTS:
        raise ValueError(f"Unsupported Premia rate product '{product_kind}'.")
    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    if validation_input.curve_dataset_path is None:
        raise ValueError("Fitted Hull-White/G2++ validation requires a curve.")
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    curves = load_parameter_rows(validation_input.curve_dataset_path, "curves")
    products = load_parameter_rows(validation_input.product_dataset_path, "products")

    inputs: list[PremiaInput] = []
    price_scales: dict[str, float] = {}
    for row in validation_input.rows:
        model = models[row.model_id]
        curve = curves[row.curve_id]
        product = products[row.product_id]
        prefix = _numbers(
            model,
            _MODEL_PARAMETERS[model_name],
            f"{model_name} model row '{row.model_id}'",
        ) + _numbers(
            curve,
            _CURVE_PARAMETERS[curve_name],
            f"{curve_name} curve row '{row.curve_id}'",
        )
        context = f"{product_kind} row '{row.product_id}'"
        notional = parameter_number(product, "notional", context, positive=True)
        if product_kind.startswith("zero_coupon_bond_"):
            suffix = (
                parameter_number(product, "strike", context, positive=True),
                parameter_number(
                    product, "option_expiry", context, positive=True
                ),
                parameter_number(
                    product, "bond_maturity", context, positive=True
                ),
                1.0,
            )
            price_scales[row.row_id] = notional
        else:
            strike = parameter_number(product, "strike", context)
            accrual = parameter_number(
                product, "accrual_period", context, positive=True
            )
            strike_factor = 1.0 + accrual * strike
            if strike_factor <= 0.0:
                raise ValueError(
                    f"{context}: 1 + accrual_period * strike must be positive."
                )
            suffix = (
                1.0 / strike_factor,
                parameter_number(product, "fixing_time", context, positive=True),
                parameter_number(product, "payment_time", context, positive=True),
                1.0,
            )
            price_scales[row.row_id] = notional * strike_factor
        inputs.append(PremiaInput(row.row_id, prefix + suffix))
    return validation_input, tuple(inputs), price_scales


def _mode(model_name: str, curve_name: str, product_kind: str) -> str:
    return f"{model_name}_{curve_name}_{_PRODUCTS[product_kind]}"


def validation_from_premia_fitted_rate_option(
    price_dataset_path: str | Path,
    model_name: str,
    curve_name: str,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> PriceValidationReport:
    """Compare one fitted Gaussian-rate dataset with a Premia formula."""

    validation_input, inputs, scales = _prepared_inputs(
        price_dataset_path, model_name, curve_name, product_kind, regime, row_ids
    )
    results = price_rows(inputs, _mode(model_name, curve_name, product_kind))
    generated = {row.row_id: row for row in validation_input.rows}
    comparisons = tuple(
        PremiaPriceComparison(
            row_id=row_id,
            generated_price=generated[row_id].generated_price,
            premia_price=result.price * scales[row_id],
            generated_standard_error=generated[row_id].generated_standard_error,
            premia_standard_error=result.standard_error,
        )
        for row_id, result in results.items()
    )
    return summarize_premia_comparisons(
        validation_input.database_id, comparisons, tolerances
    )


def reference_prices_from_premia_fitted_rate_option(
    price_dataset_path: str | Path,
    model_name: str,
    curve_name: str,
    product_kind: str,
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> tuple[PremiaFittedRateOptionReference, ...]:
    """Price one fitted-rate batch for persistence without comparing to CUDA."""

    validation_input, inputs, scales = _prepared_inputs(
        price_dataset_path, model_name, curve_name, product_kind, regime, row_ids
    )
    results = price_rows(inputs, _mode(model_name, curve_name, product_kind))
    references = []
    for row in validation_input.rows:
        if row.curve_id is None:
            raise ValueError(f"Fitted-rate row '{row.row_id}' has no curve id.")
        result = results[row.row_id]
        references.append(
            PremiaFittedRateOptionReference(
                row.row_id,
                row.model_id,
                row.curve_id,
                row.product_id,
                result.price * scales[row.row_id],
                result.standard_error,
            )
        )
    return tuple(references)


def validation_batch_from_premia_fitted_rate_option(
    price_dataset_path: str | Path,
    model_name: str,
    curve_name: str,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> BackendBatchResult:
    """Preserve successful Premia rows and expose row-local failures."""

    validation_input, inputs, scales = _prepared_inputs(
        price_dataset_path, model_name, curve_name, product_kind, regime, row_ids
    )
    results, failures = price_rows_partial(
        inputs, _mode(model_name, curve_name, product_kind)
    )
    generated = {row.row_id: row for row in validation_input.rows}
    completed = tuple(
        row.row_id for row in validation_input.rows if row.row_id in results
    )
    reports = ()
    if completed:
        comparisons = tuple(
            PremiaPriceComparison(
                row_id=row_id,
                generated_price=generated[row_id].generated_price,
                premia_price=results[row_id].price * scales[row_id],
                generated_standard_error=generated[row_id].generated_standard_error,
                premia_standard_error=results[row_id].standard_error,
            )
            for row_id in completed
        )
        reports = (
            summarize_premia_comparisons(
                validation_input.database_id, comparisons, tolerances
            ),
        )
    exceptions = tuple(
        BackendException(
            row.row_id,
            failures[row.row_id].reason,
            failures[row.row_id].status,
        )
        for row in validation_input.rows
        if row.row_id in failures
    )
    return BackendBatchResult(completed, exceptions, reports)


__all__ = (
    "PremiaFittedRateOptionReference",
    "reference_prices_from_premia_fitted_rate_option",
    "validation_batch_from_premia_fitted_rate_option",
    "validation_from_premia_fitted_rate_option",
)
