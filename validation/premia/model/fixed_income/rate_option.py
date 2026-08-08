"""Shared Premia validation for standalone Vasicek and centered OU claims."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
    # A caplet/floorlet is exactly a scaled put/call on the payment-date
    # discount bond.  Using that identity avoids accidentally comparing the
    # single-period catalogue contract with Premia's multi-reset Cap/Floor.
    "caplet": "zero_coupon_put",
    "floorlet": "zero_coupon_call",
}


def _prepared_inputs(
    price_dataset_path: str | Path,
    model_name: str,
    product_kind: str,
    regime: ValidationRegime,
    row_ids: Sequence[str] | None,
):
    """Load and convert one selected batch exactly once."""

    if model_name not in {"vasicek", "ornstein_uhlenbeck"}:
        raise ValueError(f"Unsupported Premia rate model '{model_name}'.")
    if product_kind not in _PRODUCTS:
        raise ValueError(f"Unsupported Premia rate product '{product_kind}'.")
    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    if validation_input.curve_dataset_path is not None:
        raise ValueError("Standalone Vasicek/OU validation expects no curve dataset.")
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(validation_input.product_dataset_path, "products")

    inputs: list[PremiaInput] = []
    price_scales: dict[str, float] = {}
    for row in validation_input.rows:
        model = models[row.model_id]
        product = products[row.product_id]
        model_context = f"{model_name} model row '{row.model_id}'"
        prefix = (
            parameter_number(model, "initial_state", model_context),
            parameter_number(
                model, "mean_reversion", model_context, positive=True
            ),
            parameter_number(
                model, "volatility", model_context, positive=True
            ),
            parameter_number(model, "long_term_mean", model_context)
            if model_name == "vasicek" else 0.0,
        )
        product_context = f"{product_kind} row '{row.product_id}'"
        if product_kind.startswith("zero_coupon_bond_"):
            notional = parameter_number(
                product, "notional", product_context, positive=True
            )
            suffix = (
                parameter_number(product, "strike", product_context, positive=True),
                parameter_number(
                    product, "option_expiry", product_context, positive=True
                ),
                parameter_number(
                    product, "bond_maturity", product_context, positive=True
                ),
                1.0,
            )
            price_scales[row.row_id] = notional
        else:
            notional = parameter_number(
                product, "notional", product_context, positive=True
            )
            strike = parameter_number(product, "strike", product_context)
            accrual_period = parameter_number(
                product, "accrual_period", product_context, positive=True
            )
            strike_factor = 1.0 + accrual_period * strike
            if strike_factor <= 0.0:
                raise ValueError(
                    f"{product_context}: 1 + accrual_period * strike must be positive."
                )
            suffix = (
                1.0 / strike_factor,
                parameter_number(
                    product, "fixing_time", product_context, positive=True
                ),
                parameter_number(
                    product, "payment_time", product_context, positive=True
                ),
                1.0,
            )
            price_scales[row.row_id] = notional * strike_factor
        inputs.append(PremiaInput(row.row_id, (*prefix, *suffix)))
    return validation_input, tuple(inputs), price_scales


def validation_from_premia_rate_option(
    price_dataset_path: str | Path,
    model_name: str,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> PriceValidationReport:
    """Compare a complete standalone Gaussian-rate dataset with Premia."""

    validation_input, inputs, price_scales = _prepared_inputs(
        price_dataset_path, model_name, product_kind, regime, row_ids
    )

    mode_suffix = _PRODUCTS[product_kind]
    results = price_rows(inputs, f"{model_name}_{mode_suffix}")
    comparisons = tuple(
        PremiaPriceComparison(
            row_id=row.row_id,
            generated_price=row.generated_price,
            premia_price=results[row.row_id].price * price_scales[row.row_id],
            generated_standard_error=row.generated_standard_error,
            premia_standard_error=results[row.row_id].standard_error,
        )
        for row in validation_input.rows
    )
    return summarize_premia_comparisons(
        validation_input.database_id, comparisons, tolerances
    )


def validation_batch_from_premia_rate_option(
    price_dataset_path: str | Path,
    model_name: str,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> BackendBatchResult:
    """Price once, preserving successful rows and row-local Premia failures."""

    validation_input, inputs, price_scales = _prepared_inputs(
        price_dataset_path, model_name, product_kind, regime, row_ids
    )
    mode_suffix = _PRODUCTS[product_kind]
    results, failures = price_rows_partial(inputs, f"{model_name}_{mode_suffix}")
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
                premia_price=results[row_id].price * price_scales[row_id],
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
    "validation_batch_from_premia_rate_option",
    "validation_from_premia_rate_option",
)
