"""Exact Kou reductions to native Premia vanilla, digital, and barrier claims."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from validation.premia.bridge import PremiaInput, PremiaResult, price_rows
from validation.premia.model.equity.black_scholes.path_option import (
    DirectionalValidationReport,
    summarize_directional_comparisons,
)
from validation.premia.model.equity.terminal_option import premia_model_prefix
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


SUPPORTED_PRODUCTS = frozenset(
    {
        "forward_start_call",
        "forward_start_put",
        "range_accrual",
        "up_no_touch",
        "up_one_touch",
    }
)


def _comparison(
    row: Any,
    reference: PremiaResult,
) -> PremiaPriceComparison:
    return PremiaPriceComparison(
        row_id=row.row_id,
        generated_price=row.generated_price,
        premia_price=reference.price,
        generated_standard_error=row.generated_standard_error,
        premia_standard_error=reference.standard_error,
    )


def _model_values(
    model: Mapping[str, Any], row_id: str
) -> tuple[float, ...]:
    return premia_model_prefix(model, "kou", row_id)


def _forward_start_comparisons(
    rows: Sequence[Any],
    models: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    product_kind: str,
    vanilla_method: str,
) -> list[PremiaPriceComparison]:
    side = "put" if product_kind.endswith("put") else "call"
    inputs: list[PremiaInput] = []
    scales: dict[str, float] = {}
    for row in rows:
        model = models[row.model_id]
        product = products[row.product_id]
        context = f"{product_kind} row '{row.product_id}'"
        reset = parameter_number(product, "reset_time", context, positive=True)
        maturity = parameter_number(product, "maturity", context, positive=True)
        if reset >= maturity:
            raise ValueError(f"{context}: reset_time must precede maturity.")
        moneyness = parameter_number(
            product, "moneyness", context, positive=True
        )
        values = _model_values(model, row.model_id)
        inputs.append(
            PremiaInput(
                row.row_id,
                (1.0, *values[1:], moneyness, maturity - reset),
            )
        )
        scales[row.row_id] = values[0] * math.exp(-values[2] * reset)
    references = price_rows(
        inputs, f"kou_european_{side}", vanilla_method
    )
    return [
        _comparison(
            row,
            PremiaResult(
                price=scales[row.row_id] * references[row.row_id].price,
                standard_error=(
                    scales[row.row_id]
                    * references[row.row_id].standard_error
                ),
            ),
        )
        for row in rows
    ]


def _range_accrual_comparisons(
    rows: Sequence[Any],
    models: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    digital_method: str,
) -> list[PremiaPriceComparison]:
    inputs: list[PremiaInput] = []
    observations: dict[str, list[tuple[str, str, float]]] = {}
    contracts: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        model = models[row.model_id]
        product = products[row.product_id]
        context = f"range_accrual row '{row.product_id}'"
        maturity = parameter_number(product, "maturity", context, positive=True)
        interval = parameter_number(
            product, "observation_interval", context, positive=True
        )
        lower = parameter_number(
            product, "lower_barrier", context, positive=True
        )
        upper = parameter_number(
            product, "upper_barrier", context, positive=True
        )
        if lower >= upper:
            raise ValueError(
                f"{context}: lower_barrier must be below upper_barrier."
            )
        coupon = parameter_number(
            product, "coupon_rate", context, positive=True
        )
        model_values = _model_values(model, row.model_id)
        count = max(1, math.floor(maturity / interval + 0.5))
        row_observations: list[tuple[str, str, float]] = []
        for observation in range(1, count + 1):
            time = observation * interval
            lower_id = f"{row.row_id}_lower_{observation}"
            upper_id = f"{row.row_id}_upper_{observation}"
            inputs.append(
                PremiaInput(lower_id, (*model_values, lower, time))
            )
            inputs.append(
                PremiaInput(upper_id, (*model_values, upper, time))
            )
            row_observations.append((lower_id, upper_id, time))
        observations[row.row_id] = row_observations
        contracts[row.row_id] = (
            model_values[1],
            maturity,
            coupon * interval,
        )
    digitals = price_rows(inputs, "kou_digital_call", digital_method)
    comparisons: list[PremiaPriceComparison] = []
    for row in rows:
        rate, maturity, coupon_per_observation = contracts[row.row_id]
        probability_sum = 0.0
        error_squared = 0.0
        for lower_id, upper_id, time in observations[row.row_id]:
            probability_scale = math.exp(rate * time)
            lower = digitals[lower_id]
            upper = digitals[upper_id]
            probability_sum += probability_scale * (
                lower.price - upper.price
            )
            error_squared += probability_scale * probability_scale * (
                lower.standard_error * lower.standard_error
                + upper.standard_error * upper.standard_error
            )
        discount = math.exp(-rate * maturity)
        comparisons.append(
            _comparison(
                row,
                PremiaResult(
                    price=discount
                    * (1.0 + coupon_per_observation * probability_sum),
                    standard_error=(
                        discount
                        * coupon_per_observation
                        * math.sqrt(error_squared)
                    ),
                ),
            )
        )
    return comparisons


def _touch_comparisons(
    rows: Sequence[Any],
    models: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    product_kind: str,
    barrier_method: str,
    vanilla_method: str,
) -> list[PremiaPriceComparison]:
    rebate_inputs: list[PremiaInput] = []
    vanilla_inputs: list[PremiaInput] = []
    contracts: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        model = models[row.model_id]
        product = products[row.product_id]
        context = f"{product_kind} row '{row.product_id}'"
        values = _model_values(model, row.model_id)
        barrier = parameter_number(product, "barrier", context, positive=True)
        maturity = parameter_number(product, "maturity", context, positive=True)
        cash = parameter_number(product, "cash_payoff", context, positive=True)
        if values[0] >= barrier:
            raise ValueError(f"{context}: barrier must exceed the initial spot.")
        # For K >= H, a terminal call payoff implies that the upper barrier H
        # has been hit.  The vanilla and knock-in calls therefore cancel
        # exactly, leaving only Premia's maturity-paid no-touch rebate.
        rebate_inputs.append(
            PremiaInput(
                row.row_id,
                (*values, barrier, maturity, barrier, cash),
            )
        )
        vanilla_inputs.append(
            PremiaInput(row.row_id, (*values, barrier, maturity))
        )
        contracts[row.row_id] = (values[1], maturity, cash)
    rebates = price_rows(
        rebate_inputs, "kou_up_in_call_rebate", barrier_method
    )
    vanillas = price_rows(
        vanilla_inputs, "kou_european_call", vanilla_method
    )
    comparisons: list[PremiaPriceComparison] = []
    for row in rows:
        rate, maturity, cash = contracts[row.row_id]
        rebate = rebates[row.row_id]
        vanilla = vanillas[row.row_id]
        no_touch = rebate.price - vanilla.price
        error = math.hypot(rebate.standard_error, vanilla.standard_error)
        reference = (
            no_touch
            if product_kind == "up_no_touch"
            else cash * math.exp(-rate * maturity) - no_touch
        )
        comparisons.append(
            _comparison(row, PremiaResult(reference, error))
        )
    return comparisons


def validation_from_premia_kou_specialized_option(
    price_dataset_path: str | Path,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
    *,
    vanilla_method: str = "AP_Carr_Kou",
    digital_method: str = "AP_Kou_Eu",
    barrier_method: str = "AP_Kou_Barrier_In",
) -> PriceValidationReport | DirectionalValidationReport:
    """Validate a Kou payoff that is exactly reducible to native contracts."""

    if product_kind not in SUPPORTED_PRODUCTS:
        raise ValueError(
            f"Unsupported specialized Premia Kou product '{product_kind}'."
        )
    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(
        validation_input.product_dataset_path, "products"
    )
    rows = validation_input.rows
    if product_kind in {"forward_start_call", "forward_start_put"}:
        comparisons = _forward_start_comparisons(
            rows, models, products, product_kind, vanilla_method
        )
    elif product_kind == "range_accrual":
        comparisons = _range_accrual_comparisons(
            rows, models, products, digital_method
        )
    else:
        comparisons = _touch_comparisons(
            rows,
            models,
            products,
            product_kind,
            barrier_method,
            vanilla_method,
        )
        relation = (
            "generated_at_least_reference"
            if product_kind == "up_no_touch"
            else "generated_at_most_reference"
        )
        return summarize_directional_comparisons(
            validation_input.database_id,
            comparisons,
            relation,
            tolerances,
        )
    return summarize_premia_comparisons(
        validation_input.database_id, comparisons, tolerances
    )


__all__ = (
    "SUPPORTED_PRODUCTS",
    "validation_from_premia_kou_specialized_option",
)
