"""Independent one-factor Jamshidian identities for European swaptions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.price_validation import (
    ValidationRegime,
    load_parameter_rows,
    load_price_validation_input,
    select_validation_regime,
    select_validation_row_ids,
)
from validation.quantlib.rate_option import RateModelFactory


@dataclass(frozen=True)
class QuantLibSwaptionReference:
    """One aligned source identity and its QuantLib swaption price."""

    row_id: str
    model_id: str
    product_id: str
    price: float
    standard_error: float = 0.0
    curve_id: str | None = None


def _payment_count(product: Mapping[str, Any]) -> int:
    value = positive_number(product, "payment_count", "European swaption")
    count = round(value)
    if abs(value - count) > 1.0e-9:
        raise ValueError("European swaption: payment_count must be an integer.")
    return count


def swaption_times(product: Mapping[str, Any]) -> tuple[float, ...]:
    """Return exercise and every payment time needed by a fitted curve."""

    exercise = positive_number(product, "exercise_time", "European swaption")
    interval = positive_number(product, "payment_interval", "European swaption")
    return (exercise,) + tuple(
        exercise + index * interval
        for index in range(1, _payment_count(product) + 1)
    )


def _state_boundary(
    model: Any,
    exercise: float,
    payment_times: tuple[float, ...],
    weights: tuple[float, ...],
) -> float:
    """Solve the monotone coupon-bond exercise equation in double precision."""

    def residual(state: float) -> float:
        return sum(
            weight * model.discountBond(exercise, maturity, state)
            for weight, maturity in zip(weights, payment_times)
        ) - 1.0

    lower = -0.25
    upper = 0.25
    lower_value = residual(lower)
    upper_value = residual(upper)
    for _ in range(64):
        if lower_value >= 0.0:
            break
        lower = 2.0 * lower - 0.25
        lower_value = residual(lower)
    for _ in range(64):
        if upper_value <= 0.0:
            break
        upper = 2.0 * upper + 0.25
        upper_value = residual(upper)
    if (
        not math.isfinite(lower_value)
        or not math.isfinite(upper_value)
        or lower_value < 0.0
        or upper_value > 0.0
    ):
        raise RuntimeError("Could not bracket the Jamshidian state boundary.")

    for _ in range(96):
        midpoint = 0.5 * (lower + upper)
        value = residual(midpoint)
        if value > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def swaption_price(model: Any, product: Mapping[str, Any], side: str) -> float:
    """Price one payer or receiver swaption by Jamshidian decomposition."""

    if side not in {"payer", "receiver"}:
        raise ValueError(f"Unsupported swaption side '{side}'.")
    context = "European swaption"
    notional = positive_number(product, "notional", context)
    strike = finite_number(product, "strike", context)
    if strike < 0.0:
        raise ValueError(f"{context}: strike must be non-negative.")
    accrual = positive_number(product, "accrual_fraction", context)
    times = swaption_times(product)
    exercise = times[0]
    payment_times = times[1:]
    weights = [strike * accrual] * len(payment_times)
    weights[-1] += 1.0
    weight_tuple = tuple(weights)
    boundary = _state_boundary(
        model, exercise, payment_times, weight_tuple
    )
    option_type = ql.Option.Put if side == "payer" else ql.Option.Call
    price = sum(
        weight
        * model.discountBondOption(
            option_type,
            model.discountBond(exercise, maturity, boundary),
            exercise,
            maturity,
        )
        for weight, maturity in zip(weight_tuple, payment_times)
    )
    return notional * max(float(price), 0.0)


def reference_prices_from_quantlib_swaption(
    price_dataset_path: str | Path,
    model_factory: RateModelFactory,
    side: str,
    require_curve: bool,
    regime: ValidationRegime = "all",
    row_ids: tuple[str, ...] | None = None,
) -> tuple[QuantLibSwaptionReference, ...]:
    """Price one selected batch for persistence without comparing to CUDA."""

    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    if require_curve != (validation_input.curve_dataset_path is not None):
        expected = "with" if require_curve else "without"
        raise ValueError(f"Expected a price dataset {expected} a curve reference.")
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(validation_input.product_dataset_path, "products")
    curves = (
        load_parameter_rows(validation_input.curve_dataset_path, "curves")
        if validation_input.curve_dataset_path is not None
        else None
    )
    references: list[QuantLibSwaptionReference] = []
    for row in validation_input.rows:
        product = products[row.product_id]
        curve = curves[row.curve_id] if curves is not None else None
        model = model_factory(models[row.model_id], curve, product)
        references.append(
            QuantLibSwaptionReference(
                row_id=row.row_id,
                model_id=row.model_id,
                product_id=row.product_id,
                price=swaption_price(model, product, side),
                curve_id=row.curve_id,
            )
        )
    return tuple(references)


__all__ = (
    "QuantLibSwaptionReference",
    "reference_prices_from_quantlib_swaption",
    "swaption_price",
    "swaption_times",
)
