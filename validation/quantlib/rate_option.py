"""Common QuantLib identities for analytical fixed-income price datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.price_validation import (
    PriceResultRow,
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
    load_parameter_rows,
    load_price_validation_input,
    select_validation_regime,
    select_validation_row_ids,
    validation_from_reference,
)


RateModelFactory = Callable[
    [Mapping[str, Any], Mapping[str, Any] | None, Mapping[str, Any]], Any
]


@dataclass(frozen=True)
class QuantLibRateOptionReference:
    """One aligned source identity and its QuantLib reference price."""

    row_id: str
    model_id: str
    product_id: str
    price: float
    standard_error: float = 0.0


def bond_option_times(product: Mapping[str, Any]) -> tuple[float, float]:
    """Return expiry and bond maturity for any supported rate option."""

    if "option_expiry" in product:
        return (
            positive_number(product, "option_expiry", "Bond option"),
            positive_number(product, "bond_maturity", "Bond option"),
        )
    return (
        positive_number(product, "fixing_time", "Rate option"),
        positive_number(product, "payment_time", "Rate option"),
    )


def _bond_option_price(
    model: Any,
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price one unit-notional QuantLib discount-bond option."""

    context = "Zero-coupon bond option"
    notional = positive_number(product, "notional", context)
    strike = positive_number(product, "strike", context)
    option_expiry = positive_number(product, "option_expiry", context)
    bond_maturity = positive_number(product, "bond_maturity", context)
    if bond_maturity <= option_expiry:
        raise ValueError(f"{context}: bond_maturity must exceed option_expiry.")
    return notional * model.discountBondOption(
        option_type, strike, option_expiry, bond_maturity
    )


def _rate_option_price(
    model: Any,
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Apply the caplet/floorlet identity to a discount-bond option."""

    context = "Caplet or floorlet"
    notional = positive_number(product, "notional", context)
    strike = finite_number(product, "strike", context)
    fixing_time = positive_number(product, "fixing_time", context)
    payment_time = positive_number(product, "payment_time", context)
    accrual_period = positive_number(product, "accrual_period", context)
    if payment_time <= fixing_time:
        raise ValueError(f"{context}: payment_time must exceed fixing_time.")
    strike_factor = 1.0 + accrual_period * strike
    if strike_factor <= 0.0:
        raise ValueError(f"{context}: 1 + accrual_period * strike must be positive.")
    return notional * strike_factor * model.discountBondOption(
        option_type,
        1.0 / strike_factor,
        fixing_time,
        payment_time,
    )


def validation_from_quantlib_rate_option(
    price_dataset_path: str | Path,
    model_factory: RateModelFactory,
    product_kind: str,
    require_curve: bool,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> PriceValidationReport:
    """Validate one complete analytical rate-option dataset."""

    def reference_price(
        model_parameters: Mapping[str, Any],
        curve_parameters: Mapping[str, Any] | None,
        product_parameters: Mapping[str, Any],
        _: PriceResultRow,
    ) -> float:
        model = model_factory(
            model_parameters, curve_parameters, product_parameters
        )
        if product_kind == "zero_coupon_bond_call":
            return _bond_option_price(model, product_parameters, ql.Option.Call)
        if product_kind == "zero_coupon_bond_put":
            return _bond_option_price(model, product_parameters, ql.Option.Put)
        if product_kind == "caplet":
            return _rate_option_price(model, product_parameters, ql.Option.Put)
        if product_kind == "floorlet":
            return _rate_option_price(model, product_parameters, ql.Option.Call)
        raise ValueError(f"Unknown fixed-income product kind '{product_kind}'.")

    return validation_from_reference(
        price_dataset_path,
        reference_price,
        tolerances,
        require_curve=require_curve,
        regime=regime,
        row_ids=row_ids,
    )


def reference_prices_from_quantlib_rate_option(
    price_dataset_path: str | Path,
    model_factory: RateModelFactory,
    product_kind: str,
    require_curve: bool,
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> tuple[QuantLibRateOptionReference, ...]:
    """Price one aligned batch for persistence without comparing to CUDA."""

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
    references: list[QuantLibRateOptionReference] = []
    for row in validation_input.rows:
        model_parameters = models[row.model_id]
        product = products[row.product_id]
        curve = curves[row.curve_id] if curves is not None else None
        model = model_factory(model_parameters, curve, product)
        if product_kind == "zero_coupon_bond_call":
            price = _bond_option_price(model, product, ql.Option.Call)
        elif product_kind == "zero_coupon_bond_put":
            price = _bond_option_price(model, product, ql.Option.Put)
        elif product_kind == "caplet":
            price = _rate_option_price(model, product, ql.Option.Put)
        elif product_kind == "floorlet":
            price = _rate_option_price(model, product, ql.Option.Call)
        else:
            raise ValueError(f"Unknown fixed-income product kind '{product_kind}'.")
        references.append(
            QuantLibRateOptionReference(
                row.row_id, row.model_id, row.product_id, float(price)
            )
        )
    return tuple(references)


__all__ = (
    "QuantLibRateOptionReference",
    "RateModelFactory",
    "bond_option_times",
    "reference_prices_from_quantlib_rate_option",
    "validation_from_quantlib_rate_option",
)
