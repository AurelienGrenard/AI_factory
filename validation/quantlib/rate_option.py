"""Common QuantLib identities for analytical fixed-income price datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.price_validation import (
    PriceResultRow,
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
    validation_from_reference,
)


RateModelFactory = Callable[
    [Mapping[str, Any], Mapping[str, Any] | None, Mapping[str, Any]], Any
]


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
