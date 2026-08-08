"""QuantLib analytic validation for CEV European calls and puts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.price_validation import (
    PriceResultRow, PriceValidationReport, ValidationTolerances,
    validation_from_reference,
)
from validation.quantlib.term_structure import (
    REFERENCE_DATE, flat_curve, nearest_date_from_time,
)


def _price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    context = "CEV model"
    spot = positive_number(model, "spot", context)
    risk_free_rate = finite_number(model, "risk_free_rate", context)
    dividend_yield = finite_number(model, "dividend_yield", context)
    sigma = positive_number(model, "sigma", context)
    beta = finite_number(model, "beta", context)
    if not 0.5 <= beta < 1.0:
        raise ValueError("CEV model: beta must lie in [0.5, 1).")
    maturity = positive_number(product, "maturity", "European option")
    strike = positive_number(product, "strike", "European option")

    # X_t = exp(-(r-q)t) S_t removes the drift. Its deterministic diffusion
    # scale is reduced to a constant CEV coefficient by one exact time change.
    carry = risk_free_rate - dividend_yield
    exponent = 2.0 * (beta - 1.0) * carry
    clock = maturity if abs(exponent) < 1.0e-14 else math.expm1(exponent * maturity) / exponent
    effective_sigma = sigma * math.sqrt(clock / maturity)
    transformed_strike = strike * math.exp(-carry * maturity)

    ql.Settings.instance().evaluationDate = REFERENCE_DATE
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(option_type, transformed_strike),
        ql.EuropeanExercise(nearest_date_from_time(maturity)),
    )
    option.setPricingEngine(ql.AnalyticCEVEngine(
        spot, effective_sigma, beta, flat_curve(dividend_yield)
    ))
    return option.NPV()


def validation_from_quantlib_cev_option(
    price_dataset_path: str | Path,
    option_type: int,
    tolerances: ValidationTolerances = ValidationTolerances(),
) -> PriceValidationReport:
    def reference_price(
        model: Mapping[str, Any],
        curve: Mapping[str, Any] | None,
        product: Mapping[str, Any],
        row: PriceResultRow,
    ) -> float:
        del row
        if curve is not None:
            raise ValueError("CEV equity validation expects no curve dataset.")
        return _price(model, product, option_type)

    return validation_from_reference(
        price_dataset_path, reference_price, tolerances, require_curve=False
    )
