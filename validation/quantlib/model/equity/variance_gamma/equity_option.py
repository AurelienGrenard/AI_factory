"""QuantLib recipes for terminal-payoff Variance-Gamma options."""

import math
from pathlib import Path
from typing import Any, Mapping

import QuantLib as ql
from scipy.integrate import quad

from validation.quantlib.model.equity.variance_gamma.reference import (
    quantlib_reference,
)
from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.price_validation import (
    PriceResultRow,
    PriceValidationReport,
    ValidationTolerances,
    validation_from_reference,
)


_QUADRATURE_ABSOLUTE_ERROR = 1.0e-10
_LOG_GAMMA_LOWER_BOUND = 50.0


def _price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    payoff: ql.Payoff,
) -> float:
    """Integrate QuantLib conditional Black prices over the Gamma clock.

    QuantLib's ``VarianceGammaEngine`` uses this same conditional-Black
    representation, but integrates the Gamma density directly from zero.  Its
    fixed split is inaccurate when ``maturity / nu`` is small because the
    density is then singular.  The logarithmic coordinates below remove that
    singularity without changing the reference pricing formula.
    """

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Equity option")
    # The CUDA contract is expressed in continuous synthetic time.  Unlike a
    # QuantLib instrument date, the conditional mixture can preserve that time
    # exactly; this matters for digital payoffs near a short-dated threshold.
    time = maturity
    sigma = positive_number(model, "sigma", "Variance-Gamma model")
    nu = positive_number(model, "nu", "Variance-Gamma model")
    theta = finite_number(model, "theta", "Variance-Gamma model")
    risk_free_rate = finite_number(
        model, "risk_free_rate", "Variance-Gamma model"
    )
    dividend_yield = finite_number(
        model, "dividend_yield", "Variance-Gamma model"
    )
    martingale_argument = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
    omega = math.log(martingale_argument) / nu
    gamma_shape = time / nu
    log_gamma_normalization = math.lgamma(gamma_shape)
    risk_free_discount = math.exp(-risk_free_rate * time)
    dividend_discount = math.exp(-dividend_yield * time)

    def conditional_black_price(unit_gamma_clock: float) -> float:
        gamma_clock = nu * unit_gamma_clock
        adjusted_spot = reference.spot * math.exp(
            theta * gamma_clock
            + omega * time
            + 0.5 * sigma * sigma * gamma_clock
        )
        forward = (
            adjusted_spot * dividend_discount / risk_free_discount
        )
        return ql.BlackCalculator(
            payoff,
            forward,
            sigma * math.sqrt(gamma_clock),
            risk_free_discount,
        ).value()

    # On y in (0, 1], set u = -shape*log(y).  The Gamma singularity
    # y^(shape-1) becomes an exponentially decaying, regular integrand.
    def lower_clock_integrand(u: float) -> float:
        scaled_u = u / gamma_shape
        unit_gamma_clock = (
            math.exp(-scaled_u) if scaled_u < 80.0 else 0.0
        )
        log_density_jacobian = (
            -u
            - unit_gamma_clock
            - math.log(gamma_shape)
            - log_gamma_normalization
        )
        if log_density_jacobian < -745.0:
            return 0.0
        return conditional_black_price(unit_gamma_clock) * math.exp(
            log_density_jacobian
        )

    # On y in [1, infinity), set v = log(y).  The first-moment margin
    # determines a conservative finite endpoint even for asset payoffs.
    tail_scale = max(martingale_argument, 0.05)
    effective_clock = (
        gamma_shape + 50.0 * math.sqrt(gamma_shape) + 50.0
    ) / tail_scale
    log_gamma_upper_bound = max(10.0, math.log(effective_clock))

    def upper_clock_integrand(v: float) -> float:
        unit_gamma_clock = math.exp(v)
        log_density_jacobian = (
            gamma_shape * v
            - unit_gamma_clock
            - log_gamma_normalization
        )
        if log_density_jacobian < -745.0:
            return 0.0
        return conditional_black_price(unit_gamma_clock) * math.exp(
            log_density_jacobian
        )

    lower_price = quad(
        lower_clock_integrand,
        0.0,
        _LOG_GAMMA_LOWER_BOUND,
        epsabs=_QUADRATURE_ABSOLUTE_ERROR,
        epsrel=1.0e-9,
        limit=300,
    )[0]
    upper_price = quad(
        upper_clock_integrand,
        0.0,
        log_gamma_upper_bound,
        epsabs=_QUADRATURE_ABSOLUTE_ERROR,
        epsrel=1.0e-9,
        limit=300,
    )[0]
    return lower_price + upper_price


def _payoff(product_kind: str, product: Mapping[str, Any]) -> ql.Payoff:
    context = "Equity option"
    call = ql.Option.Call
    put = ql.Option.Put
    if product_kind in {"european_call", "european_put", "straddle"}:
        option_type = call if product_kind != "european_put" else put
        return ql.PlainVanillaPayoff(
            option_type, positive_number(product, "strike", context)
        )
    if product_kind in {"digital_call", "digital_put"}:
        option_type = call if product_kind == "digital_call" else put
        return ql.CashOrNothingPayoff(
            option_type,
            positive_number(product, "strike", context),
            positive_number(product, "cash_payoff", context),
        )
    if product_kind in {
        "asset_or_nothing_call",
        "asset_or_nothing_put",
    }:
        option_type = call if product_kind.endswith("call") else put
        return ql.AssetOrNothingPayoff(
            option_type, positive_number(product, "strike", context)
        )
    if product_kind in {"gap_call", "gap_put"}:
        option_type = call if product_kind == "gap_call" else put
        return ql.GapPayoff(
            option_type,
            positive_number(product, "trigger_strike", context),
            positive_number(product, "payoff_strike", context),
        )
    raise ValueError(f"Unsupported QuantLib VG product '{product_kind}'.")


def validation_from_quantlib_variance_gamma_option(
    price_dataset_path: str | Path,
    product_kind: str,
    tolerances: ValidationTolerances,
) -> PriceValidationReport:
    """Validate one complete supported VG price dataset independently."""

    def reference_price(
        model: Mapping[str, Any],
        curve: Mapping[str, Any] | None,
        product: Mapping[str, Any],
        row: PriceResultRow,
    ) -> float:
        del row
        if curve is not None:
            raise ValueError("Variance-Gamma prices cannot reference a curve.")
        if product_kind == "straddle":
            call = ql.PlainVanillaPayoff(
                ql.Option.Call,
                positive_number(product, "strike", "Straddle"),
            )
            put = ql.PlainVanillaPayoff(
                ql.Option.Put,
                positive_number(product, "strike", "Straddle"),
            )
            return _price(model, product, call) + _price(model, product, put)
        return _price(model, product, _payoff(product_kind, product))

    return validation_from_reference(
        price_dataset_path,
        reference_price,
        tolerances,
        require_curve=False,
    )
