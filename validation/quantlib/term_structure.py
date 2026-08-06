"""Exact-time QuantLib dates and discount curves for synthetic datasets."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Any

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number


REFERENCE_DATE = ql.Date(1, ql.January, 2025)
DAY_COUNTER = ql.Actual360()


def date_from_time(time: float) -> ql.Date:
    """Map synthetic year fractions to exact Actual/360 calendar dates."""

    day_count = round(360.0 * time)
    if time < 0.0 or not math.isclose(
        day_count / 360.0, time, rel_tol=0.0, abs_tol=2.0e-6
    ):
        raise ValueError(f"Time {time} is not representable on the Actual/360 grid.")
    return REFERENCE_DATE + day_count


def nearest_date_from_time(time: float) -> ql.Date:
    """Map a positive synthetic maturity to its nearest Actual/360 date."""

    if not math.isfinite(time) or time <= 0.0:
        raise ValueError("A QuantLib maturity must be finite and positive.")
    return REFERENCE_DATE + round(360.0 * time)


def flat_curve(rate: float) -> ql.YieldTermStructureHandle:
    """Build one continuously compounded flat curve at the shared date."""

    curve = ql.FlatForward(
        REFERENCE_DATE, rate, DAY_COUNTER, ql.Continuous, ql.NoFrequency
    )
    return ql.YieldTermStructureHandle(curve)


def discount_curve(
    discount: Callable[[float], float],
    required_times: Iterable[float],
) -> ql.YieldTermStructureHandle:
    """Build a log-linear curve that is exact at every requested maturity."""

    times = sorted({0.0, *(float(time) for time in required_times)})
    dates = [date_from_time(time) for time in times]
    discounts = [float(discount(time)) for time in times]
    if any(not math.isfinite(value) or value <= 0.0 for value in discounts):
        raise ValueError("Every QuantLib discount factor must be finite and positive.")
    curve = ql.DiscountCurve(dates, discounts, DAY_COUNTER, ql.NullCalendar())
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def nelson_siegel_discount(
    parameters: Mapping[str, Any], maturity: float
) -> float:
    """Evaluate the workbench Nelson-Siegel discount factor in FP64."""

    context = "Nelson-Siegel curve"
    beta0 = finite_number(parameters, "beta0", context)
    beta1 = finite_number(parameters, "beta1", context)
    beta2 = finite_number(parameters, "beta2", context)
    tau = positive_number(parameters, "tau", context)
    if maturity == 0.0:
        return 1.0
    scaled_time = maturity / tau
    loading = -math.expm1(-scaled_time) / scaled_time
    zero_rate = beta0 + beta1 * loading + beta2 * (
        loading - math.exp(-scaled_time)
    )
    return math.exp(-maturity * zero_rate)


def svensson_discount(
    parameters: Mapping[str, Any], maturity: float
) -> float:
    """Evaluate the workbench Svensson discount factor in FP64."""

    context = "Svensson curve"
    beta0 = finite_number(parameters, "beta0", context)
    beta1 = finite_number(parameters, "beta1", context)
    beta2 = finite_number(parameters, "beta2", context)
    beta3 = finite_number(parameters, "beta3", context)
    tau1 = positive_number(parameters, "tau1", context)
    tau2 = positive_number(parameters, "tau2", context)
    if tau2 <= tau1:
        raise ValueError(f"{context}: tau2 must exceed tau1.")
    if maturity == 0.0:
        return 1.0
    scaled_time1 = maturity / tau1
    scaled_time2 = maturity / tau2
    loading1 = -math.expm1(-scaled_time1) / scaled_time1
    loading2 = -math.expm1(-scaled_time2) / scaled_time2
    zero_rate = beta0 + beta1 * loading1 + beta2 * (
        loading1 - math.exp(-scaled_time1)
    ) + beta3 * (loading2 - math.exp(-scaled_time2))
    return math.exp(-maturity * zero_rate)
