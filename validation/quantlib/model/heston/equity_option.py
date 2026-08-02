"""Common QuantLib pricing recipes for supported Heston equity options."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.model.heston.reference import HestonReference, quantlib_reference
from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.price_validation import (
    PriceResultRow,
    PriceValidationReport,
    ValidationTolerances,
    validation_from_reference,
)
from validation.quantlib.term_structure import (
    DAY_COUNTER,
    REFERENCE_DATE,
    nearest_date_from_time,
)


_TARGET_DT = 1.0 / 252.0
_FLOAT32_EPSILON = 1.1920928955078125e-7
_ASIAN_REFERENCE_SAMPLES = 4096


def _vanilla_payoff(option_type: int, product: Mapping[str, Any]) -> ql.Payoff:
    """Build one positive-strike plain-vanilla payoff."""

    return ql.PlainVanillaPayoff(
        option_type, positive_number(product, "strike", "Equity option")
    )


def _european_call_price(
    model: Mapping[str, Any], product: Mapping[str, Any]
) -> float:
    """Price one European call with QuantLib's analytic Heston engine."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "European call")
    option = ql.VanillaOption(
        _vanilla_payoff(ql.Option.Call, product),
        ql.EuropeanExercise(nearest_date_from_time(maturity)),
    )
    option.setPricingEngine(ql.AnalyticHestonEngine(reference.model))
    return option.NPV()


def _asian_fixing_dates(maturity: float, step_count: int) -> list[ql.Date]:
    """Match the workbench arithmetic average from the first step to maturity."""

    maturity_date = nearest_date_from_time(maturity)
    maturity_days = maturity_date - REFERENCE_DATE
    dates = [
        REFERENCE_DATE + round(maturity_days * step / step_count)
        for step in range(1, step_count + 1)
    ]
    if any(current <= previous for previous, current in zip(dates, dates[1:])):
        raise ValueError("Asian fixing dates must be strictly increasing.")
    return dates


def _asian_call_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    row: PriceResultRow,
) -> tuple[float, float]:
    """Price one arithmetic Asian call with an independent QuantLib MC run."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Asian call")
    step_count = max(1, math.floor(maturity / _TARGET_DT + 0.5))
    fixing_dates = _asian_fixing_dates(maturity, step_count)
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Arithmetic,
        reference.spot,
        1,
        fixing_dates,
        _vanilla_payoff(ql.Option.Call, product),
        ql.EuropeanExercise(fixing_dates[-1]),
    )
    option.setPricingEngine(
        ql.MCDiscreteArithmeticAPHestonEngine(
            reference.process,
            "pseudorandom",
            antitheticVariate=True,
            requiredSamples=_ASIAN_REFERENCE_SAMPLES,
            seed=710000000 + int(row.row_id),
            timeSteps=step_count,
            controlVariate=False,
        )
    )
    return option.NPV(), option.errorEstimate()


def _maturity_anchored_exercise_dates(
    maturity: float, exercise_interval: float
) -> list[ql.Date]:
    """Reproduce the workbench dates T-(E-1)delta through T."""

    raw_count = maturity / exercise_interval
    adjusted = raw_count - 8.0 * _FLOAT32_EPSILON * max(raw_count, 1.0)
    exercise_count = math.ceil(adjusted)
    first_exercise = maturity - (exercise_count - 1) * exercise_interval
    dates = [
        nearest_date_from_time(first_exercise + index * exercise_interval)
        for index in range(exercise_count)
    ]
    if dates[0] <= REFERENCE_DATE or any(
        current <= previous for previous, current in zip(dates, dates[1:])
    ):
        raise ValueError("American-put exercise dates must be strictly increasing.")
    return dates


def _american_put_fallback_price(
    reference: HestonReference,
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    payoff: ql.Payoff,
    exercise: ql.Exercise,
) -> float:
    """Solve rare high-v0 rows on an explicitly widened variance mesh."""

    maturity = DAY_COUNTER.yearFraction(
        REFERENCE_DATE, exercise.lastDate()
    )
    black_scholes_process = ql.FdmBlackScholesMesher.processHelper(
        reference.process.s0(),
        reference.process.riskFreeRate(),
        reference.process.dividendYield(),
        math.sqrt(positive_number(model, "theta", "Heston model")),
    )
    spot_mesher = ql.FdmBlackScholesMesher(
        100,
        black_scholes_process,
        maturity,
        positive_number(product, "strike", "American put"),
    )
    variance_mesher = ql.FdmHestonVarianceMesher(
        100, reference.process, maturity, 10, 1.0e-8
    )
    mesher = ql.FdmMesherComposite(spot_mesher, variance_mesher)
    calculator = ql.FdmLogInnerValue(payoff, mesher, 0)
    conditions = ql.FdmStepConditionComposite.vanillaComposite(
        ql.DividendSchedule(),
        exercise,
        mesher,
        calculator,
        REFERENCE_DATE,
        DAY_COUNTER,
    )
    descriptor = ql.FdmSolverDesc(
        mesher,
        ql.FdmBoundaryConditionSet(),
        conditions,
        calculator,
        maturity,
        100,
        0,
    )
    solver = ql.FdmHestonSolver(
        reference.process, descriptor, ql.FdmSchemeDesc.Hundsdorfer()
    )
    initial_variance = finite_number(model, "initial_variance", "Heston model")
    return solver.valueAt(reference.spot, initial_variance)


def _american_put_price(
    model: Mapping[str, Any], product: Mapping[str, Any]
) -> float:
    """Price the maturity-anchored Bermudan put with QuantLib's Heston PDE."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "American put")
    exercise_interval = positive_number(
        product, "exercise_interval", "American put"
    )
    exercise_dates = _maturity_anchored_exercise_dates(
        maturity, exercise_interval
    )
    payoff = _vanilla_payoff(ql.Option.Put, product)
    exercise = ql.BermudanExercise(exercise_dates)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.FdHestonVanillaEngine(reference.model, 100, 100, 50))
    try:
        pde_price = option.NPV()
    except RuntimeError as error:
        if "interpolation range" not in str(error):
            raise
        # Enlarge only the rare variance meshes that do not contain v0.
        pde_price = _american_put_fallback_price(
            reference, model, product, payoff, exercise
        )
    return max(payoff(reference.spot), pde_price)


def validation_from_quantlib_heston_option(
    price_dataset_path: str | Path,
    product_kind: str,
    tolerances: ValidationTolerances,
) -> PriceValidationReport:
    """Validate one complete Heston option dataset through a common loop."""

    def reference_price(
        model: Mapping[str, Any],
        curve: Mapping[str, Any] | None,
        product: Mapping[str, Any],
        row: PriceResultRow,
    ) -> float | tuple[float, float]:
        if curve is not None:
            raise ValueError("Heston equity prices must not reference a curve dataset.")
        if product_kind == "european_call":
            return _european_call_price(model, product)
        if product_kind == "asian_call":
            return _asian_call_price(model, product, row)
        if product_kind == "american_put":
            return _american_put_price(model, product)
        raise ValueError(f"Unknown Heston product kind '{product_kind}'.")

    return validation_from_reference(
        price_dataset_path,
        reference_price,
        tolerances,
        require_curve=False,
    )
