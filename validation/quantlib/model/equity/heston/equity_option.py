"""Common QuantLib pricing recipes for supported Heston equity options."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping

import QuantLib as ql

from validation.quantlib.model.equity.heston.reference import HestonReference, quantlib_reference
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
_AUTOCALL_REFERENCE_PAIRS = 1024
_CLIQUET_REFERENCE_PAIRS = 1024
_RANGE_ACCRUAL_REFERENCE_PAIRS = 1024


def _vanilla_payoff(option_type: int, product: Mapping[str, Any]) -> ql.Payoff:
    """Build one positive-strike plain-vanilla payoff."""

    return ql.PlainVanillaPayoff(
        option_type, positive_number(product, "strike", "Equity option")
    )


def _european_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price one European vanilla with QuantLib's analytic Heston engine."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "European option")
    option = ql.VanillaOption(
        _vanilla_payoff(option_type, product),
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


def _asian_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    row: PriceResultRow,
    option_type: int,
) -> tuple[float, float]:
    """Price one arithmetic Asian option with independent QuantLib MC."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Asian option")
    step_count = max(1, math.floor(maturity / _TARGET_DT + 0.5))
    fixing_dates = _asian_fixing_dates(maturity, step_count)
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Arithmetic,
        reference.spot,
        1,
        fixing_dates,
        _vanilla_payoff(option_type, product),
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


def _geometric_asian_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    row: PriceResultRow,
    option_type: int,
) -> tuple[float, float]:
    """Price one geometric Asian option with independent QuantLib MC."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Geometric Asian option")
    step_count = max(1, math.floor(maturity / _TARGET_DT + 0.5))
    fixing_dates = _asian_fixing_dates(maturity, step_count)
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Geometric,
        reference.spot,
        1,
        fixing_dates,
        _vanilla_payoff(option_type, product),
        ql.EuropeanExercise(fixing_dates[-1]),
    )
    option.setPricingEngine(
        ql.MCDiscreteGeometricAPHestonEngine(
            reference.process,
            "pseudorandom",
            timeSteps=step_count,
            antitheticVariate=True,
            requiredSamples=_ASIAN_REFERENCE_SAMPLES,
            seed=730000000 + int(row.row_id),
        )
    )
    return option.NPV(), option.errorEstimate()


def _forward_start_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    row: PriceResultRow,
    option_type: int,
) -> float | tuple[float, float]:
    """Price one forward-start option with independent QuantLib MC."""

    reference = quantlib_reference(model)
    reset_time = positive_number(product, "reset_time", "Forward-start option")
    maturity = positive_number(product, "maturity", "Forward-start option")
    moneyness = positive_number(product, "moneyness", "Forward-start option")
    if reset_time >= maturity:
        raise ValueError("Forward-start option: reset_time must precede maturity.")
    option = ql.ForwardVanillaOption(
        moneyness,
        nearest_date_from_time(reset_time),
        ql.PlainVanillaPayoff(option_type, 0.0),
        ql.EuropeanExercise(nearest_date_from_time(maturity)),
    )
    step_count = max(1, math.floor(maturity / _TARGET_DT + 0.5))
    option.setPricingEngine(
        ql.MCForwardEuropeanHestonEngine(
            reference.process,
            "pseudorandom",
            timeSteps=step_count,
            antitheticVariate=True,
            requiredSamples=4096,
            seed=720000000 + int(row.row_id),
        )
    )
    return option.NPV(), option.errorEstimate()


def _regular_observation_grid(
    product: Mapping[str, Any],
    context: str,
) -> tuple[float, float, int, int, list[float]]:
    """Construct the regular observation and simulation grid of one product."""

    maturity = positive_number(product, "maturity", context)
    interval = positive_number(product, "observation_interval", context)
    raw_observation_count = maturity / interval
    observation_count = round(raw_observation_count)
    tolerance = 32.0 * _FLOAT32_EPSILON * max(raw_observation_count, 1.0)
    if observation_count < 1 or abs(
        raw_observation_count - observation_count
    ) > tolerance:
        raise ValueError(
            f"{context}: maturity must be an integer multiple of "
            "observation_interval."
        )
    steps_per_observation = max(1, math.floor(interval / _TARGET_DT + 0.5))
    total_step_count = observation_count * steps_per_observation
    effective_dt = interval / steps_per_observation
    times = [effective_dt * step for step in range(1, total_step_count + 1)]
    return (
        maturity,
        interval,
        observation_count,
        steps_per_observation,
        times,
    )


def _antithetic_heston_path_price(
    reference: HestonReference,
    times: list[float],
    seed: int,
    pair_count: int,
    discounted_payoff: Callable[[ql.Path], float],
) -> tuple[float, float]:
    """Estimate one payoff and error from independent antithetic paths."""

    total_step_count = len(times)
    dimension = reference.process.factors() * total_step_count
    uniform_sequence = ql.UniformRandomSequenceGenerator(
        dimension,
        ql.UniformRandomGenerator(seed),
    )
    path_generator = ql.GaussianMultiPathGenerator(
        reference.process,
        times,
        ql.GaussianRandomSequenceGenerator(uniform_sequence),
        False,
    )

    mean = 0.0
    sum_squared_deviations = 0.0
    for pair_index in range(pair_count):
        direct = discounted_payoff(path_generator.next().value()[0])
        antithetic = discounted_payoff(path_generator.antithetic().value()[0])
        pair_value = 0.5 * (direct + antithetic)
        delta = pair_value - mean
        mean += delta / (pair_index + 1)
        sum_squared_deviations += delta * (pair_value - mean)
    variance = sum_squared_deviations / (pair_count - 1)
    return mean, math.sqrt(variance / pair_count)


def _autocall_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    row: PriceResultRow,
    product_kind: str,
) -> tuple[float, float]:
    """Price one autocall from independent antithetic QuantLib paths."""

    context = "Autocall"
    reference = quantlib_reference(model)
    _, interval, observation_count, steps_per_observation, times = (
        _regular_observation_grid(product, context)
    )
    autocall_barrier = positive_number(product, "autocall_barrier", context)
    protection_barrier = positive_number(
        product, "protection_barrier", context
    )
    annual_coupon_rate = positive_number(
        product, "annual_coupon_rate", context
    )
    coupon_barrier = None
    if product_kind != "athena_autocall":
        coupon_barrier = positive_number(product, "coupon_barrier", context)
    if not protection_barrier <= autocall_barrier:
        raise ValueError("Autocall: protection barrier exceeds autocall barrier.")
    if coupon_barrier is not None and not (
        protection_barrier <= coupon_barrier <= autocall_barrier
    ):
        raise ValueError("Autocall: coupon barrier ordering is invalid.")

    risk_free_rate = finite_number(model, "risk_free_rate", "Heston model")
    discounts = [
        math.exp(-risk_free_rate * interval * (observation + 1))
        for observation in range(observation_count)
    ]
    coupon_per_observation = annual_coupon_rate * interval

    def discounted_payoff(spot_path: ql.Path) -> float:
        present_value = 0.0
        remembered_coupon = 0.0
        accumulated_gain = 0.0
        for observation in range(observation_count):
            spot = spot_path[(observation + 1) * steps_per_observation]
            discount = discounts[observation]
            maturity_observation = observation + 1 == observation_count
            if product_kind == "athena_autocall":
                accumulated_gain += coupon_per_observation
                if not maturity_observation and spot >= autocall_barrier:
                    return discount * (1.0 + accumulated_gain)
                if maturity_observation:
                    if spot >= autocall_barrier:
                        return discount * (1.0 + accumulated_gain)
                    capital = 1.0 if spot >= protection_barrier else spot
                    return discount * capital
                continue

            if product_kind == "phoenix_memory_autocall":
                remembered_coupon += coupon_per_observation
                coupon = remembered_coupon if spot >= coupon_barrier else 0.0
                if coupon > 0.0:
                    remembered_coupon = 0.0
            else:
                coupon = (
                    coupon_per_observation if spot >= coupon_barrier else 0.0
                )
            if not maturity_observation and spot >= autocall_barrier:
                return present_value + discount * (1.0 + coupon)
            if maturity_observation:
                capital = 1.0 if spot >= protection_barrier else spot
                return present_value + discount * (capital + coupon)
            present_value += discount * coupon
        raise RuntimeError("Autocall path reached no redemption date.")

    return _antithetic_heston_path_price(
        reference,
        times,
        740000000 + int(row.row_id),
        _AUTOCALL_REFERENCE_PAIRS,
        discounted_payoff,
    )


def _cliquet_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    row: PriceResultRow,
) -> tuple[float, float]:
    """Price one periodic-return Cliquet from independent QuantLib paths."""

    context = "Cliquet"
    reference = quantlib_reference(model)
    maturity, _, observation_count, steps_per_observation, times = (
        _regular_observation_grid(product, context)
    )
    participation = positive_number(product, "participation_rate", context)
    local_floor = finite_number(product, "local_floor", context)
    local_cap = finite_number(product, "local_cap", context)
    global_floor = finite_number(product, "global_floor", context)
    global_cap = finite_number(product, "global_cap", context)
    if not local_floor < local_cap:
        raise ValueError("Cliquet: local_floor must be below local_cap.")
    if not -1.0 < global_floor < global_cap:
        raise ValueError("Cliquet: global bounds are invalid.")
    risk_free_rate = finite_number(model, "risk_free_rate", "Heston model")
    discount = math.exp(-risk_free_rate * maturity)

    def discounted_payoff(spot_path: ql.Path) -> float:
        previous_spot = spot_path[0]
        accumulated_return = 0.0
        for observation in range(observation_count):
            spot = spot_path[(observation + 1) * steps_per_observation]
            participated_return = participation * (spot / previous_spot - 1.0)
            accumulated_return += min(
                local_cap, max(local_floor, participated_return)
            )
            previous_spot = spot
        final_return = min(
            global_cap, max(global_floor, accumulated_return)
        )
        return discount * (1.0 + final_return)

    return _antithetic_heston_path_price(
        reference,
        times,
        750000000 + int(row.row_id),
        _CLIQUET_REFERENCE_PAIRS,
        discounted_payoff,
    )


def _range_accrual_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    row: PriceResultRow,
) -> tuple[float, float]:
    """Price one equity Range Accrual from independent QuantLib paths."""

    context = "Range Accrual"
    reference = quantlib_reference(model)
    maturity, interval, observation_count, steps_per_observation, times = (
        _regular_observation_grid(product, context)
    )
    lower_barrier = positive_number(product, "lower_barrier", context)
    upper_barrier = positive_number(product, "upper_barrier", context)
    coupon_rate = positive_number(product, "coupon_rate", context)
    if not lower_barrier < 1.0 < upper_barrier:
        raise ValueError("Range Accrual: normalized barriers are invalid.")
    risk_free_rate = finite_number(model, "risk_free_rate", "Heston model")
    discount = math.exp(-risk_free_rate * maturity)

    def discounted_payoff(spot_path: ql.Path) -> float:
        initial_spot = spot_path[0]
        lower_spot = initial_spot * lower_barrier
        upper_spot = initial_spot * upper_barrier
        in_range_count = 0
        for observation in range(observation_count):
            spot = spot_path[(observation + 1) * steps_per_observation]
            in_range_count += lower_spot <= spot <= upper_spot
        accrued_coupon = coupon_rate * interval * in_range_count
        return discount * (1.0 + accrued_coupon)

    return _antithetic_heston_path_price(
        reference,
        times,
        760000000 + int(row.row_id),
        _RANGE_ACCRUAL_REFERENCE_PAIRS,
        discounted_payoff,
    )


def _barrier_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    option_type: int,
    barrier_type: int,
) -> float:
    """Price one continuously monitored barrier with QuantLib's Heston PDE."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Barrier option")
    barrier = positive_number(product, "barrier", "Barrier option")
    option = ql.BarrierOption(
        barrier_type,
        barrier,
        0.0,
        _vanilla_payoff(option_type, product),
        ql.EuropeanExercise(nearest_date_from_time(maturity)),
    )
    option.setPricingEngine(ql.FdHestonBarrierEngine(reference.model, 100, 100, 50))
    return option.NPV()


def _cash_barrier_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    barrier_type: int,
) -> float:
    """Price one maturity-paid touch binary with QuantLib's Heston PDE."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Touch option")
    barrier = positive_number(product, "barrier", "Touch option")
    cash_payoff = positive_number(product, "cash_payoff", "Touch option")
    option = ql.BarrierOption(
        barrier_type,
        barrier,
        0.0,
        ql.CashOrNothingPayoff(ql.Option.Call, 1.0e-12, cash_payoff),
        ql.EuropeanExercise(nearest_date_from_time(maturity)),
    )
    option.setPricingEngine(ql.FdHestonBarrierEngine(reference.model, 100, 100, 50))
    return option.NPV()


def _double_barrier_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price one continuously monitored double knock-out with Heston PDE."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Double-barrier option")
    lower = positive_number(product, "lower_barrier", "Double-barrier option")
    upper = positive_number(product, "upper_barrier", "Double-barrier option")
    if lower >= upper:
        raise ValueError("Double-barrier option: lower barrier must be below upper.")
    option = ql.DoubleBarrierOption(
        ql.DoubleBarrier.KnockOut,
        lower,
        upper,
        0.0,
        _vanilla_payoff(option_type, product),
        ql.EuropeanExercise(nearest_date_from_time(maturity)),
    )
    option.setPricingEngine(
        ql.FdHestonDoubleBarrierEngine(reference.model, 100, 100, 50)
    )
    return option.NPV()


def _unit_cash_digitals(
    reference: HestonReference, strike: float, maturity: float
) -> tuple[float, float]:
    """Return unit call and put digitals from QuantLib's Heston CDF."""

    distribution = ql.HestonRNDCalculator(reference.process)
    probability_below = distribution.cdf(math.log(strike), maturity)
    discount = reference.process.riskFreeRate().discount(maturity)
    return (
        discount * (1.0 - probability_below),
        discount * probability_below,
    )


def _digital_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price one cash-or-nothing option from QuantLib's Heston CDF."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Digital option")
    strike = positive_number(product, "strike", "Digital option")
    call, put = _unit_cash_digitals(reference, strike, maturity)
    unit_price = call if option_type == ql.Option.Call else put
    return positive_number(product, "cash_payoff", "Digital option") * unit_price


def _asset_or_nothing_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price one asset-or-nothing option from vanilla and digital prices."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Asset-or-nothing option")
    strike = positive_number(product, "strike", "Asset-or-nothing option")
    cash_call, cash_put = _unit_cash_digitals(reference, strike, maturity)
    if option_type == ql.Option.Call:
        return _european_price(
            model, product, ql.Option.Call
        ) + strike * cash_call
    return strike * cash_put - _european_price(
        model, product, ql.Option.Put
    )


def _gap_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price one gap option from asset and cash digital components."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "Gap option")
    trigger_strike = positive_number(
        product, "trigger_strike", "Gap option"
    )
    payoff_strike = positive_number(product, "payoff_strike", "Gap option")
    vanilla_product = {"strike": trigger_strike, "maturity": maturity}
    cash_call, cash_put = _unit_cash_digitals(
        reference, trigger_strike, maturity
    )
    if option_type == ql.Option.Call:
        asset_call = _european_price(
            model, vanilla_product, ql.Option.Call
        ) + trigger_strike * cash_call
        return asset_call - payoff_strike * cash_call
    asset_put = trigger_strike * cash_put - _european_price(
        model, vanilla_product, ql.Option.Put
    )
    return payoff_strike * cash_put - asset_put


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


def _heston_finite_difference_fallback_price(
    reference: HestonReference,
    model: Mapping[str, Any],
    payoff: ql.Payoff,
    exercise: ql.Exercise,
    mesher_strike: float,
) -> float:
    """Solve rare high-v0 rows on an explicitly widened Heston mesh."""

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
        mesher_strike,
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


def _american_price(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price one maturity-anchored early-exercise option with Heston PDE."""

    reference = quantlib_reference(model)
    maturity = positive_number(product, "maturity", "American option")
    exercise_interval = positive_number(
        product, "exercise_interval", "American option"
    )
    exercise_dates = _maturity_anchored_exercise_dates(
        maturity, exercise_interval
    )
    payoff = _vanilla_payoff(option_type, product)
    exercise = ql.BermudanExercise(exercise_dates)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.FdHestonVanillaEngine(reference.model, 100, 100, 50))
    try:
        pde_price = option.NPV()
    except RuntimeError as error:
        if "interpolation range" not in str(error):
            raise
        # Enlarge only the rare variance meshes that do not contain v0.
        pde_price = _heston_finite_difference_fallback_price(
            reference,
            model,
            payoff,
            exercise,
            positive_number(product, "strike", "American option"),
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
            return _european_price(model, product, ql.Option.Call)
        if product_kind == "european_put":
            return _european_price(model, product, ql.Option.Put)
        if product_kind == "asian_call":
            return _asian_price(model, product, row, ql.Option.Call)
        if product_kind == "asian_put":
            return _asian_price(model, product, row, ql.Option.Put)
        if product_kind == "geometric_asian_call":
            return _geometric_asian_price(model, product, row, ql.Option.Call)
        if product_kind == "geometric_asian_put":
            return _geometric_asian_price(model, product, row, ql.Option.Put)
        if product_kind == "forward_start_call":
            return _forward_start_price(model, product, row, ql.Option.Call)
        if product_kind == "forward_start_put":
            return _forward_start_price(model, product, row, ql.Option.Put)
        if product_kind in {
            "athena_autocall",
            "phoenix_autocall",
            "phoenix_memory_autocall",
        }:
            return _autocall_price(model, product, row, product_kind)
        if product_kind == "cliquet":
            return _cliquet_price(model, product, row)
        if product_kind == "range_accrual":
            return _range_accrual_price(model, product, row)
        if product_kind == "up_and_out_call":
            return _barrier_price(
                model, product, ql.Option.Call, ql.Barrier.UpOut
            )
        if product_kind == "up_and_in_call":
            return _barrier_price(
                model, product, ql.Option.Call, ql.Barrier.UpIn
            )
        if product_kind == "down_and_out_put":
            return _barrier_price(
                model, product, ql.Option.Put, ql.Barrier.DownOut
            )
        if product_kind == "down_and_in_put":
            return _barrier_price(
                model, product, ql.Option.Put, ql.Barrier.DownIn
            )
        if product_kind == "up_one_touch":
            return _cash_barrier_price(model, product, ql.Barrier.UpIn)
        if product_kind == "up_no_touch":
            return _cash_barrier_price(model, product, ql.Barrier.UpOut)
        if product_kind == "double_knock_out_call":
            return _double_barrier_price(model, product, ql.Option.Call)
        if product_kind == "double_knock_out_put":
            return _double_barrier_price(model, product, ql.Option.Put)
        if product_kind == "digital_call":
            return _digital_price(model, product, ql.Option.Call)
        if product_kind == "digital_put":
            return _digital_price(model, product, ql.Option.Put)
        if product_kind == "asset_or_nothing_call":
            return _asset_or_nothing_price(model, product, ql.Option.Call)
        if product_kind == "asset_or_nothing_put":
            return _asset_or_nothing_price(model, product, ql.Option.Put)
        if product_kind == "gap_call":
            return _gap_price(model, product, ql.Option.Call)
        if product_kind == "gap_put":
            return _gap_price(model, product, ql.Option.Put)
        if product_kind == "straddle":
            return _european_price(
                model, product, ql.Option.Call
            ) + _european_price(model, product, ql.Option.Put)
        if product_kind == "american_put":
            return _american_price(model, product, ql.Option.Put)
        if product_kind == "american_call":
            return _american_price(model, product, ql.Option.Call)
        raise ValueError(f"Unknown Heston product kind '{product_kind}'.")

    return validation_from_reference(
        price_dataset_path,
        reference_price,
        tolerances,
        require_curve=False,
    )
