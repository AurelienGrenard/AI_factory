"""QuantLib references for deterministic Black-Scholes price datasets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import QuantLib as ql

from validation.quantlib.model.equity.black_scholes.reference import (
    BlackScholesReference,
    quantlib_reference,
)
from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.price_validation import (
    PriceResultRow,
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
    validation_from_reference,
)


_TARGET_DT = 1.0 / 360.0
_FLOAT32_EPSILON = 1.1920928955078125e-7
_ASIAN_REFERENCE_SAMPLES = 4096
_PATH_REFERENCE_PAIRS = 1024
_PATH_REFINEMENT_PAIRS = 8192
_NORMAL_CDF = ql.CumulativeNormalDistribution()


def _option_type(product_kind: str) -> int:
    """Return the QuantLib side encoded by a catalog product kind."""

    if product_kind.endswith("_call"):
        return ql.Option.Call
    if product_kind.endswith("_put"):
        return ql.Option.Put
    raise ValueError(f"Product kind '{product_kind}' has no option side.")


def _black_value(
    reference: BlackScholesReference,
    strike: float,
    maturity: float,
    option_type: int,
) -> float:
    """Evaluate one vanilla payoff through QuantLib's Black calculator."""

    discount = math.exp(-reference.risk_free_rate * maturity)
    forward = reference.spot * math.exp(
        (reference.risk_free_rate - reference.dividend_yield) * maturity
    )
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    return ql.BlackCalculator(
        payoff,
        forward,
        reference.volatility * math.sqrt(maturity),
        discount,
    ).value()


def _d1_d2(
    reference: BlackScholesReference, strike: float, maturity: float
) -> tuple[float, float]:
    """Return the standard Black-Scholes normal arguments."""

    standard_deviation = reference.volatility * math.sqrt(maturity)
    d1 = (
        math.log(reference.spot / strike)
        + (
            reference.risk_free_rate
            - reference.dividend_yield
            + 0.5 * reference.volatility * reference.volatility
        )
        * maturity
    ) / standard_deviation
    return d1, d1 - standard_deviation


def _european_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price a European vanilla option."""

    return _black_value(
        reference,
        positive_number(product, "strike", "European option"),
        positive_number(product, "maturity", "European option"),
        option_type,
    )


def _digital_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price a cash-or-nothing option from QuantLib's normal CDF."""

    context = "Digital option"
    strike = positive_number(product, "strike", context)
    maturity = positive_number(product, "maturity", context)
    cash_payoff = positive_number(product, "cash_payoff", context)
    _, d2 = _d1_d2(reference, strike, maturity)
    probability = _NORMAL_CDF(d2 if option_type == ql.Option.Call else -d2)
    return cash_payoff * math.exp(-reference.risk_free_rate * maturity) * probability


def _asset_or_nothing_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price an asset-or-nothing option from QuantLib's normal CDF."""

    context = "Asset-or-nothing option"
    strike = positive_number(product, "strike", context)
    maturity = positive_number(product, "maturity", context)
    d1, _ = _d1_d2(reference, strike, maturity)
    probability = _NORMAL_CDF(d1 if option_type == ql.Option.Call else -d1)
    return (
        reference.spot
        * math.exp(-reference.dividend_yield * maturity)
        * probability
    )


def _gap_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price a gap option with separate trigger and payoff strikes."""

    context = "Gap option"
    trigger = positive_number(product, "trigger_strike", context)
    payoff_strike = positive_number(product, "payoff_strike", context)
    maturity = positive_number(product, "maturity", context)
    d1, d2 = _d1_d2(reference, trigger, maturity)
    spot_discount = reference.spot * math.exp(
        -reference.dividend_yield * maturity
    )
    strike_discount = payoff_strike * math.exp(
        -reference.risk_free_rate * maturity
    )
    if option_type == ql.Option.Call:
        return spot_discount * _NORMAL_CDF(d1) - strike_discount * _NORMAL_CDF(d2)
    return strike_discount * _NORMAL_CDF(-d2) - spot_discount * _NORMAL_CDF(-d1)


def _forward_start_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price a forward-start option from its scale-invariant conditional law."""

    context = "Forward-start option"
    reset_time = positive_number(product, "reset_time", context)
    maturity = positive_number(product, "maturity", context)
    moneyness = positive_number(product, "moneyness", context)
    remaining_time = maturity - reset_time
    if remaining_time <= 0.0:
        raise ValueError(f"{context}: reset_time must precede maturity.")
    unit_forward = math.exp(
        (reference.risk_free_rate - reference.dividend_yield) * remaining_time
    )
    conditional_unit_value = ql.BlackCalculator(
        ql.PlainVanillaPayoff(option_type, moneyness),
        unit_forward,
        reference.volatility * math.sqrt(remaining_time),
        math.exp(-reference.risk_free_rate * remaining_time),
    ).value()
    return (
        reference.spot
        * math.exp(-reference.dividend_yield * reset_time)
        * conditional_unit_value
    )


def _geometric_asian_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    option_type: int,
) -> float:
    """Price the grid average including both time zero and maturity."""

    context = "Geometric Asian option"
    strike = positive_number(product, "strike", context)
    maturity = positive_number(product, "maturity", context)
    step_count = max(1, math.floor(maturity / _TARGET_DT + 0.5))
    variance = reference.volatility * reference.volatility
    log_mean = math.log(reference.spot) + 0.5 * (
        reference.risk_free_rate - reference.dividend_yield - 0.5 * variance
    ) * maturity
    log_variance = (
        variance
        * maturity
        * (2.0 * step_count + 1.0)
        / (6.0 * (step_count + 1.0))
    )
    expected_geometric_mean = math.exp(log_mean + 0.5 * log_variance)
    return ql.BlackCalculator(
        ql.PlainVanillaPayoff(option_type, strike),
        expected_geometric_mean,
        math.sqrt(log_variance),
        math.exp(-reference.risk_free_rate * maturity),
    ).value()


def _range_accrual_price(
    reference: BlackScholesReference, product: Mapping[str, Any]
) -> float:
    """Sum the exact marginal in-range probabilities on the observation grid."""

    context = "Range accrual"
    maturity = positive_number(product, "maturity", context)
    interval = positive_number(product, "observation_interval", context)
    lower = positive_number(product, "lower_barrier", context)
    upper = positive_number(product, "upper_barrier", context)
    coupon_rate = positive_number(product, "coupon_rate", context)
    if lower >= upper:
        raise ValueError(f"{context}: lower_barrier must be below upper_barrier.")
    count = math.floor(maturity / interval + 0.5)
    drift = (
        reference.risk_free_rate
        - reference.dividend_yield
        - 0.5 * reference.volatility * reference.volatility
    )
    probability_sum = 0.0
    for observation in range(1, count + 1):
        time = observation * interval
        standard_deviation = reference.volatility * math.sqrt(time)
        mean = math.log(reference.spot) + drift * time
        lower_normal = (math.log(lower) - mean) / standard_deviation
        upper_normal = (math.log(upper) - mean) / standard_deviation
        probability_sum += _NORMAL_CDF(upper_normal) - _NORMAL_CDF(lower_normal)
    return math.exp(-reference.risk_free_rate * maturity) * (
        1.0 + coupon_rate * interval * probability_sum
    )


def _asian_fixing_dates(maturity: float, step_count: int) -> list[ql.Date]:
    """Match the workbench arithmetic average after time-zero spot."""

    from validation.quantlib.term_structure import REFERENCE_DATE, nearest_date_from_time

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
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    row: PriceResultRow,
    option_type: int,
) -> tuple[float, float]:
    """Price the discretely sampled arithmetic average with QuantLib MC."""

    from validation.quantlib.term_structure import nearest_date_from_time

    maturity = positive_number(product, "maturity", "Asian option")
    step_count = max(1, math.floor(maturity / _TARGET_DT + 0.5))
    fixing_dates = _asian_fixing_dates(maturity, step_count)
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Arithmetic,
        reference.spot,
        1,
        fixing_dates,
        ql.PlainVanillaPayoff(
            option_type, positive_number(product, "strike", "Asian option")
        ),
        ql.EuropeanExercise(nearest_date_from_time(maturity)),
    )
    option.setPricingEngine(
        ql.MCDiscreteArithmeticAPEngine(
            reference.process,
            "pseudorandom",
            antitheticVariate=True,
            requiredSamples=_ASIAN_REFERENCE_SAMPLES,
            seed=810000000 + int(row.row_id),
            # QuantLib 1.43's arithmetic-Asian control variate is not
            # side-neutral for the running-average convention used here.
            # Keep the independent reference unbiased and account for its
            # larger uncertainty through errorEstimate().
            controlVariate=False,
        )
    )
    return option.NPV(), option.errorEstimate()


def _regular_observation_grid(
    product: Mapping[str, Any], context: str
) -> tuple[float, float, int, int, list[float]]:
    """Construct the exact product observation grid.

    Black-Scholes transitions are exact between two dates.  Products observed
    only on their contractual grid therefore need no artificial daily
    sub-stepping in the independent Monte Carlo reference.
    """

    maturity = positive_number(product, "maturity", context)
    interval = positive_number(product, "observation_interval", context)
    raw_count = maturity / interval
    observation_count = round(raw_count)
    tolerance = 32.0 * _FLOAT32_EPSILON * max(raw_count, 1.0)
    if observation_count < 1 or abs(raw_count - observation_count) > tolerance:
        raise ValueError(
            f"{context}: maturity must be an integer multiple of "
            "observation_interval."
        )
    steps_per_observation = 1
    times = [interval * step for step in range(1, observation_count + 1)]
    return maturity, interval, observation_count, steps_per_observation, times


def _regular_simulation_times(maturity: float) -> list[float]:
    """Match the workbench nearest-integer daily monitoring grid."""

    step_count = max(1, math.floor(maturity / _TARGET_DT + 0.5))
    return [maturity * step / step_count for step in range(1, step_count + 1)]


def _antithetic_path_price(
    reference: BlackScholesReference,
    times: list[float],
    seed: int,
    pair_count: int,
    discounted_payoff: Callable[[ql.Path], float],
) -> tuple[float, float]:
    """Estimate a path payoff using independent QuantLib antithetic paths."""

    uniform_sequence = ql.UniformRandomSequenceGenerator(
        len(times), ql.UniformRandomGenerator(seed)
    )
    path_generator = ql.GaussianPathGenerator(
        reference.process,
        ql.TimeGrid(times),
        ql.GaussianRandomSequenceGenerator(uniform_sequence),
        False,
    )
    mean = 0.0
    sum_squared_deviations = 0.0
    for pair_index in range(pair_count):
        direct = discounted_payoff(path_generator.next().value())
        antithetic = discounted_payoff(path_generator.antithetic().value())
        pair_value = 0.5 * (direct + antithetic)
        delta = pair_value - mean
        mean += delta / (pair_index + 1)
        sum_squared_deviations += delta * (pair_value - mean)
    variance = sum_squared_deviations / (pair_count - 1)
    return mean, math.sqrt(variance / pair_count)


def _refined_path_price(
    reference: BlackScholesReference,
    times: list[float],
    row: PriceResultRow,
    seed: int,
    discounted_payoff: Callable[[ql.Path], float],
    pair_count: int = _PATH_REFERENCE_PAIRS,
) -> tuple[float, float]:
    """Refine only rows whose inexpensive independent estimate is suspicious."""

    price, error = _antithetic_path_price(
        reference, times, seed, pair_count, discounted_payoff
    )
    combined_error = math.hypot(row.generated_standard_error, error)
    if abs(row.generated_price - price) <= 4.0 * combined_error:
        return price, error
    return _antithetic_path_price(
        reference,
        times,
        seed + 1_000_000_000,
        _PATH_REFINEMENT_PAIRS,
        discounted_payoff,
    )


def _autocall_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    row: PriceResultRow,
    product_kind: str,
) -> tuple[float, float]:
    """Price one catalogue autocall from QuantLib Black-Scholes paths."""

    maturity, interval, observation_count, steps_per_observation, times = (
        _regular_observation_grid(product, "Autocall")
    )
    del maturity
    autocall_barrier = positive_number(product, "autocall_barrier", "Autocall")
    protection_barrier = positive_number(
        product, "protection_barrier", "Autocall"
    )
    annual_coupon_rate = positive_number(
        product, "annual_coupon_rate", "Autocall"
    )
    coupon_barrier = (
        None
        if product_kind == "athena_autocall"
        else positive_number(product, "coupon_barrier", "Autocall")
    )
    discounts = [
        math.exp(-reference.risk_free_rate * interval * (index + 1))
        for index in range(observation_count)
    ]
    coupon_per_observation = annual_coupon_rate * interval

    def payoff(path: ql.Path) -> float:
        present_value = 0.0
        remembered_coupon = 0.0
        accumulated_gain = 0.0
        for observation in range(observation_count):
            spot = path[(observation + 1) * steps_per_observation]
            discount = discounts[observation]
            final = observation + 1 == observation_count
            if product_kind == "athena_autocall":
                accumulated_gain += coupon_per_observation
                if spot >= autocall_barrier:
                    return discount * (1.0 + accumulated_gain)
                if final:
                    capital = 1.0 if spot >= protection_barrier else spot
                    return discount * capital
                continue
            if product_kind == "phoenix_memory_autocall":
                remembered_coupon += coupon_per_observation
                coupon = remembered_coupon if spot >= coupon_barrier else 0.0
                if coupon > 0.0:
                    remembered_coupon = 0.0
            else:
                coupon = coupon_per_observation if spot >= coupon_barrier else 0.0
            if spot >= autocall_barrier and not final:
                return present_value + discount * (1.0 + coupon)
            if final:
                capital = 1.0 if spot >= protection_barrier else spot
                return present_value + discount * (capital + coupon)
            present_value += discount * coupon
        raise RuntimeError("Autocall path reached no redemption date.")

    return _refined_path_price(
        reference, times, row, 820000000 + int(row.row_id), payoff
    )


def _cliquet_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    row: PriceResultRow,
) -> tuple[float, float]:
    """Price the local/global capped Cliquet from QuantLib paths."""

    maturity, _, observation_count, steps_per_observation, times = (
        _regular_observation_grid(product, "Cliquet")
    )
    participation = positive_number(product, "participation_rate", "Cliquet")
    local_floor = finite_number(product, "local_floor", "Cliquet")
    local_cap = finite_number(product, "local_cap", "Cliquet")
    global_floor = finite_number(product, "global_floor", "Cliquet")
    global_cap = finite_number(product, "global_cap", "Cliquet")
    discount = math.exp(-reference.risk_free_rate * maturity)

    def payoff(path: ql.Path) -> float:
        previous = path[0]
        accumulated = 0.0
        for observation in range(observation_count):
            spot = path[(observation + 1) * steps_per_observation]
            local_return = participation * (spot / previous - 1.0)
            accumulated += min(local_cap, max(local_floor, local_return))
            previous = spot
        return discount * (
            1.0 + min(global_cap, max(global_floor, accumulated))
        )

    return _refined_path_price(
        reference, times, row, 830000000 + int(row.row_id), payoff
    )


def _barrier_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    row: PriceResultRow,
    option_type: int,
    barrier_type: int,
) -> tuple[float, float]:
    """Price a barrier on the exact discrete catalogue monitoring grid."""

    maturity = positive_number(product, "maturity", "Barrier option")
    barrier = positive_number(product, "barrier", "Barrier option")
    strike = positive_number(product, "strike", "Barrier option")
    discount = math.exp(-reference.risk_free_rate * maturity)
    side = 1.0 if option_type == ql.Option.Call else -1.0
    up = barrier_type in {ql.Barrier.UpIn, ql.Barrier.UpOut}
    knock_in = barrier_type in {ql.Barrier.UpIn, ql.Barrier.DownIn}

    def payoff(path: ql.Path) -> float:
        hit = any(
            path[index] >= barrier if up else path[index] <= barrier
            for index in range(len(path))
        )
        if hit != knock_in:
            return 0.0
        return discount * max(side * (path[-1] - strike), 0.0)

    return _refined_path_price(
        reference,
        _regular_simulation_times(maturity),
        row,
        840000000 + int(row.row_id),
        payoff,
    )


def _touch_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    row: PriceResultRow,
    knock_in: bool,
    pair_count: int,
) -> tuple[float, float]:
    """Price a maturity-paid touch on the discrete catalogue grid."""

    maturity = positive_number(product, "maturity", "Touch option")
    barrier = positive_number(product, "barrier", "Touch option")
    cash = positive_number(product, "cash_payoff", "Touch option")
    discounted_cash = cash * math.exp(-reference.risk_free_rate * maturity)

    def payoff(path: ql.Path) -> float:
        hit = any(path[index] >= barrier for index in range(len(path)))
        return discounted_cash if hit == knock_in else 0.0

    return _refined_path_price(
        reference,
        _regular_simulation_times(maturity),
        row,
        850000000 + int(row.row_id),
        payoff,
        pair_count,
    )


def _double_barrier_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    row: PriceResultRow,
    option_type: int,
) -> tuple[float, float]:
    """Price a double knock-out on the discrete catalogue grid."""

    maturity = positive_number(product, "maturity", "Double barrier")
    lower = positive_number(product, "lower_barrier", "Double barrier")
    upper = positive_number(product, "upper_barrier", "Double barrier")
    strike = positive_number(product, "strike", "Double barrier")
    discount = math.exp(-reference.risk_free_rate * maturity)
    side = 1.0 if option_type == ql.Option.Call else -1.0

    def payoff(path: ql.Path) -> float:
        if not all(lower < path[index] < upper for index in range(len(path))):
            return 0.0
        return discount * max(side * (path[-1] - strike), 0.0)

    return _refined_path_price(
        reference,
        _regular_simulation_times(maturity),
        row,
        860000000 + int(row.row_id),
        payoff,
    )


def _lookback_price(
    reference: BlackScholesReference,
    product: Mapping[str, Any],
    row: PriceResultRow,
) -> tuple[float, float]:
    """Price the discretely monitored fixed-strike lookback call."""

    maturity = positive_number(product, "maturity", "Lookback option")
    strike = positive_number(product, "strike", "Lookback option")
    discount = math.exp(-reference.risk_free_rate * maturity)

    def payoff(path: ql.Path) -> float:
        return discount * max(max(path[index] for index in range(len(path))) - strike, 0.0)

    return _refined_path_price(
        reference,
        _regular_simulation_times(maturity),
        row,
        870000000 + int(row.row_id),
        payoff,
    )


def validation_from_quantlib_black_scholes_option(
    price_dataset_path: str | Path,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
    path_reference_pairs: int = _PATH_REFERENCE_PAIRS,
) -> PriceValidationReport:
    """Validate one complete deterministic Black-Scholes price dataset."""

    if path_reference_pairs < 2:
        raise ValueError("A path reference requires at least two antithetic pairs.")

    def reference_price(
        model: Mapping[str, Any],
        curve: Mapping[str, Any] | None,
        product: Mapping[str, Any],
        row: PriceResultRow,
    ) -> float:
        if curve is not None:
            raise ValueError("Black-Scholes equity prices must not reference curves.")
        reference = quantlib_reference(model)
        if product_kind == "asian_call":
            return _asian_price(reference, product, row, ql.Option.Call)
        if product_kind == "asian_put":
            return _asian_price(reference, product, row, ql.Option.Put)
        if product_kind in {
            "athena_autocall", "phoenix_autocall", "phoenix_memory_autocall"
        }:
            return _autocall_price(reference, product, row, product_kind)
        if product_kind == "cliquet":
            return _cliquet_price(reference, product, row)
        if product_kind == "up_and_out_call":
            return _barrier_price(
                reference, product, row, ql.Option.Call, ql.Barrier.UpOut
            )
        if product_kind == "up_and_in_call":
            return _barrier_price(
                reference, product, row, ql.Option.Call, ql.Barrier.UpIn
            )
        if product_kind == "down_and_out_put":
            return _barrier_price(
                reference, product, row, ql.Option.Put, ql.Barrier.DownOut
            )
        if product_kind == "down_and_in_put":
            return _barrier_price(
                reference, product, row, ql.Option.Put, ql.Barrier.DownIn
            )
        if product_kind == "up_one_touch":
            return _touch_price(
                reference, product, row, True, path_reference_pairs
            )
        if product_kind == "up_no_touch":
            return _touch_price(
                reference, product, row, False, path_reference_pairs
            )
        if product_kind == "double_knock_out_call":
            return _double_barrier_price(reference, product, row, ql.Option.Call)
        if product_kind == "double_knock_out_put":
            return _double_barrier_price(reference, product, row, ql.Option.Put)
        if product_kind == "lookback_option":
            return _lookback_price(reference, product, row)
        if product_kind in {"european_call", "european_put"}:
            return _european_price(reference, product, _option_type(product_kind))
        if product_kind in {"digital_call", "digital_put"}:
            return _digital_price(reference, product, _option_type(product_kind))
        if product_kind in {
            "asset_or_nothing_call",
            "asset_or_nothing_put",
        }:
            return _asset_or_nothing_price(
                reference, product, _option_type(product_kind)
            )
        if product_kind in {"gap_call", "gap_put"}:
            return _gap_price(reference, product, _option_type(product_kind))
        if product_kind in {"forward_start_call", "forward_start_put"}:
            return _forward_start_price(
                reference, product, _option_type(product_kind)
            )
        if product_kind in {"geometric_asian_call", "geometric_asian_put"}:
            return _geometric_asian_price(
                reference, product, _option_type(product_kind)
            )
        if product_kind == "straddle":
            return _european_price(reference, product, ql.Option.Call) + _european_price(
                reference, product, ql.Option.Put
            )
        if product_kind == "range_accrual":
            return _range_accrual_price(reference, product)
        raise ValueError(f"Unknown deterministic product kind '{product_kind}'.")

    return validation_from_reference(
        price_dataset_path,
        reference_price,
        tolerances,
        require_curve=False,
        regime=regime,
        row_ids=row_ids,
    )
