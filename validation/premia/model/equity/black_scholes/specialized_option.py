"""Premia references for non-terminal Black-Scholes catalogue payoffs.

The arithmetic Asian contracts use Premia's native fixed-strike Asian engine.
The other products are exact reductions to Premia vanilla, digital, or barrier
contracts. Daily monitoring versus continuous monitoring is handled explicitly
for arithmetic Asians and touch products.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from validation.premia.bridge import PremiaInput, PremiaResult, price_rows
from validation.premia.model.equity.black_scholes.path_option import (
    DirectionalValidationReport,
    summarize_contract_difference_comparisons,
    summarize_directional_comparisons,
)
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


_TARGET_DT = 1.0 / 504.0
SUPPORTED_PRODUCTS = frozenset(
    {
        "asian_call",
        "asian_put",
        "forward_start_call",
        "forward_start_put",
        "geometric_asian_call",
        "geometric_asian_put",
        "range_accrual",
        "up_no_touch",
        "up_one_touch",
    }
)


def _model_values(model: Mapping[str, Any], row_id: str) -> tuple[float, ...]:
    """Return one Black-Scholes row in the Premia runner convention."""

    context = f"Black-Scholes model row '{row_id}'"
    return tuple(
        parameter_number(
            model,
            field,
            context,
            positive=field in {"spot", "volatility"},
        )
        for field in ("spot", "risk_free_rate", "dividend_yield", "volatility")
    )


def _scaled_error(result: PremiaResult, scale: float) -> float:
    return abs(scale) * result.standard_error


def _comparison(
    row: Any,
    reference_price: float,
    reference_error: float = 0.0,
) -> PremiaPriceComparison:
    return PremiaPriceComparison(
        row_id=row.row_id,
        generated_price=row.generated_price,
        premia_price=reference_price,
        generated_standard_error=row.generated_standard_error,
        premia_standard_error=reference_error,
    )


def _exponential_integral(rate: float, maturity: float) -> float:
    """Return the stable integral of exp(rate * t) from zero to maturity."""

    scaled_rate = rate * maturity
    if abs(scaled_rate) < 1.0e-8:
        return maturity * (
            1.0 + 0.5 * scaled_rate + scaled_rate * scaled_rate / 6.0
        )
    return math.expm1(scaled_rate) / rate


def _asian_contract_difference_bound(
    model: Mapping[str, Any], maturity: float
) -> float:
    """Bound a daily-grid Asian price against its continuous counterpart.

    The discounted call and put payoffs are one-Lipschitz in the average.  The
    returned value is therefore the exact discounted L2 distance between the
    equally weighted grid average (including both endpoints) and the continuous
    time average under Black-Scholes.
    """

    spot, rate, dividend, volatility = _model_values(model, "Asian bound")
    drift = rate - dividend
    variance = volatility * volatility
    step_count = max(1, math.floor(maturity / _TARGET_DT + 0.5))
    time_step = maturity / step_count
    times = tuple(index * time_step for index in range(step_count + 1))
    spot_squared = spot * spot

    drift_exponentials = tuple(math.exp(drift * time) for time in times)
    suffix_sums = [0.0] * (step_count + 2)
    for index in range(step_count, -1, -1):
        suffix_sums[index] = suffix_sums[index + 1] + drift_exponentials[index]
    discrete_second_moment_terms: list[float] = []
    for index, time in enumerate(times):
        diagonal = math.exp((2.0 * drift + variance) * time)
        off_diagonal = (
            2.0
            * math.exp((drift + variance) * time)
            * suffix_sums[index + 1]
        )
        discrete_second_moment_terms.append(diagonal + off_diagonal)
    discrete_second_moment = (
        spot_squared
        * math.fsum(discrete_second_moment_terms)
        / ((step_count + 1) * (step_count + 1))
    )

    drift_plus_variance = drift + variance
    discrete_continuous_terms = []
    total_drift_integral = _exponential_integral(drift, maturity)
    for time in times:
        before = math.exp(drift * time) * _exponential_integral(
            drift_plus_variance, time
        )
        after = math.exp(drift_plus_variance * time) * (
            total_drift_integral - _exponential_integral(drift, time)
        )
        discrete_continuous_terms.append(before + after)
    discrete_continuous_moment = (
        spot_squared
        * math.fsum(discrete_continuous_terms)
        / ((step_count + 1) * maturity)
    )

    if abs(drift_plus_variance * maturity) < 1.0e-7:
        # This removable singularity is rare in the catalogue.  Simpson's rule
        # avoids subtracting two nearly equal exponential integrals.
        interval_count = 256
        width = maturity / interval_count
        terms = []
        for index in range(interval_count + 1):
            time = index * width
            weight = 1 if index in {0, interval_count} else 4 if index % 2 else 2
            terms.append(
                weight
                * math.exp(drift * time)
                * _exponential_integral(drift_plus_variance, time)
            )
        continuous_second_moment = (
            2.0
            * spot_squared
            * width
            * math.fsum(terms)
            / (3.0 * maturity * maturity)
        )
    else:
        continuous_second_moment = (
            2.0
            * spot_squared
            * (
                _exponential_integral(2.0 * drift + variance, maturity)
                - total_drift_integral
            )
            / (drift_plus_variance * maturity * maturity)
        )
    squared_distance = max(
        0.0,
        discrete_second_moment
        - 2.0 * discrete_continuous_moment
        + continuous_second_moment,
    )
    return math.exp(-rate * maturity) * math.sqrt(squared_distance)


def _continuous_average_expectation(
    spot: float, rate: float, dividend: float, maturity: float
) -> float:
    return spot * _exponential_integral(rate - dividend, maturity) / maturity


def _asian_comparisons(
    rows: Sequence[Any],
    models: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    product_kind: str,
) -> tuple[list[PremiaPriceComparison], dict[str, float]]:
    inputs: list[PremiaInput] = []
    contract_bounds: dict[str, float] = {}
    price_bounds: dict[str, tuple[float, float]] = {}
    for row in rows:
        model = models[row.model_id]
        product = products[row.product_id]
        context = f"{product_kind} row '{row.product_id}'"
        strike = parameter_number(product, "strike", context, positive=True)
        maturity = parameter_number(product, "maturity", context, positive=True)
        spot, rate, dividend, _ = _model_values(model, row.model_id)
        expected_average = _continuous_average_expectation(
            spot, rate, dividend, maturity
        )
        discount = math.exp(-rate * maturity)
        if product_kind == "asian_call":
            price_bounds[row.row_id] = (
                discount * max(expected_average - strike, 0.0),
                discount * expected_average,
            )
        else:
            price_bounds[row.row_id] = (
                discount * max(strike - expected_average, 0.0),
                discount * strike,
            )
        contract_bounds[row.row_id] = _asian_contract_difference_bound(
            model, maturity
        )
        inputs.append(
            PremiaInput(
                row.row_id,
                (
                    *_model_values(model, row.model_id),
                    strike,
                    maturity,
                ),
            )
        )
    results = price_rows(inputs, f"black_scholes_{product_kind}")
    comparisons: list[PremiaPriceComparison] = []
    for row in rows:
        result = results[row.row_id]
        lower, upper = price_bounds[row.row_id]
        # A finite Monte Carlo value is still unusable as a reference when its
        # point estimate violates a model-free price bound.  A large reported
        # error does not turn a negative option value into a valid benchmark.
        bound_tolerance = 1.0e-8
        if (
            result.price < lower - bound_tolerance
            or result.price > upper + bound_tolerance
        ):
            raise RuntimeError(
                f"Premia row '{row.row_id}' failed with status 14: "
                "Asian price violates analytic no-arbitrage bounds"
            )
        comparisons.append(
            _comparison(row, result.price, result.standard_error)
        )
    return comparisons, contract_bounds


def _forward_start_comparisons(
    rows: Sequence[Any],
    models: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    product_kind: str,
) -> list[PremiaPriceComparison]:
    inputs: list[PremiaInput] = []
    scales: dict[str, float] = {}
    side = "put" if product_kind.endswith("put") else "call"
    for row in rows:
        model = models[row.model_id]
        product = products[row.product_id]
        context = f"{product_kind} row '{row.product_id}'"
        reset = parameter_number(product, "reset_time", context, positive=True)
        maturity = parameter_number(product, "maturity", context, positive=True)
        remaining = maturity - reset
        if remaining <= 0.0:
            raise ValueError(f"{context}: reset_time must precede maturity.")
        moneyness = parameter_number(product, "moneyness", context, positive=True)
        _, rate, dividend, volatility = _model_values(model, row.model_id)
        inputs.append(
            PremiaInput(
                row.row_id,
                (1.0, rate, dividend, volatility, moneyness, remaining),
            )
        )
        scales[row.row_id] = parameter_number(
            model, "spot", context, positive=True
        ) * math.exp(-dividend * reset)
    results = price_rows(inputs, f"black_scholes_european_{side}")
    return [
        _comparison(
            row,
            scales[row.row_id] * results[row.row_id].price,
            _scaled_error(results[row.row_id], scales[row.row_id]),
        )
        for row in rows
    ]


def _geometric_asian_comparisons(
    rows: Sequence[Any],
    models: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    product_kind: str,
) -> list[PremiaPriceComparison]:
    inputs: list[PremiaInput] = []
    side = "put" if product_kind.endswith("put") else "call"
    for row in rows:
        model = models[row.model_id]
        product = products[row.product_id]
        context = f"{product_kind} row '{row.product_id}'"
        strike = parameter_number(product, "strike", context, positive=True)
        maturity = parameter_number(product, "maturity", context, positive=True)
        spot, rate, dividend, volatility = _model_values(model, row.model_id)
        step_count = max(1, math.floor(maturity / _TARGET_DT + 0.5))
        variance = volatility * volatility
        log_mean = math.log(spot) + 0.5 * (
            rate - dividend - 0.5 * variance
        ) * maturity
        log_variance = (
            variance
            * maturity
            * (2.0 * step_count + 1.0)
            / (6.0 * (step_count + 1.0))
        )
        synthetic_spot = math.exp(log_mean + 0.5 * log_variance)
        synthetic_volatility = math.sqrt(log_variance / maturity)
        inputs.append(
            PremiaInput(
                row.row_id,
                (
                    synthetic_spot,
                    rate,
                    rate,
                    synthetic_volatility,
                    strike,
                    maturity,
                ),
            )
        )
    results = price_rows(inputs, f"black_scholes_european_{side}")
    return [
        _comparison(row, results[row.row_id].price, results[row.row_id].standard_error)
        for row in rows
    ]


def _range_accrual_comparisons(
    rows: Sequence[Any],
    models: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
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
        lower = parameter_number(product, "lower_barrier", context, positive=True)
        upper = parameter_number(product, "upper_barrier", context, positive=True)
        if lower >= upper:
            raise ValueError(f"{context}: lower_barrier must be below upper_barrier.")
        coupon = parameter_number(product, "coupon_rate", context, positive=True)
        model_values = _model_values(model, row.model_id)
        count = math.floor(maturity / interval + 0.5)
        row_observations: list[tuple[str, str, float]] = []
        for observation in range(1, count + 1):
            time = observation * interval
            lower_id = f"{row.row_id}_l_{observation}"
            upper_id = f"{row.row_id}_u_{observation}"
            inputs.append(PremiaInput(lower_id, (*model_values, lower, time)))
            inputs.append(PremiaInput(upper_id, (*model_values, upper, time)))
            row_observations.append((lower_id, upper_id, time))
        observations[row.row_id] = row_observations
        contracts[row.row_id] = (
            parameter_number(model, "risk_free_rate", context),
            maturity,
            coupon * interval,
        )
    results = price_rows(inputs, "black_scholes_digital_call")
    comparisons: list[PremiaPriceComparison] = []
    for row in rows:
        rate, maturity, coupon_per_observation = contracts[row.row_id]
        maturity_discount = math.exp(-rate * maturity)
        probability_sum = 0.0
        error_squared = 0.0
        for lower_id, upper_id, time in observations[row.row_id]:
            probability_scale = math.exp(rate * time)
            lower = results[lower_id]
            upper = results[upper_id]
            probability_sum += probability_scale * (lower.price - upper.price)
            error_squared += probability_scale * probability_scale * (
                lower.standard_error * lower.standard_error
                + upper.standard_error * upper.standard_error
            )
        reference = maturity_discount * (
            1.0 + coupon_per_observation * probability_sum
        )
        reference_error = (
            maturity_discount * coupon_per_observation * math.sqrt(error_squared)
        )
        comparisons.append(_comparison(row, reference, reference_error))
    return comparisons


def _touch_comparisons(
    rows: Sequence[Any],
    models: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    product_kind: str,
    monte_carlo_paths_per_price: int | None,
    standard_error_multiplier: float,
) -> list[PremiaPriceComparison]:
    rebate_inputs: list[PremiaInput] = []
    vanilla_inputs: list[PremiaInput] = []
    contracts: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        model = models[row.model_id]
        product = products[row.product_id]
        context = f"{product_kind} row '{row.product_id}'"
        spot, rate, dividend, volatility = _model_values(model, row.model_id)
        barrier = parameter_number(product, "barrier", context, positive=True)
        maturity = parameter_number(product, "maturity", context, positive=True)
        cash = parameter_number(product, "cash_payoff", context, positive=True)
        if spot >= barrier:
            raise ValueError(f"{context}: barrier must exceed the initial spot.")
        auxiliary_strike = 8.0 * max(spot, barrier)
        rebate_inputs.append(
            PremiaInput(
                row.row_id,
                (
                    spot,
                    rate,
                    dividend,
                    volatility,
                    auxiliary_strike,
                    maturity,
                    barrier,
                    cash,
                ),
            )
        )
        vanilla_inputs.append(
            PremiaInput(
                row.row_id,
                (
                    spot,
                    rate,
                    dividend,
                    volatility,
                    auxiliary_strike,
                    maturity,
                ),
            )
        )
        contracts[row.row_id] = (rate, maturity, cash)
    rebate = price_rows(rebate_inputs, "black_scholes_up_in_call_rebate")
    vanilla = price_rows(vanilla_inputs, "black_scholes_european_call")
    comparisons: list[PremiaPriceComparison] = []
    for row in rows:
        rate, maturity, cash = contracts[row.row_id]
        no_touch = rebate[row.row_id].price - vanilla[row.row_id].price
        no_touch_error = math.hypot(
            rebate[row.row_id].standard_error,
            vanilla[row.row_id].standard_error,
        )
        reference = (
            no_touch
            if product_kind == "up_no_touch"
            else cash * math.exp(-rate * maturity) - no_touch
        )
        if (
            product_kind == "up_no_touch"
            and row.generated_price == 0.0
            and row.generated_standard_error == 0.0
            and monte_carlo_paths_per_price is not None
        ):
            confidence_tail = 0.5 * math.erfc(
                standard_error_multiplier / math.sqrt(2.0)
            )
            probability_bound = -math.expm1(
                math.log(confidence_tail) / monte_carlo_paths_per_price
            )
            zero_event_price_bound = (
                cash * math.exp(-rate * maturity) * probability_bound
            )
            if reference <= zero_event_price_bound:
                raise RuntimeError(
                    f"Premia comparison for row '{row.row_id}' is inconclusive: "
                    f"zero positive payoffs in {monte_carlo_paths_per_price} "
                    "CUDA paths remain compatible with the continuous reference"
                )
        comparisons.append(_comparison(row, reference, no_touch_error))
    return comparisons


def validation_from_premia_black_scholes_specialized_option(
    price_dataset_path: str | Path,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> PriceValidationReport | DirectionalValidationReport:
    """Validate one supported Black-Scholes payoff through Premia."""

    if product_kind not in SUPPORTED_PRODUCTS:
        raise ValueError(
            f"Unsupported specialized Premia Black-Scholes product '{product_kind}'."
        )
    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(validation_input.product_dataset_path, "products")
    rows = validation_input.rows
    if product_kind in {"asian_call", "asian_put"}:
        comparisons, contract_bounds = _asian_comparisons(
            rows, models, products, product_kind
        )
        return summarize_contract_difference_comparisons(
            validation_input.database_id,
            comparisons,
            contract_bounds,
            tolerances,
        )
    elif product_kind in {"forward_start_call", "forward_start_put"}:
        comparisons = _forward_start_comparisons(
            rows, models, products, product_kind
        )
    elif product_kind in {"geometric_asian_call", "geometric_asian_put"}:
        comparisons = _geometric_asian_comparisons(
            rows, models, products, product_kind
        )
    elif product_kind == "range_accrual":
        comparisons = _range_accrual_comparisons(rows, models, products)
    else:
        comparisons = _touch_comparisons(
            rows,
            models,
            products,
            product_kind,
            validation_input.monte_carlo_paths_per_price,
            tolerances.standard_error_multiplier,
        )
        relation = (
            "generated_at_least_reference"
            if product_kind == "up_no_touch"
            else "generated_at_most_reference"
        )
        return summarize_directional_comparisons(
            validation_input.database_id, comparisons, relation, tolerances
        )
    return summarize_premia_comparisons(
        validation_input.database_id, comparisons, tolerances
    )


__all__ = (
    "SUPPORTED_PRODUCTS",
    "validation_from_premia_black_scholes_specialized_option",
)
