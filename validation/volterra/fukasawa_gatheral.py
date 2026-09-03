"""Fukasawa--Gatheral rough-SABR smile formula and campaign analysis.

The equations are transcribed from Sections 5--6 of *A rough SABR formula*,
Frontiers of Mathematical Finance 1 (2022), DOI 10.3934/fmf.2021003.
This module is an offline reference and deliberately imports no production
code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from scipy.optimize import brentq
from scipy.special import ndtr


JsonObject = dict[str, Any]


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numerical.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _validate_formula_inputs(hurst_exponent: float, rho: float) -> None:
    if not 0.0 <= hurst_exponent <= 0.5:
        raise ValueError("hurst_exponent must lie in [0, 0.5].")
    if not -1.0 < rho < 1.0:
        raise ValueError("The closed formula requires rho in (-1, 1).")


def fukasawa_gatheral_g_zero(y: float, rho: float) -> float:
    """Return the paper's explicit ``G_0(y)`` solution."""

    y = _finite(y, "y")
    rho = _finite(rho, "rho")
    _validate_formula_inputs(0.0, rho)
    if abs(y) < 1.0e-4:
        return (
            y * y
            - (4.0 * rho / 3.0) * y**3
            + 0.5 * (4.0 * rho * rho - 1.0) * y**4
        )
    orthogonal = math.sqrt(1.0 - rho * rho)
    return math.log1p(2.0 * rho * y + y * y) + (
        2.0
        * rho
        / orthogonal
        * (
            math.atan(rho / orthogonal)
            - math.atan((y + rho) / orthogonal)
        )
    )


def fukasawa_gatheral_g_half(y: float, rho: float) -> float:
    """Return the paper's explicit ``G_{1/2}(y)`` solution."""

    y = _finite(y, "y")
    rho = _finite(rho, "rho")
    _validate_formula_inputs(0.5, rho)
    if abs(y) < 1.0e-4:
        return (
            y * y
            - 0.5 * rho * y**3
            + ((15.0 * rho * rho - 4.0) / 48.0) * y**4
        )
    radical = math.sqrt(1.0 + rho * y + 0.25 * y * y)
    logarithm = math.log(
        (radical - rho - 0.5 * y) / (1.0 - rho)
    )
    return 4.0 * logarithm * logarithm


def fukasawa_gatheral_g_approximation(
    y: float,
    hurst_exponent: float,
    rho: float,
) -> float:
    """Evaluate equation (5.2), the closed approximation ``G_A``."""

    y = _finite(y, "y")
    hurst_exponent = _finite(hurst_exponent, "hurst_exponent")
    rho = _finite(rho, "rho")
    _validate_formula_inputs(hurst_exponent, rho)
    scale = 2.0 * hurst_exponent + 1.0
    denominator = 2.0 * hurst_exponent + 3.0
    result = scale * scale * (
        3.0
        * (1.0 - 2.0 * hurst_exponent)
        / denominator
        * fukasawa_gatheral_g_zero(y / scale, rho)
        + 2.0
        * hurst_exponent
        / denominator
        * fukasawa_gatheral_g_half(2.0 * y / scale, rho)
    )
    if result < 0.0 and result > -1.0e-14:
        return 0.0
    if result < 0.0:
        raise ArithmeticError("The Fukasawa--Gatheral G_A value is negative.")
    return result


def _cev_distance(
    spot: float,
    strike: float,
    beta: float,
) -> float:
    log_moneyness = math.log(strike / spot)
    if beta == 1.0:
        return log_moneyness
    one_minus_beta = 1.0 - beta
    return (
        spot ** one_minus_beta
        * math.expm1(one_minus_beta * log_moneyness)
        / one_minus_beta
    )


def fukasawa_gatheral_implied_volatility(
    spot: float,
    strike: float,
    maturity: float,
    xi_0: float,
    eta: float,
    hurst_exponent: float,
    rho: float,
    beta: float,
) -> float:
    """Evaluate equations (3.4) and (5.2) for flat ``xi_0``.

    The paper defines ``d xi / xi`` with ``eta`` and then
    ``alpha = sqrt(xi)``.  AI_factory uses exactly that same eta convention;
    no factor-two conversion is applied here.
    """

    values = {
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "xi_0": xi_0,
        "eta": eta,
        "hurst_exponent": hurst_exponent,
        "rho": rho,
        "beta": beta,
    }
    values = {name: _finite(value, name) for name, value in values.items()}
    spot = values["spot"]
    strike = values["strike"]
    maturity = values["maturity"]
    xi_0 = values["xi_0"]
    eta = values["eta"]
    hurst_exponent = values["hurst_exponent"]
    rho = values["rho"]
    beta = values["beta"]
    if min(spot, strike, maturity, xi_0) <= 0.0:
        raise ValueError("spot, strike, maturity and xi_0 must be positive.")
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if not 0.5 <= beta <= 1.0:
        raise ValueError("beta must lie in [0.5, 1].")
    _validate_formula_inputs(hurst_exponent, rho)

    initial_volatility = math.sqrt(xi_0)
    log_moneyness = math.log(strike / spot)
    cev_distance = _cev_distance(spot, strike, beta)
    if abs(log_moneyness) < 1.0e-12:
        return initial_volatility * spot ** (beta - 1.0)

    local_volatility_conversion = log_moneyness / cev_distance
    if eta == 0.0:
        return initial_volatility * local_volatility_conversion
    kernel = (
        eta
        * math.sqrt(2.0 * hurst_exponent)
        * maturity ** (hurst_exponent - 0.5)
    )
    scaled_cev_distance = kernel * cev_distance / initial_volatility
    if abs(scaled_cev_distance) < 1.0e-12:
        smile_factor = 1.0
    else:
        denominator = math.sqrt(
            fukasawa_gatheral_g_approximation(
                scaled_cev_distance, hurst_exponent, rho
            )
        )
        smile_factor = abs(scaled_cev_distance) / denominator
    return initial_volatility * local_volatility_conversion * smile_factor


def _black_scholes_price(
    volatility: float,
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_side: str,
) -> float:
    discount = math.exp(-risk_free_rate * maturity)
    dividend_discount = math.exp(-dividend_yield * maturity)
    standard_deviation = volatility * math.sqrt(maturity)
    if standard_deviation <= 0.0:
        call = max(spot * dividend_discount - strike * discount, 0.0)
    else:
        d1 = (
            math.log(spot / strike)
            + (risk_free_rate - dividend_yield) * maturity
        ) / standard_deviation + 0.5 * standard_deviation
        d2 = d1 - standard_deviation
        call = (
            spot * dividend_discount * ndtr(d1)
            - strike * discount * ndtr(d2)
        )
    if option_side == "call":
        return float(call)
    if option_side == "put":
        return float(call - spot * dividend_discount + strike * discount)
    raise ValueError("option_side must be 'call' or 'put'.")


def black_scholes_implied_volatility(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_side: str,
) -> tuple[float, float]:
    """Invert one Black--Scholes price and return implied vol and vega."""

    inputs = (price, spot, strike, maturity, risk_free_rate, dividend_yield)
    if not all(math.isfinite(float(value)) for value in inputs):
        raise ValueError("Black--Scholes inversion inputs must be finite.")
    if min(spot, strike, maturity) <= 0.0 or price < 0.0:
        raise ValueError("Black--Scholes inversion inputs are outside domain.")
    discount = math.exp(-risk_free_rate * maturity)
    dividend_discount = math.exp(-dividend_yield * maturity)
    call_lower = max(spot * dividend_discount - strike * discount, 0.0)
    if option_side == "call":
        lower = call_lower
        upper = spot * dividend_discount
    elif option_side == "put":
        lower = max(strike * discount - spot * dividend_discount, 0.0)
        upper = strike * discount
    else:
        raise ValueError("option_side must be 'call' or 'put'.")
    bound_tolerance = 2.0e-7 * max(1.0, upper)
    if price < lower - bound_tolerance or price > upper + bound_tolerance:
        raise ValueError("Price violates Black--Scholes no-arbitrage bounds.")
    bounded_price = min(max(price, lower), upper)
    if bounded_price <= lower + 1.0e-15:
        return 0.0, 0.0

    objective = lambda volatility: _black_scholes_price(
        volatility,
        spot,
        strike,
        maturity,
        risk_free_rate,
        dividend_yield,
        option_side,
    ) - bounded_price
    volatility = float(brentq(objective, 1.0e-10, 8.0, xtol=1.0e-13))
    standard_deviation = volatility * math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield) * maturity
    ) / standard_deviation + 0.5 * standard_deviation
    vega = (
        spot
        * dividend_discount
        * math.exp(-0.5 * d1 * d1)
        / math.sqrt(2.0 * math.pi)
        * math.sqrt(maturity)
    )
    return volatility, vega


def _probe_row_implied_volatility(
    row: JsonObject,
    parameters: JsonObject,
    maturity: float,
) -> tuple[float, float]:
    estimate = row.get("estimate")
    if not isinstance(estimate, dict):
        raise ValueError("Every CUDA probe row requires an estimate object.")
    side = row.get("side")
    if side not in {"call", "put"}:
        raise ValueError("Every CUDA probe row requires a call/put side.")
    price = _finite(estimate.get("price"), "price")
    standard_error = _finite(
        estimate.get("standard_error"), "standard_error"
    )
    if price < 0.0 or standard_error < 0.0:
        raise ValueError("Probe prices and standard errors must be non-negative.")
    volatility, vega = black_scholes_implied_volatility(
        price,
        _finite(parameters.get("spot"), "spot"),
        _finite(row.get("strike"), "strike"),
        maturity,
        _finite(parameters.get("risk_free_rate"), "risk_free_rate"),
        _finite(parameters.get("dividend_yield"), "dividend_yield"),
        side,
    )
    if vega <= 0.0:
        raise ValueError("A probe row has zero Black--Scholes vega.")
    return volatility, standard_error / vega


def analyze_fukasawa_gatheral_campaign(
    document: JsonObject,
    formula_allowance: float = 0.02,
    standard_error_multiplier: float = 4.0,
) -> JsonObject:
    """Compare a production CUDA campaign with paper equation (6.1)."""

    formula_allowance = _finite(formula_allowance, "formula_allowance")
    standard_error_multiplier = _finite(
        standard_error_multiplier, "standard_error_multiplier"
    )
    if formula_allowance < 0.0 or standard_error_multiplier < 0.0:
        raise ValueError("Validation allowances must be non-negative.")
    parameters = document.get("parameters")
    runs = document.get("runs")
    if (
        not isinstance(parameters, dict)
        or not isinstance(runs, list)
        or not runs
    ):
        raise ValueError("The rough-SABR probe requires parameters and runs.")
    if document.get("paper_case") != "Fukasawa--Gatheral Figure 6.4":
        raise ValueError("The CUDA probe does not identify paper Figure 6.4.")
    if document.get("eta_convention") != "d_xi_over_xi":
        raise ValueError("The CUDA probe eta convention is not the paper's.")

    grouped: dict[int, list[JsonObject]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Every CUDA probe run must be an object.")
        maturity_days = int(run.get("maturity_days"))
        grouped.setdefault(maturity_days, []).append(run)
    maturity_reports: list[JsonObject] = []
    all_rows: list[JsonObject] = []
    for maturity_days, maturity_runs in sorted(grouped.items()):
        maturity_runs.sort(key=lambda value: int(value.get("time_steps")))
        normalized_by_steps: dict[int, dict[float, tuple[float, float]]] = {}
        for run in maturity_runs:
            maturity = _finite(run.get("maturity"), "maturity")
            time_steps = int(run.get("time_steps"))
            rows = run.get("rows")
            if time_steps < 1 or not isinstance(rows, list):
                raise ValueError("A CUDA probe run has invalid steps or rows.")
            implied: list[tuple[JsonObject, float, float]] = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Every CUDA probe row must be an object.")
                volatility, volatility_error = _probe_row_implied_volatility(
                    row, parameters, maturity
                )
                implied.append((row, volatility, volatility_error))
            atm = [value for value in implied if abs(
                _finite(value[0].get("scaled_log_moneyness"), "scaled y")
            ) < 1.0e-12]
            if len(atm) != 1:
                raise ValueError("Every probe run must contain one ATM row.")
            atm_volatility = atm[0][1]
            atm_error = atm[0][2]
            normalized: dict[float, tuple[float, float]] = {}
            for row, volatility, volatility_error in implied:
                scaled_y = _finite(
                    row.get("scaled_log_moneyness"), "scaled y"
                )
                ratio = volatility / atm_volatility
                ratio_error = (
                    0.0
                    if abs(scaled_y) < 1.0e-12
                    else math.hypot(
                        volatility_error / atm_volatility,
                        volatility * atm_error / (atm_volatility**2),
                    )
                )
                normalized[scaled_y] = (ratio, ratio_error)
            normalized_by_steps[time_steps] = normalized

        finest = maturity_runs[-1]
        maturity = _finite(finest.get("maturity"), "maturity")
        finest_steps = int(finest.get("time_steps"))
        finest_rows = finest.get("rows")
        if not isinstance(finest_rows, list):
            raise ValueError("The finest CUDA probe run has no row array.")
        previous = (
            normalized_by_steps[int(maturity_runs[-2].get("time_steps"))]
            if len(maturity_runs) > 1
            else None
        )
        ordered_steps = sorted(normalized_by_steps)
        refinement_history = [
            {
                "coarse_time_steps": coarse_steps,
                "fine_time_steps": fine_steps,
                "max_abs_normalized_implied_volatility_movement": max(
                    abs(
                        normalized_by_steps[fine_steps][scaled_y][0]
                        - normalized_by_steps[coarse_steps][scaled_y][0]
                    )
                    for scaled_y in normalized_by_steps[fine_steps]
                ),
            }
            for coarse_steps, fine_steps in zip(
                ordered_steps[:-1], ordered_steps[1:], strict=True
            )
        ]
        report_rows: list[JsonObject] = []
        for row in finest_rows:
            if not isinstance(row, dict):
                raise ValueError("Every finest CUDA probe row must be an object.")
            scaled_y = _finite(row.get("scaled_log_moneyness"), "scaled y")
            normalized, normalized_error = normalized_by_steps[finest_steps][
                scaled_y
            ]
            strike = _finite(row.get("strike"), "strike")
            formula_volatility = fukasawa_gatheral_implied_volatility(
                _finite(parameters.get("spot"), "spot"),
                strike,
                maturity,
                _finite(parameters.get("xi_0"), "xi_0"),
                _finite(parameters.get("eta"), "eta"),
                _finite(parameters.get("hurst_exponent"), "hurst_exponent"),
                _finite(parameters.get("rho"), "rho"),
                _finite(parameters.get("beta"), "beta"),
            )
            formula_atm = fukasawa_gatheral_implied_volatility(
                _finite(parameters.get("spot"), "spot"),
                _finite(parameters.get("spot"), "spot"),
                maturity,
                _finite(parameters.get("xi_0"), "xi_0"),
                _finite(parameters.get("eta"), "eta"),
                _finite(parameters.get("hurst_exponent"), "hurst_exponent"),
                _finite(parameters.get("rho"), "rho"),
                _finite(parameters.get("beta"), "beta"),
            )
            formula_normalized = formula_volatility / formula_atm
            refinement = (
                abs(normalized - previous[scaled_y][0])
                if previous is not None
                else 0.0
            )
            difference = normalized - formula_normalized
            allowance = (
                formula_allowance
                + refinement
                + standard_error_multiplier * normalized_error
            )
            report_row: JsonObject = {
                "scaled_log_moneyness": scaled_y,
                "log_moneyness": _finite(
                    row.get("log_moneyness"), "log_moneyness"
                ),
                "strike": strike,
                "side": row.get("side"),
                "cuda_normalized_implied_volatility": normalized,
                "cuda_normalized_standard_error": normalized_error,
                "formula_normalized_implied_volatility": formula_normalized,
                "difference": difference,
                "refinement_indicator": refinement,
                "allowance": allowance,
                "passed": abs(difference) <= allowance,
            }
            report_rows.append(report_row)
            all_rows.append(report_row)
        maturity_reports.append({
            "maturity_days": maturity_days,
            "maturity": maturity,
            "expansion_parameter_eta_t_to_h": _finite(
                parameters.get("eta"), "eta"
            ) * maturity ** _finite(
                parameters.get("hurst_exponent"), "hurst_exponent"
            ),
            "time_step_counts": [
                int(run.get("time_steps")) for run in maturity_runs
            ],
            "finest_time_steps": finest_steps,
            "finest_atm_implied_volatility": atm_volatility,
            "finest_atm_implied_volatility_standard_error": atm_error,
            "refinement_history": refinement_history,
            "all_passed": all(bool(row["passed"]) for row in report_rows),
            "max_abs_difference": max(
                abs(float(row["difference"])) for row in report_rows
            ),
            "max_refinement_indicator": max(
                float(row["refinement_indicator"]) for row in report_rows
            ),
            "rows": report_rows,
        })
    return {
        "model": "rough_sabr",
        "reference": {
            "authors": "Masaaki Fukasawa and Jim Gatheral",
            "title": "A rough SABR formula",
            "doi": "10.3934/fmf.2021003",
            "equations": ["3.4", "5.2", "6.1"],
            "paper_case": "Figure 6.4",
        },
        "parameters": parameters,
        "numerics": document.get("numerics"),
        "formula_allowance": formula_allowance,
        "standard_error_multiplier": standard_error_multiplier,
        "summary": {
            "maturity_count": len(maturity_reports),
            "row_count": len(all_rows),
            "all_passed": all(bool(row["passed"]) for row in all_rows),
            "max_abs_normalized_implied_volatility_difference": max(
                abs(float(row["difference"])) for row in all_rows
            ),
            "max_refinement_indicator": max(
                float(row["refinement_indicator"]) for row in all_rows
            ),
        },
        "maturities": maturity_reports,
    }


def analyze_fukasawa_gatheral_probe(
    path: Path,
    formula_allowance: float = 0.02,
    standard_error_multiplier: float = 4.0,
) -> JsonObject:
    """Load and analyze one persisted CUDA campaign."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read rough-SABR CUDA probe {path}.") from error
    if not isinstance(document, dict):
        raise ValueError("The rough-SABR CUDA probe must contain an object.")
    return analyze_fukasawa_gatheral_campaign(
        document, formula_allowance, standard_error_multiplier
    )


__all__ = (
    "analyze_fukasawa_gatheral_campaign",
    "analyze_fukasawa_gatheral_probe",
    "black_scholes_implied_volatility",
    "fukasawa_gatheral_g_approximation",
    "fukasawa_gatheral_g_half",
    "fukasawa_gatheral_g_zero",
    "fukasawa_gatheral_implied_volatility",
)
