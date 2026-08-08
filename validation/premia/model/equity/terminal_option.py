"""Shared Premia validation for terminal one-asset equity payoffs.

Premia supplies model-specific vanilla and, for a subset of models, cash-
digital engines.  The other terminal payoffs below are exact static identities
of those primitive claims; no approximation or extra simulation is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from validation.premia.bridge import PremiaInput, PremiaResult, price_rows
from validation.premia.price_validation import (
    PremiaPriceComparison,
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
    load_parameter_rows,
    load_price_validation_input,
    parameter_number,
    select_validation_row_ids,
    select_validation_regime,
    summarize_premia_comparisons,
)


@dataclass(frozen=True)
class ModelSpec:
    """Premia protocol prefix and terminal engines for one model."""

    fields: tuple[str, ...]


_MODEL_SPECS = {
    "black_scholes": ModelSpec(
        ("spot", "risk_free_rate", "dividend_yield", "volatility")
    ),
    "heston": ModelSpec(
        (
            "spot", "risk_free_rate", "dividend_yield",
            "initial_variance", "kappa", "theta", "gamma", "rho",
        )
    ),
    "bates": ModelSpec(
        (
            "spot", "risk_free_rate", "dividend_yield",
            "initial_variance", "kappa", "theta", "gamma", "rho",
            "jump_intensity", "jump_log_mean", "jump_log_volatility",
        )
    ),
    "merton": ModelSpec(
        (
            "spot", "risk_free_rate", "dividend_yield", "volatility",
            "jump_intensity", "jump_log_mean", "jump_log_volatility",
        )
    ),
    "kou": ModelSpec(
        (
            "spot", "risk_free_rate", "dividend_yield", "volatility",
            "jump_intensity", "up_probability", "positive_jump_rate",
            "negative_jump_rate",
        )
    ),
    "variance_gamma": ModelSpec(
        ("spot", "risk_free_rate", "dividend_yield", "sigma", "nu", "theta")
    ),
    "normal_inverse_gaussian": ModelSpec(
        ("spot", "risk_free_rate", "dividend_yield", "alpha", "beta", "delta")
    ),
    "cev": ModelSpec(
        ("spot", "risk_free_rate", "dividend_yield", "sigma", "beta")
    ),
    "schobel_zhu": ModelSpec(
        (
            "spot", "risk_free_rate", "dividend_yield",
            "initial_volatility", "mean_reversion", "long_run_volatility",
            "volatility_of_volatility", "correlation",
        )
    ),
}

_DIGITAL_MODELS = frozenset(("black_scholes", "merton", "kou"))
_PRODUCT_KINDS = frozenset(
    (
        "european_call", "european_put", "straddle",
        "digital_call", "digital_put", "asset_or_nothing_call",
        "asset_or_nothing_put", "gap_call", "gap_put",
    )
)


def _model_prefix(
    parameters: Mapping[str, Any], model_name: str, row_id: str
) -> tuple[float, ...]:
    """Map one catalogue row to the runner's documented model convention."""

    spec = _MODEL_SPECS[model_name]
    context = f"{model_name} model row '{row_id}'"
    return tuple(parameter_number(parameters, field, context) for field in spec.fields)


def _scaled_error(result: PremiaResult, scale: float = 1.0) -> float:
    return abs(scale) * result.standard_error


def validation_from_premia_terminal_option(
    price_dataset_path: str | Path,
    model_name: str,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    vanilla_method: str | None = None,
    digital_method: str | None = None,
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> PriceValidationReport:
    """Compare a complete terminal-payoff dataset with compatible Premia engines."""

    if model_name not in _MODEL_SPECS:
        raise ValueError(f"Unsupported Premia equity model '{model_name}'.")
    if product_kind not in _PRODUCT_KINDS:
        raise ValueError(f"Unsupported Premia terminal product '{product_kind}'.")
    needs_digital = product_kind not in {"european_call", "european_put", "straddle"}
    if needs_digital and model_name not in _DIGITAL_MODELS:
        raise ValueError(
            f"Premia exposes no compatible cash-digital engine for {model_name}."
        )

    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    if validation_input.curve_dataset_path is not None:
        raise ValueError("Premia equity validation expects no curve dataset.")
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(validation_input.product_dataset_path, "products")

    # The database id starts with the exact model database id; pass the explicit
    # model name into the helper rather than relying on this representation.
    vanilla_inputs: list[PremiaInput] = []
    digital_inputs: list[PremiaInput] = []
    contracts: dict[str, tuple[float, float]] = {}
    needs_vanilla = product_kind in {
        "european_call", "european_put", "straddle",
        "asset_or_nothing_call", "asset_or_nothing_put", "gap_call", "gap_put",
    }
    for row in validation_input.rows:
        model = models[row.model_id]
        product = products[row.product_id]
        context = f"{product_kind} row '{row.product_id}'"
        trigger = parameter_number(
            product,
            "trigger_strike" if product_kind.startswith("gap_") else "strike",
            context,
            positive=True,
        )
        maturity = parameter_number(product, "maturity", context, positive=True)
        values = (*_model_prefix(model, model_name, row.model_id), trigger, maturity)
        if needs_vanilla:
            vanilla_inputs.append(PremiaInput(row.row_id, values))
        if needs_digital:
            digital_inputs.append(PremiaInput(row.row_id, values))
        payoff_strike = (
            parameter_number(product, "payoff_strike", context, positive=True)
            if product_kind.startswith("gap_") else trigger
        )
        contracts[row.row_id] = (trigger, payoff_strike)

    side = "put" if product_kind.endswith("put") else "call"
    vanilla_side = side if product_kind != "straddle" else "call"
    vanilla = (
        price_rows(
            vanilla_inputs,
            f"{model_name}_european_{vanilla_side}",
            vanilla_method,
        )
        if needs_vanilla else {}
    )
    other_vanilla = (
        price_rows(
            vanilla_inputs, f"{model_name}_european_put", vanilla_method
        )
        if product_kind == "straddle" else {}
    )
    digital_call = (
        price_rows(
            digital_inputs, f"{model_name}_digital_call", digital_method
        )
        if needs_digital else {}
    )

    comparisons: list[PremiaPriceComparison] = []
    for row in validation_input.rows:
        trigger, payoff_strike = contracts[row.row_id]
        model = models[row.model_id]
        maturity = parameter_number(
            products[row.product_id],
            "maturity",
            f"row '{row.product_id}'",
            positive=True,
        )
        discount = math.exp(
            -parameter_number(model, "risk_free_rate", f"model '{row.model_id}'")
            * maturity
        )
        vanilla_result = vanilla.get(row.row_id)
        digital_result = digital_call.get(row.row_id)
        if product_kind == "european_call" or product_kind == "european_put":
            reference = vanilla_result.price
            error = vanilla_result.standard_error
        elif product_kind == "straddle":
            put = other_vanilla[row.row_id]
            reference = vanilla_result.price + put.price
            error = math.hypot(vanilla_result.standard_error, put.standard_error)
        else:
            digital_price = digital_result.price
            digital_error = digital_result.standard_error
            if side == "put":
                digital_price = discount - digital_price
            if product_kind.startswith("digital_"):
                cash = parameter_number(
                    products[row.product_id], "cash_payoff", f"row '{row.product_id}'"
                )
                reference = cash * digital_price
                error = _scaled_error(digital_result, cash)
            elif product_kind == "asset_or_nothing_call":
                reference = vanilla_result.price + trigger * digital_price
                error = math.hypot(
                    vanilla_result.standard_error, trigger * digital_error
                )
            elif product_kind == "asset_or_nothing_put":
                reference = trigger * digital_price - vanilla_result.price
                error = math.hypot(
                    vanilla_result.standard_error, trigger * digital_error
                )
            else:
                reference = vanilla_result.price + (
                    (trigger - payoff_strike) * digital_price
                    if side == "call" else
                    (payoff_strike - trigger) * digital_price
                )
                error = math.hypot(
                    vanilla_result.standard_error,
                    abs(payoff_strike - trigger) * digital_error,
                )
        comparisons.append(
            PremiaPriceComparison(
                row_id=row.row_id,
                generated_price=row.generated_price,
                premia_price=reference,
                generated_standard_error=row.generated_standard_error,
                premia_standard_error=error,
            )
        )
    return summarize_premia_comparisons(
        validation_input.database_id, comparisons, tolerances
    )


__all__ = ("validation_from_premia_terminal_option",)
