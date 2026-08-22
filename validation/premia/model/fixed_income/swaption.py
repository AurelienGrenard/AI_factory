"""Premia adapter for physical-settlement European swaptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from validation.hierarchy import BackendException
from validation.premia.bridge import PremiaInput, price_rows_partial
from validation.premia.price_validation import (
    ValidationRegime,
    load_parameter_rows,
    load_price_validation_input,
    parameter_number,
    select_validation_regime,
    select_validation_row_ids,
)


_MODEL_PARAMETERS = {
    "ornstein_uhlenbeck": ("initial_state", "mean_reversion", "volatility"),
    "vasicek": (
        "initial_state",
        "mean_reversion",
        "volatility",
        "long_term_mean",
    ),
    "cir": ("initial_state", "mean_reversion", "volatility", "long_term_mean"),
    "hull_white": ("mean_reversion", "volatility"),
}

_CURVE_PARAMETERS = {
    "nelson_siegel": ("beta0", "beta1", "beta2", "tau"),
    "svensson": ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2"),
}

_SIDES = {"payer", "receiver"}


@dataclass(frozen=True)
class PremiaSwaptionReference:
    """One source identity and its Premia swaption reference price."""

    row_id: str
    model_id: str
    product_id: str
    price: float
    standard_error: float
    curve_id: str | None = None


def _mode(model_name: str, curve_name: str | None, side: str) -> str:
    if model_name not in _MODEL_PARAMETERS:
        raise ValueError(f"Unsupported Premia rate model '{model_name}'.")
    if side not in _SIDES:
        raise ValueError(f"Unsupported swaption side '{side}'.")
    if model_name == "hull_white":
        if curve_name not in _CURVE_PARAMETERS:
            raise ValueError("Hull-White swaptions require a supported fitted curve.")
        return f"hull_white_{curve_name}_{side}_swaption"
    if curve_name is not None:
        raise ValueError(f"{model_name} swaptions do not accept a curve dataset.")
    return f"{model_name}_{side}_swaption"


def _numbers(
    parameters: Mapping[str, object],
    names: tuple[str, ...],
    context: str,
) -> tuple[float, ...]:
    return tuple(parameter_number(parameters, name, context) for name in names)


def _prepared_inputs(
    price_dataset_path: str | Path,
    model_name: str,
    curve_name: str | None,
    side: str,
    regime: ValidationRegime,
    row_ids: Sequence[str] | None,
):
    """Prepare the subset whose regular schedule Premia can represent."""

    _mode(model_name, curve_name, side)
    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    fitted = model_name == "hull_white"
    if fitted != (validation_input.curve_dataset_path is not None):
        expected = "with" if fitted else "without"
        raise ValueError(f"{model_name} swaption validation must be {expected} a curve.")

    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(validation_input.product_dataset_path, "products")
    curves = (
        load_parameter_rows(validation_input.curve_dataset_path, "curves")
        if validation_input.curve_dataset_path is not None
        else None
    )

    inputs: list[PremiaInput] = []
    exceptions: list[BackendException] = []
    for row in validation_input.rows:
        model = models[row.model_id]
        prefix = _numbers(
            model,
            _MODEL_PARAMETERS[model_name],
            f"{model_name} model row '{row.model_id}'",
        )
        if model_name == "ornstein_uhlenbeck":
            prefix += (0.0,)
        if curves is not None:
            if row.curve_id is None:
                raise ValueError(f"Fitted-rate row '{row.row_id}' has no curve id.")
            assert curve_name is not None
            prefix += _numbers(
                curves[row.curve_id],
                _CURVE_PARAMETERS[curve_name],
                f"{curve_name} curve row '{row.curve_id}'",
            )

        product = products[row.product_id]
        context = f"European swaption row '{row.product_id}'"
        exercise = parameter_number(product, "exercise_time", context, positive=True)
        reset_period = parameter_number(
            product, "payment_interval", context, positive=True
        )
        accrual = parameter_number(
            product, "accrual_fraction", context, positive=True
        )
        payment_count = parameter_number(
            product, "payment_count", context, positive=True
        )
        rounded_count = round(payment_count)
        if abs(payment_count - rounded_count) > 1.0e-9:
            raise ValueError(f"{context}: payment_count must be an integer.")
        if abs(accrual - reset_period) > 1.0e-7:
            exceptions.append(
                BackendException(
                    row.row_id,
                    "Premia uses one reset period for both payment spacing and accrual",
                    12,
                )
            )
            continue
        strike = parameter_number(product, "strike", context)
        if strike == 0.0:
            exceptions.append(
                BackendException(
                    row.row_id,
                    "Premia's Jamshidian boundary degenerates at zero strike",
                    12,
                )
            )
            continue
        if exercise >= 50.0:
            exceptions.append(
                BackendException(
                    row.row_id,
                    "Premia's closed form is unreliable at the 50-year expiry",
                    12,
                )
            )
            continue
        if (
            model_name == "hull_white"
            and rounded_count * reset_period >= 50.0 - 1.0e-7
        ):
            exceptions.append(
                BackendException(
                    row.row_id,
                    "Premia's fitted-curve adapter is impractical at 50-year tenor",
                    12,
                )
            )
            continue
        suffix = (
            parameter_number(product, "notional", context, positive=True),
            strike,
            exercise,
            exercise + rounded_count * reset_period,
            reset_period,
        )
        inputs.append(PremiaInput(row.row_id, prefix + suffix))
    return validation_input, tuple(inputs), tuple(exceptions)


def reference_prices_from_premia_swaption_partial(
    price_dataset_path: str | Path,
    model_name: str,
    side: str,
    curve_name: str | None = None,
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
    method: str | None = None,
) -> tuple[tuple[PremiaSwaptionReference, ...], tuple[BackendException, ...]]:
    """Return successful Premia prices and every row-local technical exception."""

    validation_input, inputs, adapter_exceptions = _prepared_inputs(
        price_dataset_path, model_name, curve_name, side, regime, row_ids
    )
    mode = _mode(model_name, curve_name, side)
    batch_size = 100 if model_name == "hull_white" else max(len(inputs), 1)
    results = {}
    runner_failures = {}
    for start in range(0, len(inputs), batch_size):
        batch_results, batch_failures = price_rows_partial(
            inputs[start : start + batch_size], mode, method
        )
        results.update(batch_results)
        runner_failures.update(batch_failures)
    source_rows = {row.row_id: row for row in validation_input.rows}
    references = tuple(
        PremiaSwaptionReference(
            row_id=row_id,
            model_id=source_rows[row_id].model_id,
            product_id=source_rows[row_id].product_id,
            price=result.price,
            standard_error=result.standard_error,
            curve_id=source_rows[row_id].curve_id,
        )
        for row_id, result in results.items()
    )
    runner_exceptions = tuple(
        BackendException(row_id, failure.reason, failure.status)
        for row_id, failure in runner_failures.items()
    )
    exceptions_by_id = {
        exception.row_id: exception
        for exception in (*adapter_exceptions, *runner_exceptions)
    }
    ordered_exceptions = tuple(
        exceptions_by_id[row.row_id]
        for row in validation_input.rows
        if row.row_id in exceptions_by_id
    )
    return references, ordered_exceptions


__all__ = (
    "PremiaSwaptionReference",
    "reference_prices_from_premia_swaption_partial",
)
