"""Premia continuous references for daily-monitored Black-Scholes payoffs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from validation.premia.bridge import PremiaInput, price_rows
from validation.premia.price_validation import (
    PremiaPriceComparison,
    ValidationRegime,
    ValidationTolerances,
    load_parameter_rows,
    load_price_validation_input,
    parameter_number,
    select_validation_regime,
    select_validation_row_ids,
)
from validation.quantlib.price_validation import PriceRowDiagnostic


DirectionalRelation = Literal[
    "generated_at_least_reference", "generated_at_most_reference"
]
_BOUND_ABSOLUTE_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class DirectionalValidationReport:
    """Price-gap diagnostics whose pass condition is a proven ordering."""

    database_id: str
    row_count: int
    failed_row_ids: tuple[str, ...]
    higher_price_count: int
    lower_price_count: int
    equal_price_count: int
    mean_signed_error: float
    mean_absolute_error: float
    maximum_absolute_error: float
    maximum_absolute_error_row_id: str
    systematic_bias: bool
    row_diagnostics: tuple[PriceRowDiagnostic, ...]

    @property
    def passed(self) -> bool:
        return not self.failed_row_ids


def format_bound_report(report: DirectionalValidationReport) -> str:
    """Format the compact diagnostics used by the standalone Premia CLI."""

    return "\n".join(
        (
            f"dataset          : {report.database_id}",
            f"rows             : {report.row_count}",
            f"failed           : {len(report.failed_row_ids)}",
            f"mean signed gap  : {report.mean_signed_error:.12e}",
            f"mean absolute gap: {report.mean_absolute_error:.12e}",
            "maximum gap      : "
            f"{report.maximum_absolute_error:.12e} "
            f"(row {report.maximum_absolute_error_row_id})",
        )
    )


def summarize_directional_comparisons(
    database_id: str,
    comparisons: Sequence[PremiaPriceComparison],
    relation: DirectionalRelation,
    tolerances: ValidationTolerances,
) -> DirectionalValidationReport:
    """Retain full price-gap metrics while enforcing only the proven bound."""

    if not comparisons:
        raise ValueError("Directional Premia validation requires at least one row.")
    failed: list[str] = []
    higher = 0
    lower = 0
    equal = 0
    signed_sum = 0.0
    absolute_sum = 0.0
    maximum = -1.0
    maximum_row = comparisons[0].row_id
    row_diagnostics: list[PriceRowDiagnostic] = []
    for row in comparisons:
        signed = row.generated_price - row.premia_price
        absolute = abs(signed)
        allowance = (
            max(tolerances.absolute, _BOUND_ABSOLUTE_TOLERANCE)
            + tolerances.relative * abs(row.premia_price)
            + tolerances.standard_error_multiplier
            * math.hypot(
                row.generated_standard_error, row.premia_standard_error
            )
        )
        violation = (
            signed < -allowance
            if relation == "generated_at_least_reference"
            else signed > allowance
        )
        if violation:
            failed.append(row.row_id)
        row_diagnostics.append(
            PriceRowDiagnostic(
                row_id=row.row_id,
                generated_price=row.generated_price,
                reference_price=row.premia_price,
                generated_standard_error=row.generated_standard_error,
                reference_standard_error=row.premia_standard_error,
                allowance=allowance,
                passed=not violation,
            )
        )
        higher += signed > 0.0
        lower += signed < 0.0
        equal += signed == 0.0
        signed_sum += signed
        absolute_sum += absolute
        if absolute > maximum:
            maximum = absolute
            maximum_row = row.row_id
    row_count = len(comparisons)
    dominant = max(higher, lower) / row_count > 0.60
    return DirectionalValidationReport(
        database_id=database_id,
        row_count=row_count,
        failed_row_ids=tuple(failed),
        higher_price_count=higher,
        lower_price_count=lower,
        equal_price_count=equal,
        mean_signed_error=signed_sum / row_count,
        mean_absolute_error=absolute_sum / row_count,
        maximum_absolute_error=maximum,
        maximum_absolute_error_row_id=maximum_row,
        systematic_bias=dominant,
        row_diagnostics=tuple(row_diagnostics),
    )


def summarize_contract_difference_comparisons(
    database_id: str,
    comparisons: Sequence[PremiaPriceComparison],
    contract_allowances: Mapping[str, float],
    tolerances: ValidationTolerances,
) -> DirectionalValidationReport:
    """Accept numerical noise plus a proven row-wise contract-difference bound."""

    if not comparisons:
        raise ValueError("Premia validation requires at least one comparison.")
    comparison_ids = {row.row_id for row in comparisons}
    if comparison_ids != set(contract_allowances):
        raise ValueError("Every comparison requires one contract-difference bound.")
    failed: list[str] = []
    higher = 0
    lower = 0
    equal = 0
    signed_sum = 0.0
    absolute_sum = 0.0
    maximum = -1.0
    maximum_row = comparisons[0].row_id
    row_diagnostics: list[PriceRowDiagnostic] = []
    for row in comparisons:
        signed = row.generated_price - row.premia_price
        absolute = abs(signed)
        allowance = (
            tolerances.absolute
            + tolerances.relative * abs(row.premia_price)
            + tolerances.standard_error_multiplier
            * math.hypot(
                row.generated_standard_error, row.premia_standard_error
            )
            + contract_allowances[row.row_id]
        )
        passed = absolute <= allowance
        if not passed:
            failed.append(row.row_id)
        row_diagnostics.append(
            PriceRowDiagnostic(
                row_id=row.row_id,
                generated_price=row.generated_price,
                reference_price=row.premia_price,
                generated_standard_error=row.generated_standard_error,
                reference_standard_error=row.premia_standard_error,
                allowance=allowance,
                passed=passed,
            )
        )
        higher += signed > 0.0
        lower += signed < 0.0
        equal += signed == 0.0
        signed_sum += signed
        absolute_sum += absolute
        if absolute > maximum:
            maximum = absolute
            maximum_row = row.row_id
    row_count = len(comparisons)
    return DirectionalValidationReport(
        database_id=database_id,
        row_count=row_count,
        failed_row_ids=tuple(failed),
        higher_price_count=higher,
        lower_price_count=lower,
        equal_price_count=equal,
        mean_signed_error=signed_sum / row_count,
        mean_absolute_error=absolute_sum / row_count,
        maximum_absolute_error=maximum,
        maximum_absolute_error_row_id=maximum_row,
        systematic_bias=max(higher, lower) / row_count > 0.60,
        row_diagnostics=tuple(row_diagnostics),
    )


_PRODUCTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "up_and_out_call": (
        "black_scholes_up_and_out_call",
        ("strike", "maturity", "barrier"),
    ),
    "up_and_in_call": (
        "black_scholes_up_and_in_call",
        ("strike", "maturity", "barrier"),
    ),
    "down_and_out_put": (
        "black_scholes_down_and_out_put",
        ("strike", "maturity", "barrier"),
    ),
    "down_and_in_put": (
        "black_scholes_down_and_in_put",
        ("strike", "maturity", "barrier"),
    ),
    "double_knock_out_call": (
        "black_scholes_double_knock_out_call",
        ("strike", "maturity", "lower_barrier", "upper_barrier"),
    ),
    "double_knock_out_put": (
        "black_scholes_double_knock_out_put",
        ("strike", "maturity", "lower_barrier", "upper_barrier"),
    ),
    "lookback_option": (
        "black_scholes_lookback_option",
        ("strike", "maturity"),
    ),
}


def _model_prefix(model: Mapping[str, Any], row_id: str) -> tuple[float, ...]:
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


def validation_from_premia_black_scholes_path_option(
    price_dataset_path: str | Path,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> DirectionalValidationReport:
    """Compare a daily-grid CUDA price with its continuous Premia contract."""

    if product_kind not in _PRODUCTS:
        raise ValueError(f"Unsupported Black-Scholes Premia path product '{product_kind}'.")
    mode, fields = _PRODUCTS[product_kind]
    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(validation_input.product_dataset_path, "products")
    inputs: list[PremiaInput] = []
    for row in validation_input.rows:
        product = products[row.product_id]
        context = f"{product_kind} row '{row.product_id}'"
        suffix = tuple(
            parameter_number(product, field, context, positive=True)
            for field in fields
        )
        inputs.append(
            PremiaInput(
                row.row_id,
                (*_model_prefix(models[row.model_id], row.model_id), *suffix),
            )
        )
    results = price_rows(inputs, mode)
    comparisons = tuple(
        PremiaPriceComparison(
            row_id=row.row_id,
            generated_price=row.generated_price,
            premia_price=results[row.row_id].price,
            generated_standard_error=row.generated_standard_error,
            premia_standard_error=results[row.row_id].standard_error,
        )
        for row in validation_input.rows
    )
    relation: DirectionalRelation = (
        "generated_at_most_reference"
        if product_kind in {"up_and_in_call", "down_and_in_put", "lookback_option"}
        else "generated_at_least_reference"
    )
    return summarize_directional_comparisons(
        validation_input.database_id, comparisons, relation, tolerances
    )


__all__ = (
    "DirectionalValidationReport",
    "format_bound_report",
    "summarize_contract_difference_comparisons",
    "summarize_directional_comparisons",
    "validation_from_premia_black_scholes_path_option",
)
