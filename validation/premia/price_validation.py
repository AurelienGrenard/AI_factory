"""Premia-specific names around the shared dataset comparison machinery."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from validation.quantlib.price_validation import (
    PriceComparison,
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
    load_parameter_rows,
    load_price_validation_input,
    summarize_price_comparisons,
    select_validation_regime,
    select_validation_row_ids,
)


@dataclass(frozen=True)
class PremiaPriceComparison:
    """One generated price and its independent Premia reference."""

    row_id: str
    generated_price: float
    premia_price: float
    generated_standard_error: float = 0.0
    premia_standard_error: float = 0.0


def parameter_number(
    parameters: Mapping[str, Any],
    field: str,
    context: str,
    positive: bool = False,
) -> float:
    """Return one finite parameter with a contextual diagnostic."""

    value = parameters.get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context}: {field} must be finite.")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{context}: {field} must be positive.")
    return result


def summarize_premia_comparisons(
    database_id: str,
    comparisons: Sequence[PremiaPriceComparison],
    tolerances: ValidationTolerances = ValidationTolerances(),
) -> PriceValidationReport:
    """Apply the common row and signed-bias checks to Premia references."""

    common_rows = tuple(
        PriceComparison(
            row_id=row.row_id,
            generated_price=row.generated_price,
            quantlib_price=row.premia_price,
            generated_standard_error=row.generated_standard_error,
            quantlib_standard_error=row.premia_standard_error,
        )
        for row in comparisons
    )
    return summarize_price_comparisons(database_id, common_rows, tolerances)


def format_premia_report(report: PriceValidationReport) -> str:
    """Render one compact Premia validation report."""

    status = "PASS" if report.passed else "FAIL"
    return "\n".join(
        (
            f"Premia validation: {status}",
            f"dataset                  : {report.database_id}",
            f"rows                     : {report.row_count}",
            f"passed / failed          : {report.passed_row_count} / {report.failed_row_count}",
            f"higher / lower / equal   : {report.higher_price_count} / "
            f"{report.lower_price_count} / {report.equal_price_count}",
            f"mean signed error        : {report.mean_signed_error:.12e}",
            f"mean absolute error      : {report.mean_absolute_error:.12e}",
            f"root mean squared error  : {report.root_mean_squared_error:.12e}",
            f"RMS reported std error   : {report.root_mean_squared_standard_error:.12e}",
            f"maximum absolute error   : {report.maximum_absolute_error:.12e} "
            f"(row {report.maximum_absolute_error_row_id})",
            f"maximum relative error   : {report.maximum_relative_error:.12e} "
            f"(row {report.maximum_relative_error_row_id})",
            f"bias limit               : {report.bias_limit:.12e}",
            f"systematic bias          : {'yes' if report.systematic_bias else 'no'}",
        )
    )


def run_premia_validation_cli(
    validation: Callable[[str | Path], PriceValidationReport],
    description: str,
) -> int:
    """Run one Premia validator through the standard one-path CLI."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("price_dataset", type=Path)
    arguments = parser.parse_args()
    report = validation(arguments.price_dataset)
    print(format_premia_report(report))
    return 0 if report.passed else 1


__all__ = (
    "PremiaPriceComparison",
    "PriceValidationReport",
    "ValidationRegime",
    "ValidationTolerances",
    "load_parameter_rows",
    "load_price_validation_input",
    "parameter_number",
    "run_premia_validation_cli",
    "select_validation_regime",
    "select_validation_row_ids",
    "summarize_premia_comparisons",
)
