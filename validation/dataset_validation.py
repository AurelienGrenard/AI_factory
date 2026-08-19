"""Model-independent orchestration for persistent dataset validation reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from validation.comparison import (
    engine_coverage,
    engine_plan,
    hierarchy_price_gap_metrics,
    hierarchy_row_diagnostics,
    resolved_fallback_diagnostics,
)
from validation.hierarchy import ValidationEngine, run_validation_hierarchy
from validation.quantlib.price_validation import (
    ValidationRegime,
    load_price_validation_input,
    load_parameter_rows,
    select_validation_regime,
)
from validation.reporting import (
    DatasetValidationReport,
    SpecialRowDiagnostic,
    ValidationDisplayReport,
    display_validation_report,
    has_directional_bias,
    synchronize_validation_yaml,
    validation_fingerprint,
    write_validation_report,
)


@dataclass(frozen=True)
class ValidationPolicy:
    """Comparison and bias rules shared by one model/product contract."""

    tolerance: str = (
        "5e-7 absolute + 5e-5 relative + 5 combined standard errors"
    )
    bias_explanation: str | None = None
    enforce_directional_bias: bool = False
    enforce_statistical_bias: bool = True
    near_zero_relative_materiality: float | None = None


EngineFactory = Callable[[], Sequence[ValidationEngine]]
DatasetValidator = Callable[[str | Path], DatasetValidationReport]


def premia_row_exception(error: Exception) -> bool:
    """Recognize a row-local Premia failure, not a global runner failure."""

    message = str(error)
    return (
        message.startswith("Premia row '") and " failed with status " in message
    ) or message.startswith("Premia comparison for row '")


def quantlib_row_exception(error: Exception) -> bool:
    """Recognize a row-local QuantLib pricing failure."""

    message = str(error)
    return "reference pricing failed" in message or "invalid outputs" in message


def unavailable_engine(reference: str, method: str, reason: str) -> ValidationEngine:
    """Declare one intentionally unavailable hierarchy slot."""

    return ValidationEngine(
        reference,
        method,
        None,
        pricing_method=None,
        unavailable_reason=reason,
    )


def build_validation_section(
    price_dataset_path: str | Path,
    regime: ValidationRegime,
    engines: Sequence[ValidationEngine],
    policy: ValidationPolicy = ValidationPolicy(),
) -> ValidationDisplayReport:
    """Run one core or stress hierarchy and build its persistent report section."""

    path = Path(price_dataset_path).resolve()
    validation_input = select_validation_regime(
        load_price_validation_input(path), regime
    )
    row_ids = tuple(row.row_id for row in validation_input.rows)
    generated_prices = {
        row.row_id: row.generated_price for row in validation_input.rows
    }
    hierarchy = run_validation_hierarchy(path, regime, row_ids, engines)
    metrics = hierarchy_price_gap_metrics(hierarchy)
    special_rows, fallbacks = resolved_fallback_diagnostics(
        hierarchy.runs, generated_prices
    )
    failed_row_ids = set(metrics.failed_row_ids if metrics is not None else ())
    if policy.near_zero_relative_materiality is not None and failed_row_ids:
        models = load_parameter_rows(
            validation_input.model_dataset_path, "models"
        )
        source_rows = {row.row_id: row for row in validation_input.rows}
        accepted_near_zero: list[SpecialRowDiagnostic] = []
        for diagnostic in hierarchy_row_diagnostics(hierarchy):
            if diagnostic.row_id not in failed_row_ids:
                continue
            source = source_rows[diagnostic.row_id]
            model = models[source.model_id]
            scale_value = model.get("spot", 1.0)
            natural_scale = (
                abs(float(scale_value))
                if isinstance(scale_value, (int, float))
                else 1.0
            )
            natural_scale = max(natural_scale, 1.0e-12)
            materiality = (
                policy.near_zero_relative_materiality * natural_scale
            )
            if max(
                abs(diagnostic.generated_price),
                abs(diagnostic.reference_price),
            ) > materiality:
                continue
            failed_row_ids.remove(diagnostic.row_id)
            accepted_near_zero.append(
                SpecialRowDiagnostic(
                    row_id=diagnostic.row_id,
                    category="near_zero_materiality",
                    diagnostic=(
                        f"initial comparison failed: CUDA price = "
                        f"{diagnostic.generated_price:.12g}, reference price = "
                        f"{diagnostic.reference_price:.12g}, allowance = "
                        f"{diagnostic.allowance:.12g}"
                    ),
                    resolution=(
                        "both prices are below the declared near-zero "
                        "materiality threshold"
                    ),
                    acceptance_rule=(
                        f"max(abs(CUDA), abs(reference)) <= "
                        f"{policy.near_zero_relative_materiality:g} * natural "
                        "price scale"
                    ),
                    evidence=(
                        f"natural price scale: {natural_scale:.12g}",
                        f"materiality threshold: {materiality:.12g}",
                        f"CUDA standard error: "
                        f"{diagnostic.generated_standard_error:.12g}",
                        f"reference standard error: "
                        f"{diagnostic.reference_standard_error:.12g}",
                        f"CUDA paths: "
                        f"{validation_input.monte_carlo_paths_per_price}",
                    ),
                )
            )
        special_rows = (*special_rows, *accepted_near_zero)
    unvalidated_row_count = len(hierarchy.unresolved_row_ids)
    failed_row_ids_tuple = tuple(
        row_id for row_id in row_ids if row_id in failed_row_ids
    )
    failed_row_count = len(failed_row_ids_tuple)
    validated_row_count = metrics.row_count if metrics is not None else 0
    directional_bias = (
        has_directional_bias(metrics.higher, metrics.lower, metrics.equal)
        if metrics is not None
        else False
    )
    native_bias = policy.enforce_statistical_bias and any(
        bool(getattr(report, "systematic_bias", False))
        for run in hierarchy.runs
        for report in run.reports
    )
    expected_bias = policy.bias_explanation is not None
    reported_bias = native_bias or (
        directional_bias
        and (policy.enforce_directional_bias or expected_bias)
    )
    unexpected_bias = (
        native_bias or (policy.enforce_directional_bias and directional_bias)
    ) and not expected_bias
    status = (
        "not_available"
        if validated_row_count == 0
        else "passed"
        if (
            failed_row_count == 0
            and unvalidated_row_count == 0
            and not unexpected_bias
        )
        else "failed"
    )
    bias_explanation = None
    if reported_bias:
        bias_explanation = policy.bias_explanation
        if bias_explanation is None and policy.enforce_directional_bias:
            bias_explanation = (
                "unexpected; more than 60% of price gaps have the same sign"
            )
    return ValidationDisplayReport(
        title="Core validation" if regime == "core" else "Stress validation",
        status=status,
        database_id=validation_input.database_id,
        reference=hierarchy.primary_reference,
        pricing_method=hierarchy.primary_pricing_method,
        tolerance=policy.tolerance,
        row_count=len(row_ids),
        accepted_row_count=(
            len(row_ids)
            - len(special_rows)
            - failed_row_count
            - unvalidated_row_count
        ),
        failed_row_count=failed_row_count,
        higher_price_count=metrics.higher if metrics is not None else 0,
        lower_price_count=metrics.lower if metrics is not None else 0,
        equal_price_count=metrics.equal if metrics is not None else 0,
        mean_signed_price_gap=metrics.mean_signed if metrics is not None else None,
        mean_absolute_price_gap=(
            metrics.mean_absolute if metrics is not None else None
        ),
        maximum_absolute_price_gap=(
            metrics.maximum_absolute if metrics is not None else None
        ),
        maximum_absolute_price_gap_row_id=(
            metrics.maximum_row_id if metrics is not None else None
        ),
        systematic_bias=reported_bias,
        bias_explanation=bias_explanation,
        special_rows=special_rows,
        fallbacks=fallbacks,
        unvalidated_row_count=unvalidated_row_count,
        engine_coverage=tuple(engine_coverage(run) for run in hierarchy.runs),
        engine_plan=engine_plan(hierarchy),
        no_validation_reason=(
            "no independent validation is available via Premia or QuantLib"
            if validated_row_count == 0
            else None
        ),
        failed_row_ids=failed_row_ids_tuple,
    )


def build_dataset_validation_report(
    price_dataset_path: str | Path,
    engine_factory: EngineFactory,
    policy: ValidationPolicy = ValidationPolicy(),
) -> DatasetValidationReport:
    """Validate both catalogue regimes with one reusable engine plan."""

    path = Path(price_dataset_path).resolve()
    return DatasetValidationReport(
        validation_fingerprint=validation_fingerprint(path),
        core=build_validation_section(path, "core", engine_factory(), policy),
        stress=build_validation_section(path, "stress", engine_factory(), policy),
    )


def run_dataset_validation_cli(
    validation: DatasetValidator,
    description: str,
) -> int:
    """Persist, synchronize, and display one model/product validation report."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("price_dataset", type=Path)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    report = validation(arguments.price_dataset)
    write_validation_report(report, arguments.report)
    synchronize_validation_yaml(report, arguments.price_dataset)
    display_validation_report(report)
    return 1 if "failed" in (report.core.status, report.stress.status) else 0


__all__ = (
    "ValidationPolicy",
    "build_dataset_validation_report",
    "build_validation_section",
    "premia_row_exception",
    "quantlib_row_exception",
    "run_dataset_validation_cli",
    "unavailable_engine",
)
