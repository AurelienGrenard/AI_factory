"""Common aggregation of native Premia and QuantLib comparison reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from validation.hierarchy import EngineRun, ValidationHierarchyResult
from validation.reporting import (
    EngineCoverage,
    EngineExceptionDiagnostic,
    EnginePlanEntry,
    FallbackDiagnostic,
    SpecialRowDiagnostic,
)


@dataclass(frozen=True)
class PriceGapMetrics:
    """Backend-neutral price-gap statistics over comparable rows."""

    row_count: int
    failed_row_ids: tuple[str, ...]
    higher: int
    lower: int
    equal: int
    mean_signed: float
    mean_absolute: float
    maximum_absolute: float
    maximum_row_id: str


def native_price_gap_metrics(report: Any) -> PriceGapMetrics:
    """Adapt a native report exposing the common comparison field names."""

    higher = int(report.higher_price_count)
    lower = int(report.lower_price_count)
    equal = int(report.equal_price_count)
    return PriceGapMetrics(
        row_count=higher + lower + equal,
        failed_row_ids=tuple(report.failed_row_ids),
        higher=higher,
        lower=lower,
        equal=equal,
        mean_signed=float(report.mean_signed_error),
        mean_absolute=float(report.mean_absolute_error),
        maximum_absolute=float(report.maximum_absolute_error),
        maximum_row_id=str(report.maximum_absolute_error_row_id),
    )


def combined_price_gap_metrics(
    reports: tuple[Any, ...],
) -> PriceGapMetrics | None:
    """Combine disjoint native reports without losing their worst row."""

    metrics = tuple(native_price_gap_metrics(report) for report in reports)
    row_count = sum(metric.row_count for metric in metrics)
    if row_count == 0:
        return None
    maximum = max(metrics, key=lambda metric: metric.maximum_absolute)
    return PriceGapMetrics(
        row_count=row_count,
        failed_row_ids=tuple(
            row_id for metric in metrics for row_id in metric.failed_row_ids
        ),
        higher=sum(metric.higher for metric in metrics),
        lower=sum(metric.lower for metric in metrics),
        equal=sum(metric.equal for metric in metrics),
        mean_signed=sum(
            metric.mean_signed * metric.row_count for metric in metrics
        ) / row_count,
        mean_absolute=sum(
            metric.mean_absolute * metric.row_count for metric in metrics
        ) / row_count,
        maximum_absolute=maximum.maximum_absolute,
        maximum_row_id=maximum.maximum_row_id,
    )


def hierarchy_price_gap_metrics(
    hierarchy: ValidationHierarchyResult,
) -> PriceGapMetrics | None:
    """Combine every disjoint report produced by the row-wise hierarchy."""

    return combined_price_gap_metrics(
        tuple(report for run in hierarchy.runs for report in run.reports)
    )


def engine_coverage(run: EngineRun) -> EngineCoverage:
    """Convert one executed hierarchy step into persistent coverage."""

    metrics = combined_price_gap_metrics(run.reports)
    return EngineCoverage(
        reference=run.engine.label,
        requested_row_count=len(run.requested_row_ids),
        completed_row_count=len(run.completed_row_ids),
        failed_row_count=len(metrics.failed_row_ids) if metrics is not None else 0,
        exceptions=tuple(
            EngineExceptionDiagnostic(
                row_id=exception.row_id,
                status=exception.status,
                reason=exception.reason,
            )
            for exception in run.exceptions
        ),
    )


def engine_plan(
    hierarchy: ValidationHierarchyResult,
) -> tuple[EnginePlanEntry, ...]:
    """Convert all available and unavailable hierarchy slots for persistence."""

    return tuple(
        EnginePlanEntry(
            reference=engine.label,
            available=engine.available,
            unavailable_reason=engine.unavailable_reason,
        )
        for engine in hierarchy.engine_plan
    )


def resolved_fallback_diagnostics(
    runs: tuple[EngineRun, ...],
    generated_prices: Mapping[str, float],
) -> tuple[tuple[SpecialRowDiagnostic, ...], tuple[FallbackDiagnostic, ...]]:
    """Describe technical exceptions subsequently covered by another engine."""

    special_rows: list[SpecialRowDiagnostic] = []
    fallbacks: list[FallbackDiagnostic] = []

    def failed_row_ids(run: EngineRun) -> set[str]:
        metrics = combined_price_gap_metrics(run.reports)
        return set(metrics.failed_row_ids) if metrics is not None else set()

    for run_index, run in enumerate(runs[:-1]):
        for exception in run.exceptions:
            resolution = next(
                (
                    later
                    for later in runs[run_index + 1 :]
                    if exception.row_id in later.completed_row_ids
                ),
                None,
            )
            if resolution is None:
                continue
            fallback_passed = exception.row_id not in failed_row_ids(resolution)
            fallbacks.append(
                FallbackDiagnostic(
                    exception.row_id,
                    resolution.engine.label,
                    fallback_passed,
                )
            )
            if not fallback_passed:
                continue
            resolution_text = "generated price independently accepted by "
            if abs(generated_prices[exception.row_id]) <= 1.0e-7:
                resolution_text = "quasi-zero " + resolution_text
            special_rows.append(
                SpecialRowDiagnostic(
                    exception.row_id,
                    f"{run.engine.reference}"
                    + (
                        f" status {exception.status}"
                        if exception.status is not None
                        else ""
                    )
                    + f", {exception.reason}",
                    resolution_text + resolution.engine.label,
                )
            )
    return tuple(special_rows), tuple(fallbacks)
