"""Row-wise routing through the ordered independent-validation hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from validation.quantlib.price_validation import ValidationRegime


@dataclass(frozen=True)
class BackendException:
    """One row for which a reference engine produced no comparable price."""

    row_id: str
    reason: str
    status: int | None = None


@dataclass(frozen=True)
class BackendBatchResult:
    """Comparable rows, technical exceptions, and native backend reports."""

    completed_row_ids: tuple[str, ...]
    exceptions: tuple[BackendException, ...]
    reports: tuple[Any, ...]


BackendValidator = Callable[
    [Path, ValidationRegime, tuple[str, ...]], BackendBatchResult
]


@dataclass(frozen=True)
class ValidationEngine:
    """One explicitly compatible engine in the model/product hierarchy."""

    reference: str
    method: str
    validate: BackendValidator | None
    pricing_method: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.validate is not None and not self.pricing_method:
            raise ValueError(
                f"Available engine {self.label} requires its exact pricing method."
            )
        if self.validate is None and self.pricing_method is not None:
            raise ValueError(
                f"Unavailable engine {self.label} cannot expose a pricing method."
            )
        if self.validate is None and not self.unavailable_reason:
            raise ValueError(
                f"Unavailable engine {self.label} requires a concise reason."
            )
        if self.validate is not None and self.unavailable_reason is not None:
            raise ValueError(
                f"Available engine {self.label} cannot have an unavailable reason."
            )

    @property
    def label(self) -> str:
        return f"{self.reference} ({self.method})"

    @property
    def available(self) -> bool:
        return self.validate is not None


@dataclass(frozen=True)
class EngineRun:
    """Trace one engine's work on the rows left by its predecessor."""

    engine: ValidationEngine
    requested_row_ids: tuple[str, ...]
    completed_row_ids: tuple[str, ...]
    exceptions: tuple[BackendException, ...]
    reports: tuple[Any, ...]


@dataclass(frozen=True)
class ValidationHierarchyResult:
    """Complete ordered trace and rows left without an independent reference."""

    engine_plan: tuple[ValidationEngine, ...]
    runs: tuple[EngineRun, ...]
    unresolved_row_ids: tuple[str, ...]

    @property
    def primary_engine(self) -> ValidationEngine | None:
        """Return the engine that supplied the largest number of prices."""

        if not self.runs:
            return None
        primary = max(self.runs, key=lambda run: len(run.completed_row_ids))
        return primary.engine if primary.completed_row_ids else None

    @property
    def primary_reference(self) -> str:
        """Return the engine that supplied the largest number of prices."""

        return self.primary_engine.label if self.primary_engine is not None else "none"

    @property
    def primary_pricing_method(self) -> str | None:
        """Return the exact function or native method used for most rows."""

        return (
            self.primary_engine.pricing_method
            if self.primary_engine is not None
            else None
        )


def run_validation_hierarchy(
    price_dataset_path: str | Path,
    regime: ValidationRegime,
    row_ids: Sequence[str],
    engines: Sequence[ValidationEngine],
) -> ValidationHierarchyResult:
    """Send only technical exceptions to the next compatible engine."""

    path = Path(price_dataset_path).resolve()
    pending = tuple(row_ids)
    runs: list[EngineRun] = []
    engine_plan = tuple(engines)
    for engine in engine_plan:
        if not engine.available:
            continue
        if not pending:
            break
        assert engine.validate is not None
        result = engine.validate(path, regime, pending)
        completed = tuple(result.completed_row_ids)
        exceptions = tuple(result.exceptions)
        completed_set = set(completed)
        exception_ids = tuple(exception.row_id for exception in exceptions)
        exception_set = set(exception_ids)
        if len(completed_set) != len(completed):
            raise RuntimeError(f"{engine.label} returned duplicate completed rows.")
        if len(exception_set) != len(exception_ids):
            raise RuntimeError(f"{engine.label} returned duplicate exception rows.")
        if completed_set & exception_set:
            raise RuntimeError(
                f"{engine.label} marked rows as both completed and exceptional."
            )
        if completed_set | exception_set != set(pending):
            raise RuntimeError(
                f"{engine.label} did not partition every requested row."
            )
        runs.append(
            EngineRun(
                engine=engine,
                requested_row_ids=pending,
                completed_row_ids=completed,
                exceptions=exceptions,
                reports=tuple(result.reports),
            )
        )
        pending = exception_ids
    return ValidationHierarchyResult(engine_plan, tuple(runs), pending)


def isolate_backend_exceptions(
    row_ids: Sequence[str],
    validate_batch: Callable[[tuple[str, ...]], Any],
    is_backend_exception: Callable[[Exception], bool],
) -> BackendBatchResult:
    """Bisect a failed batch only as far as needed to identify technical rows."""

    completed: list[str] = []
    exceptions: list[BackendException] = []
    reports: list[Any] = []

    def validate(selected: tuple[str, ...]) -> None:
        try:
            report = validate_batch(selected)
        except Exception as error:
            if not is_backend_exception(error):
                raise
            if len(selected) == 1:
                exceptions.append(BackendException(selected[0], str(error)))
                return
            midpoint = len(selected) // 2
            validate(selected[:midpoint])
            validate(selected[midpoint:])
            return
        completed.extend(selected)
        reports.append(report)

    selected_rows = tuple(row_ids)
    if selected_rows:
        validate(selected_rows)
    return BackendBatchResult(
        completed_row_ids=tuple(completed),
        exceptions=tuple(exceptions),
        reports=tuple(reports),
    )
