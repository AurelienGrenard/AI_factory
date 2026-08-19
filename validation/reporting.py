"""Uniform human-readable reports for independent price validations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


DIRECTIONAL_BIAS_THRESHOLD = 0.60
REPORT_SCHEMA_VERSION = 6
ValidationStatus = Literal["passed", "failed", "not_available"]


@dataclass(frozen=True)
class SpecialRowDiagnostic:
    """Document why one accepted row required individual treatment."""

    row_id: str
    diagnostic: str
    resolution: str
    category: str = "backend_fallback"
    acceptance_rule: str = "independent fallback price is within tolerance"
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class FallbackDiagnostic:
    """Record the independent backend used after a technical failure."""

    row_id: str
    reference: str
    pricing_method: str
    passed: bool


@dataclass(frozen=True)
class EngineExceptionDiagnostic:
    """Persist one technical engine failure for later audit."""

    row_id: str
    reason: str
    status: int | None = None


@dataclass(frozen=True)
class EngineCoverage:
    """Persist how many rows one compatible reference engine attempted."""

    reference: str
    pricing_method: str
    requested_row_count: int
    completed_row_count: int
    failed_row_count: int
    exceptions: tuple[EngineExceptionDiagnostic, ...] = ()


@dataclass(frozen=True)
class EnginePlanEntry:
    """Persist one available or unavailable slot in the reference hierarchy."""

    reference: str
    pricing_method: str | None
    available: bool
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.available and not self.pricing_method:
            raise ValueError(
                "An available engine-plan entry requires its pricing method."
            )
        if not self.available and self.pricing_method is not None:
            raise ValueError(
                "An unavailable engine-plan entry cannot expose a pricing method."
            )
        if not self.available and not self.unavailable_reason:
            raise ValueError("An unavailable engine-plan entry requires a reason.")
        if self.available and self.unavailable_reason is not None:
            raise ValueError(
                "An available engine-plan entry cannot have an unavailable reason."
            )


@dataclass(frozen=True)
class ValidationDisplayReport:
    """Backend-independent data rendered by every validation notebook."""

    title: str
    status: ValidationStatus
    database_id: str
    reference: str
    pricing_method: str | None
    tolerance: str
    row_count: int
    accepted_row_count: int
    failed_row_count: int
    higher_price_count: int
    lower_price_count: int
    equal_price_count: int
    mean_signed_price_gap: float | None
    mean_absolute_price_gap: float | None
    maximum_absolute_price_gap: float | None
    maximum_absolute_price_gap_row_id: str | None
    systematic_bias: bool
    bias_explanation: str | None = None
    special_rows: tuple[SpecialRowDiagnostic, ...] = ()
    fallbacks: tuple[FallbackDiagnostic, ...] = ()
    unvalidated_row_count: int = 0
    engine_coverage: tuple[EngineCoverage, ...] = ()
    engine_plan: tuple[EnginePlanEntry, ...] = ()
    no_validation_reason: str | None = None
    failed_row_ids: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def __post_init__(self) -> None:
        counts = (
            self.row_count,
            self.accepted_row_count,
            self.failed_row_count,
            self.unvalidated_row_count,
            self.higher_price_count,
            self.lower_price_count,
            self.equal_price_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("Validation report row counts must be non-negative.")
        covered_row_count = (
            self.accepted_row_count
            + len(self.special_rows)
            + self.failed_row_count
            + self.unvalidated_row_count
        )
        if covered_row_count != self.row_count:
            raise ValueError(
                "Ordinary, special, failed, and unvalidated rows must cover "
                "the validation report."
            )
        if (
            self.higher_price_count
            + self.lower_price_count
            + self.equal_price_count
            != self.row_count - self.unvalidated_row_count
        ):
            raise ValueError(
                "Higher, lower, and equal counts must cover every validated row."
            )
        gaps = (
            self.mean_signed_price_gap,
            self.mean_absolute_price_gap,
            self.maximum_absolute_price_gap,
        )
        validated_row_count = self.row_count - self.unvalidated_row_count
        if validated_row_count == 0:
            if self.pricing_method is not None:
                raise ValueError(
                    "Unavailable validation cannot expose a pricing method."
                )
            if any(gap is not None for gap in gaps):
                raise ValueError("Unavailable validation cannot expose price gaps.")
            if self.maximum_absolute_price_gap_row_id is not None:
                raise ValueError("Unavailable validation cannot expose a maximum row.")
            if self.status != "not_available":
                raise ValueError("Unavailable validation cannot pass.")
        else:
            if not self.pricing_method:
                raise ValueError(
                    "Validated prices require the exact pricing method."
                )
            if any(gap is None or not math.isfinite(gap) for gap in gaps):
                raise ValueError("Validation report price gaps must be finite.")
            if (
                self.mean_absolute_price_gap < 0.0
                or self.maximum_absolute_price_gap < 0.0
            ):
                raise ValueError("Absolute price gaps must be non-negative.")
            if self.maximum_absolute_price_gap_row_id is None:
                raise ValueError("Validated prices require a maximum-gap row id.")
        if self.unvalidated_row_count and self.passed:
            raise ValueError("A report with unvalidated rows cannot pass.")
        if self.failed_row_count and self.passed:
            raise ValueError("A report with failed rows cannot pass.")
        if len(self.failed_row_ids) != self.failed_row_count:
            raise ValueError("The failed-row count must match failed_row_ids.")
        if len(set(self.failed_row_ids)) != len(self.failed_row_ids):
            raise ValueError("Failed row ids must be unique.")
        if validated_row_count and self.status == "not_available":
            raise ValueError("A report with validated rows cannot be unavailable.")


@dataclass(frozen=True)
class DatasetValidationReport:
    """Persistent core/stress report produced by one dataset validator."""

    validation_fingerprint: str
    core: ValidationDisplayReport
    stress: ValidationDisplayReport

    def __post_init__(self) -> None:
        if len(self.validation_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.validation_fingerprint
        ):
            raise ValueError("The validation fingerprint must contain 64 hex digits.")
        if self.core.database_id != self.stress.database_id:
            raise ValueError("Core and stress reports must describe the same dataset.")

    @property
    def passed(self) -> bool:
        """Return true only when both catalogue regimes are accepted."""

        return self.core.passed and self.stress.passed


def has_directional_bias(
    higher_price_count: int,
    lower_price_count: int,
    equal_price_count: int,
    threshold: float = DIRECTIONAL_BIAS_THRESHOLD,
) -> bool:
    """Detect a price direction followed by strictly more than 60% of rows."""

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("The directional-bias threshold must be between zero and one.")
    compared = higher_price_count + lower_price_count + equal_price_count
    return (
        compared > 0
        and max(higher_price_count, lower_price_count) / compared > threshold
    )


def format_validation_report(report: ValidationDisplayReport) -> str:
    """Render the canonical concise validation-notebook output."""

    validated_row_count = report.row_count - report.unvalidated_row_count
    if validated_row_count == 0:
        return "\n".join(
            (
                f"{report.title}: NOT AVAILABLE",
                f"dataset                            : {report.database_id}",
                "reference                          : none",
                "pricing method                     : none",
                f"rows                               : {report.row_count}",
                f"unvalidated rows                   : {report.unvalidated_row_count}",
                "validation                         : "
                + (
                    report.no_validation_reason
                    or "no independent validation is available via Premia or QuantLib"
                ),
            )
        )

    lines = [
        f"{report.title}: {'PASS' if report.passed else 'FAIL'}",
        f"dataset                            : {report.database_id}",
        f"reference                          : {report.reference}",
        f"pricing method                     : {report.pricing_method}",
        f"tolerance                          : {report.tolerance}",
        f"rows                               : {report.row_count}",
        f"accepted without special treatment : {report.accepted_row_count}",
        f"accepted after special treatment   : {len(report.special_rows)}",
        f"failed                             : {report.failed_row_count}",
        f"mean signed price gap              : {report.mean_signed_price_gap:.6e}",
        f"mean absolute price gap            : {report.mean_absolute_price_gap:.6e}",
        f"maximum absolute price gap         : {report.maximum_absolute_price_gap:.6e} "
        f"(row {report.maximum_absolute_price_gap_row_id})",
        f"higher / lower / equal             : {report.higher_price_count} / "
        f"{report.lower_price_count} / {report.equal_price_count}",
        "systematic bias                    : "
        f"{'yes' if report.systematic_bias else 'no'}",
    ]
    if report.unvalidated_row_count:
        lines.insert(
            8,
            f"unvalidated rows                   : {report.unvalidated_row_count}",
        )
    if report.bias_explanation:
        lines.append(f"bias explanation                   : {report.bias_explanation}")
    if report.failed_row_ids:
        visible = report.failed_row_ids[:10]
        suffix = (
            ""
            if len(visible) == len(report.failed_row_ids)
            else f" ... ({len(report.failed_row_ids)} total)"
        )
        lines.append(
            "failed row ids                     : "
            + ", ".join(visible)
            + suffix
        )
    for special_row in report.special_rows:
        lines.extend(
            (
                "",
                special_row.row_id,
                f"  diagnostic : {special_row.diagnostic}",
                f"  resolution : {special_row.resolution}",
                f"  rule       : {special_row.acceptance_rule}",
            )
        )
    for fallback in report.fallbacks:
        lines.extend(
            (
                "",
                f"{fallback.reference} fallback for {fallback.row_id}: "
                f"{'PASS' if fallback.passed else 'FAIL'}",
                f"  pricing method : {fallback.pricing_method}",
            )
        )
    return "\n".join(lines)


def format_dataset_validation_report(report: DatasetValidationReport) -> str:
    """Render the core and stress sections of one persisted report."""

    return "\n\n".join(
        (
            format_validation_report(report.core),
            format_validation_report(report.stress),
        )
    )


def display_validation_report(report: DatasetValidationReport) -> None:
    """Display one complete persisted report and its generated conclusion."""

    print(format_dataset_validation_report(report))
    core_accepted = (
        report.core.row_count
        - report.core.failed_row_count
        - report.core.unvalidated_row_count
    )
    stress_accepted = (
        report.stress.row_count
        - report.stress.failed_row_count
        - report.stress.unvalidated_row_count
    )
    validated_row_count = (
        report.core.row_count
        + report.stress.row_count
        - report.core.unvalidated_row_count
        - report.stress.unvalidated_row_count
    )
    if validated_row_count == 0:
        status = "NOT VALIDATED"
    else:
        status = "PASS" if report.passed else "FAIL"
    conclusion = (
        f"**{status} — {core_accepted}/{report.core.row_count} core rows and "
        f"{stress_accepted}/{report.stress.row_count} stress rows are accepted.** "
        f"Core reference: {report.core.reference}; "
        f"stress reference: {report.stress.reference}."
    )
    special_count = len(report.core.special_rows) + len(report.stress.special_rows)
    if special_count:
        conclusion += (
            f" {special_count} row(s) required special treatment; consult "
            "validation_report.json for the complete diagnostics and resolution."
        )
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
    except ImportError:
        get_ipython = None
    if get_ipython is not None and get_ipython() is not None:
        display(Markdown("## Conclusion\n\n" + conclusion))
    else:
        print("\nConclusion\n" + conclusion.replace("**", ""))


def write_validation_notebook(
    report: DatasetValidationReport,
    price_dataset_path: str | Path,
    notebook_path: str | Path,
    model_label: str,
    product_label: str,
) -> Path:
    """Write the identical compiled presentation notebook for one report.

    The notebook contains no validation logic: its sole code cell reloads the
    fingerprint-checked JSON report.  The stored output is generated from that
    same report, so GitHub renders a useful result without executing a pricer.
    """

    dataset = Path(price_dataset_path).resolve()
    destination = Path(notebook_path).resolve()
    project_root = next(
        parent for parent in (dataset.parent, *dataset.parents)
        if (parent / "CMakeLists.txt").is_file()
    )
    dataset_relative = dataset.relative_to(project_root).as_posix()
    report_relative = (destination.parent / "validation_report.json").relative_to(
        project_root
    ).as_posix()
    status = "PASS" if report.passed else "NOT VALIDATED" if (
        report.core.status == report.stress.status == "not_available"
    ) else "FAIL"
    core_accepted = (
        report.core.row_count
        - report.core.failed_row_count
        - report.core.unvalidated_row_count
    )
    stress_accepted = (
        report.stress.row_count
        - report.stress.failed_row_count
        - report.stress.unvalidated_row_count
    )
    conclusion = (
        f"**{status} — {core_accepted}/{report.core.row_count} core rows and "
        f"{stress_accepted}/{report.stress.row_count} stress rows are accepted.** "
        f"Core reference: {report.core.reference}; "
        f"stress reference: {report.stress.reference}."
    )
    special_count = len(report.core.special_rows) + len(report.stress.special_rows)
    if special_count:
        conclusion += (
            f" {special_count} row(s) required special treatment; consult "
            "validation_report.json for the complete diagnostics and resolution."
        )
    code = """from pathlib import Path
import sys

project_root = next(
    parent for parent in (Path.cwd(), *Path.cwd().parents)
    if (parent / \"CMakeLists.txt\").is_file()
)
sys.path.insert(0, str(project_root))

from validation.reporting import display_validation_report, load_validation_report

dataset = project_root / \"{dataset}\"
report_path = project_root / \"{report}\"

report = load_validation_report(report_path, dataset)
display_validation_report(report)
""".format(dataset=dataset_relative, report=report_relative)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {model_label} / {product_label}\n",
                    "\n",
                    f"Persisted independent validation of `{dataset.stem}`. "
                    "The notebook only reads the report produced by the "
                    "model/product validator; it never reruns Premia or QuantLib.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [format_dataset_validation_report(report) + "\n"],
                    },
                    {
                        "data": {"text/markdown": ["## Conclusion\n\n" + conclusion]},
                        "metadata": {},
                        "output_type": "display_data",
                    },
                ],
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _price_catalog_yaml_path(price_dataset_path: Path) -> Path:
    try:
        document = json.loads(price_dataset_path.read_text(encoding="utf-8"))
        catalog = document["catalog"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            f"Cannot resolve the catalog YAML for '{price_dataset_path}': {error}"
        ) from error
    if not isinstance(catalog, str) or not catalog:
        raise ValueError("The price dataset catalog must be a non-empty string.")
    for candidate in (price_dataset_path.parent, *price_dataset_path.parents):
        if (candidate / "CMakeLists.txt").is_file():
            return candidate / catalog / "dataset.yaml"
    raise ValueError("Could not locate the project root for the price dataset.")


def validation_fingerprint(path: str | Path) -> str:
    """Hash prices and numerical configuration while excluding volatile timing."""

    price_path = Path(path).resolve()
    try:
        price_document = json.loads(price_path.read_text(encoding="utf-8"))
        yaml_document = yaml.safe_load(
            _price_catalog_yaml_path(price_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot build the validation fingerprint: {error}") from error
    reference_fields = (
        "model_dataset",
        "curve_dataset",
        "product_dataset",
    )
    references = {
        field: price_document[field]["id"]
        for field in reference_fields
        if field in price_document
    }
    configuration_fields = (
        "summary",
        "time_grid",
        "outputs",
        "price_construction",
    )
    configuration = {
        field: yaml_document[field]
        for field in configuration_fields
        if field in yaml_document
    }
    canonical = {
        "database_id": price_document["database_id"],
        "row_count": price_document["row_count"],
        "references": references,
        "results": price_document["results"],
        "pricing_configuration": configuration,
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _overall_status(report: DatasetValidationReport) -> ValidationStatus:
    if report.passed:
        return "passed"
    if report.core.status == report.stress.status == "not_available":
        return "not_available"
    return "failed"


def _yaml_validation_block(
    report: DatasetValidationReport, notebook_path: str
) -> str:
    overall_status = _overall_status(report)
    reference = (
        report.core.reference
        if report.core.reference == report.stress.reference
        else "mixed"
    )
    lines = [
        "validation:",
        f"  status: {json.dumps(overall_status)}",
        f"  verified: {'true' if report.passed else 'false'}",
        f"  reference: {json.dumps(reference)}",
        f"  notebook: {json.dumps(notebook_path)}",
    ]
    for name, section in (("core", report.core), ("stress", report.stress)):
        lines.extend(
            (
                f"  {name}:",
                f"    status: {json.dumps(section.status)}",
                f"    row_count: {section.row_count}",
                f"    reference: {json.dumps(section.reference)}",
            )
        )
    return "\n".join(lines) + "\n"


def synchronize_validation_yaml(
    report: DatasetValidationReport,
    price_dataset_path: str | Path,
) -> Path:
    """Replace only the generated YAML validation block from a real report."""

    yaml_path = _price_catalog_yaml_path(Path(price_dataset_path).resolve())
    project_root = next(
        parent for parent in yaml_path.parents
        if (parent / "CMakeLists.txt").is_file()
    )
    notebook_path = (yaml_path.parent / "validation.ipynb").relative_to(
        project_root
    ).as_posix()
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    try:
        start = next(
            index for index, line in enumerate(lines) if line == "validation:\n"
        )
    except StopIteration as error:
        raise ValueError(
            f"Catalog YAML '{yaml_path}' has no validation block."
        ) from error
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].strip() and not lines[index][0].isspace()
        ),
        len(lines),
    )
    updated = (
        "".join(lines[:start])
        + _yaml_validation_block(report, notebook_path)
        + "".join(lines[end:])
    )
    yaml_path.write_text(updated, encoding="utf-8")
    return yaml_path


def _section_to_json(report: ValidationDisplayReport) -> dict[str, Any]:
    return {
        "title": report.title,
        "status": report.status,
        "dataset": report.database_id,
        "reference": report.reference,
        "pricing_method": report.pricing_method,
        "tolerance": report.tolerance,
        "rows": report.row_count,
        "accepted_without_special_treatment": report.accepted_row_count,
        "special_row_count": len(report.special_rows),
        "failed": report.failed_row_count,
        "failed_row_ids": list(report.failed_row_ids),
        "unvalidated": report.unvalidated_row_count,
        "mean_signed_price_gap": report.mean_signed_price_gap,
        "mean_absolute_price_gap": report.mean_absolute_price_gap,
        "maximum_absolute_price_gap": report.maximum_absolute_price_gap,
        "maximum_absolute_price_gap_row_id": (
            report.maximum_absolute_price_gap_row_id
        ),
        "higher_price_count": report.higher_price_count,
        "lower_price_count": report.lower_price_count,
        "equal_price_count": report.equal_price_count,
        "systematic_bias": report.systematic_bias,
        "bias_explanation": report.bias_explanation,
        "special_rows": [
            {
                "row_id": row.row_id,
                "diagnostic": row.diagnostic,
                "resolution": row.resolution,
                "category": row.category,
                "acceptance_rule": row.acceptance_rule,
                "evidence": list(row.evidence),
            }
            for row in report.special_rows
        ],
        "fallbacks": [
            {
                "row_id": fallback.row_id,
                "reference": fallback.reference,
                "pricing_method": fallback.pricing_method,
                "passed": fallback.passed,
            }
            for fallback in report.fallbacks
        ],
        "engine_coverage": [
            {
                "reference": engine.reference,
                "pricing_method": engine.pricing_method,
                "requested_rows": engine.requested_row_count,
                "completed_rows": engine.completed_row_count,
                "failed_rows": engine.failed_row_count,
                "exceptions": [
                    {
                        "row_id": exception.row_id,
                        "status": exception.status,
                        "reason": exception.reason,
                    }
                    for exception in engine.exceptions
                ],
            }
            for engine in report.engine_coverage
        ],
        "engine_plan": [
            {
                "reference": engine.reference,
                "pricing_method": engine.pricing_method,
                "available": engine.available,
                "unavailable_reason": engine.unavailable_reason,
            }
            for engine in report.engine_plan
        ],
        "no_validation_reason": report.no_validation_reason,
    }


def _section_from_json(document: Mapping[str, Any]) -> ValidationDisplayReport:
    special_rows = tuple(
        SpecialRowDiagnostic(
            row_id=row["row_id"],
            diagnostic=row["diagnostic"],
            resolution=row["resolution"],
            category=row.get("category", "backend_fallback"),
            acceptance_rule=row.get(
                "acceptance_rule",
                "independent fallback price is within tolerance",
            ),
            evidence=tuple(row.get("evidence", ())),
        )
        for row in document["special_rows"]
    )
    fallbacks = tuple(
        FallbackDiagnostic(
            row_id=fallback["row_id"],
            reference=fallback["reference"],
            pricing_method=fallback["pricing_method"],
            passed=fallback["passed"],
        )
        for fallback in document["fallbacks"]
    )
    engine_coverage = tuple(
        EngineCoverage(
            reference=engine["reference"],
            pricing_method=engine["pricing_method"],
            requested_row_count=engine["requested_rows"],
            completed_row_count=engine["completed_rows"],
            failed_row_count=engine["failed_rows"],
            exceptions=tuple(
                EngineExceptionDiagnostic(
                    row_id=exception["row_id"],
                    status=exception["status"],
                    reason=exception["reason"],
                )
                for exception in engine["exceptions"]
            ),
        )
        for engine in document["engine_coverage"]
    )
    engine_plan = tuple(
        EnginePlanEntry(
            reference=engine["reference"],
            pricing_method=engine["pricing_method"],
            available=engine["available"],
            unavailable_reason=engine["unavailable_reason"],
        )
        for engine in document["engine_plan"]
    )
    if document["special_row_count"] != len(special_rows):
        raise ValueError("The special-row count does not match special_rows.")
    return ValidationDisplayReport(
        title=document["title"],
        status=document["status"],
        database_id=document["dataset"],
        reference=document["reference"],
        pricing_method=document["pricing_method"],
        tolerance=document["tolerance"],
        row_count=document["rows"],
        accepted_row_count=document["accepted_without_special_treatment"],
        failed_row_count=document["failed"],
        higher_price_count=document["higher_price_count"],
        lower_price_count=document["lower_price_count"],
        equal_price_count=document["equal_price_count"],
        mean_signed_price_gap=document["mean_signed_price_gap"],
        mean_absolute_price_gap=document["mean_absolute_price_gap"],
        maximum_absolute_price_gap=document["maximum_absolute_price_gap"],
        maximum_absolute_price_gap_row_id=(
            document["maximum_absolute_price_gap_row_id"]
        ),
        systematic_bias=document["systematic_bias"],
        bias_explanation=document["bias_explanation"],
        special_rows=special_rows,
        fallbacks=fallbacks,
        unvalidated_row_count=document["unvalidated"],
        engine_coverage=engine_coverage,
        engine_plan=engine_plan,
        no_validation_reason=document["no_validation_reason"],
        failed_row_ids=tuple(document["failed_row_ids"]),
    )


def write_validation_report(
    report: DatasetValidationReport,
    path: str | Path,
) -> None:
    """Persist one stable, backend-independent validation report as JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "validation_fingerprint": report.validation_fingerprint,
        "core": _section_to_json(report.core),
        "stress": _section_to_json(report.stress),
    }
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def load_validation_report(
    report_path: str | Path,
    price_dataset_path: str | Path | None = None,
) -> DatasetValidationReport:
    """Load a report and optionally prove that it matches the current dataset."""

    path = Path(report_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read validation report '{path}': {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Validation report '{path}' must contain a JSON object.")
    if document.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Validation report '{path}' has an unsupported schema version."
        )
    try:
        report = DatasetValidationReport(
            validation_fingerprint=document["validation_fingerprint"],
            core=_section_from_json(document["core"]),
            stress=_section_from_json(document["stress"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed validation report '{path}': {error}") from error
    if price_dataset_path is not None:
        current_fingerprint = validation_fingerprint(price_dataset_path)
        if current_fingerprint != report.validation_fingerprint:
            raise ValueError(
                "The validation report is stale: its input fingerprint does not "
                "match the current prices and numerical configuration."
            )
    return report
