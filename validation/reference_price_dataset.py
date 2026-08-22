"""Persistent independent price references and fail-closed comparisons."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping, Sequence

import yaml

from validation.quantlib.price_validation import (
    CORE_ROW_COUNT,
    STRESS_ROW_COUNT,
    PriceComparison,
    PriceValidationInput,
    PriceValidationReport,
    ValidationTolerances,
    load_price_validation_input,
    summarize_price_comparisons,
)


_PRICER_ORDER = ("premia", "quantlib_specialized", "quantlib_monte_carlo")
_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class ReferencePrice:
    """One persisted reference value aligned with a source price row."""

    row_id: str
    model_id: str
    product_id: str
    price: float
    standard_error: float = 0.0
    reference_pricer_id: str = ""
    curve_id: str | None = None
    comparison_relation: str = "absolute"
    comparison_allowance: float | None = None


@dataclass(frozen=True)
class ReferenceDatasetValidation:
    """Core and stress comparisons against one immutable reference database."""

    core: PriceValidationReport
    stress: PriceValidationReport
    allow_systematic_bias: bool = False
    systematic_bias_explanation: str | None = None

    @property
    def verified(self) -> bool:
        return all(
            report.failed_row_count == 0
            and (self.allow_systematic_bias or not report.systematic_bias)
            for report in (self.core, self.stress)
        )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON '{path}': {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON '{path}' must contain an object.")
    return value


def _project_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "CMakeLists.txt").is_file() and (
            candidate / "datasets"
        ).is_dir():
            return candidate
    raise ValueError(f"Cannot locate the project root from '{path}'.")


def _semantic_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _semantic_python_source(value: Any) -> str:
    """Return a formatting-independent AST for one policy implementation."""

    try:
        source = textwrap.dedent(inspect.getsource(value))
    except (OSError, TypeError) as error:
        raise RuntimeError(
            f"Cannot inspect validation policy source for {value!r}."
        ) from error
    return ast.dump(
        ast.parse(source),
        annotate_fields=True,
        include_attributes=False,
    )


def source_fingerprints(source_price_dataset: str | Path) -> dict[str, str]:
    """Hash only semantic prices and parameters, excluding volatile timing."""

    source_path = Path(source_price_dataset).resolve()
    source_document = _read_object(source_path)
    source = load_price_validation_input(source_path)
    model_document = _read_object(source.model_dataset_path)
    product_document = _read_object(source.product_dataset_path)
    fingerprints = {
        "price_results": _semantic_fingerprint(
            {
                "database_id": source_document.get("database_id"),
                "row_count": source_document.get("row_count"),
                "results": source_document.get("results"),
            }
        ),
        "model_parameters": _semantic_fingerprint(
            {
                "database_id": model_document.get("database_id"),
                "row_count": model_document.get("row_count"),
                "models": model_document.get("models"),
            }
        ),
        "product_parameters": _semantic_fingerprint(
            {
                "database_id": product_document.get("database_id"),
                "row_count": product_document.get("row_count"),
                "products": product_document.get("products"),
            }
        ),
    }
    if source.curve_dataset_path is not None:
        curve_document = _read_object(source.curve_dataset_path)
        fingerprints["curve_parameters"] = _semantic_fingerprint(
            {
                "database_id": curve_document.get("database_id"),
                "row_count": curve_document.get("row_count"),
                "curves": curve_document.get("curves"),
            }
        )
    return fingerprints


def _tolerances_document(tolerances: ValidationTolerances) -> dict[str, float]:
    return {
        "absolute": tolerances.absolute,
        "relative": tolerances.relative,
        "relative_floor": tolerances.relative_floor,
        "standard_error_multiplier": tolerances.standard_error_multiplier,
        "bias_standard_errors": tolerances.bias_standard_errors,
    }


def _tolerances_from_document(value: Any) -> ValidationTolerances:
    if not isinstance(value, dict):
        raise ValueError("verification.tolerances must be an object.")
    fields = (
        "absolute",
        "relative",
        "relative_floor",
        "standard_error_multiplier",
        "bias_standard_errors",
    )
    if set(value) != set(fields):
        raise ValueError("verification.tolerances has an invalid field set.")
    try:
        return ValidationTolerances(**{field: float(value[field]) for field in fields})
    except (TypeError, ValueError) as error:
        raise ValueError("verification.tolerances is invalid.") from error


def validation_policy_fingerprint(
    tolerances: ValidationTolerances,
    allow_systematic_bias: bool = False,
    systematic_bias_explanation: str | None = None,
) -> str:
    """Hash the current comparison policy and its semantic implementation."""

    policy = {
        "schema": "ai_factory.reference_price_validation_policy.v1",
        "regimes": {"core": CORE_ROW_COUNT, "stress": STRESS_ROW_COUNT},
        "tolerances": _tolerances_document(tolerances),
        "systematic_bias_policy": {
            "allowed": allow_systematic_bias,
            "explanation": systematic_bias_explanation,
        },
        "implementation": {
            value.__name__: _semantic_python_source(value)
            for value in (
                ValidationTolerances,
                summarize_price_comparisons,
                ReferenceDatasetValidation,
                _tolerances_document,
                _tolerances_from_document,
                _verification_section,
                verification_document,
                _validate_verification,
                _comparison_report,
                compare_reference_dataset,
                validate_published_reference,
            )
        },
    }
    return _semantic_fingerprint(policy)


def _verification_section(
    report: PriceValidationReport,
    allow_systematic_bias: bool,
) -> dict[str, Any]:
    passed = report.failed_row_count == 0 and (
        allow_systematic_bias or not report.systematic_bias
    )
    return {
        "status": "passed" if passed else "failed",
        "row_count": report.row_count,
        "passed_row_count": report.passed_row_count,
        "failed_row_count": report.failed_row_count,
        "maximum_absolute_error": report.maximum_absolute_error,
        "maximum_absolute_error_row_id": report.maximum_absolute_error_row_id,
        "mean_signed_error": report.mean_signed_error,
        "systematic_bias": report.systematic_bias,
    }


def verification_document(
    report: ReferenceDatasetValidation,
    tolerances: ValidationTolerances,
) -> dict[str, Any]:
    """Serialize the policy and exact evidence behind the YAML boolean."""

    document = {
        "status": "passed" if report.verified else "failed",
        "tolerances": _tolerances_document(tolerances),
        "core": _verification_section(
            report.core, report.allow_systematic_bias
        ),
        "stress": _verification_section(
            report.stress, report.allow_systematic_bias
        ),
    }
    if report.allow_systematic_bias:
        if not report.systematic_bias_explanation:
            raise ValueError(
                "An accepted systematic bias requires an explanation."
            )
        document["systematic_bias_policy"] = {
            "status": "accepted_expected",
            "explanation": report.systematic_bias_explanation,
        }
    return document


def _detailed_pricer(value: Any, context: str, remaining: int) -> tuple[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a pricer object.")
    for field in ("id", "backend", "backend_version", "kind", "method"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"{context}.{field} must be a non-empty string.")
    row_priced = value.get("row_priced")
    if (
        not isinstance(row_priced, int)
        or isinstance(row_priced, bool)
        or row_priced <= 0
        or row_priced > remaining
    ):
        raise ValueError(f"{context}.row_priced exceeds the remaining regime rows.")
    return value["id"], row_priced


def _validate_pricers(
    pricers: Any,
    database_id: str,
) -> dict[str, dict[str, int]]:
    if not isinstance(pricers, dict):
        raise ValueError(
            f"Reference dataset '{database_id}': reference_pricers required."
        )
    if list(pricers) != ["core", "stress"]:
        raise ValueError("reference_pricers must contain ordered core and stress.")
    used_by_regime: dict[str, dict[str, int]] = {}
    for regime, expected in (
        ("core", CORE_ROW_COUNT),
        ("stress", STRESS_ROW_COUNT),
    ):
        section = pricers.get(regime)
        if not isinstance(section, dict):
            raise ValueError(f"Reference dataset '{database_id}': {regime} required.")
        if list(section) != ["row_count", *_PRICER_ORDER]:
            raise ValueError(f"{regime} pricers must follow the declared hierarchy.")
        if section.get("row_count") != expected:
            raise ValueError(
                f"Reference dataset '{database_id}': invalid {regime} row_count."
            )
        remaining = expected
        used: dict[str, int] = {}
        for name in _PRICER_ORDER:
            value = section[name]
            context = f"{regime}.{name}"
            if not isinstance(value, dict):
                raise ValueError(f"{context} must be an object.")
            status = value.get("status")
            if status in {"not_available", "available but not reliable"}:
                if value != {"status": status}:
                    raise ValueError(f"{context} must contain status only.")
                continue
            if status != "available":
                raise ValueError(f"{context}.status is unsupported.")
            if len(value) == 1:
                if remaining != 0:
                    raise ValueError(
                        f"{context} is required for {remaining} rows and must "
                        "describe the method used."
                    )
                continue
            identifier, row_priced = _detailed_pricer(value, context, remaining)
            if identifier in used:
                raise ValueError(f"{regime} repeats pricer id '{identifier}'.")
            used[identifier] = row_priced
            remaining -= row_priced
        if remaining != 0:
            raise ValueError(f"{regime} leaves {remaining} reference rows unpriced.")
        used_by_regime[regime] = used
    return used_by_regime


def _validate_verification(value: Any, database_id: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Reference dataset '{database_id}': verification required.")
    if value.get("status") not in {"passed", "failed"}:
        raise ValueError("verification.status must be passed or failed.")
    _tolerances_from_document(value.get("tolerances"))
    bias_policy = value.get("systematic_bias_policy")
    if bias_policy is None:
        allow_systematic_bias = False
    elif (
        not isinstance(bias_policy, dict)
        or bias_policy.get("status") != "accepted_expected"
        or not isinstance(bias_policy.get("explanation"), str)
        or not bias_policy["explanation"]
        or set(bias_policy) != {"status", "explanation"}
    ):
        raise ValueError("verification.systematic_bias_policy is unsupported.")
    else:
        allow_systematic_bias = True
    regime_passed: list[bool] = []
    for regime, expected in (
        ("core", CORE_ROW_COUNT),
        ("stress", STRESS_ROW_COUNT),
    ):
        section = value.get(regime)
        if not isinstance(section, dict):
            raise ValueError(f"verification.{regime} must be an object.")
        status = section.get("status")
        passed = section.get("passed_row_count")
        failed = section.get("failed_row_count")
        if (
            status not in {"passed", "failed"}
            or section.get("row_count") != expected
            or not isinstance(passed, int)
            or isinstance(passed, bool)
            or not isinstance(failed, int)
            or isinstance(failed, bool)
            or passed + failed != expected
            or not isinstance(section.get("systematic_bias"), bool)
        ):
            raise ValueError(f"verification.{regime} has inconsistent counts.")
        for field in ("maximum_absolute_error", "mean_signed_error"):
            number = section.get(field)
            if (
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(number)
            ):
                raise ValueError(f"verification.{regime}.{field} must be finite.")
        if section["maximum_absolute_error"] < 0.0:
            raise ValueError(
                f"verification.{regime}.maximum_absolute_error must be non-negative."
            )
        row_id = section.get("maximum_absolute_error_row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(
                f"verification.{regime}.maximum_absolute_error_row_id required."
            )
        expected_pass = failed == 0 and (
            allow_systematic_bias or not section["systematic_bias"]
        )
        if (status == "passed") != expected_pass:
            raise ValueError(f"verification.{regime}.status is inconsistent.")
        regime_passed.append(expected_pass)
    if (value["status"] == "passed") != all(regime_passed):
        raise ValueError("verification.status is inconsistent with its regimes.")


def validate_reference_document(
    document: Mapping[str, Any],
    require_validation_policy_fingerprint: bool = False,
) -> None:
    """Validate the stable envelope before a reference database is consumed."""

    database_id = document.get("database_id")
    if not isinstance(database_id, str) or not database_id:
        raise ValueError("Reference database_id must be a non-empty string.")
    if document.get("row_count") != CORE_ROW_COUNT + STRESS_ROW_COUNT:
        raise ValueError(f"Reference dataset '{database_id}' must contain 1000 rows.")
    for field in ("catalog", "url"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise ValueError(
                f"Reference dataset '{database_id}': {field} is required."
            )
    for field in ("model_dataset", "product_dataset"):
        if not isinstance(document.get(field), dict):
            raise ValueError(f"Reference dataset '{database_id}': {field} required.")

    fingerprints = document.get("source_fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError(f"Reference dataset '{database_id}': fingerprints required.")
    required_fingerprints = {
        "price_results",
        "model_parameters",
        "product_parameters",
    }
    if "curve_dataset" in document:
        required_fingerprints.add("curve_parameters")
    if set(fingerprints) != required_fingerprints or any(
        not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None
        for value in fingerprints.values()
    ):
        raise ValueError(f"Reference dataset '{database_id}': invalid fingerprints.")

    policy_fingerprint = document.get("validation_policy_fingerprint")
    if policy_fingerprint is None:
        if require_validation_policy_fingerprint:
            raise ValueError(
                f"Reference dataset '{database_id}': validation policy "
                "fingerprint required."
            )
    elif (
        not isinstance(policy_fingerprint, str)
        or _FINGERPRINT_PATTERN.fullmatch(policy_fingerprint) is None
    ):
        raise ValueError(
            f"Reference dataset '{database_id}': invalid validation policy "
            "fingerprint."
        )

    used_by_regime = _validate_pricers(
        document.get("reference_pricers"), database_id
    )
    _validate_verification(document.get("verification"), database_id)

    timing = document.get("timing")
    wall = timing.get("wall_seconds") if isinstance(timing, dict) else None
    if (
        not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or wall < 0
    ):
        raise ValueError(f"Reference dataset '{database_id}': invalid wall_seconds.")

    results = document.get("results")
    if not isinstance(results, list) or len(results) != document["row_count"]:
        raise ValueError(f"Reference dataset '{database_id}': results size mismatch.")
    identifiers: set[str] = set()
    actual_pricer_counts = {"core": Counter(), "stress": Counter()}
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise ValueError(f"Reference dataset '{database_id}': invalid result row.")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in identifiers:
            raise ValueError(
                f"Reference dataset '{database_id}': invalid/duplicate row id."
            )
        identifiers.add(row_id)
        for field in ("model_id", "product_id", "reference_pricer_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Reference row '{row_id}': {field} is required.")
        regime = "core" if index < CORE_ROW_COUNT else "stress"
        pricer_id = row["reference_pricer_id"]
        if pricer_id not in used_by_regime[regime]:
            raise ValueError(
                f"Reference row '{row_id}': pricer is not used for {regime}."
            )
        actual_pricer_counts[regime][pricer_id] += 1
        outputs = row.get("outputs")
        price = outputs.get("price") if isinstance(outputs, dict) else None
        error = (
            outputs.get("standard_error", 0.0)
            if isinstance(outputs, dict)
            else None
        )
        if not isinstance(price, (int, float)) or not math.isfinite(float(price)):
            raise ValueError(f"Reference row '{row_id}': price must be finite.")
        if (
            not isinstance(error, (int, float))
            or not math.isfinite(float(error))
            or error < 0
        ):
            raise ValueError(f"Reference row '{row_id}': invalid standard_error.")
        comparison = row.get("comparison")
        if comparison is not None:
            if not isinstance(comparison, dict) or set(comparison) != {
                "relation",
                "allowance",
            }:
                raise ValueError(
                    f"Reference row '{row_id}': invalid comparison contract."
                )
            if comparison.get("relation") not in {
                "absolute",
                "generated_at_least_reference",
                "generated_at_most_reference",
            }:
                raise ValueError(
                    f"Reference row '{row_id}': unsupported comparison relation."
                )
            allowance = comparison.get("allowance")
            if (
                not isinstance(allowance, (int, float))
                or isinstance(allowance, bool)
                or not math.isfinite(float(allowance))
                or allowance < 0.0
            ):
                raise ValueError(
                    f"Reference row '{row_id}': invalid comparison allowance."
                )
    for regime in ("core", "stress"):
        if dict(actual_pricer_counts[regime]) != used_by_regime[regime]:
            raise ValueError(f"{regime} row_priced does not match result provenance.")


def _reference_result(value: ReferencePrice) -> dict[str, Any]:
    outputs: dict[str, float] = {"price": value.price}
    if value.standard_error != 0.0:
        outputs["standard_error"] = value.standard_error
    result: dict[str, Any] = {
        "id": value.row_id,
        "model_id": value.model_id,
        "product_id": value.product_id,
        "reference_pricer_id": value.reference_pricer_id,
        "outputs": outputs,
    }
    if value.curve_id is not None:
        result["curve_id"] = value.curve_id
    if value.comparison_allowance is not None:
        result["comparison"] = {
            "relation": value.comparison_relation,
            "allowance": value.comparison_allowance,
        }
    return result


def _comparison_report(
    source: PriceValidationInput,
    reference_results: Sequence[Mapping[str, Any]],
    tolerances: ValidationTolerances,
    allow_systematic_bias: bool = False,
    systematic_bias_explanation: str | None = None,
) -> ReferenceDatasetValidation:
    if len(reference_results) != len(source.rows):
        raise ValueError("Source and reference result counts differ.")
    comparisons: list[PriceComparison] = []
    for source_row, reference in zip(source.rows, reference_results):
        if reference.get("id") != source_row.row_id:
            raise ValueError("Source and reference row order differs.")
        if (
            reference.get("model_id") != source_row.model_id
            or reference.get("product_id") != source_row.product_id
            or reference.get("curve_id") != source_row.curve_id
        ):
            raise ValueError(
                f"Reference row '{source_row.row_id}' is not source-aligned."
            )
        outputs = reference.get("outputs")
        comparison = reference.get("comparison")
        comparisons.append(
            PriceComparison(
                source_row.row_id,
                source_row.generated_price,
                float(outputs["price"]),
                source_row.generated_standard_error,
                float(outputs.get("standard_error", 0.0)),
                (
                    str(comparison["relation"])
                    if isinstance(comparison, dict)
                    else "absolute"
                ),
                (
                    float(comparison["allowance"])
                    if isinstance(comparison, dict)
                    else None
                ),
            )
        )
    if len(comparisons) != CORE_ROW_COUNT + STRESS_ROW_COUNT:
        raise ValueError(
            "Source/reference comparison requires 900 core + 100 stress rows."
        )
    return ReferenceDatasetValidation(
        summarize_price_comparisons(
            source.database_id, comparisons[:CORE_ROW_COUNT], tolerances
        ),
        summarize_price_comparisons(
            source.database_id, comparisons[CORE_ROW_COUNT:], tolerances
        ),
        allow_systematic_bias,
        systematic_bias_explanation,
    )


def compare_reference_prices(
    source_price_dataset: str | Path,
    prices: Sequence[ReferencePrice],
    tolerances: ValidationTolerances = ValidationTolerances(),
    allow_systematic_bias: bool = False,
    systematic_bias_explanation: str | None = None,
) -> ReferenceDatasetValidation:
    """Compare generated reference values before they are persisted."""

    source = load_price_validation_input(source_price_dataset)
    return _comparison_report(
        source,
        tuple(_reference_result(value) for value in prices),
        tolerances,
        allow_systematic_bias,
        systematic_bias_explanation,
    )


def build_reference_document(
    source_price_dataset: str | Path,
    destination: str | Path,
    prices: Sequence[ReferencePrice],
    reference_pricers: Mapping[str, Any],
    report: ReferenceDatasetValidation,
    tolerances: ValidationTolerances,
    wall_seconds: float,
) -> dict[str, Any]:
    """Copy source identity metadata and attach independent reference prices."""

    source_path = Path(source_price_dataset).resolve()
    root = _project_root(source_path)
    source = _read_object(source_path)
    destination_path = Path(destination).resolve()
    try:
        relative_text = destination_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Reference prices must live inside the project.") from error
    prefix = "validation/datasets/"
    if not relative_text.startswith(prefix):
        raise ValueError("Reference prices must live below validation/datasets/.")
    document: dict[str, Any] = {
        "database_id": source["database_id"],
        "catalog": source["catalog"],
        "url": "https://datasets.ai-factory.example/v1/validation/"
        + relative_text[len(prefix):],
        "row_count": len(prices),
        "model_dataset": source["model_dataset"],
        "product_dataset": source["product_dataset"],
    }
    if "curve_dataset" in source:
        document["curve_dataset"] = source["curve_dataset"]
    document.update(
        {
            "source_fingerprints": source_fingerprints(source_path),
            "validation_policy_fingerprint": validation_policy_fingerprint(
                tolerances,
                report.allow_systematic_bias,
                report.systematic_bias_explanation,
            ),
            "reference_pricers": dict(reference_pricers),
            "verification": verification_document(report, tolerances),
            "timing": {"wall_seconds": wall_seconds},
            "results": [_reference_result(value) for value in prices],
        }
    )
    validate_reference_document(document)
    return document


def write_reference_document(document: Mapping[str, Any], path: str | Path) -> None:
    """Validate and write one deterministic, human-readable reference JSON."""

    validate_reference_document(
        document, require_validation_policy_fingerprint=True
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _validate_envelope(
    source_path: Path,
    reference_document: Mapping[str, Any],
) -> None:
    source_document = _read_object(source_path)
    for field in (
        "database_id",
        "catalog",
        "row_count",
        "model_dataset",
        "product_dataset",
        "curve_dataset",
    ):
        if reference_document.get(field) != source_document.get(field):
            raise ValueError(f"Source and reference {field} values differ.")
    expected_fingerprints = source_fingerprints(source_path)
    if reference_document.get("source_fingerprints") != expected_fingerprints:
        raise ValueError("Reference source fingerprints are stale.")


def compare_reference_dataset(
    source_price_dataset: str | Path,
    reference_price_dataset: str | Path,
    tolerances: ValidationTolerances | None = None,
    require_validation_policy_fingerprint: bool = False,
) -> ReferenceDatasetValidation:
    """Compare aligned source and cache, checking persisted verification."""

    source_path = Path(source_price_dataset).resolve()
    source = load_price_validation_input(source_path)
    reference_document = _read_object(Path(reference_price_dataset).resolve())
    validate_reference_document(
        reference_document,
        require_validation_policy_fingerprint=require_validation_policy_fingerprint,
    )
    _validate_envelope(source_path, reference_document)
    stored_tolerances = _tolerances_from_document(
        reference_document["verification"]["tolerances"]
    )
    bias_policy = reference_document["verification"].get(
        "systematic_bias_policy"
    )
    allow_systematic_bias = isinstance(bias_policy, dict)
    systematic_bias_explanation = (
        bias_policy["explanation"] if isinstance(bias_policy, dict) else None
    )
    if tolerances is not None and _tolerances_document(tolerances) != (
        _tolerances_document(stored_tolerances)
    ):
        raise ValueError("Requested tolerances differ from the persisted policy.")
    effective_tolerances = (
        stored_tolerances if tolerances is None else tolerances
    )
    expected_policy_fingerprint = validation_policy_fingerprint(
        effective_tolerances,
        allow_systematic_bias,
        systematic_bias_explanation,
    )
    stored_policy_fingerprint = reference_document.get(
        "validation_policy_fingerprint"
    )
    if (
        stored_policy_fingerprint is not None
        and stored_policy_fingerprint != expected_policy_fingerprint
    ):
        raise ValueError("Reference validation policy fingerprint is stale.")
    report = _comparison_report(
        source,
        reference_document["results"],
        effective_tolerances,
        allow_systematic_bias,
        systematic_bias_explanation,
    )
    expected_verification = verification_document(
        report, effective_tolerances
    )
    if reference_document["verification"] != expected_verification:
        raise ValueError(
            "Persisted verification does not match the current comparison."
        )
    return report


def validate_published_reference(
    source_price_dataset: str | Path,
    reference_price_dataset: str | Path,
    tolerances: ValidationTolerances | None = None,
    require_validation_policy_fingerprint: bool = False,
) -> ReferenceDatasetValidation:
    """Fail closed unless source, cache, verification, and YAML all agree."""

    source_path = Path(source_price_dataset).resolve()
    reference_path = Path(reference_price_dataset).resolve()
    report = compare_reference_dataset(
        source_path,
        reference_path,
        tolerances,
        require_validation_policy_fingerprint,
    )
    if not report.verified:
        raise ValueError("A failed reference dataset cannot be published.")
    root = _project_root(source_path)
    try:
        reference_relative = reference_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Published reference must live inside the project.") from error
    prefix = "validation/datasets/"
    if not reference_relative.startswith(prefix):
        raise ValueError("Published reference must live below validation/datasets/.")
    reference_document = _read_object(reference_path)
    expected_url = (
        "https://datasets.ai-factory.example/v1/validation/"
        + reference_relative[len(prefix):]
    )
    if reference_document.get("url") != expected_url:
        raise ValueError("Published reference URL contradicts its repository path.")
    source_document = _read_object(source_path)
    catalog_path = root / source_document["catalog"] / "dataset.yaml"
    try:
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"Cannot read catalog YAML '{catalog_path}': {error}"
        ) from error
    expected = {
        "status": "available",
        "verified": True,
        "dataset": reference_relative,
    }
    if not isinstance(catalog, dict) or catalog.get("validation") != expected:
        raise ValueError("Catalog validation metadata contradicts the reference cache.")
    return report


def refresh_validation_policy_fingerprint(
    source_price_dataset: str | Path,
    reference_price_dataset: str | Path,
    tolerances: ValidationTolerances = ValidationTolerances(),
) -> ReferenceDatasetValidation:
    """Revalidate cached prices and publish the current policy fingerprint."""

    source_path = Path(source_price_dataset).resolve()
    reference_path = Path(reference_price_dataset).resolve()
    reference_document = _read_object(reference_path)
    validate_reference_document(reference_document)
    _validate_envelope(source_path, reference_document)
    verification = reference_document["verification"]
    bias_policy = verification.get("systematic_bias_policy")
    allow_systematic_bias = isinstance(bias_policy, dict)
    systematic_bias_explanation = (
        bias_policy["explanation"] if allow_systematic_bias else None
    )
    report = _comparison_report(
        load_price_validation_input(source_path),
        reference_document["results"],
        tolerances,
        allow_systematic_bias,
        systematic_bias_explanation,
    )
    if not report.verified:
        raise ValueError(
            "Reference prices fail the current validation policy; regenerate "
            "them with the independent backend."
        )
    policy_fingerprint = validation_policy_fingerprint(
        tolerances,
        allow_systematic_bias,
        systematic_bias_explanation,
    )
    reference_document["verification"] = verification_document(
        report, tolerances
    )
    ordered_document: dict[str, Any] = {}
    for field, value in reference_document.items():
        if field == "validation_policy_fingerprint":
            continue
        ordered_document[field] = value
        if field == "source_fingerprints":
            ordered_document["validation_policy_fingerprint"] = (
                policy_fingerprint
            )
    write_reference_document(ordered_document, reference_path)
    return validate_published_reference(
        source_path,
        reference_path,
        tolerances,
        require_validation_policy_fingerprint=True,
    )


def synchronize_catalog_validation(
    source_price_dataset: str | Path,
    reference_price_dataset: str | Path,
    verified: bool,
) -> None:
    """Replace one catalogue validation block with the compact cache contract."""

    source_path = Path(source_price_dataset).resolve()
    root = _project_root(source_path)
    source_document = _read_object(source_path)
    catalog_path = root / source_document["catalog"] / "dataset.yaml"
    try:
        reference_relative = (
            Path(reference_price_dataset).resolve().relative_to(root).as_posix()
        )
    except ValueError as error:
        raise ValueError("Reference prices must live inside the project.") from error
    text = catalog_path.read_text(encoding="utf-8")
    marker = "\nvalidation:\n"
    if marker not in text:
        raise ValueError(f"Catalog YAML '{catalog_path}' has no validation block.")
    start = text.index(marker) + 1
    end = len(text)
    for next_marker in ("\noutputs:\n", "\nmodel_dataset:\n"):
        position = text.find(next_marker, start)
        if position >= 0:
            end = min(end, position + 1)
    block = "\n".join(
        (
            "validation:",
            '  status: "available"',
            f"  verified: {'true' if verified else 'false'}",
            f"  dataset: {json.dumps(reference_relative)}",
        )
    ) + "\n"
    catalog_path.write_text(text[:start] + block + text[end:], encoding="utf-8")


def format_reference_validation(
    report: ReferenceDatasetValidation,
    label: str,
) -> str:
    """Render the two cached comparisons for logs and command-line use."""

    sections = []
    for title, result in (("Core", report.core), ("Stress", report.stress)):
        sections.append(
            "\n".join(
                (
                    f"{title} {label} reference: "
                    f"{'PASS' if result.passed else 'FAIL'}",
                    f"rows                     : {result.row_count}",
                    f"passed / failed          : {result.passed_row_count} / "
                    f"{result.failed_row_count}",
                    f"maximum absolute error   : "
                    f"{result.maximum_absolute_error:.12e} "
                    f"({result.maximum_absolute_error_row_id})",
                    f"mean signed error        : {result.mean_signed_error:.12e}",
                    "systematic bias          : "
                    f"{'yes' if result.systematic_bias else 'no'}",
                )
            )
        )
    sections.append(f"Dataset verified: {'true' if report.verified else 'false'}")
    return "\n\n".join(sections)


__all__ = (
    "ReferenceDatasetValidation",
    "ReferencePrice",
    "build_reference_document",
    "compare_reference_dataset",
    "compare_reference_prices",
    "format_reference_validation",
    "refresh_validation_policy_fingerprint",
    "source_fingerprints",
    "synchronize_catalog_validation",
    "validate_published_reference",
    "validate_reference_document",
    "validation_policy_fingerprint",
    "verification_document",
    "write_reference_document",
)
