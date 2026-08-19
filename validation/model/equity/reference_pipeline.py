"""Shared persistence contract for equity reference-price datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from validation.quantlib.price_validation import ValidationTolerances
from validation.reference_price_dataset import (
    ReferenceDatasetValidation,
    ReferencePrice,
    build_reference_document,
    compare_reference_prices,
    format_reference_validation,
    synchronize_catalog_validation,
    validate_published_reference,
    write_reference_document,
)


REGIME_ROW_COUNTS = (("core", 900), ("stress", 100))
ReferenceGenerator = Callable[[Path, Path], ReferenceDatasetValidation]


def detailed_pricer(
    identifier: str,
    backend: str,
    backend_version: str,
    kind: str,
    method: str,
    row_priced: int,
) -> dict[str, Any]:
    """Build metadata for an engine that actually supplied reference rows."""

    return {
        "status": "available",
        "id": identifier,
        "backend": backend,
        "backend_version": backend_version,
        "kind": kind,
        "method": method,
        "row_priced": row_priced,
    }


def persist_generated_reference(
    source_price_dataset: str | Path,
    reference_price_dataset: str | Path,
    prices: Sequence[ReferencePrice],
    reference_pricers: Mapping[str, Any],
    wall_seconds: float,
    tolerances: ValidationTolerances = ValidationTolerances(),
    allow_systematic_bias: bool = False,
    systematic_bias_explanation: str | None = None,
) -> ReferenceDatasetValidation:
    """Verify generated prices, build the stable envelope, and write the cache."""

    report = compare_reference_prices(
        source_price_dataset,
        prices,
        tolerances,
        allow_systematic_bias,
        systematic_bias_explanation,
    )
    document = build_reference_document(
        source_price_dataset,
        reference_price_dataset,
        prices,
        reference_pricers,
        report,
        tolerances,
        wall_seconds,
    )
    write_reference_document(document, reference_price_dataset)
    return report


def validate_cached_reference(
    source_price_dataset: str | Path,
    reference_price_dataset: str | Path,
) -> ReferenceDatasetValidation:
    """Fail closed on cache, fingerprints, verification, or YAML drift."""

    return validate_published_reference(
        source_price_dataset, reference_price_dataset
    )


def run_reference_cli(
    description: str,
    label: str,
    generator: ReferenceGenerator,
) -> int:
    """Expose explicit regeneration and cache-only validation uniformly."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("price_dataset", type=Path)
    parser.add_argument("reference_dataset", type=Path)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Rerun external reference engines and replace the cache.",
    )
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Return a non-zero status when either regime is not verified.",
    )
    arguments = parser.parse_args()
    if arguments.generate:
        report = generator(arguments.price_dataset, arguments.reference_dataset)
        synchronize_catalog_validation(
            arguments.price_dataset,
            arguments.reference_dataset,
            report.verified,
        )
        report = validate_cached_reference(
            arguments.price_dataset, arguments.reference_dataset
        )
    else:
        report = validate_cached_reference(
            arguments.price_dataset, arguments.reference_dataset
        )
    print(format_reference_validation(report, label))
    return int(arguments.require_verified and not report.verified)


__all__ = (
    "REGIME_ROW_COUNTS",
    "detailed_pricer",
    "persist_generated_reference",
    "run_reference_cli",
    "validate_cached_reference",
)
