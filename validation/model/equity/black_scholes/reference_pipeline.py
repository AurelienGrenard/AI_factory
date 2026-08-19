"""Cache-only publication interface for Black-Scholes price references."""

from __future__ import annotations

from pathlib import Path

from validation.model.equity.reference_pipeline import (
    run_reference_cli,
    validate_cached_reference,
)
from validation.reference_price_dataset import ReferenceDatasetValidation


PRODUCT_KINDS = frozenset(
    {
        "asian_call",
        "asian_put",
        "asset_or_nothing_call",
        "asset_or_nothing_put",
        "athena_autocall",
        "cliquet",
        "digital_call",
        "digital_put",
        "double_knock_out_call",
        "double_knock_out_put",
        "down_and_in_put",
        "down_and_out_put",
        "european_call",
        "european_put",
        "forward_start_call",
        "forward_start_put",
        "gap_call",
        "gap_put",
        "geometric_asian_call",
        "geometric_asian_put",
        "lookback_option",
        "phoenix_autocall",
        "phoenix_memory_autocall",
        "range_accrual",
        "straddle",
        "up_and_in_call",
        "up_and_out_call",
        "up_no_touch",
        "up_one_touch",
    }
)


def validate_product_kind(product_kind: str) -> None:
    """Reject names outside the Black-Scholes publication set."""

    if product_kind not in PRODUCT_KINDS:
        raise ValueError(f"Unsupported Black-Scholes product '{product_kind}'.")


def product_folder(product_kind: str) -> str:
    """Map a singular API product name to its catalogue directory."""

    validate_product_kind(product_kind)
    return {
        "up_no_touch": "up_no_touches",
        "up_one_touch": "up_one_touches",
    }.get(product_kind, product_kind + "s")


def validate_dataset(
    source_price_dataset: str | Path,
    reference_price_dataset: str | Path,
) -> ReferenceDatasetValidation:
    """Validate Black-Scholes prices without importing an external backend."""

    return validate_cached_reference(
        source_price_dataset, reference_price_dataset
    )


def run_product_validation_cli(product_kind: str) -> int:
    """Use the cache by default and import external engines only on generation."""

    validate_product_kind(product_kind)

    def generate(source: Path, destination: Path) -> ReferenceDatasetValidation:
        from validation.model.equity.black_scholes.validation import (
            generate_reference_dataset,
        )

        return generate_reference_dataset(source, destination, product_kind)

    return run_reference_cli(
        f"Validate one Black-Scholes {product_kind} dataset.",
        "Black-Scholes",
        generate,
    )


__all__ = (
    "PRODUCT_KINDS",
    "product_folder",
    "run_product_validation_cli",
    "validate_dataset",
    "validate_product_kind",
)
