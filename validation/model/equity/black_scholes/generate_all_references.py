"""Explicitly regenerate every persistent Black-Scholes price reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.model.equity.black_scholes.reference_pipeline import (
    PRODUCT_KINDS,
    product_folder,
)
from validation.model.equity.black_scholes.validation import (
    generate_reference_dataset,
)
from validation.reference_price_dataset import synchronize_catalog_validation


def generate_all(root: Path, selected_products: set[str]) -> None:
    """Regenerate selected products through their ordered reference hierarchy."""

    source_root = root / "datasets/model/equity/markovian/black_scholes/prices"
    destination_root = root / "validation/datasets/price/equity/black_scholes"
    for product_kind in sorted(selected_products):
        folder = product_folder(product_kind)
        stem = f"black_scholes_01__{folder}_01__01.json"
        source = source_root / folder / stem
        destination = destination_root / folder / stem
        report = generate_reference_dataset(source, destination, product_kind)
        synchronize_catalog_validation(source, destination, report.verified)
        if not report.verified:
            raise RuntimeError(
                f"Generated reference failed validation: {source}"
            )
        print(f"verified {destination.relative_to(root)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product",
        action="append",
        choices=sorted(PRODUCT_KINDS),
        help="Regenerate only this product; repeat to select several.",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    generate_all(root, set(arguments.product or PRODUCT_KINDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
