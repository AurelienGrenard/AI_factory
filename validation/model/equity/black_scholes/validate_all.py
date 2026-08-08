"""Regenerate every persisted Black-Scholes validation artifact."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from validation.model.equity.black_scholes.validation import (
    _PRODUCT_KINDS,
    validate_dataset,
)
from validation.reporting import (
    synchronize_validation_yaml,
    write_validation_report,
)


def _validate_product(root: Path, product_kind: str) -> tuple[str, bool]:
    folder = {
        "up_no_touch": "up_no_touches",
        "up_one_touch": "up_one_touches",
    }.get(product_kind, product_kind + "s")
    database_id = f"black_scholes_01__{folder}_01__01"
    dataset = (
        root / "datasets/price/equity/black_scholes" / folder
        / f"{database_id}.json"
    )
    catalog = root / "catalog/price/equity/black_scholes" / folder / database_id
    report = validate_dataset(dataset, product_kind)
    write_validation_report(report, catalog / "validation_report.json")
    synchronize_validation_yaml(report, dataset)
    return product_kind, report.passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of independent datasets validated concurrently",
    )
    parser.add_argument(
        "--product",
        action="append",
        choices=sorted(_PRODUCT_KINDS),
        help="validate only this product; may be repeated",
    )
    arguments = parser.parse_args()
    if arguments.jobs < 1:
        parser.error("--jobs must be strictly positive")

    root = Path(__file__).resolve().parents[4]
    products = arguments.product or sorted(_PRODUCT_KINDS)
    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=arguments.jobs) as executor:
        futures = {
            executor.submit(_validate_product, root, product): product
            for product in products
        }
        for future in as_completed(futures):
            product_kind, passed = future.result()
            status = "PASS" if passed else "FAIL"
            print(f"{product_kind}: {status}", flush=True)
            if not passed:
                failed.append(product_kind)
    if failed:
        print("Failed Black-Scholes validations: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
