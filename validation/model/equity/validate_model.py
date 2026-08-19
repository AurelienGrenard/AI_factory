"""Common bulk runner for one published equity-model validation tree."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
from pathlib import Path

from validation.model.equity.stochastic_equity import product_folder
from validation.reporting import (
    synchronize_validation_yaml,
    write_validation_notebook,
    write_validation_report,
)


def _validate_product(
    root: Path,
    model_name: str,
    product_kind: str,
) -> tuple[str, str]:
    validation = importlib.import_module(
        f"validation.model.equity.{model_name}.validation"
    )
    folder = product_folder(product_kind)
    database_id = f"{model_name}_01__{folder}_01__01"
    dataset = (
        root
        / "datasets/price/equity"
        / model_name
        / folder
        / f"{database_id}.json"
    )
    catalog = (
        root / "catalog/price/equity" / model_name / folder / database_id
    )
    report = validation.validate_dataset(dataset, product_kind)
    write_validation_report(report, catalog / "validation_report.json")
    synchronize_validation_yaml(report, dataset)
    write_validation_notebook(
        report,
        dataset,
        catalog / "validation.ipynb",
        model_name.replace("_", " ").title(),
        product_kind.replace("_", " ").title(),
    )
    status = (
        "passed"
        if report.passed
        else "not_available"
        if report.core.status == report.stress.status == "not_available"
        else "failed"
    )
    return product_kind, status


def run_all_validations(
    model_name: str,
    product_kinds: frozenset[str],
) -> int:
    """Validate and publish every requested product of one equity model."""

    parser = argparse.ArgumentParser(
        description=f"Regenerate every persisted {model_name} validation artifact."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of independent datasets validated concurrently",
    )
    parser.add_argument(
        "--product",
        action="append",
        choices=sorted(product_kinds),
        help="validate only this product; may be repeated",
    )
    arguments = parser.parse_args()
    if arguments.jobs < 1:
        parser.error("--jobs must be strictly positive")

    root = Path(__file__).resolve().parents[3]
    products = arguments.product or sorted(product_kinds)
    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=arguments.jobs) as executor:
        futures = {
            executor.submit(_validate_product, root, model_name, product): product
            for product in products
        }
        for future in as_completed(futures):
            product_kind, status = future.result()
            print(f"{product_kind}: {status.upper()}", flush=True)
            if status == "failed":
                failed.append(product_kind)
    if failed:
        print(f"Failed {model_name} validations: " + ", ".join(sorted(failed)))
        return 1
    return 0


__all__ = ("run_all_validations",)
