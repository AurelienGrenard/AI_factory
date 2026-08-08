"""Validate one deterministic Black-Scholes price dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.quantlib.model.equity.black_scholes.equity_option import (
    validation_from_quantlib_black_scholes_option,
)
from validation.quantlib.price_validation import format_validation_report


def main() -> int:
    """Parse the product kind and print the common validation report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("price_dataset", type=Path)
    parser.add_argument("product_kind")
    parser.add_argument("--regime", choices=("all", "core", "stress"), default="all")
    parser.add_argument("--row-id", action="append")
    arguments = parser.parse_args()
    report = validation_from_quantlib_black_scholes_option(
        arguments.price_dataset,
        arguments.product_kind,
        regime=arguments.regime,
        row_ids=arguments.row_id,
    )
    print(format_validation_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
