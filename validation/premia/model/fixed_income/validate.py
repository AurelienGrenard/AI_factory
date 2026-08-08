"""Validate one compatible fixed-income dataset against Premia."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.premia.model.fixed_income.rate_option import (
    validation_from_premia_rate_option,
)
from validation.premia.price_validation import format_premia_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("price_dataset", type=Path)
    parser.add_argument("model")
    parser.add_argument("product")
    arguments = parser.parse_args()
    report = validation_from_premia_rate_option(
        arguments.price_dataset,
        arguments.model,
        arguments.product,
    )
    print(format_premia_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
