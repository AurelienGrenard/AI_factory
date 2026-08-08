"""Validate one Black-Scholes discrete path dataset against a Premia bound."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.premia.model.equity.black_scholes.path_option import (
    format_bound_report,
    validation_from_premia_black_scholes_path_option,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("price_dataset", type=Path)
    parser.add_argument("product_kind")
    parser.add_argument("--regime", choices=("all", "core", "stress"), default="all")
    arguments = parser.parse_args()
    report = validation_from_premia_black_scholes_path_option(
        arguments.price_dataset,
        arguments.product_kind,
        regime=arguments.regime,
    )
    print(format_bound_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
