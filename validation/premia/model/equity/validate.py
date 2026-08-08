"""Validate one compatible terminal equity dataset against Premia."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.premia.model.equity.terminal_option import (
    validation_from_premia_terminal_option,
)
from validation.premia.price_validation import format_premia_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("price_dataset", type=Path)
    parser.add_argument("model")
    parser.add_argument("product")
    parser.add_argument("--vanilla-method")
    parser.add_argument("--digital-method")
    parser.add_argument("--regime", choices=("all", "core", "stress"), default="all")
    arguments = parser.parse_args()
    report = validation_from_premia_terminal_option(
        arguments.price_dataset,
        arguments.model,
        arguments.product,
        vanilla_method=arguments.vanilla_method,
        digital_method=arguments.digital_method,
        regime=arguments.regime,
    )
    print(format_premia_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
