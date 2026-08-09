"""Validate Premia-covered rows and fall back row-wise to QuantLib MC."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.premia.model.equity.black_scholes.path_option import (
    format_bound_report,
    validation_from_premia_black_scholes_path_option,
)
from validation.quantlib.model.equity.black_scholes.equity_option import (
    validation_from_quantlib_black_scholes_option,
)
from validation.quantlib.price_validation import (
    ValidationTolerances,
    format_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("price_dataset", type=Path)
    parser.add_argument("product_kind")
    parser.add_argument("--regime", choices=("core", "stress"), required=True)
    arguments = parser.parse_args()

    premia = validation_from_premia_black_scholes_path_option(
        arguments.price_dataset,
        arguments.product_kind,
        regime=arguments.regime,
    )
    print(format_bound_report(premia))
    if premia.failed_row_count != 0:
        print("Hybrid validation: FAIL (finite Premia result violated the bound)")
        return 1
    failed_row_ids = [row_id for row_id, _, _ in premia.backend_failure_rows]
    if not failed_row_ids:
        print("Hybrid validation: PASS (Premia covered every requested row)")
        return 0

    print("QuantLib row-level fallback:")
    quantlib = validation_from_quantlib_black_scholes_option(
        arguments.price_dataset,
        arguments.product_kind,
        ValidationTolerances(),
        regime=arguments.regime,
        row_ids=failed_row_ids,
    )
    print(format_validation_report(quantlib))
    print(
        "Hybrid validation: "
        + ("PASS" if quantlib.passed else "FAIL")
        + f" (QuantLib fallback rows: {len(failed_row_ids)})"
    )
    return 0 if quantlib.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
