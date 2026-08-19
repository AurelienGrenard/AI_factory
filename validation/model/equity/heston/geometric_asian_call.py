"""Validate one heston geometric asian call dataset."""

from validation.model.equity.heston.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("geometric_asian_call"))
