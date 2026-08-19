"""Validate one bates up one touch dataset."""

from validation.model.equity.bates.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("up_one_touch"))
