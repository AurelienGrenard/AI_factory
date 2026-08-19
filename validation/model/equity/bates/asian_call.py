"""Validate one bates asian call dataset."""

from validation.model.equity.bates.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("asian_call"))
