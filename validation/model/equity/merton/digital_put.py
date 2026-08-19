"""Validate one merton digital put dataset."""

from validation.model.equity.merton.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("digital_put"))
