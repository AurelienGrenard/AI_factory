"""Validate one kou digital put dataset."""

from validation.model.equity.kou.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("digital_put"))
