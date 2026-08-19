"""Validate one bates double knock out put dataset."""

from validation.model.equity.bates.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("double_knock_out_put"))
