"""Validate one kou forward start put dataset."""

from validation.model.equity.kou.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("forward_start_put"))
