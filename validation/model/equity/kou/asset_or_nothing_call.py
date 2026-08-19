"""Validate one kou asset or nothing call dataset."""

from validation.model.equity.kou.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("asset_or_nothing_call"))
