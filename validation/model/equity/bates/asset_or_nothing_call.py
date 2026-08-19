"""Validate one bates asset or nothing call dataset."""

from validation.model.equity.bates.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("asset_or_nothing_call"))
