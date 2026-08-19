"""Validate one heston athena autocall dataset."""

from validation.model.equity.heston.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("athena_autocall"))
