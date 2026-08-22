"""Validate one CIR European receiver-swaption price dataset."""

from validation.model.fixed_income.swaption import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("cir", "receiver"))
