"""Validate one CIR floorlet price dataset."""

from validation.model.fixed_income.cir_rate import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("floorlet"))
