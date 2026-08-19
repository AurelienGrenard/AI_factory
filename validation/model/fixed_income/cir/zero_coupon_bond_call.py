"""Validate one CIR zero-coupon bond call price dataset."""

from validation.model.fixed_income.cir_rate import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("zero_coupon_bond_call"))
