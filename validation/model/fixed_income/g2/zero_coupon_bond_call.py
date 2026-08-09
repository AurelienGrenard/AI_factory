"""Validate one standalone G2 zero-coupon bond call price dataset."""

from validation.model.fixed_income.gaussian_rate import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("g2", "zero_coupon_bond_call"))
