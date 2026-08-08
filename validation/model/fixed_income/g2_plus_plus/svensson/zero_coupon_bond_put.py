"""Validate one g2_plus_plus/svensson zero coupon bond put dataset."""

from validation.model.fixed_income.fitted_gaussian_rate import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(
        run_product_validation_cli("g2_plus_plus", "svensson", "zero_coupon_bond_put")
    )
