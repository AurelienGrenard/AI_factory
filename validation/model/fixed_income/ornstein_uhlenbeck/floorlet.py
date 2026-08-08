"""Validate one Ornstein-Uhlenbeck floorlet price dataset."""

from validation.model.fixed_income.gaussian_rate import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(
        run_product_validation_cli("ornstein_uhlenbeck", "floorlet")
    )
