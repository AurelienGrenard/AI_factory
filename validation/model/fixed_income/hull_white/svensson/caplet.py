"""Validate one hull_white/svensson caplet dataset."""

from validation.model.fixed_income.fitted_gaussian_rate import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(
        run_product_validation_cli("hull_white", "svensson", "caplet")
    )
