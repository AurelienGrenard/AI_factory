"""Validate one Hull-White/Svensson European payer-swaption price dataset."""

from validation.model.fixed_income.swaption import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(
        run_product_validation_cli("hull_white", "payer", "svensson")
    )
