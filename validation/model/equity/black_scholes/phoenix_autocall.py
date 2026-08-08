"""Validate one Black-Scholes phoenix-autocall price dataset."""

from validation.model.equity.black_scholes.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("phoenix_autocall"))
