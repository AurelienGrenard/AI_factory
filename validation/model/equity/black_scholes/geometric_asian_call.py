"""Validate one Black-Scholes geometric-asian-call price dataset."""

from validation.model.equity.black_scholes.reference_pipeline import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("geometric_asian_call"))
