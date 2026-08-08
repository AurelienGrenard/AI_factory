"""Validate one Black-Scholes cliquet price dataset."""

from validation.model.equity.black_scholes.validation import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("cliquet"))
