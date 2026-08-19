"""Validate one Black-Scholes double-knock-out-put price dataset."""

from validation.model.equity.black_scholes.reference_pipeline import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("double_knock_out_put"))
