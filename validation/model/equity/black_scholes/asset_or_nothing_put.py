"""Validate one Black-Scholes asset-or-nothing-put price dataset."""

from validation.model.equity.black_scholes.reference_pipeline import run_product_validation_cli


if __name__ == "__main__":
    raise SystemExit(run_product_validation_cli("asset_or_nothing_put"))
