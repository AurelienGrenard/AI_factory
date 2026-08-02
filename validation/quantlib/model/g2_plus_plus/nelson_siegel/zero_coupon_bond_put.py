"""Validate G2++/Nelson-Siegel zero-coupon bond puts."""

from validation.quantlib.model.g2_plus_plus.nelson_siegel.reference import quantlib_model
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli
from validation.quantlib.rate_option import validation_from_quantlib_rate_option


def validation_from_quantlib(path, tolerances=ValidationTolerances()):
    return validation_from_quantlib_rate_option(
        path, quantlib_model, "zero_coupon_bond_put", True, tolerances
    )


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
