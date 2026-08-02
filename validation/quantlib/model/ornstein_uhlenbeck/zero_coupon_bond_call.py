"""Validate OU zero-coupon bond calls against QuantLib's Vasicek formula."""

from validation.quantlib.model.ornstein_uhlenbeck.reference import quantlib_model
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli
from validation.quantlib.rate_option import validation_from_quantlib_rate_option


def validation_from_quantlib(path, tolerances=ValidationTolerances()):
    """Compare every generated OU bond-call price with QuantLib Vasicek."""

    return validation_from_quantlib_rate_option(
        path, quantlib_model, "zero_coupon_bond_call", False, tolerances
    )


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
