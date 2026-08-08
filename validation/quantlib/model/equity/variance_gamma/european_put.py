"""Validate Variance-Gamma European puts against QuantLib's analytic engine."""

from validation.quantlib.model.equity.variance_gamma.equity_option import (
    validation_from_quantlib_variance_gamma_option,
)
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli


def validation_from_quantlib(path, tolerances=ValidationTolerances()):
    """Compare every generated European-put price with QuantLib."""

    return validation_from_quantlib_variance_gamma_option(path, "european_put", tolerances)


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
