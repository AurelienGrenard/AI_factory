"""Validate Heston European calls against QuantLib's analytic engine."""

from validation.quantlib.model.equity.heston.equity_option import (
    validation_from_quantlib_heston_option,
)
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli


def validation_from_quantlib(path, tolerances=ValidationTolerances()):
    """Compare every generated European-call price with QuantLib."""

    return validation_from_quantlib_heston_option(
        path, "european_call", tolerances
    )


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
