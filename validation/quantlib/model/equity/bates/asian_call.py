"""Validate Bates arithmetic Asian calls against QuantLib Monte Carlo."""

from validation.quantlib.model.equity.bates.equity_option import (
    validation_from_quantlib_bates_option,
)
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli


_TOLERANCES = ValidationTolerances(
    absolute=2.0e-5, relative=2.0e-3,
    standard_error_multiplier=5.0, bias_standard_errors=5.0,
)


def validation_from_quantlib(path, tolerances=_TOLERANCES):
    """Compare every generated Asian-call price with independent QuantLib MC."""

    return validation_from_quantlib_bates_option(path, "asian_call", tolerances)


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
