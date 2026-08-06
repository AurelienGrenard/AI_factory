"""Compare discrete up one-touches with QuantLib's continuous Heston PDE."""

from validation.quantlib.model.equity.heston.equity_option import (
    validation_from_quantlib_heston_option,
)
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli


_TOLERANCES = ValidationTolerances(absolute=3.0e-3, relative=1.5e-1)


def validation_from_quantlib(path, tolerances=_TOLERANCES):
    """Compare every maturity-paid up one-touch with QuantLib."""

    return validation_from_quantlib_heston_option(path, "up_one_touch", tolerances)


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
