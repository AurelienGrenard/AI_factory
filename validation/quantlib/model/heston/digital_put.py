"""Validate Heston digital puts against QuantLib's Heston distribution."""

from validation.quantlib.model.heston.equity_option import (
    validation_from_quantlib_heston_option,
)
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli


_TOLERANCES = ValidationTolerances(absolute=2.0e-4, relative=2.0e-3)


def validation_from_quantlib(path, tolerances=_TOLERANCES):
    """Compare every generated digital-put price with QuantLib."""

    return validation_from_quantlib_heston_option(path, "digital_put", tolerances)


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
