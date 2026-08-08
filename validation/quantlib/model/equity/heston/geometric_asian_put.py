"""Validate Heston geometric Asian puts against independent QuantLib MC."""

from validation.quantlib.model.equity.heston.equity_option import (
    validation_from_quantlib_heston_option,
)
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli


_TOLERANCES = ValidationTolerances(absolute=2.0e-5, relative=2.0e-3)


def validation_from_quantlib(path, tolerances=_TOLERANCES):
    """Compare every geometric-Asian-put price with independent QuantLib MC."""

    return validation_from_quantlib_heston_option(
        path, "geometric_asian_put", tolerances
    )


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
