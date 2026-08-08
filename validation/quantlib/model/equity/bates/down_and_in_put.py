"""Validate discrete Bates down-and-in puts with independent QuantLib paths."""

from validation.quantlib.model.equity.bates.equity_option import (
    validation_from_quantlib_bates_option,
)
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli


_TOLERANCES = ValidationTolerances(
    absolute=2.0e-3, relative=1.5e-1,
    standard_error_multiplier=5.0, bias_standard_errors=5.0,
)


def validation_from_quantlib(path, tolerances=_TOLERANCES):
    """Compare every discretely monitored down-and-in put with QuantLib."""

    return validation_from_quantlib_bates_option(
        path, "down_and_in_put", tolerances
    )


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
