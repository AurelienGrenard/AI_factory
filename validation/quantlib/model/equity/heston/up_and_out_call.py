"""Validate discrete Heston up-and-out calls with independent QuantLib paths."""

from validation.quantlib.model.equity.heston.equity_option import (
    validation_from_quantlib_heston_option,
)
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli


_TOLERANCES = ValidationTolerances(absolute=2.0e-3, relative=1.5e-1)


def validation_from_quantlib(path, tolerances=_TOLERANCES):
    """Compare every discretely monitored up-and-out-call price with QuantLib."""

    return validation_from_quantlib_heston_option(
        path, "up_and_out_call", tolerances
    )


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
