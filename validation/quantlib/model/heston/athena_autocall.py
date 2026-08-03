"""Validate Heston Athena autocalls against independent QuantLib paths."""

from validation.quantlib.model.heston.equity_option import (
    validation_from_quantlib_heston_option,
)
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli


_TOLERANCES = ValidationTolerances(
    absolute=5.0e-4,
    relative=2.0e-3,
    standard_error_multiplier=5.0,
    bias_standard_errors=5.0,
)


def validation_from_quantlib(path, tolerances=_TOLERANCES):
    """Compare every Athena-autocall price with independent QuantLib MC."""

    return validation_from_quantlib_heston_option(
        path, "athena_autocall", tolerances
    )


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
