"""Validate CEV European calls against QuantLib's analytic CEV engine."""

import QuantLib as ql

from validation.quantlib.model.equity.cev.european_option import (
    validation_from_quantlib_cev_option,
)
from validation.quantlib.price_validation import (
    ValidationTolerances,
    run_validation_cli,
)

# A finite MC sample can contain no payoff in a very rare OTM tail. This
# model-specific floor remains far below economically material dataset prices.
_TOLERANCES = ValidationTolerances(absolute=2.0e-6)


def validation_from_quantlib(path, tolerances=_TOLERANCES):
    return validation_from_quantlib_cev_option(path, ql.Option.Call, tolerances)


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
