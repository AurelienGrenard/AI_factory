"""Validate Merton European-call datasets against Premia's closed formula."""

from validation.premia.model.equity.merton.european_option import (
    validation_from_premia_merton_option,
)
from validation.premia.price_validation import (
    ValidationTolerances,
    run_premia_validation_cli,
)


def validation_from_premia(path, tolerances=ValidationTolerances()):
    """Compare every generated European-call price with Premia."""

    return validation_from_premia_merton_option(path, "call", tolerances)


if __name__ == "__main__":
    raise SystemExit(run_premia_validation_cli(validation_from_premia, __doc__))
