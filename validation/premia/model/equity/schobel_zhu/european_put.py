"""Validate Schobel-Zhu European puts against Premia's Stein approximation."""

from validation.premia.model.equity.schobel_zhu.european_option import (
    validation_from_premia_schobel_zhu_option,
)
from validation.premia.price_validation import (
    ValidationTolerances,
    run_premia_validation_cli,
)


def validation_from_premia(path, tolerances=ValidationTolerances()):
    """Compare every generated European-put price with Premia."""

    return validation_from_premia_schobel_zhu_option(path, "put", tolerances)


if __name__ == "__main__":
    raise SystemExit(run_premia_validation_cli(validation_from_premia, __doc__))
