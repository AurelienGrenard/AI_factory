"""Validate Vasicek caplets against the QuantLib Vasicek bond-put identity."""

from validation.quantlib.model.vasicek.reference import quantlib_model
from validation.quantlib.price_validation import ValidationTolerances, run_validation_cli
from validation.quantlib.rate_option import validation_from_quantlib_rate_option


def validation_from_quantlib(price_dataset_path, tolerances=ValidationTolerances()):
    """Validate every Vasicek caplet row."""

    return validation_from_quantlib_rate_option(
        price_dataset_path, quantlib_model, "caplet", False, tolerances
    )


if __name__ == "__main__":
    raise SystemExit(run_validation_cli(validation_from_quantlib, __doc__))
