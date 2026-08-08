"""Shared Premia validation for Heston European calls and puts."""

from __future__ import annotations

from pathlib import Path

from validation.premia.bridge import (
    HestonEuropeanOptionInput,
    price_heston_european_options,
)
from validation.premia.price_validation import (
    PremiaPriceComparison,
    PriceValidationReport,
    ValidationTolerances,
    load_parameter_rows,
    load_price_validation_input,
    parameter_number,
    summarize_premia_comparisons,
)


def validation_from_premia_heston_option(
    price_dataset_path: str | Path,
    option_type: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
) -> PriceValidationReport:
    """Compare every generated Heston European option with Premia's formula."""

    validation_input = load_price_validation_input(price_dataset_path)
    if validation_input.curve_dataset_path is not None:
        raise ValueError("Premia Heston equity validation expects no curve dataset.")
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(validation_input.product_dataset_path, "products")

    premia_inputs: list[HestonEuropeanOptionInput] = []
    for row in validation_input.rows:
        try:
            model = models[row.model_id]
            product = products[row.product_id]
        except KeyError as error:
            raise ValueError(
                f"Price row '{row.row_id}': unknown source row id '{error.args[0]}'."
            ) from error
        model_context = f"Heston model row '{row.model_id}'"
        product_context = f"European option row '{row.product_id}'"
        initial_variance = parameter_number(
            model, "initial_variance", model_context
        )
        if initial_variance < 0.0:
            raise ValueError(f"{model_context}: initial_variance must be non-negative.")
        rho = parameter_number(model, "rho", model_context)
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"{model_context}: rho must lie in [-1, 1].")
        premia_inputs.append(
            HestonEuropeanOptionInput(
                row_id=row.row_id,
                spot=parameter_number(
                    model, "spot", model_context, positive=True
                ),
                risk_free_rate=parameter_number(
                    model, "risk_free_rate", model_context
                ),
                dividend_yield=parameter_number(
                    model, "dividend_yield", model_context
                ),
                initial_variance=initial_variance,
                kappa=parameter_number(
                    model, "kappa", model_context, positive=True
                ),
                theta=parameter_number(
                    model, "theta", model_context, positive=True
                ),
                gamma=parameter_number(
                    model, "gamma", model_context, positive=True
                ),
                rho=rho,
                strike=parameter_number(
                    product, "strike", product_context, positive=True
                ),
                maturity=parameter_number(
                    product, "maturity", product_context, positive=True
                ),
            )
        )

    premia_prices = price_heston_european_options(premia_inputs, option_type)
    comparisons = tuple(
        PremiaPriceComparison(
            row_id=row.row_id,
            generated_price=row.generated_price,
            premia_price=premia_prices[row.row_id],
            generated_standard_error=row.generated_standard_error,
        )
        for row in validation_input.rows
    )
    return summarize_premia_comparisons(
        validation_input.database_id, comparisons, tolerances
    )
