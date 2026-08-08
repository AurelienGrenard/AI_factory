"""Shared Premia validation for Schobel-Zhu European calls and puts."""

from pathlib import Path

from validation.premia.bridge import (
    SchobelZhuEuropeanOptionInput,
    price_schobel_zhu_european_options,
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


def validation_from_premia_schobel_zhu_option(
    price_dataset_path: str | Path,
    option_type: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
) -> PriceValidationReport:
    """Compare Schobel-Zhu European options with Premia's approximation."""

    validation_input = load_price_validation_input(price_dataset_path)
    if validation_input.curve_dataset_path is not None:
        raise ValueError("Premia Schobel-Zhu validation expects no curve dataset.")
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(
        validation_input.product_dataset_path,
        "products",
    )

    premia_inputs: list[SchobelZhuEuropeanOptionInput] = []
    for row in validation_input.rows:
        try:
            model = models[row.model_id]
            product = products[row.product_id]
        except KeyError as error:
            raise ValueError(
                f"Price row '{row.row_id}': unknown source row id "
                f"'{error.args[0]}'."
            ) from error
        model_context = f"Schobel-Zhu model row '{row.model_id}'"
        product_context = f"European option row '{row.product_id}'"
        premia_inputs.append(
            SchobelZhuEuropeanOptionInput(
                row_id=row.row_id,
                spot=parameter_number(
                    model, "spot", model_context, positive=True
                ),
                risk_free_rate=parameter_number(
                    model,
                    "risk_free_rate",
                    model_context,
                ),
                dividend_yield=parameter_number(
                    model,
                    "dividend_yield",
                    model_context,
                ),
                initial_volatility=parameter_number(
                    model,
                    "initial_volatility",
                    model_context,
                    positive=True,
                ),
                mean_reversion=parameter_number(
                    model,
                    "mean_reversion",
                    model_context,
                    positive=True,
                ),
                long_run_volatility=parameter_number(
                    model,
                    "long_run_volatility",
                    model_context,
                    positive=True,
                ),
                volatility_of_volatility=parameter_number(
                    model,
                    "volatility_of_volatility",
                    model_context,
                    positive=True,
                ),
                correlation=parameter_number(
                    model,
                    "correlation",
                    model_context,
                ),
                strike=parameter_number(
                    product,
                    "strike",
                    product_context,
                    positive=True,
                ),
                maturity=parameter_number(
                    product,
                    "maturity",
                    product_context,
                    positive=True,
                ),
            )
        )

    premia_prices = price_schobel_zhu_european_options(
        premia_inputs,
        option_type,
    )
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
        validation_input.database_id,
        comparisons,
        tolerances,
    )
