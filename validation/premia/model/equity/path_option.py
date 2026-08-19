"""Premia references for path-dependent Merton and Kou equity options."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from validation.hierarchy import (
    BackendBatchResult,
    BackendException,
)
from validation.premia.bridge import (
    PremiaInput,
    PremiaResult,
    price_rows_partial,
)
from validation.premia.model.equity.black_scholes.path_option import (
    DirectionalValidationReport,
    summarize_directional_comparisons,
)
from validation.premia.model.equity.terminal_option import premia_model_prefix
from validation.premia.price_validation import (
    PremiaPriceComparison,
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
    load_parameter_rows,
    load_price_validation_input,
    parameter_number,
    select_validation_regime,
    select_validation_row_ids,
    summarize_premia_comparisons,
)


_DIRECT_PRODUCTS = {
    "merton": {
        "asian_call": ("merton_asian_call", ("strike", "maturity")),
        "asian_put": ("merton_asian_put", ("strike", "maturity")),
        "lookback_option": (
            "merton_lookback_option",
            ("strike", "maturity"),
        ),
        "up_and_out_call": (
            "merton_up_and_out_call",
            ("strike", "maturity", "barrier"),
        ),
        "down_and_out_put": (
            "merton_down_and_out_put",
            ("strike", "maturity", "barrier"),
        ),
    },
    "kou": {
        "asian_call": ("kou_asian_call", ("strike", "maturity")),
        "asian_put": ("kou_asian_put", ("strike", "maturity")),
        "lookback_option": ("kou_lookback_option", ("strike", "maturity")),
        "up_and_out_call": (
            "kou_up_and_out_call",
            ("strike", "maturity", "barrier"),
        ),
        "up_and_in_call": (
            "kou_up_and_in_call",
            ("strike", "maturity", "barrier"),
        ),
        "down_and_out_put": (
            "kou_down_and_out_put",
            ("strike", "maturity", "barrier"),
        ),
        "down_and_in_put": (
            "kou_down_and_in_put",
            ("strike", "maturity", "barrier"),
        ),
    },
    "heston": {
        "american_call": ("heston_american_call", ("strike", "maturity")),
        "american_put": ("heston_american_put", ("strike", "maturity")),
        "asian_call": ("heston_asian_call", ("strike", "maturity")),
        "up_and_out_call": (
            "heston_up_and_out_call",
            ("strike", "maturity", "barrier"),
        ),
        "down_and_out_put": (
            "heston_down_and_out_put",
            ("strike", "maturity", "barrier"),
        ),
    },
    "bates": {
        "american_call": ("bates_american_call", ("strike", "maturity")),
        "american_put": ("bates_american_put", ("strike", "maturity")),
        "asian_call": ("bates_asian_call", ("strike", "maturity")),
        "asian_put": ("bates_asian_put", ("strike", "maturity")),
        "up_and_out_call": (
            "bates_up_and_out_call",
            ("strike", "maturity", "barrier"),
        ),
        "down_and_out_put": (
            "bates_down_and_out_put",
            ("strike", "maturity", "barrier"),
        ),
    },
}

_IN_PARITY = {
    model_name: {
        "up_and_in_call": ("up_and_out_call", "call"),
        "down_and_in_put": ("down_and_out_put", "put"),
    }
    for model_name in ("merton", "heston", "bates")
}


def _combined_result(vanilla: PremiaResult, out: PremiaResult) -> PremiaResult:
    return PremiaResult(
        price=vanilla.price - out.price,
        standard_error=math.hypot(
            vanilla.standard_error, out.standard_error
        ),
    )


def _workbench_average_expectation(
    spot: float,
    rate: float,
    dividend: float,
    maturity: float,
    step_count: int,
) -> float:
    """Return E[mean(S)] on the published endpoint grid, including time zero."""

    growth_per_step = (rate - dividend) * maturity / step_count
    if abs(growth_per_step) < 1.0e-8:
        return spot
    return (
        spot
        * math.expm1((step_count + 1) * growth_per_step)
        / ((step_count + 1) * math.expm1(growth_per_step))
    )


def validation_batch_from_premia_path_option(
    price_dataset_path: str | Path,
    model_name: str,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
    method: str | None = None,
    asian_put_via_call_parity: bool = True,
) -> BackendBatchResult:
    """Price one Premia batch once and preserve its row-local failures."""

    if model_name not in _DIRECT_PRODUCTS:
        raise ValueError(f"Unsupported Premia path model '{model_name}'.")
    direct_kind = product_kind
    parity_side = None
    if product_kind in _IN_PARITY.get(model_name, {}):
        direct_kind, parity_side = _IN_PARITY[model_name][product_kind]
    asian_put_parity = (
        asian_put_via_call_parity
        and model_name in {"merton", "kou", "heston"}
        and product_kind == "asian_put"
    )
    if asian_put_parity:
        direct_kind = "asian_call"
    try:
        mode, fields = _DIRECT_PRODUCTS[model_name][direct_kind]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Premia {model_name} path product '{product_kind}'."
        ) from error

    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(
        validation_input.product_dataset_path, "products"
    )
    path_inputs: list[PremiaInput] = []
    vanilla_inputs: list[PremiaInput] = []
    asian_step_counts: dict[str, int] = {}
    for row in validation_input.rows:
        model = models[row.model_id]
        product = products[row.product_id]
        context = f"{product_kind} row '{row.product_id}'"
        suffix = tuple(
            parameter_number(product, field, context, positive=True)
            for field in fields
        )
        prefix = premia_model_prefix(model, model_name, row.model_id)
        if model_name == "kou" and product_kind in {"asian_call", "asian_put"}:
            # The published Kou Asian datasets use exact increments on the
            # nearest 1/252 endpoint grid.  Premia receives the same number of
            # future monitoring dates; its separate initial-average field
            # already represents S(0).
            asian_step_counts[row.row_id] = max(
                1, math.floor(252.0 * suffix[1] + 0.5)
            )
            suffix = (*suffix, float(asian_step_counts[row.row_id]))
        path_inputs.append(PremiaInput(row.row_id, (*prefix, *suffix)))
        if parity_side is not None:
            vanilla_inputs.append(
                PremiaInput(row.row_id, (*prefix, suffix[0], suffix[1]))
            )

    references, path_failures = price_rows_partial(path_inputs, mode, method)
    exceptions = {
        row_id: BackendException(
            row_id,
            f"Premia row '{row_id}' failed with status {failure.status}: "
            f"{failure.reason}.",
            failure.status,
        )
        for row_id, failure in path_failures.items()
    }
    if parity_side is not None:
        vanilla, vanilla_failures = price_rows_partial(
            vanilla_inputs, f"{model_name}_european_{parity_side}"
        )
        exceptions.update(
            {
                row_id: BackendException(
                    row_id,
                    f"Premia row '{row_id}' failed with status "
                    f"{failure.status}: {failure.reason}.",
                    failure.status,
                )
                for row_id, failure in vanilla_failures.items()
            }
        )
        references = {
            row_id: _combined_result(vanilla[row_id], references[row_id])
            for row_id in references
            if row_id in vanilla
        }
    if asian_put_parity:
        adjusted: dict[str, PremiaResult] = {}
        for row in validation_input.rows:
            if row.row_id not in references:
                continue
            model = models[row.model_id]
            product = products[row.product_id]
            context = f"asian_put row '{row.product_id}'"
            maturity = parameter_number(
                product, "maturity", context, positive=True
            )
            strike = parameter_number(
                product, "strike", context, positive=True
            )
            spot = parameter_number(
                model, "spot", f"model '{row.model_id}'", positive=True
            )
            rate = parameter_number(
                model, "risk_free_rate", f"model '{row.model_id}'"
            )
            dividend = parameter_number(
                model, "dividend_yield", f"model '{row.model_id}'"
            )
            expected_average = _workbench_average_expectation(
                spot,
                rate,
                dividend,
                maturity,
                asian_step_counts.get(
                    row.row_id, max(1, math.floor(360.0 * maturity + 0.5))
                ),
            )
            discount = math.exp(-rate * maturity)
            call = references[row.row_id]
            adjusted[row.row_id] = PremiaResult(
                price=call.price - discount * expected_average + discount * strike,
                standard_error=call.standard_error,
            )
        references = adjusted
    if product_kind in {"asian_call", "asian_put"}:
        for row in validation_input.rows:
            if row.row_id not in references:
                continue
            model = models[row.model_id]
            product = products[row.product_id]
            context = f"{product_kind} row '{row.product_id}'"
            maturity = parameter_number(
                product, "maturity", context, positive=True
            )
            strike = parameter_number(
                product, "strike", context, positive=True
            )
            spot = parameter_number(
                model, "spot", f"model '{row.model_id}'", positive=True
            )
            rate = parameter_number(
                model, "risk_free_rate", f"model '{row.model_id}'"
            )
            dividend = parameter_number(
                model, "dividend_yield", f"model '{row.model_id}'"
            )
            discount = math.exp(-rate * maturity)
            upper = discount * (
                _workbench_average_expectation(
                    spot,
                    rate,
                    dividend,
                    maturity,
                    asian_step_counts.get(
                        row.row_id,
                        max(1, math.floor(360.0 * maturity + 0.5)),
                    ),
                )
                if product_kind == "asian_call"
                else strike
            )
            reference = references[row.row_id].price
            if reference < -1.0e-8 or reference > upper + 1.0e-6:
                exceptions[row.row_id] = BackendException(
                    row.row_id,
                    f"Premia row '{row.row_id}' failed with status 14: "
                    "Asian price violates analytic no-arbitrage bounds.",
                    14,
                )
                del references[row.row_id]
    comparisons = tuple(
        PremiaPriceComparison(
            row_id=row.row_id,
            generated_price=row.generated_price,
            premia_price=references[row.row_id].price,
            generated_standard_error=row.generated_standard_error,
            premia_standard_error=references[row.row_id].standard_error,
        )
        for row in validation_input.rows
        if row.row_id in references
    )
    directional_product = product_kind in {
        "up_and_out_call",
        "down_and_out_put",
        "up_and_in_call",
        "down_and_in_put",
        "lookback_option",
        "american_call",
        "american_put",
    }
    if model_name == "bates" and product_kind in {
        "up_and_out_call",
        "down_and_out_put",
        "up_and_in_call",
        "down_and_in_put",
    }:
        directional_product = False
    if model_name == "merton" and product_kind == "lookback_option":
        directional_product = False
    report: PriceValidationReport | DirectionalValidationReport | None = None
    if directional_product and comparisons:
        relation = (
            "generated_at_most_reference"
            if product_kind in {
                "up_and_in_call",
                "down_and_in_put",
                "lookback_option",
                "american_call",
                "american_put",
            }
            else "generated_at_least_reference"
        )
        report = summarize_directional_comparisons(
            validation_input.database_id,
            comparisons,
            relation,
            tolerances,
        )
    elif comparisons:
        report = summarize_premia_comparisons(
            validation_input.database_id, comparisons, tolerances
        )
    completed = tuple(
        row.row_id for row in validation_input.rows if row.row_id in references
    )
    failed = tuple(
        exceptions[row.row_id]
        for row in validation_input.rows
        if row.row_id in exceptions
    )
    return BackendBatchResult(
        completed_row_ids=completed,
        exceptions=failed,
        reports=() if report is None else (report,),
    )


def validation_from_premia_path_option(
    price_dataset_path: str | Path,
    model_name: str,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
    method: str | None = None,
    asian_put_via_call_parity: bool = True,
) -> PriceValidationReport | DirectionalValidationReport:
    """Standalone all-or-nothing facade used by backend diagnostics."""

    result = validation_batch_from_premia_path_option(
        price_dataset_path,
        model_name,
        product_kind,
        tolerances,
        regime,
        row_ids,
        method,
        asian_put_via_call_parity,
    )
    if result.exceptions:
        raise RuntimeError(result.exceptions[0].reason)
    if len(result.reports) != 1:
        raise RuntimeError("Premia path validation returned no comparable rows.")
    return result.reports[0]


__all__ = (
    "validation_batch_from_premia_path_option",
    "validation_from_premia_path_option",
)
