"""Adapters from AI_factory JSON datasets to independent Volterra references."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Literal

from validation.volterra.common import certify_price
from validation.volterra.rough_bergomi import (
    RoughBergomiParameters,
    exact_gaussian_european_option_price,
    hybrid_european_option_price,
)


JsonObject = dict[str, Any]


def _read_json(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON dataset {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON dataset {path} must contain an object.")
    return value


def _rows_by_id(document: JsonObject, key: str, path: Path) -> JsonObject:
    rows = document.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a '{key}' array.")
    indexed: JsonObject = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"Every {path}:{key} row must have a string id.")
        identifier = row["id"]
        if identifier in indexed:
            raise ValueError(f"Duplicate id {identifier!r} in {path}:{key}.")
        indexed[identifier] = row
    declared_count = document.get("row_count")
    if declared_count is not None and declared_count != len(rows):
        raise ValueError(
            f"{path} declares {declared_count} rows but contains {len(rows)}."
        )
    return indexed


def _time_convention(document: JsonObject, path: Path) -> tuple[str, float]:
    convention = document.get("time_convention")
    if not isinstance(convention, dict):
        raise ValueError(f"{path} has no time_convention object.")
    unit = convention.get("unit")
    days_per_year = convention.get("days_per_year")
    if not isinstance(unit, str):
        raise ValueError(f"{path} time convention has no string unit.")
    try:
        year_length = float(days_per_year)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path} time convention has no numerical days_per_year."
        ) from error
    if not math.isfinite(year_length) or year_length <= 0.0:
        raise ValueError(f"{path} days_per_year must be positive and finite.")
    return unit, year_length


def _maturity_in_years(value: Any, unit: str, days_per_year: float) -> float:
    try:
        maturity = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Product maturity must be numerical.") from error
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("Product maturity must be positive and finite.")
    if unit in {"business_day", "calendar_day", "day"}:
        return maturity / days_per_year
    if unit in {"year", "years"}:
        return maturity
    raise ValueError(f"Unsupported product maturity unit {unit!r}.")


def _parameter(parameters: JsonObject, name: str) -> float:
    try:
        value = float(parameters[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid rough-Bergomi parameter {name!r}.") \
            from error
    if not math.isfinite(value):
        raise ValueError(f"Rough-Bergomi parameter {name!r} must be finite.")
    return value


def _certification_document(value: Any) -> JsonObject:
    document = asdict(value)
    if not math.isfinite(document["z_score"]):
        document["z_score"] = None
    return document


def _reference_document(value: Any) -> JsonObject:
    return asdict(value)


def validate_rough_bergomi_dataset(
    price_path: Path,
    model_path: Path,
    product_path: Path,
    option_side: Literal["call", "put"],
    row_offset: int = 0,
    row_limit: int = 1,
    target_dt: float = 1.0 / 360.0,
    antithetic_pair_count: int = 16_384,
    batch_pair_count: int = 2048,
    seed: int = 4_820_001,
    include_exact_grid: bool = False,
    exact_maximum_step_count: int = 360,
) -> JsonObject:
    """Validate selected aligned price rows without importing production code."""

    if option_side not in {"call", "put"}:
        raise ValueError("option_side must be 'call' or 'put'.")
    if row_offset < 0 or row_limit < 1:
        raise ValueError("row_offset must be non-negative and row_limit positive.")
    if not math.isfinite(target_dt) or target_dt <= 0.0:
        raise ValueError("target_dt must be positive and finite.")
    if exact_maximum_step_count < 1:
        raise ValueError("exact_maximum_step_count must be positive.")

    price_document = _read_json(price_path)
    model_document = _read_json(model_path)
    product_document = _read_json(product_path)
    model_rows = _rows_by_id(model_document, "models", model_path)
    product_rows = _rows_by_id(product_document, "products", product_path)
    price_rows = price_document.get("results")
    if not isinstance(price_rows, list):
        raise ValueError(f"{price_path} must contain a 'results' array.")
    declared_count = price_document.get("row_count")
    if declared_count is not None and declared_count != len(price_rows):
        raise ValueError(
            f"{price_path} declares {declared_count} rows but contains "
            f"{len(price_rows)}."
        )

    product_unit, product_year_length = _time_convention(
        product_document, product_path
    )
    price_unit, price_year_length = _time_convention(price_document, price_path)
    if (product_unit, product_year_length) != (price_unit, price_year_length):
        raise ValueError("Price and product time conventions do not match.")

    selected = price_rows[row_offset : row_offset + row_limit]
    if not selected:
        raise ValueError("The requested row selection is empty.")
    output_rows: list[JsonObject] = []
    for selected_index, result in enumerate(selected):
        if not isinstance(result, dict):
            raise ValueError("Every price result must be an object.")
        result_id = result.get("id")
        model_id = result.get("model_id")
        product_id = result.get("product_id")
        if not all(isinstance(value, str) for value in (
            result_id, model_id, product_id
        )):
            raise ValueError(
                "Every price result must have string row/model/product ids."
            )
        try:
            model_row = model_rows[model_id]
            product_row = product_rows[product_id]
        except KeyError as error:
            raise ValueError(
                f"Price row {result_id} refers to unknown id {error.args[0]!r}."
            ) from error
        model_parameters = model_row.get("parameters")
        product_parameters = product_row.get("parameters")
        outputs = result.get("outputs")
        if not all(isinstance(value, dict) for value in (
            model_parameters, product_parameters, outputs
        )):
            raise ValueError(f"Price row {result_id} has malformed parameters.")

        parameters = RoughBergomiParameters(
            spot=_parameter(model_parameters, "spot"),
            risk_free_rate=_parameter(model_parameters, "risk_free_rate"),
            dividend_yield=_parameter(model_parameters, "dividend_yield"),
            xi_0=_parameter(model_parameters, "xi_0"),
            eta=_parameter(model_parameters, "eta"),
            hurst_exponent=_parameter(model_parameters, "hurst_exponent"),
            rho=_parameter(model_parameters, "rho"),
        )
        strike = _parameter(product_parameters, "strike")
        maturity = _maturity_in_years(
            product_parameters.get("maturity"),
            product_unit,
            product_year_length,
        )
        step_count = max(1, math.floor(maturity / target_dt + 0.5))
        try:
            generated_price = float(outputs["price"])
            generated_standard_error = float(outputs["standard_error"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Price row {result_id} has invalid price outputs."
            ) from error

        row_seed = seed + row_offset + selected_index
        hybrid = hybrid_european_option_price(
            parameters,
            strike,
            maturity,
            option_side,
            step_count,
            antithetic_pair_count,
            row_seed,
            batch_pair_count,
            "fft",
        )
        row: JsonObject = {
            "id": result_id,
            "model_id": model_id,
            "product_id": product_id,
            "maturity_years": maturity,
            "strike": strike,
            "time_steps": step_count,
            "effective_dt": maturity / step_count,
            "generated": {
                "price": generated_price,
                "standard_error": generated_standard_error,
            },
            "hybrid_numpy_fft": _reference_document(hybrid),
            "generated_vs_hybrid": _certification_document(
                certify_price(
                    generated_price,
                    generated_standard_error,
                    hybrid.price,
                    hybrid.standard_error,
                )
            ),
        }
        if include_exact_grid:
            if step_count <= exact_maximum_step_count:
                exact = exact_gaussian_european_option_price(
                    parameters,
                    strike,
                    maturity,
                    option_side,
                    step_count,
                    antithetic_pair_count,
                    row_seed + 1_000_000,
                    batch_pair_count,
                )
                row["exact_gaussian_grid"] = _reference_document(exact)
                row["hybrid_vs_exact_gaussian"] = _certification_document(
                    certify_price(
                        hybrid.price,
                        hybrid.standard_error,
                        exact.price,
                        exact.standard_error,
                    )
                )
                row["generated_vs_exact_gaussian"] = _certification_document(
                    certify_price(
                        generated_price,
                        generated_standard_error,
                        exact.price,
                        exact.standard_error,
                    )
                )
            else:
                row["exact_gaussian_grid"] = {
                    "skipped": True,
                    "reason": (
                        f"time_steps={step_count} exceeds "
                        f"exact_maximum_step_count={exact_maximum_step_count}"
                    ),
                }
        output_rows.append(row)

    hybrid_pass_count = sum(
        bool(row["generated_vs_hybrid"]["passed"]) for row in output_rows
    )
    exact_decisions = [
        row["generated_vs_exact_gaussian"]
        for row in output_rows
        if "generated_vs_exact_gaussian" in row
    ]
    return {
        "model": "rough_bergomi",
        "source_price_dataset": str(price_path),
        "source_model_dataset": str(model_path),
        "source_product_dataset": str(product_path),
        "option_side": option_side,
        "target_dt": target_dt,
        "reference_antithetic_pairs": antithetic_pair_count,
        "selection": {
            "row_offset": row_offset,
            "row_limit": row_limit,
            "validated_row_count": len(output_rows),
        },
        "summary": {
            "generated_vs_hybrid_pass_count": hybrid_pass_count,
            "generated_vs_hybrid_all_passed": (
                hybrid_pass_count == len(output_rows)
            ),
            "generated_vs_exact_decision_count": len(exact_decisions),
            "generated_vs_exact_all_passed": (
                all(bool(value["passed"]) for value in exact_decisions)
                if exact_decisions
                else None
            ),
        },
        "rows": output_rows,
    }


__all__ = ("validate_rough_bergomi_dataset",)
