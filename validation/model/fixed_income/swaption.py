"""Persistent independent references for one-factor European swaptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

from validation.model.fixed_income.reference_pipeline import (
    REGIME_ROW_COUNTS,
    detailed_pricer,
    persist_generated_reference,
    run_reference_cli,
    validate_cached_reference,
)
from validation.reference_price_dataset import (
    ReferenceDatasetValidation,
    ReferencePrice,
)


_SIDES = {"payer", "receiver"}


@dataclass(frozen=True)
class SwaptionModelSpec:
    """Independent-engine mapping for one supported one-factor rate model."""

    premia_model: str | None
    curve_name: str | None
    require_curve: bool
    quantlib_pricer_id: str


_MODELS: Mapping[tuple[str, str | None], SwaptionModelSpec] = {
    ("ornstein_uhlenbeck", None): SwaptionModelSpec(
        "ornstein_uhlenbeck", None, False, "quantlib_ou_jamshidian_swaption"
    ),
    ("vasicek", None): SwaptionModelSpec(
        "vasicek", None, False, "quantlib_vasicek_jamshidian_swaption"
    ),
    ("cir", None): SwaptionModelSpec(
        None, None, False, "quantlib_cir_jamshidian_swaption"
    ),
    ("hull_white", "nelson_siegel"): SwaptionModelSpec(
        "hull_white",
        "nelson_siegel",
        True,
        "quantlib_hull_white_jamshidian_swaption",
    ),
    ("hull_white", "svensson"): SwaptionModelSpec(
        "hull_white",
        "svensson",
        True,
        "quantlib_hull_white_jamshidian_swaption",
    ),
}


def _spec(model_name: str, curve_name: str | None) -> SwaptionModelSpec:
    try:
        return _MODELS[(model_name, curve_name)]
    except KeyError as error:
        raise ValueError(
            f"Unsupported swaption model/curve '{model_name}/{curve_name}'."
        ) from error


def _validate_side(side: str) -> None:
    if side not in _SIDES:
        raise ValueError(f"Unsupported swaption side '{side}'.")


def _premia_identity(model_name: str, side: str) -> tuple[str, str]:
    if model_name == "hull_white":
        return (
            f"premia_cf_hull_white_1d_{side}_swaption",
            f"CF_HullWhite1d_{side.capitalize()}Swaption",
        )
    suffix = " (centered OU mapping)" if model_name == "ornstein_uhlenbeck" else ""
    return (
        f"premia_cf_vasicek_{side}_swaption"
        + ("_centered_ou_mapping" if suffix else ""),
        f"CF_Vasicek1d_{side.capitalize()}Swaption{suffix}",
    )


def reference_pricers(
    model_name: str,
    curve_name: str | None,
    side: str,
    backend_version: str,
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Describe each callable engine and the rows it actually supplied."""

    spec = _spec(model_name, curve_name)
    _validate_side(side)
    sections: dict[str, Any] = {}
    for regime, row_count in REGIME_ROW_COUNTS:
        regime_counts = counts[regime]
        if spec.premia_model is None:
            premia: dict[str, Any] = {
                "status": "available but not reliable"
            }
        elif regime_counts["premia"]:
            identifier, method = _premia_identity(model_name, side)
            premia = detailed_pricer(
                identifier,
                "Premia",
                "19",
                method,
                regime_counts["premia"],
            )
        else:
            premia = {"status": "available"}
        quantlib = (
            detailed_pricer(
                spec.quantlib_pricer_id,
                "QuantLib",
                backend_version,
                "discountBondOption-based Jamshidian decomposition",
                regime_counts["quantlib"],
            )
            if regime_counts["quantlib"]
            else {"status": "available"}
        )
        sections[regime] = {
            "row_count": row_count,
            "premia": premia,
            "quantlib_specialized": quantlib,
            "quantlib_monte_carlo": {"status": "not_available"},
        }
    return sections


def _quantlib_factory(model_name: str, curve_name: str | None):
    """Load only the QuantLib model factory selected for this dataset."""

    if model_name == "ornstein_uhlenbeck":
        from validation.quantlib.model.fixed_income.ornstein_uhlenbeck.reference import (
            quantlib_model,
        )
    elif model_name == "vasicek":
        from validation.quantlib.model.fixed_income.vasicek.reference import (
            quantlib_model,
        )
    elif model_name == "cir":
        from validation.quantlib.model.fixed_income.cir.reference import quantlib_model
    elif model_name == "hull_white" and curve_name == "nelson_siegel":
        from validation.quantlib.model.fixed_income.hull_white.nelson_siegel.reference import (
            quantlib_model,
        )
    elif model_name == "hull_white" and curve_name == "svensson":
        from validation.quantlib.model.fixed_income.hull_white.svensson.reference import (
            quantlib_model,
        )
    else:  # Guarded by _spec; retained for a local diagnostic.
        raise ValueError(f"No QuantLib factory for '{model_name}/{curve_name}'.")
    return quantlib_model


def generate_reference_dataset(
    price_dataset_path: str | Path,
    reference_dataset_path: str | Path,
    model_name: str,
    side: str,
    curve_name: str | None = None,
) -> ReferenceDatasetValidation:
    """Run the ordered Premia/QuantLib hierarchy and persist all 1000 rows."""

    import QuantLib as ql

    from validation.premia.model.fixed_income.swaption import (
        reference_prices_from_premia_swaption_partial,
    )
    from validation.quantlib.price_validation import load_price_validation_input
    from validation.quantlib.swaption import (
        reference_prices_from_quantlib_swaption,
    )

    spec = _spec(model_name, curve_name)
    _validate_side(side)
    started = time.perf_counter()
    prices_by_id: dict[str, ReferencePrice] = {}
    fallback_ids: tuple[str, ...] | None = None

    if spec.premia_model is not None:
        premia_rows, exceptions = reference_prices_from_premia_swaption_partial(
            price_dataset_path,
            spec.premia_model,
            side,
            curve_name=spec.curve_name,
        )
        premia_id, _ = _premia_identity(model_name, side)
        prices_by_id.update(
            {
                row.row_id: ReferencePrice(
                    row.row_id,
                    row.model_id,
                    row.product_id,
                    row.price,
                    row.standard_error,
                    premia_id,
                    row.curve_id,
                )
                for row in premia_rows
            }
        )
        fallback_ids = tuple(exception.row_id for exception in exceptions)

    if fallback_ids is None or fallback_ids:
        quantlib_rows = reference_prices_from_quantlib_swaption(
            price_dataset_path,
            _quantlib_factory(model_name, curve_name),
            side,
            spec.require_curve,
            row_ids=fallback_ids,
        )
        for row in quantlib_rows:
            if row.row_id in prices_by_id:
                raise RuntimeError(f"Duplicate swaption reference row '{row.row_id}'.")
            prices_by_id[row.row_id] = ReferencePrice(
                row.row_id,
                row.model_id,
                row.product_id,
                row.price,
                row.standard_error,
                spec.quantlib_pricer_id,
                row.curve_id,
            )

    source = load_price_validation_input(price_dataset_path)
    source_ids = tuple(row.row_id for row in source.rows)
    if set(prices_by_id) != set(source_ids):
        missing = sorted(set(source_ids) - set(prices_by_id))
        extra = sorted(set(prices_by_id) - set(source_ids))
        raise RuntimeError(
            f"Swaption reference rows do not match source; missing={missing}, extra={extra}."
        )
    rows = tuple(prices_by_id[row_id] for row_id in source_ids)
    counts = {
        regime: {
            "premia": sum(
                row.reference_pricer_id.startswith("premia_") for row in selected
            ),
            "quantlib": sum(
                row.reference_pricer_id.startswith("quantlib_") for row in selected
            ),
        }
        for regime, selected in (
            ("core", rows[:900]),
            ("stress", rows[900:]),
        )
    }
    return persist_generated_reference(
        price_dataset_path,
        reference_dataset_path,
        rows,
        reference_pricers(model_name, curve_name, side, ql.__version__, counts),
        time.perf_counter() - started,
    )


def validate_dataset(
    price_dataset_path: str | Path,
    reference_dataset_path: str | Path,
) -> ReferenceDatasetValidation:
    """Validate one swaption cache without rerunning external engines."""

    return validate_cached_reference(price_dataset_path, reference_dataset_path)


def run_product_validation_cli(
    model_name: str,
    side: str,
    curve_name: str | None = None,
) -> int:
    """Generate explicitly or compare the immutable reference by default."""

    _spec(model_name, curve_name)
    _validate_side(side)
    label = model_name if curve_name is None else f"{model_name}/{curve_name}"
    return run_reference_cli(
        f"Validate one {label} European {side} swaption dataset.",
        label,
        lambda source, destination: generate_reference_dataset(
            source, destination, model_name, side, curve_name
        ),
    )


__all__ = (
    "_MODELS",
    "generate_reference_dataset",
    "reference_pricers",
    "run_product_validation_cli",
    "validate_dataset",
)
