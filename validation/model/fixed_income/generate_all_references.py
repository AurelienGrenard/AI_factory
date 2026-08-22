"""Explicitly regenerate every persistent fixed-income price reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.model.fixed_income.cir_rate import (
    generate_reference_dataset as generate_cir,
)
from validation.model.fixed_income.fitted_gaussian_rate import (
    generate_reference_dataset as generate_fitted,
)
from validation.model.fixed_income.gaussian_rate import (
    generate_reference_dataset as generate_standalone,
)
from validation.model.fixed_income.swaption import (
    generate_reference_dataset as generate_swaption,
)
from validation.reference_price_dataset import (
    ReferenceDatasetValidation,
    synchronize_catalog_validation,
)


_PRODUCTS = (
    ("caplet", "caplets"),
    ("floorlet", "floorlets"),
    ("zero_coupon_bond_call", "zero_coupon_bond_calls"),
    ("zero_coupon_bond_put", "zero_coupon_bond_puts"),
)
_MODELS = (
    "cir",
    "g2",
    "g2_plus_plus",
    "hull_white",
    "ornstein_uhlenbeck",
    "vasicek",
)
_SWAPTION_SIDES = ("payer", "receiver")


def _persist(
    source: Path,
    destination: Path,
    report: ReferenceDatasetValidation,
) -> None:
    """Publish the compact YAML only after both regimes pass."""

    synchronize_catalog_validation(source, destination, report.verified)
    if not report.verified:
        raise RuntimeError(f"Generated reference failed validation: {source}")
    print(f"verified {destination.relative_to(Path.cwd())}", flush=True)


def generate_all(root: Path, selected_models: set[str]) -> None:
    """Regenerate the selected model families using their declared engines."""

    source_root = root / "datasets/model/fixed_income"
    destination_root = root / "validation/datasets/price/fixed_income"
    for model_name in ("cir", "g2", "ornstein_uhlenbeck", "vasicek"):
        if model_name not in selected_models:
            continue
        for product_kind, folder in _PRODUCTS:
            stem = f"{model_name}_01__{folder}_01__01.json"
            source = source_root / model_name / "prices" / folder / stem
            destination = destination_root / model_name / folder / stem
            if model_name == "cir":
                report = generate_cir(source, destination, product_kind)
            else:
                report = generate_standalone(
                    source,
                    destination,
                    model_name,
                    product_kind,
                )
            _persist(source, destination, report)

    for model_name in ("cir", "ornstein_uhlenbeck", "vasicek"):
        if model_name not in selected_models:
            continue
        for side in _SWAPTION_SIDES:
            folder = f"european_{side}_swaptions"
            stem = f"{model_name}_01__{folder}_01__01.json"
            source = source_root / model_name / "prices" / folder / stem
            destination = destination_root / model_name / folder / stem
            report = generate_swaption(
                source, destination, model_name, side
            )
            _persist(source, destination, report)

    for model_name in ("g2_plus_plus", "hull_white"):
        if model_name not in selected_models:
            continue
        for curve_name in ("nelson_siegel", "svensson"):
            for product_kind, folder in _PRODUCTS:
                stem = (
                    f"{model_name}_01__{curve_name}_01__{folder}_01__01.json"
                )
                source = (
                    source_root
                    / model_name
                    / "prices"
                    / curve_name
                    / folder
                    / stem
                )
                destination = (
                    destination_root / model_name / curve_name / folder / stem
                )
                report = generate_fitted(
                    source,
                    destination,
                    model_name,
                    curve_name,
                    product_kind,
                )
                _persist(source, destination, report)

    if "hull_white" in selected_models:
        for curve_name in ("nelson_siegel", "svensson"):
            for side in _SWAPTION_SIDES:
                folder = f"european_{side}_swaptions"
                stem = (
                    f"hull_white_01__{curve_name}_01__{folder}_01__01.json"
                )
                source = (
                    source_root
                    / "hull_white"
                    / "prices"
                    / curve_name
                    / folder
                    / stem
                )
                destination = (
                    destination_root
                    / "hull_white"
                    / curve_name
                    / folder
                    / stem
                )
                report = generate_swaption(
                    source,
                    destination,
                    "hull_white",
                    side,
                    curve_name,
                )
                _persist(source, destination, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=_MODELS,
        dest="models",
        help="Regenerate only this model family; repeat to select several.",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    selected = set(arguments.models or _MODELS)
    generate_all(root, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
