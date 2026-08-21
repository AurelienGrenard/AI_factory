"""Tests for persistent fitted Gaussian-rate references."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from validation.model.fixed_income.fitted_gaussian_rate import (
    _PRODUCT_KINDS,
    _SPECS,
    reference_pricers,
    validate_dataset,
)
from validation.reference_price_dataset import validate_reference_document


ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {
    "caplet": "caplets",
    "floorlet": "floorlets",
    "zero_coupon_bond_call": "zero_coupon_bond_calls",
    "zero_coupon_bond_put": "zero_coupon_bond_puts",
}


class FittedGaussianRateValidationTest(unittest.TestCase):
    @staticmethod
    def _paths(
        model_name: str,
        curve_name: str,
        folder: str,
    ) -> tuple[Path, Path]:
        name = f"{model_name}_01__{curve_name}_01__{folder}_01__01.json"
        relative = Path(model_name) / curve_name / folder / name
        return (
            ROOT
            / "datasets/model/fixed_income"
            / model_name
            / "prices"
            / curve_name
            / folder
            / name,
            ROOT / "validation/datasets/price/fixed_income" / relative,
        )

    def test_every_cache_verifies_core_and_stress(self) -> None:
        self.assertEqual(
            set(_SPECS),
            {
                ("hull_white", "nelson_siegel"),
                ("hull_white", "svensson"),
                ("g2_plus_plus", "nelson_siegel"),
                ("g2_plus_plus", "svensson"),
            },
        )
        self.assertEqual(len(_PRODUCT_KINDS), 4)
        for model_name, curve_name in _SPECS:
            for folder in FOLDERS.values():
                source, reference = self._paths(model_name, curve_name, folder)
                report = validate_dataset(source, reference)
                self.assertTrue(
                    report.verified,
                    f"{model_name}/{curve_name}/{folder}",
                )
                self.assertEqual(report.core.passed_row_count, 900)
                self.assertEqual(report.stress.passed_row_count, 100)

    def test_fitted_metadata_includes_curve_identity_and_fingerprint(self) -> None:
        for model_name, curve_name in _SPECS:
            for product_kind, folder in FOLDERS.items():
                _, path = self._paths(model_name, curve_name, folder)
                document = json.loads(path.read_text(encoding="utf-8"))
                validate_reference_document(document)
                self.assertIn("curve_dataset", document)
                self.assertIn("curve_parameters", document["source_fingerprints"])
                self.assertTrue(
                    all("curve_id" in row for row in document["results"])
                )
                self.assertEqual(
                    document["reference_pricers"],
                    reference_pricers(model_name, curve_name, product_kind),
                )
                for regime in ("core", "stress"):
                    pricers = document["reference_pricers"][regime]
                    self.assertGreater(len(pricers["premia"]), 1)
                    self.assertEqual(
                        pricers["quantlib_specialized"],
                        {"status": "available"},
                    )
                    self.assertEqual(
                        pricers["quantlib_monte_carlo"],
                        {"status": "not_available"},
                    )


if __name__ == "__main__":
    unittest.main()
