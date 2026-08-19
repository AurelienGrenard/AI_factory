"""Tests for persistent standalone Gaussian-rate references."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from validation.model.fixed_income.gaussian_rate import (
    _MODELS,
    _PRODUCT_KINDS,
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


class GaussianRateValidationTest(unittest.TestCase):
    @staticmethod
    def _paths(model_name: str, folder: str) -> tuple[Path, Path]:
        name = f"{model_name}_01__{folder}_01__01.json"
        return (
            ROOT / "datasets/price/fixed_income" / model_name / folder / name,
            ROOT
            / "validation/datasets/price/fixed_income"
            / model_name
            / folder
            / name,
        )

    def test_every_cache_verifies_core_and_stress(self) -> None:
        self.assertEqual(set(_MODELS), {"vasicek", "ornstein_uhlenbeck", "g2"})
        self.assertEqual(len(_PRODUCT_KINDS), 4)
        for model_name in _MODELS:
            for folder in FOLDERS.values():
                source, reference = self._paths(model_name, folder)
                report = validate_dataset(source, reference)
                self.assertTrue(report.verified, f"{model_name}/{folder}")
                self.assertEqual(report.core.passed_row_count, 900)
                self.assertEqual(report.stress.passed_row_count, 100)

    def test_metadata_records_only_the_used_engine_in_detail(self) -> None:
        for model_name in _MODELS:
            for product_kind, folder in FOLDERS.items():
                _, path = self._paths(model_name, folder)
                document = json.loads(path.read_text(encoding="utf-8"))
                validate_reference_document(document)
                self.assertEqual(document["verification"]["status"], "passed")
                for regime in ("core", "stress"):
                    pricers = document["reference_pricers"][regime]
                    if model_name == "g2":
                        self.assertEqual(
                            pricers["premia"], {"status": "not_available"}
                        )
                        self.assertEqual(
                            pricers["quantlib_specialized"]["id"],
                            "quantlib_g2_discount_bond_option",
                        )
                    else:
                        self.assertGreater(len(pricers["premia"]), 1)
                        self.assertEqual(
                            pricers["quantlib_specialized"],
                            {"status": "available"},
                        )
                    self.assertEqual(
                        pricers["quantlib_monte_carlo"],
                        {"status": "not_available"},
                    )
                expected = reference_pricers(
                    model_name,
                    product_kind,
                    "1.43" if model_name == "g2" else "19",
                )
                self.assertEqual(document["reference_pricers"], expected)


if __name__ == "__main__":
    unittest.main()
