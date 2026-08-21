"""Tests for cached CIR references and their explicit backend audit."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from validation.model.fixed_income.cir_rate import validate_dataset
from validation.reference_price_dataset import (
    compare_reference_dataset,
    validate_reference_document,
)


ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = (
    "caplets",
    "floorlets",
    "zero_coupon_bond_calls",
    "zero_coupon_bond_puts",
)


class CirRateValidationTest(unittest.TestCase):
    @staticmethod
    def _paths(folder: str) -> tuple[Path, Path]:
        name = f"cir_01__{folder}_01__01.json"
        return (
            ROOT / "datasets/model/fixed_income/cir/prices" / folder / name,
            ROOT / "validation/datasets/price/fixed_income/cir" / folder / name,
        )

    def test_all_cached_quantlib_references_verify_core_and_stress(self) -> None:
        for folder in PRODUCTS:
            source, reference = self._paths(folder)
            report = validate_dataset(source, reference)
            self.assertTrue(report.verified, folder)
            self.assertEqual(
                (report.core.row_count, report.stress.row_count), (900, 100)
            )
            self.assertEqual(
                (report.core.failed_row_count, report.stress.failed_row_count),
                (0, 0),
            )

    def test_metadata_distinguishes_availability_from_reliability(self) -> None:
        for folder in PRODUCTS:
            _, path = self._paths(folder)
            document = json.loads(path.read_text(encoding="utf-8"))
            validate_reference_document(document)
            self.assertEqual(
                set(document["source_fingerprints"]),
                {"price_results", "model_parameters", "product_parameters"},
            )
            self.assertEqual(document["verification"]["status"], "passed")
            for regime in ("core", "stress"):
                section = document["reference_pricers"][regime]
                self.assertEqual(
                    list(section),
                    [
                        "row_count",
                        "premia",
                        "quantlib_specialized",
                        "quantlib_monte_carlo",
                    ],
                )
                self.assertEqual(
                    section["premia"],
                    {"status": "available but not reliable"},
                )
                self.assertEqual(
                    section["quantlib_specialized"]["status"], "available"
                )
                self.assertEqual(
                    section["quantlib_specialized"]["backend"], "QuantLib"
                )
                self.assertEqual(
                    section["quantlib_specialized"]["id"],
                    "quantlib_cir_discount_bond_option",
                )
                self.assertEqual(
                    section["quantlib_specialized"]["backend_version"], "1.43"
                )
                self.assertEqual(
                    section["quantlib_specialized"]["kind"],
                    "specialized_pricer",
                )
                self.assertEqual(
                    section["quantlib_specialized"]["method"],
                    "CoxIngersollRoss.discountBondOption with OTM-tail parity "
                    "stabilization",
                )
                self.assertEqual(
                    section["quantlib_specialized"]["row_priced"],
                    section["row_count"],
                )
                self.assertEqual(
                    section["quantlib_monte_carlo"],
                    {"status": "not_available"},
                )

    def test_catalogs_only_expose_the_cached_validation_contract(self) -> None:
        for folder in PRODUCTS:
            stem = f"cir_01__{folder}_01__01"
            catalog = (
                ROOT
                / "catalog/model/fixed_income/cir/prices"
                / folder
                / stem
            )
            document = yaml.safe_load((catalog / "dataset.yaml").read_text())
            self.assertEqual(
                document["validation"],
                {
                    "status": "available",
                    "verified": True,
                    "dataset": f"validation/datasets/price/fixed_income/cir/"
                    f"{folder}/{stem}.json",
                },
            )
            self.assertFalse((catalog / "validation.ipynb").exists())
            self.assertFalse((catalog / "validation_report.json").exists())

    def test_multiple_used_pricers_are_supported_and_counted(self) -> None:
        _, reference = self._paths("caplets")
        document = json.loads(reference.read_text(encoding="utf-8"))
        mixed = copy.deepcopy(document)
        core = mixed["reference_pricers"]["core"]
        core["premia"] = {
            "status": "available",
            "id": "premia_example",
            "backend": "Premia",
            "backend_version": "19",
            "kind": "specialized_pricer",
            "method": "example",
            "row_priced": 1,
        }
        core["quantlib_specialized"]["row_priced"] = 899
        mixed["results"][0]["reference_pricer_id"] = "premia_example"
        validate_reference_document(mixed)

        core["premia"]["row_priced"] = 2
        with self.assertRaisesRegex(ValueError, "row_priced"):
            validate_reference_document(mixed)

    def test_stale_fingerprint_and_falsified_verification_fail_closed(self) -> None:
        source, reference = self._paths("caplets")
        document = json.loads(reference.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / reference.name

            stale = copy.deepcopy(document)
            stale["source_fingerprints"]["price_results"] = "sha256:" + "0" * 64
            path.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprints are stale"):
                compare_reference_dataset(source, path)

            falsified = copy.deepcopy(document)
            falsified["verification"]["core"]["maximum_absolute_error"] = 0.0
            path.write_text(json.dumps(falsified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verification does not match"):
                compare_reference_dataset(source, path)


if __name__ == "__main__":
    unittest.main()
