"""Publication tests for persistent Black-Scholes reference prices."""

from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from validation.model.equity.black_scholes.reference_pipeline import (
    PRODUCT_KINDS,
    product_folder,
    validate_dataset,
)
from validation.reference_price_dataset import validate_reference_document


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BIAS_PRODUCTS = {
    "asian_call",
    "asian_put",
    "double_knock_out_call",
    "double_knock_out_put",
    "down_and_in_put",
    "down_and_out_put",
    "lookback_option",
    "up_and_in_call",
    "up_and_out_call",
    "up_no_touch",
    "up_one_touch",
}
QUANTLIB_ONLY_PRODUCTS = {
    "athena_autocall",
    "cliquet",
    "phoenix_autocall",
    "phoenix_memory_autocall",
}


class BlackScholesReferencePipelineTest(unittest.TestCase):
    @staticmethod
    def _paths(product_kind: str) -> tuple[Path, Path, Path]:
        folder = product_folder(product_kind)
        stem = f"black_scholes_01__{folder}_01__01"
        source = (
            ROOT
            / "datasets/price/equity/black_scholes"
            / folder
            / f"{stem}.json"
        )
        reference = (
            ROOT
            / "validation/datasets/price/equity/black_scholes"
            / folder
            / f"{stem}.json"
        )
        catalog = ROOT / "catalog/price/equity/black_scholes" / folder / stem
        return source, reference, catalog

    def test_all_29_catalogs_publish_only_a_verified_cache(self) -> None:
        self.assertEqual(len(PRODUCT_KINDS), 29)
        for product_kind in PRODUCT_KINDS:
            source, reference, catalog = self._paths(product_kind)
            report = validate_dataset(source, reference)
            self.assertTrue(report.verified, product_kind)
            self.assertEqual(
                (report.core.row_count, report.stress.row_count),
                (900, 100),
            )
            document = yaml.safe_load(
                (catalog / "dataset.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                document["validation"],
                {
                    "status": "available",
                    "verified": True,
                    "dataset": reference.relative_to(ROOT).as_posix(),
                },
            )
            self.assertFalse((catalog / "validation.ipynb").exists())
            self.assertFalse((catalog / "validation_report.json").exists())

    def test_reference_hierarchy_and_expected_bias_are_explicit(self) -> None:
        for product_kind in PRODUCT_KINDS:
            _, reference, _ = self._paths(product_kind)
            document = json.loads(reference.read_text(encoding="utf-8"))
            validate_reference_document(document)
            for regime, row_count in (("core", 900), ("stress", 100)):
                pricers = document["reference_pricers"][regime]
                self.assertEqual(
                    list(pricers),
                    [
                        "row_count",
                        "premia",
                        "quantlib_specialized",
                        "quantlib_monte_carlo",
                    ],
                )
                self.assertEqual(pricers["row_count"], row_count)
                used = sum(
                    pricers[name].get("row_priced", 0)
                    for name in (
                        "premia",
                        "quantlib_specialized",
                        "quantlib_monte_carlo",
                    )
                )
                self.assertEqual(used, row_count)
                if product_kind in QUANTLIB_ONLY_PRODUCTS:
                    self.assertEqual(
                        pricers["premia"], {"status": "not_available"}
                    )
                    self.assertEqual(
                        pricers["quantlib_monte_carlo"]["row_priced"],
                        row_count,
                    )
                else:
                    self.assertGreater(pricers["premia"]["row_priced"], 0)

            bias_policy = document["verification"].get(
                "systematic_bias_policy"
            )
            if product_kind in EXPECTED_BIAS_PRODUCTS:
                self.assertEqual(bias_policy["status"], "accepted_expected")
                self.assertTrue(bias_policy["explanation"])
                self.assertIn(
                    "continuous",
                    document["reference_pricers"]["core"]["premia"][
                        "method"
                    ],
                )
            else:
                self.assertIsNone(bias_policy)
            self.assertTrue(
                all("comparison" in row for row in document["results"])
            )

    def test_cache_validation_never_imports_external_pricers(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "QuantLib" or name.startswith("validation.premia"):
                raise AssertionError(f"Unexpected external backend import: {name}")
            return original_import(name, *args, **kwargs)

        representatives = (
            "european_call",
            "up_and_out_call",
            "athena_autocall",
        )
        with patch("builtins.__import__", side_effect=guarded_import):
            for product_kind in representatives:
                source, reference, _ = self._paths(product_kind)
                self.assertTrue(validate_dataset(source, reference).verified)

    def test_expected_bias_requires_a_non_empty_explanation(self) -> None:
        _, reference, _ = self._paths("up_and_out_call")
        document = json.loads(reference.read_text(encoding="utf-8"))
        invalid = copy.deepcopy(document)
        invalid["verification"]["systematic_bias_policy"]["explanation"] = ""
        with self.assertRaisesRegex(ValueError, "systematic_bias_policy"):
            validate_reference_document(invalid)


if __name__ == "__main__":
    unittest.main()
