"""The delivered catalogue contains inputs, aligned prices and gradients."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/codegen/pricing_bindings"))
from capability_manifest import (  # noqa: E402
    AVAILABLE_DATASET_SPECS,
    PRICE_GRADIENT_DATASET_SPECS,
)


class PublicCatalogInventoryTest(unittest.TestCase):
    def test_published_generator_selection(self) -> None:
        manifest = json.loads((ROOT / "catalog/manifest.json").read_text())
        rows = manifest["datasets"]
        sources = {row["source"] for row in rows}
        self.assertEqual(len(rows), manifest["count"])
        self.assertEqual(len(sources), len(rows))
        self.assertEqual(
            {row["kind"] for row in rows},
            {"model_parameters", "product_parameters", "curve_parameters", "prices", "price_gradients"},
        )
        selected = [
            spec for spec in AVAILABLE_DATASET_SPECS
            if spec.dataset_kind in {"model_parameters", "product_parameters", "curve_parameters"}
            or (spec.dataset_kind == "prices" and spec.construction == "aligned")
            or (
                spec.dataset_kind == "price_gradients"
                and spec.construction == "aligned"
                and (spec.product != "bermudan_swaption"
                     or spec.exercise_replay == "frozen_regression_policy")
            )
        ]
        expected = {
            spec.generator_path.removeprefix("catalog/").removesuffix("/generator.cpp")
            for spec in selected
        }
        self.assertEqual(sources, expected)
        for row in rows:
            leaf = ROOT / row["catalogPath"]
            self.assertTrue((leaf / "generator.cpp").is_file(), leaf)
            self.assertTrue((leaf / "recipe.yaml").is_file(), leaf)

    def test_bermudan_public_policy_is_frozen_regression(self) -> None:
        manifest = json.loads((ROOT / "catalog/manifest.json").read_text())
        bermudans = [row for row in manifest["datasets"]
                     if row["kind"] == "price_gradients"
                     and "/fixed_income/" in row["source"]
                     and "/bermudan_" in row["source"]]
        self.assertEqual(len(bermudans), 26)
        self.assertTrue(all(row["id"].endswith("_frozen_policy") for row in bermudans))
        self.assertEqual(
            len([spec for spec in PRICE_GRADIENT_DATASET_SPECS
                 if spec.asset_class == "fixed_income"
                 and spec.construction == "aligned"
                 and spec.product != "bermudan_swaption"]),
            78,
        )


if __name__ == "__main__":
    unittest.main()
