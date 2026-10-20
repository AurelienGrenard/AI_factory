"""The delivered catalogue contains inputs, aligned prices and gradients."""

import json
from pathlib import Path
import re
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/codegen/pricing_bindings"))
from tools.datasets.multi_asset_specs import REFERENCE_DATASET_SPECS  # noqa: E402
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
            {"model_parameters", "product_parameters", "curve_parameters",
             "prices", "price_gradients", "samples"},
        )
        selected = [
            spec for spec in AVAILABLE_DATASET_SPECS
            if spec.dataset_kind in {"model_parameters", "product_parameters", "curve_parameters", "samples"}
            or (spec.dataset_kind == "prices" and spec.construction == "aligned")
            or (
                spec.dataset_kind == "price_gradients"
                and spec.construction == "aligned"
                and (spec.product != "bermudan_swaption"
                     or spec.exercise_replay == "frozen_regression_policy")
            )
        ]
        selected.extend(REFERENCE_DATASET_SPECS)
        expected = {
            spec.generator_path.removeprefix("catalog/").removesuffix("/generator.cpp")
            for spec in selected
            if (ROOT / spec.generator_path).is_file()
        }
        self.assertEqual(sources, expected)
        self.assertFalse(any(source.startswith(("correlation/", "multi_asset/"))
                             for source in sources))
        for row in rows:
            leaf = ROOT / row["catalogPath"]
            self.assertTrue((leaf / "generator.cpp").is_file(), leaf)
            self.assertTrue((leaf / "recipe.yaml").is_file(), leaf)
            if row["kind"] == "price_gradients":
                self.assertNotIn("url", row, leaf)
                self.assertNotIn("url", yaml.safe_load((leaf / "recipe.yaml").read_text()), leaf)
            receipt = leaf / "generation.yaml"
            if receipt.exists():
                self.assertTrue((ROOT / row["output"]).is_file(), leaf)
            if row["kind"] == "samples" or (
                row["kind"] == "price_gradients"
                and "/equity/markovian/" in row["source"]
            ):
                self.assertFalse(receipt.exists(), leaf)
        self.assertEqual(sum(row["kind"] == "samples" for row in rows), 50)
        self.assertEqual(sum(row["kind"] == "price_gradients" for row in rows), 655)
        self.assertEqual(sum(row["kind"] == "price_gradients"
                             and row["source"].startswith("model/equity/rough/")
                             for row in rows), 174)
        self.assertFalse(any("diagonal_hessian" in row["source"]
                             or "full_hessian" in row["source"] for row in rows))
        self.assertFalse(any(row["kind"] == "price_gradients"
                             and "/american_" in row["source"]
                             and row["id"].endswith("_frozen_policy") for row in rows))

    def test_bermudan_public_policy_is_frozen_regression(self) -> None:
        manifest = json.loads((ROOT / "catalog/manifest.json").read_text())
        bermudans = [row for row in manifest["datasets"]
                     if row["kind"] == "price_gradients"
                     and "/fixed_income/" in row["source"]
                     and "/bermudan_" in row["source"]]
        self.assertTrue(all(row["id"].endswith("_frozen_policy") for row in bermudans))
        planned_bermudans = [spec for spec in PRICE_GRADIENT_DATASET_SPECS
                             if spec.asset_class == "fixed_income"
                             and spec.construction == "aligned"
                             and spec.product == "bermudan_swaption"]
        self.assertEqual(len(planned_bermudans), 26)
        self.assertTrue(all(spec.exercise_replay == "frozen_regression_policy"
                            for spec in planned_bermudans))
        self.assertEqual(
            len(bermudans),
            sum(spec.asset_class == "fixed_income"
                and spec.construction == "aligned"
                and spec.product == "bermudan_swaption"
                and (ROOT / spec.generator_path).is_file()
                for spec in PRICE_GRADIENT_DATASET_SPECS),
        )
        self.assertEqual(
            len([spec for spec in PRICE_GRADIENT_DATASET_SPECS
                 if spec.asset_class == "fixed_income"
                 and spec.construction == "aligned"
                 and spec.product != "bermudan_swaption"]),
            91,  # 13 model/curve compositions times seven non-Bermudan products.
        )

    def test_fixed_income_gradient_products_cover_all_contract_coordinates(self) -> None:
        expected = {
            "rate_option": {"product.strike", "product.maturity_years"},
            "zero_coupon_bond_option": {"product.strike", "product.maturity_years"},
            "zero_coupon_bond_up_and_out": {
                "product.strike", "product.barrier", "product.maturity_years"
            },
            "european_swaption": {"product.strike", "product.maturity_years"},
            "bermudan_swaption": {"product.strike"},
        }
        checked = 0
        for spec in PRICE_GRADIENT_DATASET_SPECS:
            if spec.asset_class != "fixed_income" or spec.construction != "aligned":
                continue
            recipe_path = ROOT / spec.recipe_yaml_path
            if not recipe_path.is_file():
                continue
            recipe = yaml.safe_load(recipe_path.read_text())
            coordinates = {
                item["parameter"] for item in recipe["sensitivity"]["parameters"]
                if item["parameter"].startswith("product.")
            }
            self.assertEqual(coordinates, expected[spec.product], recipe_path)
            checked += 1
        self.assertEqual(checked, 117)

    def test_exact_schedule_sensitivities_declare_native_calendar(self) -> None:
        checked = 0
        for spec in PRICE_GRADIENT_DATASET_SPECS:
            generator = ROOT / spec.generator_path
            if not generator.is_file() or not generator.is_relative_to(ROOT / "catalog"):
                continue
            recipe = yaml.safe_load((ROOT / spec.recipe_yaml_path).read_text())
            kind = recipe.get("time_representation", {}).get("kind")
            native_schedule = "recipe.exact_schedule_transition = true;" in generator.read_text()
            self.assertEqual(native_schedule, kind == "exact_schedule_transition", spec.generator_path)
            checked += 1
        self.assertEqual(checked, sum((ROOT / spec.generator_path).is_file()
                                      for spec in PRICE_GRADIENT_DATASET_SPECS))

    def test_fixed_income_price_generator_url_versions_match_recipes(self) -> None:
        helper_path = ROOT / "tools/pricing/bermudan_swaption_price_generation.cuh"
        helper = helper_path.read_text()
        bermudan_prefixes = re.findall(
            r'"(https://datasets\.ai-factory\.example/v\d+/)" \+ relative', helper
        )
        self.assertEqual(bermudan_prefixes, ["https://datasets.ai-factory.example/v2/"] * 2)
        checked = 0
        for spec in AVAILABLE_DATASET_SPECS:
            if (spec.asset_class != "fixed_income" or spec.dataset_kind != "prices"
                    or spec.construction != "aligned"):
                continue
            recipe = yaml.safe_load((ROOT / spec.recipe_yaml_path).read_text())
            generator = (ROOT / spec.generator_path).read_text()
            expected_version = re.search(r'/v\d+/', spec.url).group()
            generated_versions = set(re.findall(
                r'https://datasets\.ai-factory\.example(/v\d+/)', generator
            ))
            if spec.product == "bermudan_swaption":
                self.assertFalse(generated_versions, spec.generator_path)
                self.assertEqual(expected_version, "/v2/", spec.generator_path)
            else:
                self.assertEqual(generated_versions, {expected_version}, spec.generator_path)
            checked += 1
        self.assertEqual(checked, 117)


if __name__ == "__main__":
    unittest.main()
