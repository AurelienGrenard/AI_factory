"""Small structural checks for the public catalogue."""

import json
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


class PublicCatalogInventoryTest(unittest.TestCase):
    def test_manifest_entries_point_to_generators_and_recipes(self):
        manifest = json.loads((ROOT / "catalog/manifest.json").read_text())
        rows = manifest["datasets"]
        self.assertEqual(len(rows), manifest["count"])
        self.assertEqual(len({row["source"] for row in rows}), len(rows))
        for row in rows:
            with self.subTest(source=row["source"]):
                leaf = ROOT / row["catalogPath"]
                generator = leaf / "generator.cpp"
                recipe_path = leaf / "recipe.yaml"
                self.assertTrue(generator.is_file())
                self.assertTrue(recipe_path.is_file())
                recipe = yaml.safe_load(recipe_path.read_text())
                self.assertEqual(recipe["generator"], generator.relative_to(ROOT).as_posix())
                self.assertEqual(recipe["output"]["path"], row["output"])

    def test_bermudan_gradients_declare_frozen_policy(self):
        manifest = json.loads((ROOT / "catalog/manifest.json").read_text())
        for row in manifest["datasets"]:
            if (row["kind"] == "price_gradients"
                    and "/fixed_income/" in row["source"]
                    and "/bermudan_" in row["source"]):
                with self.subTest(source=row["source"]):
                    recipe = yaml.safe_load(
                        (ROOT / row["catalogPath"] / "recipe.yaml").read_text()
                    )
                    self.assertEqual(recipe["exercise_replay"], "frozen_regression_policy")


if __name__ == "__main__":
    unittest.main()
