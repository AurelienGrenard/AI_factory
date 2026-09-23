"""Guard the publication boundary between catalogue, datasets, and local work."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


class RepositoryLayoutTest(unittest.TestCase):
    def test_datasets_contains_no_campaign_or_experiment_workspace(self) -> None:
        datasets = ROOT / "datasets"
        self.assertFalse((datasets / "generation-runs").exists())
        self.assertFalse((datasets / "experiments").exists())
        if datasets.exists():
            self.assertLessEqual(
                {path.name for path in datasets.iterdir()},
                {"curve", "model", "product"},
            )

    def test_local_workspaces_are_ignored_together(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/work/", ignore)
        self.assertIn("/datasets/", ignore)
        self.assertNotIn("/experiments/", ignore)

    def test_catalog_generators_have_canonical_recipes_only(self) -> None:
        catalog = ROOT / "catalog"
        obsolete = tuple(catalog.rglob("dataset.yaml"))
        self.assertEqual(obsolete, ())
        generators = tuple(catalog.rglob("generator.cpp"))
        self.assertGreater(len(generators), 0)
        for generator in generators:
            recipe = generator.with_name("recipe.yaml")
            self.assertTrue(recipe.is_file(), generator.relative_to(ROOT))
            document = yaml.safe_load(recipe.read_text(encoding="utf-8"))
            self.assertEqual(document.get("schema_version"), 1)
            self.assertNotIn("validation", document)
            self.assertNotIn("generation", document)

    def test_generation_receipts_do_not_repeat_recipe_semantics(self) -> None:
        forbidden = {
            "dataset_id",
            "database_id",
            "validation",
            "outputs",
            "sensitivity",
            "time_grid",
            "construction",
            "numerical_method",
        }
        for path in (ROOT / "catalog").rglob("generation.yaml"):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(document.get("schema_version"), 1)
            self.assertTrue(forbidden.isdisjoint(document), path.relative_to(ROOT))


if __name__ == "__main__":
    unittest.main()
