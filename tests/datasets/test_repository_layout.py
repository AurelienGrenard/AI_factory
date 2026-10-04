"""Guard the publication boundary between catalogue, datasets, and local work."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


def source_catalog_files(pattern: str):
    return (path for folder in (ROOT / "catalog", ROOT / "work/catalog")
            for path in folder.rglob(pattern))


class RepositoryLayoutTest(unittest.TestCase):
    def test_local_experiments_are_explicit_and_isolated_from_aggregates(self) -> None:
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        catalog_cmake = (ROOT / "cmake/AIFactoryCatalog.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn("AI_FACTORY_ENABLE_LOCAL_EXPERIMENTS", root_cmake)
        self.assertIn("if(AI_FACTORY_ENABLE_LOCAL_EXPERIMENTS)", root_cmake)
        self.assertNotIn(
            'if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/work/experiments/CMakeLists.txt")',
            root_cmake,
        )
        for function_name in (
            "add_parameter_generator",
            "add_price_generator",
            "add_sample_generator",
        ):
            body = catalog_cmake.split(f"function({function_name}", 1)[1].split(
                "endfunction()", 1
            )[0]
            self.assertIn("if(NOT AI_FACTORY_REGISTERING_LOCAL_EXPERIMENTS)", body)

    def test_datasets_contains_no_campaign_or_experiment_workspace(self) -> None:
        datasets = ROOT / "datasets"
        self.assertFalse((datasets / "generation-runs").exists())
        self.assertFalse((datasets / "experiments").exists())
        if datasets.exists():
            self.assertLessEqual(
                {path.name for path in datasets.iterdir()},
                {"prod", "other"},
            )

    def test_local_workspaces_are_ignored_together(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/work/", ignore)
        self.assertIn("/datasets/", ignore)
        self.assertNotIn("/experiments/", ignore)

    def test_catalog_generators_have_canonical_recipes_only(self) -> None:
        catalog = ROOT / "catalog"
        obsolete = tuple(source_catalog_files("dataset.yaml"))
        self.assertEqual(obsolete, ())
        generators = tuple(source_catalog_files("generator.cpp"))
        self.assertGreater(len(generators), 0)
        for generator in generators:
            recipe = generator.with_name("recipe.yaml")
            self.assertTrue(recipe.is_file(), generator.relative_to(ROOT))
            document = yaml.safe_load(recipe.read_text(encoding="utf-8"))
            self.assertEqual(document.get("schema_version"), 1)
            self.assertNotIn("validation", document)
            self.assertNotIn("generation", document)

    def test_active_dataset_taxonomy_uses_price_gradients(self) -> None:
        for root_name in ("catalog", "datasets"):
            root = ROOT / root_name
            if not root.exists():
                continue
            obsolete = [
                path.relative_to(ROOT)
                for path in root.rglob("*")
                if path.is_dir()
                and path.name in {"price_delta", "price_sensitivities"}
            ]
            self.assertEqual(obsolete, [])

        gradient_recipes = tuple(
            path for path in source_catalog_files("recipe.yaml")
            if "price_gradients" in path.parts
        )
        self.assertEqual(len(gradient_recipes),
                         1020 if (ROOT / "work/catalog").exists() else 0)
        for recipe in gradient_recipes:
            document = yaml.safe_load(recipe.read_text(encoding="utf-8"))
            self.assertEqual(document.get("kind"), "price_gradients", recipe)
            self.assertTrue(
                document["dataset_id"].endswith(
                    (
                        "_price_gradient_diagonal_hessian",
                        "_price_gradient_diagonal_hessian_frozen_policy",
                    )
                ),
                recipe,
            )
            self.assertEqual(
                document["sensitivity"]["orders"],
                ["first", "diagonal_second"],
                recipe,
            )

    def test_recipes_are_readable_dataset_identity_cards(self) -> None:
        obsolete_fields = {
            "profile",
            "rng_mapping_version",
            "launch_profile",
            "sensitivity_execution",
        }

        def field_names(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from field_names(child)
            elif isinstance(value, list):
                for child in value:
                    yield from field_names(child)

        for path in source_catalog_files("recipe.yaml"):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertIsInstance(document.get("row_count"), int, path)
            self.assertGreater(document["row_count"], 0, path)
            self.assertTrue(
                obsolete_fields.isdisjoint(field_names(document)),
                path.relative_to(ROOT),
            )
            if document.get("kind") in {"prices", "price_gradients"}:
                self.assertNotIn("construction", document, path)
            if "random_number_generator" in document:
                self.assertEqual(document["random_number_generator"], "philox")

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
        for path in source_catalog_files("generation.yaml"):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(document.get("schema_version"), 1)
            self.assertTrue(forbidden.isdisjoint(document), path.relative_to(ROOT))


if __name__ == "__main__":
    unittest.main()
