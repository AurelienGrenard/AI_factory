"""Minimal checks for recipe and generation metadata."""

from pathlib import Path
import unittest

from tools.datasets.metadata_schemas import load_document, validate_document


ROOT = Path(__file__).resolve().parents[2]


class MetadataSchemaTest(unittest.TestCase):
    def test_current_recipe_and_generation_shapes(self):
        recipe = {
            "schema_version": 1, "kind": "prices", "dataset_id": "example_01",
            "generator": "catalog/model/example/generator.cpp",
            "output": {"path": "datasets/model/example.json", "format": "json"},
            "row_count": 1,
        }
        generation = {
            "schema_version": 4, "status": "complete", "inputs": {},
            "output": {"path": "datasets/model/example.json", "sha256": "a" * 64},
            "hardware": {"device": "gpu", "name": "V100"},
            "software": {"cmake": "3.31", "cxx_compiler": "14.2",
                         "cuda_compiler": "12.4"},
            "timing": {"wall_seconds": 0.001, "duration": "1 ms"},
        }
        validate_document(recipe, "recipe")
        validate_document(generation, "generation")
        recipe["generation_details"] = "catalog/model/example/generation.yaml"
        with self.assertRaises(ValueError):
            validate_document(recipe, "recipe")
        generation["output"]["sha256"] = "bad"
        with self.assertRaises(ValueError):
            validate_document(generation, "generation")

    def test_catalogue_recipes_name_existing_generators(self):
        # Unpublished work can have partial receipts, so recipe metadata is
        # checked independently from generation.yaml and the current manifest.
        recipes = list((ROOT / "catalog").rglob("recipe.yaml"))
        self.assertTrue(recipes)
        for path in recipes:
            with self.subTest(recipe=str(path.relative_to(ROOT))):
                recipe = load_document(path)
                validate_document(recipe, "recipe", path)
                self.assertEqual(
                    recipe["generator"],
                    (path.relative_to(ROOT).parent / "generator.cpp").as_posix(),
                )
                self.assertTrue((path.parent / "generator.cpp").is_file())


if __name__ == "__main__":
    unittest.main()
