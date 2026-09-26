"""Exercise every metadata schema and validate all tracked catalogue instances."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from tools.datasets.metadata_schemas import (
    schema_validator,
    validate_document,
    validate_repository,
)


ROOT = Path(__file__).resolve().parents[2]


VALID_DOCUMENTS = {
    "recipe": {
        "schema_version": 1,
        "kind": "prices",
        "dataset_id": "model_01__product_01__01",
        "generator": "generator.cpp",
        "output": {"path": "datasets/model/example.json", "format": "json"},
        "generation_output": "catalog/model/example/generation.yaml",
        "url": "https://datasets.ai-factory.example/v1/model/example.json",
    },
    "generation": {
        "schema_version": 1,
        "status": "complete",
        "recipe": {
            "path": "catalog/model/example/recipe.yaml",
            "sha256": "a" * 64,
        },
        "artifact": {"row_count": 1, "sha256": "b" * 64},
        "execution": {},
        "timing": {"wall_seconds": 1.0, "kernel_seconds": 0.5},
        "record_sha256": "c" * 64,
    },
    "validation": {
        "schema_version": 1,
        "status": "pending",
        "verified": False,
    },
    "campaign": {
        "version": 4,
        "root": "/repository",
        "build": "/repository/build",
        "publish": False,
        "revision": "d" * 40,
        "jobs": [
            {
                "target": "generate_example",
                "generator": "catalog/example/generator.cpp",
                "recipe": "catalog/example/recipe.yaml",
                "generation": "catalog/example/generation.yaml",
                "dataset": "datasets/example.json",
                "state": "pending",
            }
        ],
    },
    "experiment": {
        "schema_version": 1,
        "experiment_id": "example_v1",
        "purpose": "Exercise the schema.",
        "status": "planned",
        "inputs": [],
        "commands": [],
        "outputs": [],
    },
}


class MetadataSchemaTest(unittest.TestCase):
    def test_all_schemas_and_catalogue_instances_are_valid(self) -> None:
        for kind, document in VALID_DOCUMENTS.items():
            with self.subTest(kind=kind):
                schema_validator(kind)
                validate_document(document, kind)
        counts = validate_repository(ROOT)
        self.assertEqual(counts["recipe"], 5080)
        self.assertEqual(counts["generation"], 720)
        self.assertEqual(counts["validation"], 619)

    def test_each_schema_rejects_a_broken_contract(self) -> None:
        mutations = {
            "recipe": ("generator", "handwritten.cpp"),
            "generation": ("status", "running"),
            "validation": ("verified", "false"),
            "campaign": ("version", 3),
            "experiment": ("status", "finished"),
        }
        for kind, (field, value) in mutations.items():
            document = deepcopy(VALID_DOCUMENTS[kind])
            document[field] = value
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, f"invalid {kind} metadata"):
                    validate_document(document, kind)

    def test_generation_cannot_repeat_recipe_or_validation_fields(self) -> None:
        for field in ("dataset_id", "validation", "time_grid"):
            document = deepcopy(VALID_DOCUMENTS["generation"])
            document[field] = {}
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "invalid generation metadata"):
                    validate_document(document, "generation")

    def test_recipe_requires_an_https_json_url(self) -> None:
        for value in (None, "http://datasets.example/test.json", "https://datasets.example/test"):
            document = deepcopy(VALID_DOCUMENTS["recipe"])
            if value is None:
                document.pop("url")
            else:
                document["url"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "invalid recipe metadata"):
                    validate_document(document, "recipe")


if __name__ == "__main__":
    unittest.main()
