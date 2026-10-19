"""Exercise every metadata schema and validate all tracked catalogue instances."""

from __future__ import annotations

from copy import deepcopy
import json
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
        "row_count": 1,
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
        manifest = json.loads((ROOT / "catalog/manifest.json").read_text())
        self.assertEqual(counts["recipe"], manifest["count"])
        # A local worktree may contain untracked generation receipts from runs
        # that have not been published in the committed manifest yet.
        self.assertGreaterEqual(
            counts["generation"],
            sum(row["ready"] for row in manifest["datasets"]),
        )
        self.assertEqual(counts["validation"], 0)

    def test_editable_generation_v2_has_current_hashes_without_record_lock(self) -> None:
        document = {
            "schema_version": 2, "status": "complete",
            "recipe": {"path": "catalog/model/example/recipe.yaml"},
            "artifact": {"path": "datasets/model/example.json",
                         "row_count": 1, "sha256": "a" * 64},
            "inputs": {"model": {"path": "datasets/model/input.json",
                                 "sha256": "b" * 64}},
            "environment": {"hardware": {"device": "gpu", "name": "V100"},
                            "software": {"cmake": "3.31", "cxx_compiler": "14.2", "cuda_compiler": "12.4"}},
            "curation": {"source_sha256": "c" * 64,
                         "removed_gradients": ["model.risk_free_rate"]},
        }
        validate_document(document, "generation")
        document["record_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            validate_document(document, "generation")

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

    def test_generation_accepts_an_experimental_work_recipe(self) -> None:
        document = deepcopy(VALID_DOCUMENTS["generation"])
        document["recipe"]["path"] = (
            "work/generation/example/recipe.yaml"
        )
        validate_document(document, "generation")

    def test_generation_cannot_repeat_recipe_or_validation_fields(self) -> None:
        for field in ("dataset_id", "validation", "time_grid"):
            document = deepcopy(VALID_DOCUMENTS["generation"])
            document[field] = {}
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "invalid generation metadata"):
                    validate_document(document, "generation")

    def test_recipe_rejects_execution_only_fields(self) -> None:
        for field, value in (
            ("rng_mapping_version", "philox_source_step_v2"),
            ("launch_profile", "candidate"),
            ("sensitivity_execution", {"default": "mono"}),
        ):
            document = deepcopy(VALID_DOCUMENTS["recipe"])
            document[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError, "invalid recipe metadata"
                ):
                    validate_document(document, "recipe")

    def test_recipe_requires_a_positive_row_count(self) -> None:
        for value in (None, 0, -1):
            document = deepcopy(VALID_DOCUMENTS["recipe"])
            if value is None:
                document.pop("row_count")
            else:
                document["row_count"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "invalid recipe metadata"
                ):
                    validate_document(document, "recipe")

    def test_full_hessian_recipe_keeps_bumps_and_orders_without_url(self) -> None:
        document = deepcopy(VALID_DOCUMENTS["recipe"])
        document["kind"] = "price_gradients"
        document.pop("url")
        document["option_side"] = "call"
        document["sensitivity"] = {
            "method": "finite_difference_shared_innovations",
            "parameters": [{
                "parameter": "model.spot", "displacement": 0.005,
                "scale": "relative", "boundary": "central_then_one_sided_order2",
            }],
            "orders": ["first", "diagonal_second", "mixed_second"],
            "mixed_second": "all",
            "source_price_recipe": "catalog/model/example/prices/call/recipe.yaml",
        }
        validate_document(document, "recipe")
        document["url"] = "https://datasets.example/test.json"
        with self.assertRaisesRegex(ValueError, "invalid recipe metadata"):
            validate_document(document, "recipe")

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
