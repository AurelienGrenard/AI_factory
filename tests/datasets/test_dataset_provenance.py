"""Published metadata tracks current content, without freezing historic code bytes."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools.datasets import check_dataset_compatibility as checker
from tools.datasets import dataset_provenance as provenance
from tools.datasets import refresh_generation
from tools.datasets.artifact_publication import digest


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset_rel = "datasets/prices.json"
        self.dataset = self.root / self.dataset_rel
        self.dataset.parent.mkdir()
        self.document = {
            "database_id": "prices", "row_count": 1,
            "model_family": "test", "product_family": "test",
            "model_dataset": {"id": "model"},
            "product_dataset": {"id": "product"},
            "results": [{"model_id": "m1", "product_id": "p1",
                         "price": 1.25, "standard_error": 0.01}],
        }
        self.save_dataset()
        self.recipe_rel = "catalog/example/recipe.yaml"
        self.recipe_path = self.root / self.recipe_rel
        self.recipe_path.parent.mkdir(parents=True)
        self.recipe = {
            "schema_version": 1, "kind": "prices", "dataset_id": "prices",
            "generator": "catalog/example/generator.cpp", "row_count": 1,
            "output": {"path": self.dataset_rel, "format": "json"},
            "paths_per_price": 1_048_576, "seeds": {"dynamics": 123},
        }
        self.save_recipe()
        self.generation_path = self.root / "catalog/example/generation.yaml"
        self.native = {"schema_version": 1, "status": "complete",
                       "artifact": {"row_count": 1},
                       "execution": {"paths_per_price": 1_048_576},
                       "timing": {"wall_seconds": 1.0, "kernel_seconds": .9}}
        self.generation = provenance.generation_record(
            self.native, self.recipe_rel, self.dataset_rel, {}, self.root,
            {"device": "gpu", "name": "test GPU", "host": "test", "cpu_model": "test CPU"},
            {"cmake": "3.31", "cxx_compiler": "14.2", "cuda_compiler": "13.2"})
        self.save_generation()

    def save_dataset(self):
        self.dataset.write_text(json.dumps(self.document))

    def save_recipe(self):
        self.recipe_path.write_text(yaml.safe_dump(self.recipe))

    def save_generation(self):
        self.generation_path.write_text(yaml.safe_dump(self.generation))

    def assess(self):
        return provenance.assess(self.recipe, self.generation, self.dataset, self.root)

    def test_current_metadata_is_compatible_without_numerical_certification(self):
        result = self.assess()
        self.assertEqual(result["status"], "compatible")
        self.assertEqual(result["certification"], "not_assessed")
        self.assertNotIn("record_sha256", self.generation)
        self.assertNotIn("execution", self.generation)

    def test_recipe_and_hardware_may_change_without_generation_gate(self):
        self.recipe["seeds"]["dynamics"] = 124
        self.generation["hardware"]["name"] = "corrected GPU name"
        self.assertEqual(self.assess()["status"], "compatible")

    def test_json_edit_needs_only_current_output_sha(self):
        self.document["results"][0]["price"] = 1.5
        self.save_dataset()
        self.assertEqual(self.assess()["status"], "invalid")
        self.generation["output"]["sha256"] = digest(self.dataset)
        self.assertEqual(self.assess()["status"], "compatible")

    def test_recipe_must_describe_current_json(self):
        self.recipe["row_count"] = 2
        self.assertEqual(self.assess()["status"], "invalid")
        self.recipe["row_count"] = 1
        self.recipe["output"]["path"] = "datasets/another.json"
        self.assertEqual(self.assess()["status"], "invalid")

    def test_input_sha_is_current_metadata_not_an_immutable_gate(self):
        relative = "datasets/model/example.json"
        parameter = self.root / relative
        parameter.parent.mkdir(parents=True)
        parameter.write_text('{"database_id":"example","row_count":1,'
                             '"models":[{"id":"000001","parameters":{"x":1}}]}')
        self.document["model_dataset"] = {"id": "example"}
        self.save_dataset()
        self.generation["output"]["sha256"] = digest(self.dataset)
        self.recipe["inputs"] = {"model": relative}
        self.generation["inputs"] = {"model": {"path": relative, "sha256": digest(parameter)}}
        self.assertEqual(self.assess()["status"], "compatible")
        parameter.write_text('{"database_id":"example","row_count":1,'
                             '"models":[{"id":"000001","parameters":{"x":2}}]}')
        self.assertEqual(self.assess()["status"], "review_required")
        self.generation["inputs"]["model"]["sha256"] = digest(parameter)
        self.assertEqual(self.assess()["status"], "compatible")

    def test_refresh_preserves_historical_input_hash_after_parameter_replacement(self):
        relative = "datasets/model/example.json"
        parameter = self.root / relative
        parameter.parent.mkdir(parents=True)
        parameter.write_text('{"database_id":"example","row_count":1,'
                             '"models":[{"id":"000001","parameters":{"x":1}}]}')
        old_hash = digest(parameter)
        self.recipe["inputs"] = {"model": relative}
        self.generation["inputs"] = {"model": {"path": relative, "sha256": old_hash}}
        self.document["model_dataset"] = {
            "id": "example"
        }
        self.save_dataset()
        self.generation["output"]["sha256"] = digest(self.dataset)
        parameter.write_text('{"database_id":"example","row_count":1,'
                             '"models":[{"id":"000001","parameters":{"x":2}}]}')
        refreshed = refresh_generation.refresh(
            self.recipe, self.generation, self.dataset, self.root, []
        )
        self.assertEqual(refreshed["inputs"]["model"], {"path": relative, "sha256": old_hash})
        self.assertEqual(
            provenance.assess(self.recipe, refreshed, self.dataset, self.root)["status"],
            "review_required",
        )

    def test_gradients_can_be_curated_with_recipe_and_sha_update(self):
        self.recipe["kind"] = "price_gradients"
        self.recipe["sensitivity"] = {"parameters": [
            {"parameter": "model.spot"}, {"parameter": "model.risk_free_rate"}]}
        self.document = {
            "database_id": "prices", "model_family": "test",
            "product_family": "european_option",
            "row_count": 1,
            "model_dataset": {"id": "model"},
            "product_dataset": {"id": "product"},
            "results": [{"model_id": "m1", "product_id": "p1", "price": 1.25,
                         "standard_error": .01,
                         "gradients": {"model.spot": .4,
                                       "model.risk_free_rate": -.1},
                         "gradient_standard_errors": {"model.spot": .02,
                                                      "model.risk_free_rate": .03}}],
        }
        self.save_dataset()
        self.generation["output"]["sha256"] = digest(self.dataset)
        self.assertEqual(self.assess()["status"], "compatible")
        self.recipe["sensitivity"]["parameters"].pop()
        row = self.document["results"][0]
        row["gradients"].pop("model.risk_free_rate")
        row["gradient_standard_errors"].pop("model.risk_free_rate")
        self.save_dataset()
        self.generation["output"]["sha256"] = digest(self.dataset)
        self.assertEqual(self.assess()["status"], "compatible")
        row["gradients"]["model.risk_free_rate"] = -.1
        self.save_dataset()
        self.generation["output"]["sha256"] = digest(self.dataset)
        self.assertEqual(self.assess()["status"], "invalid")

    def test_legacy_receipt_does_not_lock_recipe_or_execution(self):
        old = {**self.native, "recipe": {"path": self.recipe_rel,
                                          "sha256": "a" * 64},
               "artifact": {"row_count": 1, "sha256": digest(self.dataset)},
               "inputs": {}, "provenance": {}, "record_sha256": "b" * 64}
        self.generation = old
        self.recipe["seeds"]["dynamics"] = 456
        self.assertEqual(self.assess()["status"], "compatible")

    def test_invalid_or_missing_receipt(self):
        for receipt in (None, {}, {"schema_version": 99}):
            self.generation = receipt
            self.assertEqual(self.assess()["status"], "review_required")

    def test_refresh_migrates_legacy_receipt_without_inventing_versions(self):
        previous = {**self.native,
                    "recipe": {"path": self.recipe_rel, "sha256": "a" * 64},
                    "artifact": {"row_count": 1, "sha256": digest(self.dataset)},
                    "provenance": {"runtime_observation": {
                        "stdout": "index, name, uuid, driver_version\n"
                                  "0, Test GPU, gpu-1, 123.45\n"}},
                    "record_sha256": "b" * 64}
        record = refresh_generation.refresh(
            self.recipe, previous, self.dataset, self.root, [])
        self.assertEqual(record["schema_version"], 4)
        self.assertEqual(record["hardware"]["name"], "Test GPU")
        self.assertNotIn("software", record)
        self.assertEqual(record["output"]["sha256"], digest(self.dataset))

    def test_refresh_accepts_a_coherent_json_edit_without_special_permission(self):
        self.document["results"][0]["price"] = 1.5
        self.save_dataset()
        record = refresh_generation.refresh(
            self.recipe, self.generation, self.dataset, self.root, [])
        self.assertEqual(record["output"]["sha256"], digest(self.dataset))
        self.assertNotIn("curation", record)

    def test_samples_are_checked_as_a_stream(self):
        self.recipe.update(kind="samples", dataset_id="prices", row_count=2,
                           shape={"parameter_count": 1, "paths_per_parameter": 2},
                           maturity_sampling={"minimum_days": 1,
                                              "maximum_days": 10})
        self.dataset.write_text("""{
"database_id": "prices",
"model_family": "test",
"row_count": 2,
"parameters": [
{"id":"000001","values":{"spot":1.0}}
],
"results": [
{"parameter_id":"000001","maturity_days":1,"values":{"spot":1.0}},
{"parameter_id":"000001","maturity_days":2,"values":{"spot":1.1}}
]
}""")
        self.generation["output"]["sha256"] = digest(self.dataset)
        self.assertEqual(self.assess()["status"], "compatible")

    def test_native_receipt_cannot_be_enriched_twice(self):
        work = self.root
        job = {"generation": "catalog/example/generation.yaml"}
        with self.assertRaisesRegex(ValueError, "native version-one"):
            provenance.attach_generation(work, job, {"build": str(self.root)})

    def test_source_archive_preserves_actual_bytes(self):
        (self.root / "src").mkdir()
        source = self.root / "src/changed.cuh"
        source.write_text("// dirty and untracked content\n")
        archive = self.root / "sources.tar.gz"
        with patch.object(
            provenance.subprocess, "check_output", return_value=b"src/changed.cuh\0"
        ):
            checksum = provenance.snapshot_sources(self.root, archive)
        self.assertEqual(checksum, digest(archive))
        with tarfile.open(archive) as snapshot:
            self.assertEqual(snapshot.extractfile("src/changed.cuh").read(), source.read_bytes())

    def test_cli_is_read_only_and_does_not_need_build(self):
        args = ["checker", "--recipe", str(self.recipe_path),
                "--generation", str(self.generation_path),
                "--dataset", str(self.dataset), "--root", str(self.root)]
        before = (digest(self.recipe_path), digest(self.generation_path),
                  digest(self.dataset))
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(), 0)
        self.assertEqual((digest(self.recipe_path), digest(self.generation_path),
                          digest(self.dataset)), before)


if __name__ == "__main__":
    unittest.main()
