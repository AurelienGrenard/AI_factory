"""Check split recipe/generation provenance and conservative reuse decisions."""

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
from tools.datasets.artifact_publication import digest


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "prices.json"
        self.dataset.write_text('{"results": [{"price": 1.25}]}')
        self.recipe_path = self.root / "recipe.yaml"
        self.recipe = {
            "schema_version": 1,
            "kind": "prices",
            "dataset_id": "prices",
            "generator": "generator.cpp",
            "output": {"path": "prices.json", "format": "json"},
            "generation_output": "generation.yaml",
            "paths_per_price": 1_048_576,
            "seeds": {"dynamics": 123},
        }
        self.recipe_path.write_text(yaml.safe_dump(self.recipe))
        self.generation_path = self.root / "generation.yaml"
        self.generation_path.write_text(yaml.safe_dump({
            "schema_version": 1,
            "status": "complete",
            "artifact": {"row_count": 1},
            "execution": {"paths_per_price": 1_048_576},
            "timing": {"wall_seconds": 1.0, "kernel_seconds": .9},
        }))
        self.parameters = {
            "database_id": "model_01", "row_count": 2,
            "catalog": "old/location",
            "models": [
                {"id": "1", "parameters": {"spot": 1.0}},
                {"id": "2", "parameters": {"spot": 2.0}},
            ],
        }
        (self.root / "parameters.json").write_text(json.dumps(self.parameters))
        self.job = {
            "dataset": "prices.json", "generation": "generation.yaml",
            "recipe": "recipe.yaml", "generator": "generator.cpp",
            "inputs": ["parameters.json"],
            "binary_sha256": "b" * 64,
            "generator_sha256": "c" * 64,
            "recipe_sha256": digest(self.recipe_path),
            "semantic_inputs": provenance.input_fingerprints(
                self.root, ["parameters.json"]
            ),
            "declared_method": {"engine": "MC"},
            "launch_plan": {"paths_per_price": 1_048_576},
        }
        self.state = {
            "revision": "clean-revision",
            "source_archive_sha256": "d" * 64,
            "build_hashes": {
                "CMakeCache.txt": "e" * 64,
                "build.ninja": "f" * 64,
            },
            "input_hashes": {
                "parameters.json": digest(self.root / "parameters.json")
            },
        }
        provenance.attach_generation(self.root, self.job, self.state)
        self.generation = yaml.safe_load(self.generation_path.read_text())
        self.candidate = {
            "recipe_sha256": digest(self.recipe_path),
            "inputs": copy.deepcopy(self.generation["inputs"]),
            "execution": {
                key: copy.deepcopy(self.generation["execution"][key])
                for key in (
                    "binary_sha256", "generator_sha256", "declared_method",
                    "launch_plan", "build_hashes",
                )
            },
        }

    def assess(self, candidate=True):
        return provenance.assess(
            self.recipe,
            self.generation,
            self.dataset,
            self.candidate if candidate else None,
            digest(self.recipe_path),
        )

    def test_matching_generation_is_compatible_without_certification(self):
        result = self.assess()
        self.assertEqual(result["status"], "compatible")
        self.assertEqual(result["certification"], "not_assessed")

    def test_recipe_change_requires_new_dataset(self):
        self.recipe["seeds"]["dynamics"] = 124
        self.recipe_path.write_text(yaml.safe_dump(self.recipe))
        self.assertEqual(self.assess()["status"], "regenerate")

    def test_parameter_content_and_order_are_semantic(self):
        for change in ("value", "order", "new_field"):
            document = copy.deepcopy(self.parameters)
            if change == "value":
                document["models"][0]["parameters"]["spot"] = 1.1
            elif change == "order":
                document["models"].reverse()
            else:
                document["numerical_convention"] = "changed"
            path = self.root / "new.json"
            path.write_text(json.dumps(document))
            self.candidate["inputs"] = provenance.input_fingerprints(
                self.root, ["new.json"]
            )
            self.assertEqual(self.assess()["status"], "regenerate", change)

    def test_parameter_locations_and_formatting_are_not_semantic(self):
        document = copy.deepcopy(self.parameters)
        document.update(catalog="new/location", url="new/url", timing={"wall": 9})
        path = self.root / "new.json"
        path.write_text(json.dumps(document, indent=4, sort_keys=True))
        self.candidate["inputs"] = provenance.input_fingerprints(
            self.root, ["new.json"]
        )
        self.assertEqual(self.assess()["status"], "compatible")

    def test_execution_changes_require_review(self):
        for key in self.candidate["execution"]:
            old = self.candidate["execution"][key]
            self.candidate["execution"][key] = "changed"
            self.assertEqual(self.assess()["status"], "review_required", key)
            self.candidate["execution"][key] = old

    def test_corruption_is_invalid(self):
        original = self.dataset.read_bytes()
        self.dataset.write_text("corrupt")
        self.assertEqual(self.assess()["status"], "invalid")
        self.dataset.write_bytes(original)
        self.generation["execution"]["binary_sha256"] = "changed"
        self.assertEqual(self.assess()["status"], "invalid")

    def test_missing_or_unknown_receipt_requires_review(self):
        for generation in (None, {}, {"schema_version": 99}):
            self.generation = generation
            self.assertEqual(self.assess()["status"], "review_required")

    def test_duplicate_and_invalid_parameter_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            provenance.input_fingerprints(
                self.root, ["parameters.json", "parameters.json"]
            )
        for document in ({}, {**self.parameters, "row_count": 3},
                         {**self.parameters, "extra": float("nan")}):
            path = self.root / "bad.json"
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                provenance.parameter_fingerprint(path)

    def test_native_receipt_cannot_be_enriched_twice(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            provenance.attach_generation(self.root, self.job, self.state)

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

    def test_cli_is_read_only_and_checks_integrity_before_build(self):
        args = [
            "checker", "--recipe", str(self.recipe_path),
            "--generation", str(self.generation_path),
            "--dataset", str(self.dataset), "--target", "generate_fixture",
            "--build", str(self.root / "missing-build"),
        ]
        before = (digest(self.recipe_path), digest(self.generation_path), digest(self.dataset))
        with patch.object(checker, "candidate_descriptor", return_value=self.candidate) as describe:
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 0)
        describe.assert_called_once()
        self.assertEqual(
            (digest(self.recipe_path), digest(self.generation_path), digest(self.dataset)),
            before,
        )


if __name__ == "__main__":
    unittest.main()
