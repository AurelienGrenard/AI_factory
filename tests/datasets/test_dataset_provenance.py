"""Check dataset integrity, path-independent inputs and conservative reuse decisions."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools.datasets import dataset_provenance as provenance
from tools.datasets.artifact_publication import digest
from tools.datasets import check_dataset_compatibility as checker


class ProvenanceTests(unittest.TestCase):
    def test_price_delta_bump_and_method_are_semantic(self):
        job = {"kind": "price_delta", "identity": "heston/european_option", "dataset": "paired.json",
               "rows": 2, "sample_shape": None, "rng_stream_seeds": {"dynamics": 719},
               "launch_plan": {"paths_per_price": 1048576}, "time_grid": {"steps_per_year": 504},
               "sensitivity": {"parameter": "spot", "relative_full_width": .01, "method": "centered_crn"}}
        original = provenance.fingerprint(provenance.specification(job))
        for key, value in (("relative_full_width", .02), ("method", "independent_bumps")):
            changed = copy.deepcopy(job)
            changed["sensitivity"][key] = value
            self.assertNotEqual(original, provenance.fingerprint(provenance.specification(changed)))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "prices.json"
        self.dataset.write_text('{"results": [{"price": 1.25}]}')
        self.parameters = {"database_id": "model_01", "row_count": 2,
                           "catalog": "old/location", "models": [
                               {"id": "1", "parameters": {"spot": 1.0}},
                               {"id": "2", "parameters": {"spot": 2.0}}]}
        (self.root / "parameters.json").write_text(json.dumps(self.parameters))
        self.catalog_path = self.root / "dataset.yaml"
        self.catalog_path.write_text(yaml.safe_dump({
            "database_id": "prices", "row_count": 1, "catalog": "old/catalog", "url": "old/url",
            "summary": {"monte_carlo_paths_per_price": 1048576, "threads_per_block": 256},
            "time_grid": {"steps_per_year": 504}, "validation": {"verified": False},
            "model_dataset": {"id": "model_01", "catalog": "old/model", "url": "old/model/url"},
            "timing": {"kernel_seconds": 1.0},
        }))
        self.job = {"kind": "prices", "identity": "model/product", "dataset": "prices.json",
                    "catalog": "dataset.yaml", "rows": 1, "sample_shape": None,
                    "rng_stream_seeds": {"dynamics": 123}, "declared_method": {"engine": "MC"},
                    "launch_plan": {"paths_per_price": 1048576, "threads_per_block": 256},
                    "recipe": "recipe.cpp", "inputs": ["parameters.json"],
                    "binary_sha256": "b" * 64, "recipe_sha256": "c" * 64,
                    "semantic_inputs": provenance.input_fingerprints(self.root, ["parameters.json"])}
        self.state = {"revision": "entry-worktree", "source_archive_sha256": "d" * 64,
                      "build_hashes": {"CMakeCache.txt": "e" * 64, "build.ninja": "f" * 64},
                      "input_hashes": {"parameters.json": digest(self.root / "parameters.json")}}
        provenance.attach_generation(self.root, self.job, self.state)
        self.catalog = yaml.safe_load(self.catalog_path.read_text())
        record = self.catalog["generation"]
        self.candidate = {key: copy.deepcopy(record[key]) for key in ("specification", "inputs", "execution")}

    def status(self):
        return provenance.assess(self.catalog, self.dataset, self.candidate)["status"]

    def test_matching_generation_does_not_claim_certification(self):
        before = self.dataset.read_bytes()
        result = provenance.assess(self.catalog, self.dataset, self.candidate)
        self.assertEqual(result["status"], "compatible")
        self.assertEqual(result["certification"], "not_assessed")
        self.assertEqual(self.dataset.read_bytes(), before)
        self.assertEqual(self.catalog["validation"], {"verified": False})

    def test_relocation_and_documentation_do_not_require_regeneration(self):
        moved = self.root / "moved.json"
        self.dataset.rename(moved)
        self.dataset = moved
        self.catalog.update(catalog="new/catalog", url="new/url", title="new title", timing={"kernel_seconds": 99})
        self.catalog["model_dataset"].update(catalog="new/model", url="new/model/url")
        self.assertEqual(self.status(), "compatible")
        # Exact source history is recorded but never used as a global invalidator.
        (self.root / "renamed_codegen.py").write_text("# renderer reorganisation\n")
        self.assertEqual(self.status(), "compatible")

    def test_parameter_locations_formatting_and_timing_are_not_semantics(self):
        self.parameters.update(catalog="renamed", url="new", timing={"wall": 99})
        (self.root / "new.json").write_text(json.dumps(self.parameters, indent=4, sort_keys=True))
        self.candidate["inputs"] = provenance.input_fingerprints(self.root, ["new.json"])
        self.assertEqual(self.status(), "compatible")

    def test_parameter_values_order_and_unknown_semantic_fields_invalidate(self):
        for change in ("value", "order", "new_field"):
            document = copy.deepcopy(self.parameters)
            if change == "value":
                document["models"][0]["parameters"]["spot"] = 1.1
            elif change == "order":
                document["models"].reverse()
            else:
                document["new_numerical_convention"] = "changed"
            (self.root / "new.json").write_text(json.dumps(document))
            self.candidate["inputs"] = provenance.input_fingerprints(self.root, ["new.json"])
            self.assertEqual(self.status(), "regenerate", change)

    def test_seed_paths_and_shape_changes_require_a_new_dataset(self):
        for key, value in (("rng_stream_seeds", {"dynamics": 124}), ("paths_per_price", 65536),
                           ("sample_shape", [12000, 250]), ("rows", 2)):
            with self.subTest(key=key):
                old = self.candidate["specification"][key]
                self.candidate["specification"][key] = value
                self.assertEqual(self.status(), "regenerate")
                self.candidate["specification"][key] = old

    def test_kernel_precision_recipe_geometry_and_build_changes_require_review(self):
        for key in self.candidate["execution"]:
            with self.subTest(key=key):
                old = self.candidate["execution"][key]
                self.candidate["execution"][key] = "changed"
                self.assertEqual(self.status(), "review_required")
                self.candidate["execution"][key] = old

    def test_corrupted_output_and_metadata_are_invalid(self):
        original = self.dataset.read_bytes()
        self.dataset.write_text("changed payoff result")
        self.assertEqual(self.status(), "invalid")
        self.dataset.write_bytes(original)
        self.catalog["time_grid"]["steps_per_year"] = 252
        self.assertEqual(self.status(), "invalid")

    def test_record_checksum_and_missing_output_fail_closed(self):
        self.catalog["generation"]["execution"]["binary_sha256"] = "modified"
        self.assertEqual(self.status(), "invalid")
        self.catalog = yaml.safe_load(self.catalog_path.read_text())
        self.dataset.unlink()
        self.assertEqual(self.status(), "invalid")

    def test_malformed_schema_is_not_accepted_with_a_recomputed_checksum(self):
        record = self.catalog["generation"]
        del record["dataset_sha256"]
        record["record_sha256"] = provenance.fingerprint({k: v for k, v in record.items() if k != "record_sha256"})
        self.assertEqual(self.status(), "invalid")

    def test_legacy_and_unknown_versions_require_review_not_invented_history(self):
        for record in (None, {}, {"schema_version": 99}):
            self.catalog["generation"] = record
            self.assertEqual(self.status(), "review_required")

    def test_certification_is_separate(self):
        self.catalog["validation"] = {"verified": True, "status": "available"}
        self.assertEqual(self.status(), "compatible")
        self.assertEqual(provenance.assess(self.catalog, self.dataset, self.candidate)["certification"], "not_assessed")

    def test_unknown_duplicate_and_nonfinite_parameter_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            provenance.input_fingerprints(self.root, ["parameters.json", "parameters.json"])
        for document in ({}, {**self.parameters, "row_count": 3},
                         {**self.parameters, "extra": float("nan")}):
            (self.root / "bad.json").write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                provenance.parameter_fingerprint(self.root / "bad.json")

    def test_samples_have_the_same_record_without_price_paths(self):
        catalog = {k: v for k, v in self.catalog.items() if k != "generation"}
        self.catalog_path.write_text(yaml.safe_dump(catalog))
        self.job.update(kind="samples", sample_shape=[1, 2], rows=2, inputs=[], semantic_inputs={})
        del self.job["launch_plan"]
        provenance.attach_generation(self.root, self.job, self.state)
        record = yaml.safe_load(self.catalog_path.read_text())["generation"]
        self.assertEqual(record["specification"]["sample_shape"], [1, 2])
        self.assertIsNone(record["specification"]["paths_per_price"])
        self.assertEqual(record["inputs"], {})

    def test_no_overwrite_of_existing_generation_history(self):
        with self.assertRaisesRegex(ValueError, "already contains"):
            provenance.attach_generation(self.root, self.job, self.state)

    def test_source_archive_preserves_actual_file_bytes(self):
        (self.root / "src").mkdir()
        (self.root / "src/changed.cuh").write_text("// dirty and untracked content\n")
        archive = self.root / "sources.tar.gz"
        with patch.object(provenance.subprocess, "check_output", return_value=b"src/changed.cuh\0"):
            checksum = provenance.snapshot_sources(self.root, archive)
        self.assertEqual(checksum, digest(archive))
        with tarfile.open(archive) as snapshot:
            self.assertEqual(snapshot.extractfile("src/changed.cuh").read(), (self.root / "src/changed.cuh").read_bytes())

    def test_read_only_cli_and_integrity_before_build_lookup(self):
        args = ["checker", "--catalog", str(self.catalog_path), "--dataset", str(self.dataset),
                "--target", "generate_fixture", "--build", str(self.root / "missing-build")]
        for corrupt in (False, True):
            if corrupt:
                self.dataset.write_text("corrupt")
            hashes = (digest(self.dataset), digest(self.catalog_path))
            output = io.StringIO()
            with patch.object(checker, "candidate_descriptor", return_value=self.candidate) as describe:
                with patch("sys.argv", args), contextlib.redirect_stdout(output):
                    code = checker.main()
            self.assertEqual(code, 4 if corrupt else 0)
            self.assertEqual(describe.call_count, 0 if corrupt else 1)
            self.assertEqual((digest(self.dataset), digest(self.catalog_path)), hashes)
            self.assertEqual(json.loads(output.getvalue())["certification"], "not_assessed")

    def test_candidate_uses_live_inputs_and_compiled_price_plan_without_generation(self):
        job = {**self.job, "target": "generate_fixture"}
        (self.root / "generate_fixture").write_text("binary")
        (self.root / "recipe.cpp").write_text("recipe")
        for name in ("CMakeCache.txt", "build.ninja"):
            (self.root / name).write_text("build")
        with patch.object(checker, "inventory", return_value=[job]):
            with patch.object(checker, "require_current_build"):
                with patch.object(checker.subprocess, "check_output", return_value=json.dumps(job["launch_plan"])) as command:
                    result = checker.candidate_descriptor(self.root, self.root, job["target"])
        self.assertEqual(result["specification"]["rows"], 2)
        self.assertEqual(result["specification"]["paths_per_price"], 1048576)
        self.assertEqual(result["inputs"], self.job["semantic_inputs"])
        self.assertEqual(Path(command.call_args.args[0][0]).name, "inspect_pricing_launch_plan")

    def test_legacy_cli_does_not_inspect_or_modify_the_current_build(self):
        catalog = {k: v for k, v in self.catalog.items() if k != "generation"}
        self.catalog_path.write_text(yaml.safe_dump(catalog))
        args = ["checker", "--catalog", str(self.catalog_path), "--dataset", str(self.dataset),
                "--target", "generate_fixture"]
        with patch.object(checker, "candidate_descriptor") as describe:
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 2)
            describe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
