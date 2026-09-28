"""Check catalogue staging and resume without CUDA or independent references."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from tools.datasets import generate_catalog as campaign
from tools.datasets.artifact_publication import digest


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        base = "catalog/model/equity/markovian/test/prices/calls/test"
        self.job = {"target": "generate_test", "kind": "prices", "rows": 2, "inputs": [],
                    "dataset": "datasets/model/equity/markovian/test/prices/calls/test.json",
                    "url": "https://datasets.ai-factory.example/v1/model/equity/markovian/test/prices/calls/test.json",
                    "generator": f"{base}/generator.cpp",
                    "recipe": f"{base}/recipe.yaml",
                    "generation": f"{base}/generation.yaml",
                    "validation": f"{base}/validation.yaml",
                    "row_count": 2,
                    "launch_plan": {"paths_per_price": 32}, "state": "pending", "previous": {}}
        self.recipe = {"schema_version": 1, "kind": "prices", "dataset_id": "test",
                       "generator": "generator.cpp",
                       "output": {"path": self.job["dataset"], "format": "json"},
                       "generation_output": self.job["generation"], "url": self.job["url"],
                       "row_count": 2}
        self.receipt = {"schema_version": 1, "status": "complete",
                        "artifact": {"row_count": 2},
                        "execution": {"paths_per_price": 32},
                        "timing": {"wall_seconds": .1, "kernel_seconds": .05}}
        self.document = {"database_id": "test", "url": self.job["url"], "row_count": 2, "results": [
            {"id": f"{i:06d}", "outputs": {"price": 0.1, "standard_error": 0.01}} for i in (1, 2)]}
        for key in ("dataset", "generation"):
            self.job["previous"][self.job[key]] = None
        binary = self.run / "bin" / self.job["target"]
        binary.parent.mkdir(parents=True)
        binary.write_text("frozen binary; mocked execution")
        self.job["binary_sha256"] = digest(binary)
        self.job.update(identity="test/calls", sample_shape=None, semantic_inputs={},
                        rng_stream_seeds={"dynamics": 123},
                        declared_method={"engine": "test", "construction": "aligned"},
                        checkpoint="jobs/generate_test/checkpoint", checkpoint_id="a" * 64)
        generator = self.run / "sources" / self.job["generator"]
        generator.parent.mkdir(parents=True)
        generator.write_text("frozen generator")
        self.job["generator_sha256"] = digest(generator)
        recipe = self.run / "sources" / self.job["recipe"]
        recipe.write_text(yaml.safe_dump(self.recipe))
        self.job["recipe_sha256"] = digest(recipe)
        (self.run / "sources.tar.gz").write_text("frozen source archive")
        self.state = {"version": 4, "root": str(self.root), "build": str(self.root / "build"),
                      "publish": False, "jobs": [self.job],
                      "input_hashes": {}, "build_hashes": {}, "revision": "a" * 40,
                      "source_archive_sha256": digest(self.run / "sources.tar.gz")}
        telemetry = patch.object(campaign, "gpu_observation", return_value={"unavailable": "unit test"})
        telemetry.start()
        self.addCleanup(telemetry.stop)

    def generator(self, _binary, work, _logs, progress=None,
                  checkpoint=None, checkpoint_id=None):
        if progress is not None:
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(json.dumps({"state": "complete"}))
        if checkpoint is not None:
            self.assertEqual(checkpoint_id, "a" * 64)
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "results.checkpoint").write_text("mock checkpoint")
        recipe = work / self.job["recipe"]
        recipe.parent.mkdir(parents=True, exist_ok=True)
        recipe.write_text(yaml.safe_dump(self.recipe))
        for key, text in (("dataset", json.dumps(self.document)), ("generation", yaml.safe_dump(self.receipt))):
            path = work / self.job[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return 0.125

    def test_inventory(self):
        jobs = campaign.inventory(campaign.ROOT, {"prices", "samples"}, set(), set())
        from capability_manifest import AVAILABLE_DATASET_SPECS
        self.assertEqual({job["target"] for job in jobs}, {spec.cmake_target for spec in AVAILABLE_DATASET_SPECS
                         if spec.dataset_kind in {"prices", "samples"}})
        self.assertEqual(len(jobs), len({job["target"] for job in jobs}))

    def test_gradient_plan_uses_recipe_path_count(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        for name, rows in (("models.json", 3), ("products.json", 5)):
            (inputs / name).write_text(json.dumps({"row_count": rows}))
        job = {
            "kind": "price_sensitivities",
            "target": "generate_gradient_test",
            "row_count": 15,
            "inputs": ["models.json", "products.json"],
            "identity": "heston/european_option",
            "declared_method": {"construction": "cartesian"},
            "sensitivity": {"parameters": [{"parameter": "model.kappa"}]},
            "paths_per_price": 262144,
        }
        plan = {"paths_per_price": 262144}
        with (patch.object(campaign.subprocess, "check_output", return_value=json.dumps(plan)) as inspect,
              patch.object(campaign, "input_fingerprints", return_value={})):
            description = campaign.describe_job(inputs, self.root / "bin", job)
        self.assertEqual(description["rows"], 15)
        self.assertEqual(description["launch_plan"], plan)
        self.assertEqual(
            inspect.call_args.args[0],
            [str(self.root / "bin/inspect_pricing_launch_plan"),
             "heston/european_option", "15", "262144", "--price-gradients", "1"],
        )

    def test_full_hessian_plan_uses_mixed_graph_inspector(self):
        inputs = self.root / "mixed-inputs"
        inputs.mkdir()
        for name, rows in (("models.json", 2), ("products.json", 2)):
            (inputs / name).write_text(json.dumps({"row_count": rows}))
        job = {
            "kind": "price_sensitivities",
            "target": "generate_full_hessian_test",
            "row_count": 2,
            "inputs": ["models.json", "products.json"],
            "identity": "heston/european_option",
            "declared_method": {"construction": "aligned"},
            "sensitivity": {
                "parameters": [
                    {"parameter": "model.kappa"},
                    {"parameter": "product.strike"},
                ],
                "orders": [
                    "first",
                    "diagonal_second",
                    "mixed_second",
                ],
            },
            "paths_per_price": 262144,
            "maximum_resident_prices": 16,
        }
        plan = {"sensitivity_strategy": "mixed_node_graph"}
        with (patch.object(
                  campaign.subprocess,
                  "check_output",
                  return_value=json.dumps(plan),
              ) as inspect,
              patch.object(campaign, "input_fingerprints", return_value={})):
            description = campaign.describe_job(
                inputs, self.root / "bin", job
            )
        self.assertEqual(description["rows"], 2)
        self.assertEqual(description["launch_plan"], plan)
        self.assertEqual(
            inspect.call_args.args[0],
            [
                str(self.root / "bin/inspect_pricing_launch_plan"),
                "heston/european_option",
                "2",
                "262144",
                "16",
                "--price-gradients",
                "2",
                "--mixed",
            ],
        )

    def test_diagonal_node_graph_plan_records_strategy_and_batch_limit(self):
        inputs = self.root / "diagonal-inputs"
        inputs.mkdir()
        for name, rows in (("models.json", 4), ("products.json", 3)):
            (inputs / name).write_text(json.dumps({"row_count": rows}))
        job = {
            "kind": "price_sensitivities",
            "target": "generate_diagonal_node_graph_test",
            "row_count": 12,
            "inputs": ["models.json", "products.json"],
            "identity": "heston/european_option",
            "declared_method": {"construction": "cartesian"},
            "sensitivity": {
                "parameters": [
                    {"parameter": "model.kappa"},
                    {"parameter": "product.strike"},
                ],
                "orders": ["first", "diagonal_second"],
            },
            "sensitivity_strategy": "node_graph",
            "paths_per_price": 262144,
            "maximum_resident_prices": 256,
        }
        plan = {"sensitivity_strategy": "node_graph"}
        with (patch.object(
                  campaign.subprocess,
                  "check_output",
                  return_value=json.dumps(plan),
              ) as inspect,
              patch.object(campaign, "input_fingerprints", return_value={})):
            description = campaign.describe_job(
                inputs, self.root / "bin", job
            )
        self.assertEqual(description["rows"], 12)
        self.assertEqual(description["launch_plan"], plan)
        self.assertEqual(
            inspect.call_args.args[0],
            [
                str(self.root / "bin/inspect_pricing_launch_plan"),
                "heston/european_option",
                "12",
                "262144",
                "256",
                "--price-gradients",
                "2",
                "--diagonal",
                "--node-graph",
            ],
        )

    def test_runner_exposes_progress_sidecar_path(self):
        binary = self.root / "progress-generator"
        binary.write_text(
            "#!/bin/sh\n"
            "printf '{\"state\":\"complete\"}\\n' > "
            '"$AI_FACTORY_GENERATION_PROGRESS"\n'
            "printf '{\"event\":\"progress\",\"state\":\"complete\"}\\n' >> "
            '"$AI_FACTORY_GENERATION_PROGRESS_LOG"\n'
            'mkdir -p "$AI_FACTORY_GENERATION_CHECKPOINT_DIR"\n'
            'printf "%s" "$AI_FACTORY_GENERATION_CHECKPOINT_ID" > '
            '"$AI_FACTORY_GENERATION_CHECKPOINT_DIR/identity"\n'
        )
        os.chmod(binary, 0o755)
        work = self.root / "progress-work"
        logs = self.root / "progress-logs"
        progress = self.root / "progress.json"
        checkpoint = self.root / "progress-checkpoint"
        work.mkdir()
        logs.mkdir()
        campaign.run_generator(binary, work, logs, progress, checkpoint, "b" * 64)
        self.assertEqual(json.loads(progress.read_text())["state"], "complete")
        events = [json.loads(line) for line in (logs / "progress.jsonl").read_text().splitlines()]
        self.assertEqual([event["event"] for event in events],
                         ["generator_started", "progress", "generator_exited"])
        self.assertEqual(events[-1]["returncode"], 0)
        self.assertEqual((checkpoint / "identity").read_text(), "b" * 64)

    def test_pilot_does_not_publish_or_rerun(self):
        with patch.object(campaign, "run_generator", side_effect=self.generator) as run:
            campaign.execute(self.run, self.state)
            campaign.execute(self.run, self.state)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.job["progress"], "jobs/generate_test/progress.json")
        self.assertEqual(self.job["progress_journal"],
                         "jobs/generate_test/attempt-001/progress.jsonl")
        self.assertEqual(self.job["state"], "complete")
        self.assertTrue(self.job["checkpoint_cleared"])
        self.assertFalse((self.run / self.job["checkpoint"]).exists())
        events = [json.loads(line) for line in
                  (self.run / self.job["progress_journal"]).read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "job_complete")
        metadata = yaml.safe_load((Path(self.job["work"]) / self.job["generation"]).read_text())
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["artifact"]["sha256"], digest(Path(self.job["work"]) / self.job["dataset"]))
        self.assertFalse((self.root / self.job["dataset"]).exists())
        (Path(self.job["work"]) / self.job["dataset"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Completed output changed"):
            campaign.execute(self.run, self.state)

    def test_spot_sensitivity_inventory_preserves_the_narrow_selection(self):
        jobs = campaign.inventory(
            campaign.ROOT, {"price_sensitivities"}, set(), set()
        )
        from capability_manifest import (
            PRICE_GRADIENT_DATASET_SPECS,
            PRICE_GRADIENT_SOURCE_BY_GENERATOR,
            resolve_rng_domain,
        )
        spot_specs = {
            spec.generator_path: spec
            for spec in PRICE_GRADIENT_DATASET_SPECS
            if spec.sensitivity_parameters == ("model.spot",)
        }
        spot_jobs = [job for job in jobs if job["generator"] in spot_specs]
        self.assertEqual(len(spot_jobs), len(spot_specs))
        for job in spot_jobs:
            self.assertEqual(
                [item["parameter"] for item in job["sensitivity"]["parameters"]],
                ["model.spot"],
            )
            source = PRICE_GRADIENT_SOURCE_BY_GENERATOR[job["generator"]]
            self.assertEqual(
                job["sensitivity"]["source_price_recipe"],
                source.recipe_yaml_path,
            )
            if source.engine not in {
                "equity_closed_form", "fixed_income_closed_form"
            }:
                self.assertEqual(
                    job["rng_stream_seeds"]["dynamics"],
                    resolve_rng_domain(source).seed("dynamics"),
                )

        cartesian = next(
            job for job in spot_jobs
            if "/heston/" in job["generator"]
            and "/european_calls/" in job["generator"]
            and "_cartesian_" in job["generator"]
        )
        self.assertEqual(cartesian["declared_method"]["construction"], "cartesian")

    def test_publication_rejects_a_dataset_url_different_from_the_recipe(self):
        work = self.root / "wrong-url"
        self.document["url"] = "https://datasets.example/wrong.json"
        self.generator(None, work, None)
        with self.assertRaisesRegex(ValueError, "Dataset URL"):
            campaign.check_outputs(work, self.job)

    def test_generation_cannot_claim_validation_and_path_count_is_checked(self):
        for validation, paths in (({"verified": True}, 32), (None, 16)):
            self.receipt["execution"]["paths_per_price"] = paths
            if validation is None:
                self.receipt.pop("validation", None)
            else:
                self.receipt["validation"] = validation
            with patch.object(campaign, "run_generator", side_effect=self.generator):
                with self.assertRaises(ValueError):
                    campaign.execute(self.run, self.state)
            self.assertFalse((self.root / self.job["dataset"]).exists())
        self.assertEqual(self.job["attempt"], 2)

    def test_publication_resumes_without_rerunning_generator(self):
        self.state["publish"] = True
        with patch.object(campaign, "run_generator", side_effect=self.generator) as run:
            with patch.object(campaign, "publish_pair", side_effect=InterruptedError("stopped")):
                with self.assertRaises(InterruptedError):
                    campaign.execute(self.run, self.state)
            self.assertEqual(self.job["state"], "staged")
            campaign.execute(self.run, self.state)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(json.loads((self.root / self.job["dataset"]).read_text()), self.document)

    def test_changed_binary_is_rejected(self):
        (self.run / "bin" / self.job["target"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Frozen executable changed"):
            campaign.execute(self.run, self.state)

    def test_changed_source_archive_and_recipe_are_rejected(self):
        for relative, message in (("sources.tar.gz", "source archive"),
                                  (f"sources/{self.job['generator']}", "generator"),
                                  (f"sources/{self.job['recipe']}", "recipe")):
            path = self.run / relative
            original = path.read_bytes()
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, message):
                campaign.execute(self.run, self.state)
            path.write_bytes(original)

    def test_missing_standard_error_does_not_publish(self):
        self.state["publish"] = True
        del self.document["results"][0]["outputs"]["standard_error"]
        with patch.object(campaign, "run_generator", side_effect=self.generator):
            with self.assertRaisesRegex(ValueError, "standard error"):
                campaign.execute(self.run, self.state)
        self.assertFalse((self.root / self.job["dataset"]).exists())

    def test_failed_dataset_reuses_checkpoint_only_on_explicit_execute(self):
        checkpoint_path = self.run / self.job["checkpoint"]

        def interrupted(_binary, _work, _logs, _progress, checkpoint, checkpoint_id):
            self.assertEqual(checkpoint, checkpoint_path)
            self.assertEqual(checkpoint_id, self.job["checkpoint_id"])
            checkpoint.mkdir(parents=True)
            (checkpoint / "durable-prefix").write_text("two batches")
            raise KeyboardInterrupt

        with patch.object(campaign, "run_generator", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                campaign.execute(self.run, self.state)
        self.assertEqual(self.job["state"], "interrupted")
        self.assertEqual((checkpoint_path / "durable-prefix").read_text(), "two batches")
        first_journal = self.run / self.job["progress_journal"]
        first_contents = first_journal.read_text()
        self.assertEqual(json.loads(first_contents.splitlines()[-1])["event"], "job_interrupted")
        def resumed(*arguments):
            self.assertEqual((checkpoint_path / "durable-prefix").read_text(), "two batches")
            return self.generator(*arguments)

        with patch.object(campaign, "run_generator", side_effect=resumed) as run:
            campaign.execute(self.run, self.state)
            campaign.execute(self.run, self.state)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.job["attempt"], 2)
        self.assertEqual(first_journal.read_text(), first_contents)
        self.assertEqual(json.loads((self.run / self.job["progress_journal"])
                                    .read_text().splitlines()[-1])["event"], "job_complete")
        self.assertNotIn("last_error", self.job)
        checkpoint_arguments = run.call_args.args
        self.assertEqual(checkpoint_arguments[4], self.run / self.job["checkpoint"])
        self.assertEqual(checkpoint_arguments[5], self.job["checkpoint_id"])

    def test_streamed_samples(self):
        path = self.root / "samples.json"
        job = {"rows": 2, "dataset": "samples.json", "sample_shape": [1, 2]}
        recipe = {"maturity_sampling": {"minimum_days": 63, "maximum_days": 504}}
        envelope = {"database_id": "samples", "row_count": 2,
                    "construction": {"parameter_count": 1, "paths_per_parameter": 2}}
        rows = [{"id": f"{i:06d}", "parameters": {"x": 1.0}, "values": {"spot": 1.1},
                 "maturity_days": 252, "T": 1.0} for i in (1, 2)]
        contents = json.dumps(envelope)[:-1] + ',\n  "samples": [\n'
        contents += ",\n".join(json.dumps(row) for row in rows) + "\n  ]\n}\n"
        path.write_text(contents)
        campaign.check_samples(path, job, recipe)
        for invalid in (contents[:-6], contents.replace('"spot": 1.1', '"spot": NaN')):
            path.write_text(invalid)
            with self.assertRaises(ValueError):
                campaign.check_samples(path, job, recipe)


if __name__ == "__main__":
    unittest.main()
