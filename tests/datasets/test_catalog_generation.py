"""Check catalogue staging and resume without CUDA or independent references."""
import json
import copy
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
        self.job = {"target": "generate_test", "kind": "prices", "rows": 2, "inputs": [],
                    "dataset": "datasets/model/equity/markovian/test/prices/calls/test.json",
                    "catalog": "catalog/model/equity/markovian/test/prices/calls/test/dataset.yaml",
                    "launch_plan": {"paths_per_price": 32}, "state": "pending", "previous": {}}
        self.catalog = {"database_id": "test", "row_count": 2,
                        "summary": {"monte_carlo_paths_per_price": 32},
                        "price_construction": {"method": "Aligned"},
                        "validation": {"status": "pending", "verified": False,
                                       "dataset": "validation/datasets/price/equity/markovian/test/calls/test.json"}}
        self.document = {"database_id": "test", "row_count": 2, "results": [
            {"id": f"{i:06d}", "outputs": {"price": 0.1, "standard_error": 0.01}} for i in (1, 2)]}
        for key in ("dataset", "catalog"):
            path = self.root / self.job[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("old artifact")
            self.job["previous"][self.job[key]] = digest(path)
        binary = self.run / "bin" / self.job["target"]
        binary.parent.mkdir(parents=True)
        binary.write_text("frozen binary; mocked execution")
        self.job["binary_sha256"] = digest(binary)
        self.job.update(identity="test/calls", sample_shape=None, recipe="recipe.cpp", semantic_inputs={},
                        rng_stream_seeds={"dynamics": 123},
                        declared_method={"engine": "test", "construction": "aligned"},
                        checkpoint="jobs/generate_test/checkpoint", checkpoint_id="a" * 64)
        recipe = self.run / "sources/recipe.cpp"
        recipe.parent.mkdir(parents=True)
        recipe.write_text("frozen recipe")
        self.job["recipe_sha256"] = digest(recipe)
        (self.run / "sources.tar.gz").write_text("frozen source archive")
        self.state = {"version": 3, "root": str(self.root), "publish": False, "jobs": [self.job],
                      "input_hashes": {}, "build_hashes": {}, "revision": "unit-test",
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
        for key, text in (("dataset", json.dumps(self.document)), ("catalog", yaml.safe_dump(self.catalog))):
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
        metadata = yaml.safe_load((Path(self.job["work"]) / self.job["catalog"]).read_text())
        self.assertEqual(metadata["generation"]["schema_version"], 1)
        self.assertEqual(metadata["generation"]["dataset_sha256"], digest(Path(self.job["work"]) / self.job["dataset"]))
        self.assertEqual(digest(self.root / self.job["dataset"]), self.job["previous"][self.job["dataset"]])
        (Path(self.job["work"]) / self.job["dataset"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Completed output changed"):
            campaign.execute(self.run, self.state)

    def test_price_delta_inventory_inherits_seeds_and_records_bump(self):
        jobs = campaign.inventory(campaign.ROOT, {"price_delta"}, set(), set())
        from capability_manifest import PRICE_DELTA_DATASET_SPECS, PRICE_DELTA_SOURCE_BY_RECIPE, resolve_rng_domain
        self.assertEqual(len(jobs), len(PRICE_DELTA_DATASET_SPECS))
        for job in jobs:
            self.assertEqual(job["sensitivity"]["relative_full_width"], .01)
            source = PRICE_DELTA_SOURCE_BY_RECIPE[job["recipe"]]
            self.assertEqual(job["sensitivity"]["source_price_recipe"], source.recipe_path)
            if source.engine != "equity_closed_form":
                self.assertEqual(job["rng_stream_seeds"]["dynamics"], resolve_rng_domain(source).seed("dynamics"))

        cartesian = next(job for job in jobs
                         if job["target"] == "generate_heston_european_calls_01_cartesian_price_delta")
        self.assertEqual(cartesian["declared_method"]["construction"], "cartesian")

    def test_price_delta_publication_checks_paired_outputs_and_contract(self):
        self.job.update(kind="price_delta", sensitivity={"parameter": "spot", "method": "centered_crn",
                        "relative_full_width": .01, "source_price_recipe": "original.cpp"}, time_grid=None)
        self.catalog["validation"] = {"status": "pending", "verified": False}
        self.catalog["sensitivity"] = self.job["sensitivity"].copy()
        self.document["sensitivity"] = self.job["sensitivity"].copy()
        self.catalog["summary"]["seed"] = 123
        self.document["summary"] = self.catalog["summary"].copy()
        for row in self.document["results"]:
            row["outputs"].update(delta=.5, delta_standard_error=.02)
            row["spot_bump"] = {"lower": .995, "upper": 1.005, "represented_width": 1.005 - .995}
        work = self.root / "paired"
        self.generator(None, work, None)
        self.assertEqual(len(campaign.check_outputs(work, self.job)), 2)
        self.document["sensitivity"]["relative_full_width"] = .02
        self.generator(None, work, None)
        with self.assertRaisesRegex(ValueError, "sensitivity"):
            campaign.check_outputs(work, self.job)

    def test_price_delta_preparation_and_fft_geometry_are_checked(self):
        self.job.update(kind="price_delta", sensitivity={}, time_grid=None,
                        preparation={"method": "hybrid_fft", "shared_convolution": True})
        self.job["launch_plan"]["path_chunk_size"] = 65536
        self.catalog["validation"] = {"status": "pending", "verified": False}
        self.catalog["summary"].update(seed=123, path_chunk_size=65536,
                                       preparation=self.job["preparation"].copy())
        self.document["summary"] = copy.deepcopy(self.catalog["summary"])
        for row in self.document["results"]:
            row["outputs"].update(delta=.5, delta_standard_error=.02)
            row["spot_bump"] = {"lower": .995, "upper": 1.005, "represented_width": 1.005 - .995}
        work = self.root / "rough-paired"
        self.generator(None, work, None)
        campaign.check_outputs(work, self.job)
        self.document["summary"]["preparation"]["shared_convolution"] = False
        self.generator(None, work, None)
        with self.assertRaisesRegex(ValueError, "preparation"):
            campaign.check_outputs(work, self.job)
        self.document["summary"]["preparation"]["shared_convolution"] = True
        self.document["summary"]["path_chunk_size"] = 8192
        self.generator(None, work, None)
        with self.assertRaisesRegex(ValueError, "geometry"):
            campaign.check_outputs(work, self.job)

    def test_certification_and_path_count_guards(self):
        for certified, paths in ((True, 32), (False, 16)):
            self.catalog["validation"]["verified"] = certified
            self.catalog["summary"]["monte_carlo_paths_per_price"] = paths
            with patch.object(campaign, "run_generator", side_effect=self.generator):
                with self.assertRaises(ValueError):
                    campaign.execute(self.run, self.state)
            self.assertEqual(digest(self.root / self.job["dataset"]), self.job["previous"][self.job["dataset"]])
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
        for relative, message in (("sources.tar.gz", "source archive"), ("sources/recipe.cpp", "recipe")):
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
        self.assertEqual(digest(self.root / self.job["dataset"]), self.job["previous"][self.job["dataset"]])

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
        catalog = {"construction": {"maturity_sampling": {"minimum_days": 63, "maximum_days": 504}}}
        envelope = {"database_id": "samples", "row_count": 2,
                    "construction": {"parameter_count": 1, "paths_per_parameter": 2}}
        rows = [{"id": f"{i:06d}", "parameters": {"x": 1.0}, "values": {"spot": 1.1},
                 "maturity_days": 252, "T": 1.0} for i in (1, 2)]
        contents = json.dumps(envelope)[:-1] + ',\n  "samples": [\n'
        contents += ",\n".join(json.dumps(row) for row in rows) + "\n  ]\n}\n"
        path.write_text(contents)
        campaign.check_samples(path, job, catalog)
        for invalid in (contents[:-6], contents.replace('"spot": 1.1', '"spot": NaN')):
            path.write_text(invalid)
            with self.assertRaises(ValueError):
                campaign.check_samples(path, job, catalog)


if __name__ == "__main__":
    unittest.main()
