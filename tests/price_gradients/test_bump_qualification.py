"""Fail-closed tests for the cached bump-qualification decision layer."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("qualify_bumps.py")
SPEC = importlib.util.spec_from_file_location("qualify_bumps", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BumpQualificationTests(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "schema_version": 1, "policy_id": "test-v1", "status": "candidate",
            "scope": {"models": ["m"]},
            "execution": {"seeds": [1, 2, 3], "minimum_independent_seeds": 3,
                          "bump_factors": [1.0], "paths_per_price": 16,
                          "threads_per_block": 256, "sensitivity_batch_size": 1,
                          "refinements": {"m": [1]}},
            "criteria": {"reference_price_convergence_absolute": 2e-7,
                         "reference_gradient_convergence_absolute": 1e-3,
                         "reference_derivative_convergence_absolute": 2e-3,
                         "monte_carlo_standard_errors": 6.0,
                         "total_error_absolute": 2e-4, "total_error_relative": 5e-3,
                         "stencil_bias_absolute": 2e-4, "stencil_bias_relative": 5e-3,
                         "fp32_error_absolute": 2e-4, "fp32_error_relative": 5e-3},
            "base_bumps": {"m": {"x": {}}},
            "cases": {"m": [{"id": "c", "classes": ["ordinary"]}]},
            "domain_policy": {"required_case_classes": ["ordinary"]},
        }
        rows = []
        for seed in (1, 2, 3):
            rows.append({"model": "m", "coordinate": "x", "case_id": "c",
                         "case_classes": ["ordinary"], "factor": 1.0, "refinement": 1,
                         "seed": seed, "paths": 16, "threads_per_block": 256,
                         "sensitivity_batch_size": 1, "dt": 1/504, "mc": 1.0001, "mc_se": .001,
                         "reference_stencil": 1.0, "reference_price_convergence": 1e-8,
                         "reference_gradient_convergence": 1e-5, "derivative": 1.0,
                         "reference_derivative_convergence": 1e-5,
                         "reference_derivative_stencil": "centered",
                         "reference_convergence_applicable": True, "stencil_bias": 0.0})
        self.analysis = {"schema_version": 2, "policy_id": "test-v1",
                         "execution_eligible": True, "rows": rows}

    def test_passing_candidate_is_reported(self):
        result = MODULE.qualify(self.policy, self.analysis)
        self.assertEqual(result["summary"]["failed_case_count"], 0)
        self.assertEqual(result["recommendations"][0]["status"], "qualified_candidate")

    def test_stable_but_biased_stencil_fails(self):
        analysis = copy.deepcopy(self.analysis)
        for row in analysis["rows"]:
            row["reference_stencil"] = 1.1
            row["stencil_bias"] = .1
            row["mc"] = 1.1
        result = MODULE.qualify(self.policy, analysis)
        self.assertFalse(result["cases"][0]["checks"]["stencil_bias"])
        self.assertEqual(result["recommendations"][0]["status"], "unqualified")

    def test_failed_case_is_retained(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["rows"][0]["mc"] = 2.0
        result = MODULE.qualify(self.policy, analysis)
        self.assertEqual(len(result["cases"]), 1)
        self.assertFalse(result["cases"][0]["pass"])

    def test_incomplete_seed_set_fails_closed(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["rows"].pop()
        with self.assertRaises(ValueError):
            MODULE.qualify(self.policy, analysis)

    def test_missing_refinement_fails_closed(self):
        policy = copy.deepcopy(self.policy)
        policy["execution"]["refinements"]["m"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "surface mismatch"):
            MODULE.qualify(policy, self.analysis)

    def test_missing_case_fails_closed(self):
        policy = copy.deepcopy(self.policy)
        policy["cases"]["m"].append({"id": "stress", "classes": ["ordinary"]})
        with self.assertRaisesRegex(ValueError, "surface mismatch"):
            MODULE.qualify(policy, self.analysis)

    def test_duplicate_seed_observation_fails_closed(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["rows"].append(copy.deepcopy(analysis["rows"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate analyzed"):
            MODULE.qualify(self.policy, analysis)

    def test_smoke_execution_cannot_qualify(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["execution_eligible"] = False
        result = MODULE.qualify(self.policy, analysis)
        self.assertEqual(result["recommendations"][0]["status"], "ineligible_execution")

    def test_versioned_policy_covers_every_required_case_class(self):
        policy = json.loads(Path(__file__).with_name("qualification_policy_v2.json").read_text())
        required = set(policy["domain_policy"]["required_case_classes"])
        for model in policy["scope"]["models"]:
            covered = {name for case in policy["cases"][model] for name in case["classes"]}
            self.assertTrue(required <= covered, f"{model} lacks {sorted(required - covered)}")

    def test_cli_rejects_analysis_from_another_policy_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy_path, analysis_path, output_path = root/"policy.json", root/"analysis.json", root/"out.json"
            policy_path.write_text(json.dumps(self.policy))
            analysis = copy.deepcopy(self.analysis)
            analysis["fingerprints"] = {"native": "sha256:" + "0"*64,
                                        "policy": "sha256:" + "f"*64}
            analysis_path.write_text(json.dumps(analysis))
            with self.assertRaisesRegex(ValueError, "supplied policy"):
                MODULE.main(policy_path, analysis_path, output_path)


if __name__ == "__main__":
    unittest.main()
