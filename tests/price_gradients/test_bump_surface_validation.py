"""Exact native-campaign inventory checks before reference regeneration."""
import copy
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("analyze_bumps.py")
SPEC = importlib.util.spec_from_file_location("analyze_bumps", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeSurfaceValidationTests(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "policy_id": "surface-v2", "scope": {"models": ["m"]},
            "base_bumps": {"m": {"x": {}}},
            "cases": {"m": [{"id": "c", "classes": ["ordinary"]}]},
            "execution": {"native_schema_version": 2, "paths_per_price": 16,
                          "threads_per_block": 256, "sensitivity_batch_size": 1,
                          "seeds": [1, 2], "bump_factors": [1.0],
                          "refinements": {"m": [1, 2]}},
        }
        self.records = []
        for refinement in (1, 2):
            for seed in (1, 2):
                self.records.append({
                    "schema_version": 2, "policy_id": "surface-v2", "model": "m",
                    "refinement": refinement, "factor": 1.0, "seed": seed,
                    "paths": 16, "threads_per_block": 256, "sensitivity_batch_size": 1,
                    "dt": 1/(504*refinement), "coordinates": ["x"],
                    "case_ids": ["c"], "case_classes": [["ordinary"]],
                    "rows": 1, "k": 1, "scenarios": [{}, {}, {}], "stencils": [{}],
                    "price": [0.], "price_se": [0.], "gradient": [0.], "gradient_se": [0.],
                })

    def test_complete_surface_passes(self):
        MODULE._validate_surface(self.records, self.policy)

    def test_missing_grid_fails(self):
        with self.assertRaisesRegex(ValueError, "surface mismatch"):
            MODULE._validate_surface([r for r in self.records if r["refinement"] == 1], self.policy)

    def test_missing_case_fails(self):
        policy = copy.deepcopy(self.policy)
        policy["cases"]["m"].append({"id": "stress", "classes": ["stress"]})
        with self.assertRaisesRegex(ValueError, "case identity mismatch"):
            MODULE._validate_surface(self.records, policy)

    def test_duplicate_seed_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate native"):
            MODULE._validate_surface(self.records + [copy.deepcopy(self.records[0])], self.policy)

    def test_effective_launch_mismatch_fails(self):
        records = copy.deepcopy(self.records)
        records[0]["threads_per_block"] = 128
        with self.assertRaisesRegex(ValueError, "launch configuration"):
            MODULE._validate_surface(records, self.policy)


if __name__ == "__main__":
    unittest.main()
