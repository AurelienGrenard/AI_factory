"""CPU checks for Jamshidian workload coverage and strategy selection."""
from pathlib import Path
import unittest
from unittest.mock import patch

import run_jamshidian_strategy as runner


class StrategyTest(unittest.TestCase):
    def test_inventory_covers_every_one_factor_composition(self):
        self.assertEqual({(x["model"], x["curve"]) for x in runner.compositions()}, {
            ("cir", ""), ("vasicek", ""), ("ornstein_uhlenbeck", ""),
            ("hull_white", "nelson_siegel"), ("hull_white", "svensson"),
            ("cir_plus_plus", "nelson_siegel"), ("cir_plus_plus", "svensson")})

    def test_geometries_are_unique_and_bounded(self):
        jobs = runner.broad_jobs(runner.SIZES)
        keys = {(j["rows"], j["strategy"], j["threads"], j["blocks"]) for j in jobs}
        self.assertEqual(len(keys), len(jobs))
        for j in jobs:
            self.assertIn(j["threads"], (64, 128, 256, 512))
            self.assertLessEqual(j["blocks"], j["rows"])
            self.assertGreater(j["blocks"], 0)

    def test_full_grid_adapts_to_price_count(self):
        self.assertEqual(runner.geometry(1048576, "scalar", 256, "full")["blocks"], 4096)
        self.assertEqual(runner.geometry(1048576, "cooperative", 128, "full")["blocks"], 1048576)
        self.assertEqual(runner.geometry(1048576, "cooperative", 128, "1024")["blocks"], 1024)

    def test_failed_or_power_ineligible_results_cannot_select_candidates(self):
        good = {"model": "cir", "curve": "", "profile": "catalogue", "side": "payer",
                "status": "passed", "timing_eligible": True}
        self.assertEqual(runner.matching([good, {**good, "status": "numerical_failure"},
            {**good, "timing_eligible": False}, {**good, "status": "reference_incomplete"}],
            {"model": "cir", "curve": ""}, "catalogue", "payer"), [good])

    def test_thermal_activity_is_not_an_execution_veto(self):
        snapshot = {"power_source": "external_power", "temperature_c": 95,
                    "throttle": {"hardware_thermal_slowdown": "Active", "hardware_power_brake": "Not Active"},
                    "concurrent_compute_processes": []}
        self.assertIsNone(runner.execution_issue(snapshot, Path("/tmp/probe")))
        snapshot["power_source"] = "battery"
        self.assertIsNotNone(runner.execution_issue(snapshot, Path("/tmp/probe")))

    def test_wsl_anonymous_process_requires_verified_ownership(self):
        snapshot = {"power_source": "external_power", "throttle": {},
                    "concurrent_compute_processes": ["123, [Not Found]"]}
        with patch.object(runner, "own_benchmark_process", return_value=True):
            self.assertIsNone(runner.execution_issue(snapshot, Path("/tmp/probe"), running=True))
            self.assertIsNotNone(runner.execution_issue(snapshot, Path("/tmp/probe")))
        with patch.object(runner, "own_benchmark_process", return_value=False):
            self.assertIsNotNone(runner.execution_issue(snapshot, Path("/tmp/probe"), running=True))

    def test_exited_pid_is_allowed_only_after_prior_ownership_verification(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError), patch.object(Path, "exists", return_value=False):
            self.assertTrue(runner.own_benchmark_process("123, [Not Found]", Path("/tmp/probe"), {123}))
            self.assertFalse(runner.own_benchmark_process("123, [Not Found]", Path("/tmp/probe"), set()))


if __name__ == "__main__":
    unittest.main()
