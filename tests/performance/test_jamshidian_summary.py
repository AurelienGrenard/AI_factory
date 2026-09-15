"""Qualification rejects incomplete numerics, noisy timings, and weak repeats."""
import copy
import unittest

from tools.performance.summarize_jamshidian_strategy import summarize


def record(repeat):
    return {"model": "cir_plus_plus", "curve": "svensson", "profile": "catalogue",
        "side": "payer", "stage": "confirm", "repeat": repeat,
        "job": {"rows": 1000, "strategy": "cooperative", "threads": 128,
                "blocks": 1000, "grid_policy": "full"},
        "gpu_samples_ms": [1.0] * 21, "warmups": 5, "repetitions": 21,
        "kernel": {"median_ms": 1.0, "coefficient_of_variation": .01},
        "public_api": {"median_ms": 1.02}, "status": "passed",
        "numerically_eligible": True, "timing_eligible": True,
        "numerics": {"reference_invalid_rows": 0, "failed_rows": 0, "maximum_absolute_error": 0},
        "diagnostics": {"resources": {"registers_per_thread": 64},
            "compiled_resources": {"stack_frame_bytes": 0, "local_bytes_per_thread": 0,
                "sass_local_load_instructions": 0, "sass_local_store_instructions": 0},
            "launch": {"dynamic_shared_bytes_per_block": 7200},
            "occupancy": {"theoretical": .5}}, "evidence": f"campaign-{repeat}"}


class SummaryTest(unittest.TestCase):
    def test_three_clean_confirmations_qualify(self):
        self.assertTrue(summarize([record(i) for i in range(3)])[0]["confirmed_candidate"])

    def test_screening_repetition_and_reference_failures_do_not_qualify(self):
        for field, value in (("stage", "screen"), ("repeat", 0), ("repetitions", 3),
                             ("status", "reference_incomplete"), ("timing_eligible", False)):
            rows = [record(i) for i in range(3)]
            for row in rows:
                row[field] = value
            self.assertFalse(summarize(rows)[0]["confirmed_candidate"], field)

    def test_noise_or_cross_run_drift_does_not_qualify(self):
        rows = [record(i) for i in range(3)]
        noisy = copy.deepcopy(rows)
        noisy[0]["kernel"]["coefficient_of_variation"] = .06
        self.assertFalse(summarize(noisy)[0]["confirmed_candidate"])
        rows[0]["kernel"]["median_ms"] = 2.0
        self.assertFalse(summarize(rows)[0]["confirmed_candidate"])


if __name__ == "__main__":
    unittest.main()
