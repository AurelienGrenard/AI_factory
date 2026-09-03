"""Fail-closed tests for the CUDA performance baseline checker."""

from __future__ import annotations

import copy
import unittest

from validation.performance.check_baseline import compare
from validation.performance.run_baseline import best_stable_manifest


ENVIRONMENT = {
    "gpu": "test GPU",
    "compute_capability": "8.9",
    "sm_count": 1,
    "driver_version": 13020,
    "runtime_version": 13030,
    "cuda_compiler_version": "13.3.73",
}


def measurement(variant: str) -> dict[str, object]:
    return {
        "finding": "PERF-TEST",
        "benchmark": "manifest",
        "variant": variant,
        "configuration": {"row_count": 8},
        "protocol_version": 1,
        "environment": copy.deepcopy(ENVIRONMENT),
        "kernel": {
            "median_ms": 10.0,
            "coefficient_of_variation": 0.01,
        },
        "wall": {"median_ms": 11.0},
    }


def baseline() -> dict[str, object]:
    references = [measurement("one"), measurement("two")]
    for reference in references:
        reference.pop("protocol_version")
        reference.pop("environment")
    return {
        "protocol_version": 1,
        "environment": copy.deepcopy(ENVIRONMENT),
        "decision_policy": {
            "minimum_effect_size": 0.05,
            "maximum_coefficient_of_variation": 0.05,
            "maximum_publication_coefficient_of_variation": 0.10,
        },
        "measurements": references,
    }


class PerformanceBaselineCheckerTest(unittest.TestCase):
    def assert_rejected(
        self,
        reference: dict[str, object],
        candidates: list[dict[str, object]],
        fragment: str,
    ) -> None:
        failures, _, _ = compare(reference, candidates)
        self.assertTrue(
            any(fragment in failure for failure in failures),
            failures,
        )

    def test_accepts_complete_manifest_once(self) -> None:
        failures, inconclusive, informational = compare(
            baseline(), [measurement("one"), measurement("two")]
        )
        self.assertEqual(failures, [])
        self.assertEqual(inconclusive, [])
        self.assertEqual(informational, [])

    def test_rejects_partial_manifest(self) -> None:
        self.assert_rejected(
            baseline(), [measurement("one")], "missing candidate measurement"
        )

    def test_rejects_duplicate_candidate(self) -> None:
        self.assert_rejected(
            baseline(),
            [measurement("one"), measurement("one"), measurement("two")],
            "duplicate candidate measurement",
        )

    def test_rejects_unknown_candidate(self) -> None:
        self.assert_rejected(
            baseline(),
            [measurement("one"), measurement("two"), measurement("three")],
            "no matching baseline",
        )

    def test_rejects_environment_mismatch(self) -> None:
        candidates = [measurement("one"), measurement("two")]
        candidates[0]["environment"]["gpu"] = "another GPU"
        self.assert_rejected(
            baseline(), candidates, "environment field gpu mismatch"
        )

    def test_rejects_protocol_mismatch(self) -> None:
        candidates = [measurement("one"), measurement("two")]
        candidates[0]["protocol_version"] = 2
        self.assert_rejected(
            baseline(), candidates, "protocol version mismatch"
        )

    def test_marks_noise_inconclusive(self) -> None:
        candidates = [measurement("one"), measurement("two")]
        candidates[0]["kernel"]["coefficient_of_variation"] = 0.06
        failures, inconclusive, informational = compare(
            baseline(), candidates
        )
        self.assertEqual(failures, [])
        self.assertEqual(len(inconclusive), 1)
        self.assertEqual(informational, [])

    def test_reports_declared_informational_noise_without_blocking(self) -> None:
        reference = baseline()
        reference["measurements"][0]["comparison_policy"] = "informational"
        candidates = [measurement("one"), measurement("two")]
        candidates[0]["kernel"]["coefficient_of_variation"] = 0.06
        failures, inconclusive, informational = compare(
            reference, candidates
        )
        self.assertEqual(failures, [])
        self.assertEqual(inconclusive, [])
        self.assertEqual(len(informational), 1)

    def test_requires_and_compares_publication_wall_when_baselined(self) -> None:
        reference = baseline()
        reference["measurements"][0]["publication_wall"] = {
            "median_ms": 20.0,
            "coefficient_of_variation": 0.01,
        }
        candidates = [measurement("one"), measurement("two")]
        self.assert_rejected(
            reference,
            candidates,
            "missing candidate timing publication_wall",
        )
        candidates[0]["publication_wall"] = {
            "median_ms": 22.0,
            "coefficient_of_variation": 0.01,
        }
        self.assert_rejected(
            reference,
            candidates,
            "publication_wall median regression",
        )

    def test_selects_lowest_median_stable_campaign(self) -> None:
        first = measurement("one")
        second = copy.deepcopy(first)
        third = copy.deepcopy(first)
        first["kernel"]["median_ms"] = 9.0
        first["kernel"]["coefficient_of_variation"] = 0.08
        second["kernel"]["median_ms"] = 11.0
        third["kernel"]["median_ms"] = 10.0
        selected = best_stable_manifest(
            [[first], [second], [third]], 0.05
        )
        self.assertEqual(selected[0]["kernel"]["median_ms"], 10.0)

    def test_publication_uses_its_declared_host_noise_limit(self) -> None:
        reference = baseline()
        reference["measurements"][0]["publication_wall"] = {
            "median_ms": 20.0,
            "coefficient_of_variation": 0.08,
        }
        candidates = [measurement("one"), measurement("two")]
        candidates[0]["publication_wall"] = {
            "median_ms": 20.0,
            "coefficient_of_variation": 0.09,
        }
        failures, inconclusive, _ = compare(reference, candidates)
        self.assertEqual(failures, [])
        self.assertEqual(inconclusive, [])


if __name__ == "__main__":
    unittest.main()
