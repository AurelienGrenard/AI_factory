"""Fail-closed tests for the CUDA performance baseline checker."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.performance.check_baseline import compare
from tools.performance.profile_kernel import ncu_command, select_profile_target
from tools.performance.run_baseline import (
    aggregate_campaigns,
    campaign_series_can_complete,
    initialize_baseline,
    load_raw_campaigns,
    parse_nvidia_power_limits,
    select_initialization_manifest,
    stabilize_thermal_environment,
    validate_campaign_preflight,
    validate_preflight,
    write_audit_reports,
    write_rebaseline_diff,
    write_raw_campaigns,
)


ENVIRONMENT = {
    "gpu": "test GPU",
    "compute_capability": "8.9",
    "sm_count": 1,
    "memory_bytes": 1_000_000,
    "driver_version": 13020,
    "runtime_version": 13030,
    "cuda_compiler_version": "13.3.73",
}


def timing(median: float) -> dict[str, float]:
    return {
        "minimum_ms": median * 0.9,
        "median_ms": median,
        "p95_ms": median * 1.1,
        "mean_ms": median,
        "standard_deviation_ms": median * 0.01,
        "coefficient_of_variation": 0.01,
    }


def diagnostic(variant: str) -> dict[str, object]:
    compiled = {
        "compiled_symbol": f"_Z_test_kernel_{variant}",
        "registers_per_thread": 32,
        "stack_frame_bytes": 0,
        "static_shared_bytes_per_block": 0,
        "local_bytes_per_thread": 0,
        "sass_code_bytes": 256,
        "sass_instruction_count": 16,
        "sass_local_load_instructions": 0,
        "sass_local_store_instructions": 0,
    }
    return {
        "type": "cuda_kernel_launch_diagnostics",
        "kernel": "test_kernel",
        "variant": variant,
        "compiled_symbol": compiled["compiled_symbol"],
        "compiled_resources": compiled,
        "device": {
            "index": 0,
            "name": "test GPU",
            "compute_capability": "8.9",
        },
        "launch": {
            "grid_block_count": 1,
            "grid": [1, 1, 1],
            "threads_per_block": 128,
            "block": [128, 1, 1],
            "dynamic_shared_bytes_per_block": 0,
        },
        "resources": {
            "registers_per_thread": 32,
            "static_shared_bytes_per_block": 0,
            "local_bytes_per_thread": 0,
            "maximum_threads_per_block": 1024,
            "maximum_dynamic_shared_bytes_per_block": 49152,
        },
        "occupancy": {
            "active_blocks_per_multiprocessor": 4,
            "active_warps_per_multiprocessor": 16,
            "maximum_warps_per_multiprocessor": 64,
            "theoretical": 0.25,
        },
        "code": {"binary_version": 89, "ptx_version": 89},
    }


def timing_budgets() -> dict[str, dict[str, object]]:
    kernel_regression = {
        "rule": "regression",
        "maximum_regression": 0.05,
        "maximum_coefficient_of_variation": 0.05,
    }
    host_regression = {
        "rule": "regression",
        "maximum_regression": 0.05,
        "maximum_coefficient_of_variation": 0.10,
    }
    return {
        "kernel": copy.deepcopy(kernel_regression),
        "public_api": copy.deepcopy(host_regression),
        "pipeline": {"rule": "record_only"},
    }


def measurement(variant: str) -> dict[str, object]:
    return {
        "measurement_id": variant,
        "command_id": "test_command",
        "finding": "PERF-TEST",
        "benchmark": "manifest",
        "variant": variant,
        "configuration": {"row_count": 8},
        "protocol_version": 3,
        "protocol": {
            "warmups": 5,
            "repetitions": 21,
            "primary_statistic": "median_ms",
            "tail_statistic": "p95_ms",
            "minimum_accepted_gain": 0.05,
            "maximum_noise_coefficient": 0.05,
            "maximum_host_noise_coefficient": 0.10,
            "maximum_publication_noise_coefficient": 0.10,
            "wall_semantics": "wall",
        },
        "environment": copy.deepcopy(ENVIRONMENT),
        "kernel": timing(10.0),
        "public_api": timing(11.0),
        "pipeline": timing(11.0),
        "numerical_check": {"finite": True, "price": 1.0},
        "device_memory": {
            "tracked_live_bytes": 1024,
            "observed_resident_bytes": 4096,
            "free_margin_bytes": 995_904,
            "total_bytes": 1_000_000,
        },
        "binary": {
            "executable_bytes": 10_000,
            "executable_sha256": "a" * 64,
            "command_sha256": "b" * 64,
        },
        "resources": [diagnostic(variant)],
        "campaign_aggregation": {
            "method": (
                "median_of_campaign_medians_and_variation_conservative_tail"
            ),
            "attempt_count": 3,
        },
    }


def baseline() -> dict[str, object]:
    references = [measurement("one"), measurement("two")]
    for reference in references:
        reference["id"] = reference.pop("measurement_id")
        reference.pop("protocol_version")
        reference.pop("environment")
        reference["comparison_policy"] = "blocking"
        for resource in reference["resources"]:
            resource["compiled_resource_budgets"] = {
                field: {
                    "rule": "maximum",
                    "value": value,
                    "reason": "test budget",
                }
                for field, value in resource["compiled_resources"].items()
                if field != "compiled_symbol"
            }
        reference["timing_budgets"] = timing_budgets()
        reference["numerical_budgets"] = {
            "finite": {"rule": "exact"},
            "price": {
                "rule": "relative",
                "absolute_tolerance": 1.0e-7,
                "relative_tolerance": 1.0e-6,
            },
        }
        reference["device_memory_budgets"] = {
            "tracked_live_bytes": {"rule": "maximum", "value": 1024},
            "observed_resident_bytes": {"rule": "maximum", "value": 8192},
            "free_margin_bytes": {"rule": "minimum", "value": 900_000},
            "total_bytes": {"rule": "exact"},
        }
        reference["binary_budgets"] = {
            "executable_bytes": {"rule": "maximum", "value": 11_000},
            "executable_sha256": {"rule": "informational"},
            "command_sha256": {"rule": "exact"},
        }
    decisions = [
        {
            "id": f"decision_{index}",
            "outcome": "accept",
            "selected": ["one"],
            "rejected": ["two"],
            "rationale": "test decision",
        }
        for index in range(4)
    ]
    return {
        "protocol_version": 3,
        "environment": copy.deepcopy(ENVIRONMENT),
        "architecture_profile": {
            "compute_capability": "8.9",
            "qualification": "performance_profiled",
            "tuning_profile_id": "test_profile",
        },
        "decision_policy": {
            "warmups": 5,
            "repetitions": 21,
            "primary_statistic": "median_ms",
            "tail_statistic": "p95_ms",
            "maximum_timing_regression": 0.05,
            "maximum_coefficient_of_variation": 0.05,
            "maximum_host_coefficient_of_variation": 0.10,
            "maximum_publication_coefficient_of_variation": 0.10,
            "wall_semantics": "wall",
            "campaign_attempts": 3,
            "maximum_campaign_attempts": 5,
            "campaign_aggregation": (
                "median_of_campaign_medians_and_variation_conservative_tail"
            ),
            "preflight": {
                "accepted_power_sources": ["external_power", "no_battery"],
                "maximum_temperature_c": 85,
                "minimum_current_power_limit_w": 140.0,
                "retry_cooldown_seconds": 0,
                "thermal_stabilization": {
                    "command_id": "test_command",
                    "minimum_runs": 3,
                    "maximum_runs": 3,
                    "minimum_duration_seconds": 0,
                    "temperature_window": 2,
                    "maximum_temperature_range_c": 2,
                },
                "concurrent_compute_processes": "forbidden",
                "forbidden_throttle_reasons": [
                    "hardware_slowdown",
                    "hardware_thermal_slowdown",
                    "software_thermal_slowdown",
                ],
            },
            "timing_reports": {
                "kernel": "kernel",
                "public_api": "public_api",
                "pipeline": "pipeline",
                "publication": "publication_wall",
            },
        },
        "commands": [
            {
                "id": "test_command",
                "executable": "test_benchmark",
                "arguments": [],
            }
        ],
        "audit_reports": {
            "generic_cuda": ["test_command"],
            "model_sampling": [],
            "early_exercise": [],
            "rough": [],
        },
        "decisions": decisions,
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
            any(fragment in failure for failure in failures), failures
        )

    def candidates(self) -> list[dict[str, object]]:
        return [measurement("one"), measurement("two")]

    def test_accepts_complete_manifest_once(self) -> None:
        failures, inconclusive, informational = compare(
            baseline(), self.candidates()
        )
        self.assertEqual(failures, [])
        self.assertEqual(inconclusive, [])
        self.assertEqual(informational, [])

    def test_rejects_partial_duplicate_and_unknown_manifests(self) -> None:
        reference = baseline()
        self.assert_rejected(
            reference, [measurement("one")], "missing candidate measurement"
        )
        self.assert_rejected(
            reference,
            [measurement("one"), measurement("one"), measurement("two")],
            "duplicate candidate measurement",
        )
        self.assert_rejected(
            reference,
            [measurement("one"), measurement("two"), measurement("three")],
            "no matching baseline",
        )

    def test_rejects_environment_protocol_command_and_id_mismatch(self) -> None:
        for field, value, fragment in (
            ("environment", {**ENVIRONMENT, "gpu": "other"}, "environment mismatch"),
            ("protocol_version", 4, "protocol version mismatch"),
            ("command_id", "other", "command id mismatch"),
            ("measurement_id", "other", "measurement id mismatch"),
        ):
            candidates = self.candidates()
            candidates[0][field] = value
            self.assert_rejected(baseline(), candidates, fragment)

    def test_marks_blocking_noise_inconclusive(self) -> None:
        candidates = self.candidates()
        candidates[0]["kernel"]["coefficient_of_variation"] = 0.06
        failures, inconclusive, informational = compare(baseline(), candidates)
        self.assertEqual(failures, [])
        self.assertEqual(len(inconclusive), 1)
        self.assertEqual(informational, [])

    def test_applies_the_distinct_host_noise_budget(self) -> None:
        candidates = self.candidates()
        candidates[0]["public_api"]["coefficient_of_variation"] = 0.08
        failures, inconclusive, informational = compare(baseline(), candidates)
        self.assertEqual(failures, [])
        self.assertEqual(inconclusive, [])
        self.assertEqual(informational, [])

        candidates[0]["public_api"]["coefficient_of_variation"] = 0.11
        failures, inconclusive, informational = compare(baseline(), candidates)
        self.assertEqual(failures, [])
        self.assertEqual(len(inconclusive), 1)
        self.assertEqual(informational, [])

    def test_reports_informational_noise_without_blocking(self) -> None:
        reference = baseline()
        reference["measurements"][0]["comparison_policy"] = "informational"
        candidates = self.candidates()
        candidates[0]["kernel"]["coefficient_of_variation"] = 0.06
        failures, inconclusive, informational = compare(reference, candidates)
        self.assertEqual(failures, [])
        self.assertEqual(inconclusive, [])
        self.assertEqual(len(informational), 1)

    def test_blocks_median_p95_and_public_wall_regressions(self) -> None:
        for field in ("median_ms", "p95_ms"):
            candidates = self.candidates()
            candidates[0]["kernel"][field] *= 1.06
            self.assert_rejected(baseline(), candidates, field)

        reference = baseline()
        reference["measurements"][0]["publication_wall"] = timing(20.0)
        reference["measurements"][0]["timing_budgets"]["publication_wall"] = {
            "rule": "regression",
            "maximum_regression": 0.05,
            "maximum_coefficient_of_variation": 0.10,
        }
        candidates = self.candidates()
        candidates[0]["publication_wall"] = timing(22.0)
        self.assert_rejected(reference, candidates, "publication_wall median_ms")

    def test_rejects_missing_or_unbudgeted_numerical_fields(self) -> None:
        candidates = self.candidates()
        candidates[0]["numerical_check"].pop("finite")
        self.assert_rejected(baseline(), candidates, "unbudgeted")
        candidates = self.candidates()
        candidates[0]["numerical_check"]["unknown"] = 4
        self.assert_rejected(baseline(), candidates, "unbudgeted")

    def test_enforces_exact_relative_and_maximum_budgets(self) -> None:
        candidates = self.candidates()
        candidates[0]["numerical_check"]["finite"] = False
        self.assert_rejected(baseline(), candidates, "finite")
        candidates = self.candidates()
        candidates[0]["numerical_check"]["price"] = 1.01
        self.assert_rejected(baseline(), candidates, "price")
        candidates = self.candidates()
        candidates[0]["device_memory"]["tracked_live_bytes"] = 1025
        self.assert_rejected(baseline(), candidates, "tracked_live_bytes")

    def test_blocks_register_local_occupancy_and_code_size_regressions(self) -> None:
        mutations = (
            ("resources", "registers_per_thread", 33, "registers_per_thread"),
            ("resources", "local_bytes_per_thread", 8, "local_bytes_per_thread"),
            ("occupancy", "theoretical", 0.125, "occupancy theoretical"),
        )
        for section, field, value, fragment in mutations:
            candidates = self.candidates()
            candidates[0]["resources"][0][section][field] = value
            self.assert_rejected(baseline(), candidates, fragment)
        candidates = self.candidates()
        candidates[0]["binary"]["executable_bytes"] = 11_001
        self.assert_rejected(baseline(), candidates, "executable_bytes")

    def test_rejects_manifest_without_four_decisions(self) -> None:
        reference = baseline()
        reference["decisions"] = reference["decisions"][:3]
        self.assert_rejected(reference, self.candidates(), "four explicit decisions")

    def test_rejects_incomplete_power_preflight_policy(self) -> None:
        for field in (
            "minimum_current_power_limit_w",
            "retry_cooldown_seconds",
        ):
            reference = baseline()
            reference["decision_policy"]["preflight"].pop(field)
            self.assert_rejected(
                reference,
                self.candidates(),
                "fail-closed power and stability preflight required",
            )

    def test_stops_when_campaign_series_can_no_longer_succeed(self) -> None:
        self.assertTrue(campaign_series_can_complete(0, 2, 3, 5))
        self.assertFalse(campaign_series_can_complete(0, 3, 3, 5))
        self.assertTrue(campaign_series_can_complete(2, 4, 3, 5))

    def test_protocol_initialization_preserves_numerical_contracts(self) -> None:
        predecessor = baseline()
        for reference in predecessor["measurements"]:
            reference["id"] = "test_command__" + reference["id"]
        initialized = initialize_baseline(
            predecessor, self.candidates(), "protocol migration"
        )
        self.assertEqual(
            [row["id"] for row in initialized["measurements"]],
            [row["id"] for row in predecessor["measurements"]],
        )
        self.assertEqual(
            initialized["measurements"][0]["numerical_budgets"],
            predecessor["measurements"][0]["numerical_budgets"],
        )
        self.assertIn(
            "device_memory_budgets", initialized["measurements"][0]
        )

    def test_protocol_initialization_allows_only_protocol_field_migration(
        self,
    ) -> None:
        predecessor = baseline()
        for reference in predecessor["measurements"]:
            reference["id"] = "test_command__" + reference["id"]
        candidates = self.candidates()
        candidates[0]["configuration"]["new_protocol_field"] = 4
        selected = select_initialization_manifest(predecessor, candidates)
        self.assertEqual(
            [row["measurement_id"] for row in selected],
            ["test_command__one", "test_command__two"],
        )
        candidates[0]["benchmark"] = "different"
        with self.assertRaisesRegex(ValueError, "stable identity field"):
            select_initialization_manifest(predecessor, candidates)

    def test_aggregates_every_campaign_without_best_of_n(self) -> None:
        reference = baseline()
        first = self.candidates()
        second = self.candidates()
        third = self.candidates()
        first[0]["kernel"]["median_ms"] = 9.0
        first[0]["kernel"]["coefficient_of_variation"] = 0.08
        second[0]["kernel"]["coefficient_of_variation"] = 0.02
        third[0]["kernel"]["coefficient_of_variation"] = 0.03
        second[0]["kernel"]["median_ms"] = 11.0
        third[0]["kernel"]["median_ms"] = 10.0
        first[1]["kernel"]["median_ms"] = 12.0
        second[1]["kernel"]["median_ms"] = 8.0
        third[1]["kernel"]["median_ms"] = 10.0
        first[0]["numerical_check"]["price"] = 0.9
        second[0]["numerical_check"]["price"] = 1.1
        first[0]["device_memory"]["free_margin_bytes"] = 990_000
        second[0]["device_memory"]["free_margin_bytes"] = 980_000
        first[0].pop("campaign_aggregation")
        second[0].pop("campaign_aggregation")
        third[0].pop("campaign_aggregation")
        first[1].pop("campaign_aggregation")
        second[1].pop("campaign_aggregation")
        third[1].pop("campaign_aggregation")
        aggregated = aggregate_campaigns([first, second, third], reference)
        self.assertEqual(aggregated[0]["kernel"]["median_ms"], 10.0)
        self.assertEqual(aggregated[1]["kernel"]["median_ms"], 10.0)
        self.assertEqual(aggregated[0]["numerical_check"]["price"], 1.0)
        self.assertEqual(
            aggregated[0]["device_memory"]["free_margin_bytes"], 980_000
        )
        self.assertEqual(
            aggregated[0]["kernel"]["p95_ms"],
            max(row[0]["kernel"]["p95_ms"] for row in (first, second, third)),
        )
        self.assertEqual(
            aggregated[0]["kernel"]["coefficient_of_variation"], 0.03
        )

    def test_campaign_noise_requires_two_stable_campaigns(self) -> None:
        reference = baseline()
        first = self.candidates()
        second = self.candidates()
        third = self.candidates()
        first[0]["kernel"]["coefficient_of_variation"] = 0.08
        second[0]["kernel"]["coefficient_of_variation"] = 0.07
        third[0]["kernel"]["coefficient_of_variation"] = 0.01
        aggregated = aggregate_campaigns(
            [first, second, third], reference
        )
        failures, inconclusive, informational = compare(reference, aggregated)
        self.assertEqual(failures, [])
        self.assertEqual(len(inconclusive), 1)
        self.assertEqual(informational, [])

    def test_retains_every_raw_campaign_with_hashes(self) -> None:
        attempts = [
            {
                "status": "eligible",
                "reason": None,
                "preflight": {},
                "measurements": self.candidates(),
            }
            for _ in range(3)
        ]
        with TemporaryDirectory() as temporary:
            directory = write_raw_campaigns(
                Path(temporary) / "candidate.ndjson", attempts
            )
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["campaign_count"], 3)
            self.assertEqual(manifest["eligible_campaign_count"], 3)
            self.assertEqual(len(manifest["files"]), 3)
            self.assertTrue(all(
                (directory / entry["path"]).is_file()
                and len(entry["sha256"]) == 64
                and entry["measurement_count"] == 2
                for entry in manifest["files"]
            ))

    def test_updates_one_raw_campaign_directory_incrementally(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate.ndjson"
            first = [{
                "status": "rejected_environment",
                "reason": "battery",
                "preflight": {},
                "measurements": None,
            }]
            directory = write_raw_campaigns(output, first)
            second = [*first, {
                "status": "eligible",
                "reason": None,
                "preflight": {},
                "measurements": self.candidates(),
            }]
            self.assertEqual(
                write_raw_campaigns(output, second, directory), directory
            )
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["campaign_count"], 2)
            self.assertEqual(manifest["eligible_campaign_count"], 1)
            self.assertFalse((directory / "campaign_01.ndjson").exists())
            self.assertTrue((directory / "campaign_02.ndjson").exists())
            loaded = load_raw_campaigns(directory)
            self.assertEqual([row["status"] for row in loaded], [
                "rejected_environment",
                "eligible",
            ])
            self.assertEqual(len(loaded[1]["measurements"]), 2)

    def test_refuses_tampered_raw_campaign_resume(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate.ndjson"
            directory = write_raw_campaigns(output, [{
                "status": "eligible",
                "reason": None,
                "preflight": {},
                "measurements": self.candidates(),
            }])
            payload = directory / "campaign_01.ndjson"
            payload.write_text(payload.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_raw_campaigns(directory)

    def test_materializes_exactly_four_partitioned_reports(self) -> None:
        reference = baseline()
        candidates = self.candidates()
        with TemporaryDirectory() as temporary:
            directory = write_raw_campaigns(
                Path(temporary) / "candidate.ndjson",
                [{
                    "status": "eligible",
                    "reason": None,
                    "preflight": {},
                    "measurements": candidates,
                }],
            )
            write_audit_reports(directory, reference, candidates)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(len(manifest["audit_reports"]), 4)
            self.assertEqual(
                sum(entry["measurement_count"] for entry in manifest["audit_reports"]),
                len(candidates),
            )

    def test_preflight_rejects_battery_temperature_and_throttle(self) -> None:
        reference = baseline()
        snapshot = {
            "gpu": "test GPU",
            "power_source": "external_power",
            "temperature_c": 70,
            "power_limits_w": {"current": 150.0},
            "concurrent_compute_processes": [],
            "throttle": {
                "hardware_slowdown": "Not Active",
                "hardware_thermal_slowdown": "Not Active",
                "software_thermal_slowdown": "Not Active",
            },
        }
        validate_preflight(reference, snapshot)
        battery = {**snapshot, "power_source": "battery"}
        with self.assertRaisesRegex(ValueError, "power source"):
            validate_preflight(reference, battery)
        throttled = copy.deepcopy(snapshot)
        throttled["throttle"]["hardware_thermal_slowdown"] = "Active"
        with self.assertRaisesRegex(ValueError, "throttling"):
            validate_preflight(reference, throttled)
        low_power = copy.deepcopy(snapshot)
        low_power["power_limits_w"]["current"] = 55.0
        with self.assertRaisesRegex(ValueError, "power limit"):
            validate_preflight(reference, low_power)
        warm = {**snapshot, "temperature_c": 80}
        validate_campaign_preflight(reference, snapshot, warm)
        hot = {**snapshot, "temperature_c": 86}
        with self.assertRaisesRegex(ValueError, "temperature"):
            validate_campaign_preflight(reference, snapshot, hot)

    def test_stabilizes_before_the_official_preflight(self) -> None:
        reference = baseline()
        cold = {
            "gpu": "test GPU",
            "power_source": "external_power",
            "temperature_c": 55,
            "power_limits_w": {"current": 150.0},
            "concurrent_compute_processes": [],
            "throttle": {
                "hardware_slowdown": "Not Active",
                "hardware_thermal_slowdown": "Not Active",
                "software_thermal_slowdown": "Not Active",
            },
        }
        warming = {**cold, "temperature_c": 64}
        warm = {**cold, "temperature_c": 65}
        evidence: list[dict[str, object]] = []
        with (
            patch(
                "tools.performance.run_baseline.collect_preflight",
                side_effect=[cold, cold, warming, warm],
            ),
            patch(
                "tools.performance.run_baseline.subprocess.run",
                return_value=SimpleNamespace(stdout=b"warmup"),
            ) as run,
        ):
            snapshot = stabilize_thermal_environment(
                reference, Path("build"), evidence
            )
        self.assertEqual(snapshot["temperature_c"], 65)
        self.assertEqual(len(evidence), 4)
        self.assertEqual(run.call_count, 3)

    def test_extracts_nvidia_xml_power_limits(self) -> None:
        xml = """<nvidia_smi_log><gpu><gpu_power_readings>
          <current_power_limit>150.00 W</current_power_limit>
          <default_power_limit>150.00 W</default_power_limit>
          <min_power_limit>5.00 W</min_power_limit>
          <max_power_limit>175.00 W</max_power_limit>
        </gpu_power_readings></gpu></nvidia_smi_log>"""
        self.assertEqual(parse_nvidia_power_limits(xml), {
            "current": 150.0,
            "default": 150.0,
            "minimum": 5.0,
            "maximum": 175.0,
        })
        with self.assertRaisesRegex(ValueError, "one GPU"):
            parse_nvidia_power_limits("<nvidia_smi_log/>")

    def test_writes_exhaustive_hashed_rebaseline_diff(self) -> None:
        predecessor = baseline()
        successor = copy.deepcopy(predecessor)
        successor["measurements"][0]["kernel"]["median_ms"] = 9.5
        successor["rebaseline_reason"] = "protocol initialization"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            predecessor_path = root / "predecessor.json"
            diff_path = root / "diff.json"
            predecessor_path.write_text(json.dumps(predecessor, indent=2) + "\n")
            payload = write_rebaseline_diff(
                predecessor_path,
                successor,
                diff_path,
                "protocol initialization",
                "approved for test",
                "protocol_initialization",
            )
            report = json.loads(diff_path.read_text())
            self.assertEqual(json.loads(payload), successor)
            self.assertEqual(report["change_count"], 2)
            self.assertEqual(report["approval"], "approved for test")
            self.assertEqual(
                report["changes"][0]["path"],
                "$.measurements[0].kernel.median_ms",
            )
            self.assertEqual(
                report["successor"]["sha256"],
                hashlib.sha256(payload.encode()).hexdigest(),
            )

    def test_allows_ids_only_for_first_protocol_initialization(self) -> None:
        predecessor = baseline()
        predecessor["measurements"] = []
        successor = baseline()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            predecessor_path = root / "predecessor.json"
            predecessor_path.write_text(json.dumps(predecessor) + "\n")
            write_rebaseline_diff(
                predecessor_path,
                successor,
                root / "initialization.json",
                "first protocol initialization",
                "approved for test",
                "protocol_initialization",
            )
            with self.assertRaisesRegex(ValueError, "measurement ids differ"):
                write_rebaseline_diff(
                    predecessor_path,
                    successor,
                    root / "ordinary.json",
                    "ordinary rebaseline",
                    "approved for test",
                    "regression_checked_rebaseline",
                )

    def test_resolves_profile_target_from_manifest_and_candidate(self) -> None:
        manifest = baseline()
        candidate = measurement("one")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "test_benchmark"
            executable.write_bytes(b"profile executable")
            candidate["binary"]["executable_sha256"] = hashlib.sha256(
                executable.read_bytes()
            ).hexdigest()
            target = select_profile_target(
                manifest, [candidate], root, "one", 0
            )
            self.assertEqual(target["scope"], "generic_cuda")
            self.assertEqual(target["compiled_symbol"], "_Z_test_kernel_one")
            command = ncu_command(
                "/usr/bin/ncu", target, root / "report", "detailed"
            )
            self.assertEqual(command[-1], str(executable))
            self.assertIn("_Z_test_kernel_one", command)
            self.assertEqual(command[command.index("--kill") + 1], "1")


if __name__ == "__main__":
    unittest.main()
