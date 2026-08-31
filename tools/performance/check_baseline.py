#!/usr/bin/env python3
"""Fail-closed comparison for the versioned CUDA performance manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


TIMING_FIELDS = ("median_ms", "p95_ms", "coefficient_of_variation")
ENVIRONMENT_FIELDS = (
    "gpu",
    "compute_capability",
    "sm_count",
    "memory_bytes",
    "driver_version",
    "runtime_version",
    "cuda_compiler_version",
)


def measurement_key(measurement: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        measurement["finding"],
        measurement["benchmark"],
        measurement["variant"],
        json.dumps(measurement.get("configuration", {}), sort_keys=True),
    )


def diagnostic_key(diagnostic: dict[str, Any]) -> tuple[str, str, str]:
    return (
        diagnostic["kernel"],
        diagnostic["variant"],
        json.dumps(diagnostic["launch"], sort_keys=True),
    )


def load_candidate(path: Path) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid benchmark JSON: {error}"
            ) from error
        if document.get("schema") != "ai_factory_cuda_performance_baseline":
            raise ValueError(
                f"{path}:{line_number}: unexpected performance record"
            )
        measurements.append(document)
    if not measurements:
        raise ValueError(f"{path}: no CUDA performance measurements found")
    return measurements


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _manifest_failures(baseline: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if baseline.get("protocol_version") != 3:
        failures.append("manifest: protocol_version must be 3")
    decision_policy = baseline.get("decision_policy", {})
    if decision_policy.get("campaign_attempts", 0) < 3:
        failures.append("manifest: at least three complete campaigns are required")
    if decision_policy.get("maximum_campaign_attempts", 0) < decision_policy.get(
        "campaign_attempts", 0
    ):
        failures.append("manifest: campaign retry bound is invalid")
    if decision_policy.get("campaign_aggregation") != (
        "median_of_all_campaign_medians_conservative_tail"
    ):
        failures.append("manifest: campaign aggregation is missing or opportunistic")
    preflight = decision_policy.get("preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("accepted_power_sources")
            != ["external_power", "no_battery"]
        or not isinstance(preflight.get("maximum_temperature_c"), int)
        or not isinstance(preflight.get("maximum_temperature_delta_c"), int)
        or preflight.get("concurrent_compute_processes") != "forbidden"
        or preflight.get("forbidden_throttle_reasons") != [
            "hardware_slowdown",
            "hardware_thermal_slowdown",
            "software_thermal_slowdown",
        ]
    ):
        failures.append("manifest: fail-closed power and stability preflight required")
    if decision_policy.get("timing_reports") != {
        "kernel": "kernel",
        "public_api": "public_api",
        "pipeline": "pipeline",
        "publication": "publication_wall",
    }:
        failures.append("manifest: four timing-report boundaries are required")
    architecture_profile = baseline.get("architecture_profile")
    if not isinstance(architecture_profile, dict):
        failures.append("manifest: architecture profile is required")
    elif (
        architecture_profile.get("compute_capability")
        != baseline.get("environment", {}).get("compute_capability")
        or architecture_profile.get("qualification")
        != "performance_profiled"
        or not architecture_profile.get("tuning_profile_id")
    ):
        failures.append("manifest: architecture profile is incompatible")
    commands = baseline.get("commands")
    measurements = baseline.get("measurements")
    decisions = baseline.get("decisions")
    if not isinstance(commands, list) or not commands:
        failures.append("manifest: commands must be a non-empty list")
        commands = []
    if not isinstance(measurements, list) or not measurements:
        failures.append("manifest: measurements must be a non-empty list")
        measurements = []
    if not isinstance(decisions, list) or len(decisions) < 4:
        failures.append("manifest: at least four explicit decisions are required")
        decisions = []

    command_ids: set[str] = set()
    for command in commands:
        if not isinstance(command, dict):
            failures.append("manifest: every command must be an object")
            continue
        command_id = command.get("id")
        if not isinstance(command_id, str) or not command_id:
            failures.append("manifest: every command requires a non-empty id")
        elif command_id in command_ids:
            failures.append(f"manifest: duplicate command id {command_id}")
        else:
            command_ids.add(command_id)
        if not isinstance(command.get("executable"), str):
            failures.append(f"manifest command {command_id}: missing executable")
        arguments = command.get("arguments")
        if not isinstance(arguments, list) or not all(
            isinstance(argument, str) for argument in arguments
        ):
            failures.append(f"manifest command {command_id}: invalid arguments")

    audit_reports = baseline.get("audit_reports")
    if not isinstance(audit_reports, dict) or set(audit_reports) != {
        "generic_cuda", "model_sampling", "early_exercise", "rough",
    }:
        failures.append("manifest: exactly four named audit reports are required")
    else:
        reported_commands = [
            command_id
            for report in audit_reports.values()
            if isinstance(report, list)
            for command_id in report
        ]
        if (
            any(not isinstance(report, list)
                for report in audit_reports.values())
            or len(reported_commands) != len(set(reported_commands))
            or set(reported_commands) != command_ids
        ):
            failures.append(
                "manifest: audit reports must partition every command exactly once"
            )

    measurement_ids: set[str] = set()
    semantic_keys: set[tuple[str, str, str, str]] = set()
    referenced_commands: set[str] = set()
    for measurement in measurements:
        if not isinstance(measurement, dict):
            failures.append("manifest: every measurement must be an object")
            continue
        measurement_id = measurement.get("id")
        if not isinstance(measurement_id, str) or not measurement_id:
            failures.append("manifest: every measurement requires a non-empty id")
        elif measurement_id in measurement_ids:
            failures.append(f"manifest: duplicate measurement id {measurement_id}")
        else:
            measurement_ids.add(measurement_id)
        try:
            key = measurement_key(measurement)
        except (KeyError, TypeError) as error:
            failures.append(
                f"manifest measurement {measurement_id}: invalid key: {error}"
            )
            continue
        if key in semantic_keys:
            failures.append(f"manifest: duplicate measurement key {key}")
        semantic_keys.add(key)
        command_id = measurement.get("command_id")
        if command_id not in command_ids:
            failures.append(
                f"manifest measurement {measurement_id}: unknown command_id {command_id}"
            )
        elif isinstance(command_id, str):
            referenced_commands.add(command_id)
        if measurement.get("comparison_policy") not in (
            "blocking",
            "informational",
        ):
            failures.append(
                f"manifest measurement {measurement_id}: invalid comparison_policy"
            )
        protocol = measurement.get("protocol")
        expected_protocol = {
            "warmups": decision_policy.get("warmups"),
            "repetitions": decision_policy.get("repetitions"),
            "primary_statistic": decision_policy.get("primary_statistic"),
            "tail_statistic": decision_policy.get("tail_statistic"),
            "minimum_accepted_gain": decision_policy.get(
                "maximum_timing_regression"
            ),
            "maximum_noise_coefficient": decision_policy.get(
                "maximum_coefficient_of_variation"
            ),
            "maximum_publication_noise_coefficient": decision_policy.get(
                "maximum_publication_coefficient_of_variation"
            ),
            "wall_semantics": decision_policy.get("wall_semantics"),
        }
        if not isinstance(protocol, dict) or protocol != expected_protocol:
            failures.append(
                f"manifest measurement {measurement_id}: protocol differs from "
                "the authoritative decision policy"
            )
        aggregation = measurement.get("campaign_aggregation")
        if aggregation is not None and aggregation != {
            "method": decision_policy.get("campaign_aggregation"),
            "attempt_count": decision_policy.get("campaign_attempts"),
        }:
            failures.append(
                f"manifest measurement {measurement_id}: invalid campaign aggregate"
            )
        timing_budgets = measurement.get("timing_budgets")
        if not isinstance(timing_budgets, dict):
            failures.append(
                f"manifest measurement {measurement_id}: missing timing_budgets"
            )
        else:
            expected_sections = {"kernel", "public_api", "pipeline"}
            if "publication_wall" in measurement:
                expected_sections.add("publication_wall")
            if set(timing_budgets) != expected_sections:
                failures.append(
                    f"manifest measurement {measurement_id}: timing budgets do not "
                    "cover exactly the emitted timing sections"
                )
        numerical = measurement.get("numerical_check")
        budgets = measurement.get("numerical_budgets")
        if not isinstance(numerical, dict) or not isinstance(budgets, dict):
            failures.append(
                f"manifest measurement {measurement_id}: missing numerical budgets"
            )
        elif set(numerical) != set(budgets):
            failures.append(
                f"manifest measurement {measurement_id}: numerical budgets do not "
                "cover exactly the emitted fields"
            )
        if not isinstance(measurement.get("resources"), list) or not measurement.get(
            "resources"
        ):
            failures.append(
                f"manifest measurement {measurement_id}: no launch resources"
            )
        else:
            for diagnostic in measurement["resources"]:
                compiled = diagnostic.get("compiled_resources", {})
                budgets = diagnostic.get("compiled_resource_budgets", {})
                required = {
                    "registers_per_thread",
                    "stack_frame_bytes",
                    "static_shared_bytes_per_block",
                    "local_bytes_per_thread",
                    "sass_code_bytes",
                    "sass_instruction_count",
                    "sass_local_load_instructions",
                    "sass_local_store_instructions",
                }
                if (
                    not diagnostic.get("compiled_symbol")
                    or not isinstance(compiled, dict)
                    or not isinstance(budgets, dict)
                    or not required <= set(compiled)
                    or set(budgets) != required
                ):
                    failures.append(
                        f"manifest measurement {measurement_id}: incomplete "
                        "compiled-resource budgets"
                    )
        for field in (
            "device_memory",
            "device_memory_budgets",
            "binary",
            "binary_budgets",
        ):
            if not isinstance(measurement.get(field), dict):
                failures.append(
                    f"manifest measurement {measurement_id}: missing {field}"
                )

    for command_id in sorted(command_ids - referenced_commands):
        failures.append(f"manifest command {command_id}: emits no declared measurement")

    decision_ids: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            failures.append("manifest: every decision must be an object")
            continue
        decision_id = decision.get("id")
        if not isinstance(decision_id, str) or not decision_id:
            failures.append("manifest: every decision requires a non-empty id")
        elif decision_id in decision_ids:
            failures.append(f"manifest: duplicate decision id {decision_id}")
        else:
            decision_ids.add(decision_id)
        selected = decision.get("selected")
        rejected = decision.get("rejected")
        rationale = decision.get("rationale")
        outcome = decision.get("outcome")
        if not isinstance(selected, list) or not selected:
            failures.append(f"manifest decision {decision_id}: selected is empty")
            selected = []
        if not isinstance(rejected, list) or not rejected:
            failures.append(f"manifest decision {decision_id}: rejected is empty")
            rejected = []
        if not isinstance(rationale, str) or not rationale:
            failures.append(f"manifest decision {decision_id}: missing rationale")
        if outcome not in {"accept", "reject", "inconclusive", "unavailable"}:
            failures.append(f"manifest decision {decision_id}: invalid outcome")
        for reference in selected + rejected:
            if reference not in measurement_ids:
                failures.append(
                    f"manifest decision {decision_id}: unknown measurement {reference}"
                )
    return failures


def _append(
    policy: str,
    message: str,
    failures: list[str],
    informational: list[str],
) -> None:
    (informational if policy == "informational" else failures).append(message)


def _compare_budgeted_values(
    key: tuple[str, str, str, str],
    label: str,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    budgets: dict[str, Any],
    policy: str,
    failures: list[str],
    informational: list[str],
) -> None:
    if set(reference) != set(candidate) or set(reference) != set(budgets):
        _append(
            policy,
            f"{key}: {label} fields are missing, unknown, or unbudgeted",
            failures,
            informational,
        )
        return
    for field, budget in budgets.items():
        rule = budget.get("rule") if isinstance(budget, dict) else None
        observed = candidate[field]
        expected = reference[field]
        message = f"{key}: {label}.{field} violates {rule} budget"
        if rule == "informational":
            if isinstance(observed, float) and not math.isfinite(observed):
                _append(policy, message, failures, informational)
        elif rule == "exact":
            if observed != expected:
                _append(policy, message, failures, informational)
        elif rule == "maximum":
            maximum = budget.get("value")
            if (
                not _finite_number(observed)
                or not _finite_number(maximum)
                or float(observed) > float(maximum)
            ):
                _append(policy, message, failures, informational)
        elif rule == "minimum":
            minimum = budget.get("value")
            if (
                not _finite_number(observed)
                or not _finite_number(minimum)
                or float(observed) < float(minimum)
            ):
                _append(policy, message, failures, informational)
        elif rule == "relative":
            absolute = budget.get("absolute_tolerance", 0.0)
            relative = budget.get("relative_tolerance", 0.0)
            if not all(
                _finite_number(value)
                for value in (observed, expected, absolute, relative)
            ) or abs(float(observed) - float(expected)) > (
                float(absolute) + float(relative) * abs(float(expected))
            ):
                _append(policy, message, failures, informational)
        else:
            _append(policy, message, failures, informational)


def _compare_resources(
    key: tuple[str, str, str, str],
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    policy: str,
    failures: list[str],
    informational: list[str],
) -> None:
    try:
        references = {diagnostic_key(row): row for row in reference_rows}
        candidates = {diagnostic_key(row): row for row in candidate_rows}
    except (KeyError, TypeError) as error:
        _append(
            policy,
            f"{key}: malformed launch resources: {error}",
            failures,
            informational,
        )
        return
    if len(references) != len(reference_rows) or len(candidates) != len(candidate_rows):
        _append(policy, f"{key}: duplicate launch resources", failures, informational)
        return
    if set(references) != set(candidates):
        _append(
            policy,
            f"{key}: launch resource set changed",
            failures,
            informational,
        )
        return
    maximum_fields = (
        "registers_per_thread",
        "static_shared_bytes_per_block",
        "local_bytes_per_thread",
    )
    minimum_fields = (
        "maximum_threads_per_block",
        "maximum_dynamic_shared_bytes_per_block",
    )
    for resource_key, reference in references.items():
        candidate = candidates[resource_key]
        if (
            candidate.get("device") != reference.get("device")
            or candidate.get("code") != reference.get("code")
        ):
            _append(
                policy,
                f"{key}: {resource_key} device or code target changed",
                failures,
                informational,
            )
            continue
        if candidate.get("compiled_symbol") != reference.get("compiled_symbol"):
            _append(
                policy,
                f"{key}: {resource_key} compiled symbol changed",
                failures,
                informational,
            )
            continue
        compiled_reference = reference.get("compiled_resources", {})
        compiled_candidate = candidate.get("compiled_resources", {})
        compiled_budgets = reference.get("compiled_resource_budgets", {})
        if (
            set(compiled_reference) != set(compiled_candidate)
            or set(compiled_budgets) != set(compiled_reference) - {
                "compiled_symbol"
            }
        ):
            _append(
                policy,
                f"{key}: {resource_key} compiled resources are incomplete",
                failures,
                informational,
            )
            continue
        if compiled_candidate.get("compiled_symbol") != compiled_reference.get(
            "compiled_symbol"
        ):
            _append(
                policy,
                f"{key}: {resource_key} compiled resource symbol changed",
                failures,
                informational,
            )
        for field, budget in compiled_budgets.items():
            if (
                budget.get("rule") != "maximum"
                or not _finite_number(budget.get("value"))
                or not _finite_number(compiled_candidate.get(field))
                or compiled_candidate[field] > budget["value"]
            ):
                _append(
                    policy,
                    f"{key}: {resource_key} compiled {field} exceeds budget",
                    failures,
                    informational,
                )
        for field in maximum_fields:
            if candidate["resources"][field] > reference["resources"][field]:
                _append(
                    policy,
                    f"{key}: {resource_key} {field} increased",
                    failures,
                    informational,
                )
        for field in minimum_fields:
            if candidate["resources"][field] < reference["resources"][field]:
                _append(
                    policy,
                    f"{key}: {resource_key} {field} decreased",
                    failures,
                    informational,
                )
        for field in (
            "active_blocks_per_multiprocessor",
            "active_warps_per_multiprocessor",
            "theoretical",
        ):
            if candidate["occupancy"][field] < reference["occupancy"][field]:
                _append(
                    policy,
                    f"{key}: {resource_key} occupancy {field} decreased",
                    failures,
                    informational,
                )


def compare(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    failures = _manifest_failures(baseline)
    inconclusive: list[str] = []
    informational: list[str] = []
    if failures:
        return failures, inconclusive, informational

    expected_protocol = baseline["protocol_version"]
    references = {
        measurement_key(measurement): measurement
        for measurement in baseline["measurements"]
    }
    candidate_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        try:
            key = measurement_key(candidate)
        except (KeyError, TypeError) as error:
            failures.append(f"candidate: invalid measurement key: {error}")
            continue
        if key in candidate_index:
            failures.append(f"{key}: duplicate candidate measurement")
        else:
            candidate_index[key] = candidate

    for key in sorted(references.keys() - candidate_index.keys()):
        failures.append(f"{key}: missing candidate measurement")
    for key in sorted(candidate_index.keys() - references.keys()):
        failures.append(f"{key}: no matching baseline")

    for key in sorted(references.keys() & candidate_index.keys()):
        candidate = candidate_index[key]
        reference = references[key]
        policy = reference["comparison_policy"]
        if candidate.get("measurement_id") != reference["id"]:
            failures.append(f"{key}: measurement id mismatch")
            continue
        if candidate.get("command_id") != reference["command_id"]:
            failures.append(f"{key}: command id mismatch")
            continue
        if candidate.get("protocol_version") != expected_protocol:
            failures.append(f"{key}: protocol version mismatch")
            continue
        if candidate.get("protocol") != reference["protocol"]:
            failures.append(f"{key}: public timing protocol mismatch")
            continue
        if candidate.get("campaign_aggregation") != {
            "method": baseline["decision_policy"]["campaign_aggregation"],
            "attempt_count": baseline["decision_policy"]["campaign_attempts"],
        }:
            failures.append(f"{key}: missing complete-campaign aggregation")
            continue
        candidate_environment = candidate.get("environment", {})
        if any(
            candidate_environment.get(field) != baseline["environment"].get(field)
            for field in ENVIRONMENT_FIELDS
        ):
            failures.append(f"{key}: environment mismatch")
            continue

        for section, budget in reference["timing_budgets"].items():
            if section not in candidate or section not in reference:
                failures.append(f"{key}: missing timing section {section}")
                continue
            candidate_timing = candidate[section]
            reference_timing = reference[section]
            if not all(
                field in candidate_timing
                and field in reference_timing
                and _finite_number(candidate_timing[field])
                and _finite_number(reference_timing[field])
                for field in TIMING_FIELDS
            ):
                failures.append(
                    f"{key}: incomplete or non-finite timing {section}"
                )
                continue
            if budget.get("rule") == "record_only":
                continue
            if budget.get("rule") != "regression":
                failures.append(f"{key}: unknown timing budget for {section}")
                continue
            maximum_noise = budget["maximum_coefficient_of_variation"]
            if (
                candidate_timing["coefficient_of_variation"] > maximum_noise
                or reference_timing["coefficient_of_variation"] > maximum_noise
            ):
                message = (
                    f"{key}: {section} coefficient of variation exceeds "
                    f"{maximum_noise:.1%}"
                )
                if policy == "informational":
                    informational.append(message)
                else:
                    inconclusive.append(message)
                continue
            maximum_regression = budget["maximum_regression"]
            for statistic in ("median_ms", "p95_ms"):
                regression = (
                    candidate_timing[statistic] / reference_timing[statistic] - 1.0
                )
                if regression > maximum_regression:
                    _append(
                        policy,
                        f"{key}: {section} {statistic} regression {regression:.2%} "
                        f"exceeds {maximum_regression:.2%}",
                        failures,
                        informational,
                    )

        _compare_budgeted_values(
            key,
            "numerical_check",
            reference["numerical_check"],
            candidate.get("numerical_check", {}),
            reference["numerical_budgets"],
            "blocking",
            failures,
            informational,
        )
        _compare_budgeted_values(
            key,
            "device_memory",
            reference["device_memory"],
            candidate.get("device_memory", {}),
            reference["device_memory_budgets"],
            "blocking",
            failures,
            informational,
        )
        _compare_budgeted_values(
            key,
            "binary",
            reference["binary"],
            candidate.get("binary", {}),
            reference["binary_budgets"],
            "blocking",
            failures,
            informational,
        )
        _compare_resources(
            key,
            reference["resources"],
            candidate.get("resources", []),
            "blocking",
            failures,
            informational,
        )
        if (
            candidate["kernel"]["median_ms"]
            > candidate["public_api"]["median_ms"]
        ):
            failures.append(
                f"{key}: kernel median exceeds enclosing public API median"
            )
        if candidate["kernel"]["p95_ms"] > candidate["public_api"]["p95_ms"]:
            failures.append(
                f"{key}: kernel p95 exceeds enclosing public API p95"
            )
    return failures, inconclusive, informational


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    arguments = parser.parse_args()
    try:
        baseline = json.loads(arguments.baseline.read_text())
        candidates = load_candidate(arguments.candidate)
        failures, inconclusive, informational = compare(baseline, candidates)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"baseline check error: {error}", file=sys.stderr)
        return 2

    for message in informational:
        print(f"INFO: {message}")
    for message in inconclusive:
        print(f"INCONCLUSIVE: {message}", file=sys.stderr)
    for message in failures:
        print(f"FAIL: {message}", file=sys.stderr)
    if failures:
        return 1
    if inconclusive:
        print(
            "INCONCLUSIVE: rerun the complete manifest before accepting it",
            file=sys.stderr,
        )
        return 3
    print(
        f"PASS: {len(candidates)} measurement(s), 0 blocking inconclusive, "
        f"{len(informational)} informational"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
