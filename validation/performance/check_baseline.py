#!/usr/bin/env python3
"""Compare benchmark NDJSON with a versioned CUDA performance baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def measurement_key(measurement: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        measurement["finding"],
        measurement["benchmark"],
        measurement["variant"],
        json.dumps(measurement.get("configuration", {}), sort_keys=True),
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
        if document.get("schema") == "ai_factory_cuda_performance_baseline":
            measurements.append(document)
    if not measurements:
        raise ValueError(f"{path}: no CUDA performance measurements found")
    return measurements


def compare(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    failures: list[str] = []
    inconclusive: list[str] = []
    informational: list[str] = []
    expected_protocol = baseline["protocol_version"]
    minimum_gain = baseline["decision_policy"]["minimum_effect_size"]
    maximum_noise = baseline["decision_policy"][
        "maximum_coefficient_of_variation"
    ]
    maximum_publication_noise = baseline["decision_policy"].get(
        "maximum_publication_coefficient_of_variation",
        maximum_noise,
    )
    references: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for measurement in baseline["measurements"]:
        key = measurement_key(measurement)
        if key in references:
            failures.append(f"{key}: duplicate baseline measurement")
        else:
            references[key] = measurement

    candidate_index: dict[
        tuple[str, str, str, str], dict[str, Any]
    ] = {}
    for candidate in candidates:
        key = measurement_key(candidate)
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
        comparison_policy = reference.get("comparison_policy", "blocking")
        if comparison_policy not in ("blocking", "informational"):
            failures.append(f"{key}: unknown comparison policy")
            continue
        if candidate.get("protocol_version") != expected_protocol:
            failures.append(f"{key}: protocol version mismatch")
            continue
        candidate_environment = candidate.get("environment", {})
        for field in (
            "gpu",
            "compute_capability",
            "sm_count",
            "driver_version",
            "runtime_version",
            "cuda_compiler_version",
        ):
            if (
                candidate_environment.get(field)
                != baseline["environment"][field]
            ):
                failures.append(f"{key}: environment field {field} mismatch")
                break
        else:
            field = None
        if field is not None:
            continue
        kernel = candidate["kernel"]
        wall = candidate["wall"]
        if kernel["median_ms"] > wall["median_ms"]:
            failures.append(f"{key}: kernel median exceeds enclosing wall median")
            continue
        timing_sections = ["kernel"]
        if "publication_wall" in reference:
            timing_sections.append("publication_wall")
        for timing_section in timing_sections:
            if timing_section not in candidate:
                failures.append(
                    f"{key}: missing candidate timing {timing_section}"
                )
                continue
            candidate_timing = candidate[timing_section]
            reference_timing = reference[timing_section]
            section_maximum_noise = (
                maximum_publication_noise
                if timing_section == "publication_wall"
                else maximum_noise
            )
            if (candidate_timing["coefficient_of_variation"]
                    > section_maximum_noise
                    or reference_timing["coefficient_of_variation"]
                        > section_maximum_noise):
                message = (
                    f"{key}: {timing_section} coefficient of variation "
                    f"exceeds {section_maximum_noise:.1%}"
                )
                if comparison_policy == "informational":
                    informational.append(message)
                else:
                    inconclusive.append(message)
                continue
            regression = (
                candidate_timing["median_ms"]
                / reference_timing["median_ms"] - 1.0
            )
            if regression > minimum_gain:
                message = (
                    f"{key}: {timing_section} median regression "
                    f"{regression:.2%} exceeds {minimum_gain:.2%}"
                )
                if comparison_policy == "informational":
                    informational.append(message)
                else:
                    failures.append(message)
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
        print(f"INCONCLUSIVE: {message}")
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
