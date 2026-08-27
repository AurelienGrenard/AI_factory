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
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    inconclusive: list[str] = []
    expected_protocol = baseline["protocol_version"]
    minimum_gain = baseline["decision_policy"]["minimum_effect_size"]
    maximum_noise = baseline["decision_policy"][
        "maximum_coefficient_of_variation"
    ]
    references = {
        measurement_key(measurement): measurement
        for measurement in baseline["measurements"]
    }

    for candidate in candidates:
        key = measurement_key(candidate)
        if key not in references:
            failures.append(f"{key}: no matching baseline")
            continue
        reference = references[key]
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
        reference_kernel = reference["kernel"]
        if (kernel["coefficient_of_variation"] > maximum_noise
                or reference_kernel["coefficient_of_variation"]
                    > maximum_noise):
            inconclusive.append(
                f"{key}: coefficient of variation exceeds {maximum_noise:.1%}"
            )
            continue
        regression = (
            kernel["median_ms"] / reference_kernel["median_ms"] - 1.0
        )
        if regression > minimum_gain:
            failures.append(
                f"{key}: median regression {regression:.2%} exceeds "
                f"{minimum_gain:.2%}"
            )
    return failures, inconclusive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    arguments = parser.parse_args()
    try:
        baseline = json.loads(arguments.baseline.read_text())
        candidates = load_candidate(arguments.candidate)
        failures, inconclusive = compare(baseline, candidates)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"baseline check error: {error}", file=sys.stderr)
        return 2

    for message in inconclusive:
        print(f"INCONCLUSIVE: {message}")
    for message in failures:
        print(f"FAIL: {message}", file=sys.stderr)
    if failures:
        return 1
    print(
        f"PASS: {len(candidates)} measurement(s), "
        f"{len(inconclusive)} inconclusive"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
