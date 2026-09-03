#!/usr/bin/env python3
"""Run the complete versioned CUDA performance manifest and gate it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .check_baseline import compare, measurement_key
except ImportError:
    from check_baseline import compare, measurement_key


def commands(build_directory: Path) -> list[list[str]]:
    generic = build_directory / "ai_factory_generic_kernel_benchmark"
    volterra = build_directory / "ai_factory_volterra_kernel_benchmark"
    samples = build_directory / "ai_factory_model_sample_benchmark"
    return [
        [str(generic), "index"],
        [str(generic), "geometry"],
        [str(generic), "accumulation"],
        [str(generic), "overhead", "1024"],
        [str(generic), "ragged", "regular"],
        [str(generic), "ragged", "homogeneous"],
        [str(generic), "ragged", "heterogeneous"],
        [str(build_directory / "ai_factory_cir_kernel_benchmark_noinline")],
        [str(build_directory / "ai_factory_early_exercise_benchmark")],
        [str(samples)],
        [str(volterra), "rough_bergomi", "252", "1", "1", "65536", "21", "65536"],
        [str(volterra), "rough_bergomi", "252", "1", "1", "4096", "21", "2097152"],
        [str(volterra), "rough_bergomi", "252", "1", "1", "16384", "21", "2097152"],
        [str(volterra), "rough_bergomi", "252", "1", "1", "65536", "21", "2097152"],
        [str(volterra), "rough_bergomi", "252", "1", "8", "65536", "21", "1048576"],
        [str(volterra), "rough_bergomi", "4096", "1", "1", "65536", "21", "65536"],
    ]


def run_commands(command_list: list[list[str]]) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for command in command_list:
        executable = Path(command[0])
        if not executable.is_file():
            raise FileNotFoundError(
                f"missing performance executable: {executable}"
            )
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            document = json.loads(line)
            if document.get("schema") == "ai_factory_cuda_performance_baseline":
                measurements.append(document)
    return measurements


def select_manifest(
    baseline: dict[str, Any],
    raw_measurements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    expected = [
        measurement_key(measurement)
        for measurement in baseline["measurements"]
    ]
    if len(expected) != len(set(expected)):
        raise ValueError("baseline manifest contains duplicate keys")

    expected_set = set(expected)
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    ignored = 0
    for measurement in raw_measurements:
        key = measurement_key(measurement)
        if key not in expected_set:
            ignored += 1
            continue
        if key in selected:
            raise ValueError(f"benchmark commands emitted duplicate key {key}")
        selected[key] = measurement
    missing = expected_set - selected.keys()
    if missing:
        raise ValueError(f"benchmark commands omitted manifest keys: {missing}")
    return [selected[key] for key in expected], ignored


def best_stable_manifest(
    attempts: list[list[dict[str, Any]]],
    maximum_noise: float,
    maximum_publication_noise: float | None = None,
) -> list[dict[str, Any]]:
    if not attempts:
        raise ValueError("at least one performance attempt is required")
    result: list[dict[str, Any]] = []
    for measurement_index in range(len(attempts[0])):
        rows = [attempt[measurement_index] for attempt in attempts]
        key = measurement_key(rows[0])
        if any(measurement_key(row) != key for row in rows[1:]):
            raise ValueError("performance attempts have inconsistent ordering")
        publication_noise = (
            maximum_noise
            if maximum_publication_noise is None
            else maximum_publication_noise
        )
        stable = [
            row for row in rows
            if all(
                row[section]["coefficient_of_variation"] <= (
                    publication_noise
                    if section == "publication_wall"
                    else maximum_noise
                )
                for section in ("kernel", "publication_wall")
                if section in row
            )
        ]
        pool = stable if stable else rows
        result.append(min(pool, key=lambda row: row["kernel"]["median_ms"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts", type=int, default=3)
    arguments = parser.parse_args()

    try:
        if arguments.attempts < 1:
            raise ValueError("attempt count must be positive")
        baseline = json.loads(arguments.baseline.read_text())
        attempts: list[list[dict[str, Any]]] = []
        ignored = 0
        for attempt in range(arguments.attempts):
            print(
                f"Performance campaign {attempt + 1}/{arguments.attempts}",
                flush=True,
            )
            raw = run_commands(commands(arguments.build_dir))
            selected, attempt_ignored = select_manifest(baseline, raw)
            attempts.append(selected)
            ignored += attempt_ignored
        candidate = best_stable_manifest(
            attempts,
            baseline["decision_policy"][
                "maximum_coefficient_of_variation"
            ],
            baseline["decision_policy"].get(
                "maximum_publication_coefficient_of_variation"
            ),
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n"
                    for row in candidate)
        )
        failures, inconclusive, informational = compare(baseline, candidate)
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"performance gate error: {error}", file=sys.stderr)
        return 2

    print(
        f"Captured {len(candidate)} best stable manifest measurements from "
        f"{arguments.attempts} campaign(s); "
        f"ignored {ignored} explicit experiment measurement(s)."
    )
    for message in informational:
        print(f"INFO: {message}")
    for message in inconclusive:
        print(f"INCONCLUSIVE: {message}", file=sys.stderr)
    for message in failures:
        print(f"FAIL: {message}", file=sys.stderr)
    if failures:
        return 1
    if inconclusive:
        return 3
    print("PASS: complete blocking performance manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
