#!/usr/bin/env python3
"""Run the authoritative CUDA performance manifest and gate it."""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    from .check_baseline import compare, measurement_key
except ImportError:
    from check_baseline import compare, measurement_key


def commands(
    baseline: dict[str, Any], build_directory: Path
) -> list[tuple[str, list[str]]]:
    """Resolve commands exclusively from the versioned manifest."""
    result: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for specification in baseline["commands"]:
        command_id = specification["id"]
        executable_name = specification["executable"]
        arguments = specification["arguments"]
        if command_id in seen:
            raise ValueError(f"duplicate performance command id: {command_id}")
        if (
            not isinstance(command_id, str)
            or not command_id
            or not isinstance(executable_name, str)
            or not executable_name
            or Path(executable_name).name != executable_name
            or not isinstance(arguments, list)
            or not all(isinstance(argument, str) for argument in arguments)
        ):
            raise ValueError(f"invalid performance command: {specification}")
        seen.add(command_id)
        result.append(
            (
                command_id,
                [str(build_directory / executable_name), *arguments],
            )
        )
    if not result:
        raise ValueError("performance manifest contains no commands")
    return result


def validate_build_configuration(
    baseline: dict[str, Any], build_directory: Path
) -> None:
    cache_path = build_directory / "CMakeCache.txt"
    values: dict[str, str] = {}
    for line in cache_path.read_text().splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        values[key] = value
    environment = baseline["environment"]
    if values.get("CMAKE_BUILD_TYPE") != environment["build_type"]:
        raise ValueError("performance build type differs from the manifest")
    architectures = ";".join(
        str(architecture) for architecture in environment["cuda_architectures"]
    )
    if values.get("CMAKE_CUDA_ARCHITECTURES") != architectures:
        raise ValueError("CUDA architectures differ from the manifest")
    if values.get("AI_FACTORY_CUDA_TUNING_PROFILE_ID") != baseline[
        "architecture_profile"
    ]["tuning_profile_id"]:
        raise ValueError("CUDA tuning profile differs from the manifest")
    cuda_flags = values.get("CMAKE_CUDA_FLAGS", "")
    if environment["fast_math"] or "--use_fast_math" in cuda_flags:
        raise ValueError("performance campaigns require fast math to be disabled")


def _power_source() -> str:
    supplies = Path("/sys/class/power_supply")
    if supplies.is_dir():
        online = [
            path.read_text().strip()
            for path in supplies.glob("*/online")
            if path.is_file()
        ]
        if online:
            return "external_power" if "1" in online else "battery"
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        return "unavailable"
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_Battery).BatteryStatus",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not values:
        return "no_battery"
    statuses = {int(value) for value in values}
    return (
        "battery"
        if statuses & {1, 4, 5}
        else "external_power"
        if statuses <= {2, 3, 6, 7, 8, 9, 11}
        else "unknown"
    )


def parse_nvidia_power_limits(xml_status: str) -> dict[str, float | None]:
    """Extract one GPU's power envelope from `nvidia-smi -q -x`."""
    try:
        root = ET.fromstring(xml_status)
    except ET.ParseError as error:
        raise ValueError("invalid NVIDIA XML status") from error
    gpu_nodes = root.findall("gpu")
    if len(gpu_nodes) != 1:
        raise ValueError("performance profile requires one GPU power record")

    def watts(tag: str) -> float | None:
        value = gpu_nodes[0].findtext(f"gpu_power_readings/{tag}", "N/A")
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?) W", value)
        return float(match.group(1)) if match else None

    return {
        "current": watts("current_power_limit"),
        "default": watts("default_power_limit"),
        "minimum": watts("min_power_limit"),
        "maximum": watts("max_power_limit"),
    }


def collect_preflight() -> dict[str, Any]:
    fields = (
        "name,pci.bus_id,pstate,power.draw,power.limit,"
        "clocks.current.graphics,clocks.current.memory,temperature.gpu,"
        "clocks_throttle_reasons.active,"
        "clocks_throttle_reasons.gpu_idle,"
        "clocks_throttle_reasons.sw_power_cap,"
        "clocks_throttle_reasons.hw_slowdown,"
        "clocks_throttle_reasons.hw_thermal_slowdown,"
        "clocks_throttle_reasons.sw_thermal_slowdown"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("performance profile requires exactly one visible GPU")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 14:
        raise ValueError("unexpected nvidia-smi preflight output")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    xml_status = subprocess.run(
        ["nvidia-smi", "-q", "-x"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "gpu": values[0],
        "pci_bus_id": values[1],
        "performance_state": values[2],
        "power_draw_w": values[3],
        "power_limit_w": values[4],
        "graphics_clock_mhz": values[5],
        "memory_clock_mhz": values[6],
        "temperature_c": int(values[7]),
        "throttle_reasons": values[8],
        "throttle": {
            "gpu_idle": values[9],
            "software_power_cap": values[10],
            "hardware_slowdown": values[11],
            "hardware_thermal_slowdown": values[12],
            "software_thermal_slowdown": values[13],
        },
        "power_limits_w": parse_nvidia_power_limits(xml_status.stdout),
        "power_source": _power_source(),
        "concurrent_compute_processes": [
            line.strip() for line in processes.stdout.splitlines() if line.strip()
        ],
    }


def validate_preflight(baseline: dict[str, Any], snapshot: dict[str, Any]) -> None:
    policy = baseline["decision_policy"]["preflight"]
    if snapshot["gpu"] != baseline["environment"]["gpu"]:
        raise ValueError("preflight GPU differs from the baseline profile")
    if snapshot["power_source"] not in policy["accepted_power_sources"]:
        raise ValueError(
            f"preflight power source is {snapshot['power_source']}; external "
            "power is required"
        )
    if snapshot["temperature_c"] > policy["maximum_temperature_c"]:
        raise ValueError("preflight GPU temperature exceeds the manifest bound")
    minimum_power_limit = policy.get("minimum_current_power_limit_w")
    current_power_limit = snapshot.get("power_limits_w", {}).get("current")
    if minimum_power_limit is not None and (
        current_power_limit is None or current_power_limit < minimum_power_limit
    ):
        raise ValueError(
            "preflight GPU power limit is below the manifest bound: "
            f"{current_power_limit} W < {minimum_power_limit} W"
        )
    if snapshot["concurrent_compute_processes"]:
        raise ValueError("preflight found concurrent GPU compute processes")
    forbidden = policy["forbidden_throttle_reasons"]
    active = [
        reason
        for reason in forbidden
        if snapshot["throttle"].get(reason) == "Active"
    ]
    if active:
        raise ValueError(
            "preflight found forbidden GPU throttling: " + ", ".join(active)
        )


def validate_campaign_preflight(
    baseline: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    """Reject environmental violations without inspecting benchmark results."""
    validate_preflight(baseline, before)
    validate_preflight(baseline, after)
    if before["power_source"] != after["power_source"]:
        raise ValueError("campaign power source changed")


def stabilize_thermal_environment(
    baseline: dict[str, Any],
    build_directory: Path,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reach the profile's warm operating range before the official snapshot."""
    policy = baseline["decision_policy"]["preflight"]["thermal_stabilization"]
    command_id = policy["command_id"]
    matches = [
        command for identifier, command in commands(baseline, build_directory)
        if identifier == command_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"thermal stabilization command is not unique: {command_id}"
        )
    command = matches[0]
    snapshot = collect_preflight()
    validate_preflight(baseline, snapshot)
    evidence.append({"phase": "initial", "snapshot": snapshot})
    environment = os.environ.copy()
    environment.pop("AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS", None)
    temperatures: list[int] = []
    started_at = time.monotonic()
    for run in range(policy["maximum_runs"]):
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        snapshot = collect_preflight()
        validate_preflight(baseline, snapshot)
        elapsed_seconds = time.monotonic() - started_at
        evidence.append({
            "phase": "after_stabilization_run",
            "run": run + 1,
            "command_id": command_id,
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "elapsed_seconds": elapsed_seconds,
            "snapshot": snapshot,
        })
        temperatures.append(snapshot["temperature_c"])
        window = temperatures[-policy["temperature_window"]:]
        if (
            run + 1 >= policy["minimum_runs"]
            and elapsed_seconds >= policy["minimum_duration_seconds"]
            and len(window) == policy["temperature_window"]
            and max(window) - min(window)
                <= policy["maximum_temperature_range_c"]
        ):
            return snapshot
    raise ValueError(
        "thermal stabilization did not converge within the declared runs"
    )


def campaign_series_can_complete(
    eligible_count: int,
    completed_count: int,
    required_count: int,
    maximum_count: int,
) -> bool:
    """Return whether the remaining declared attempts can still succeed."""
    return eligible_count + maximum_count - completed_count >= required_count


_COMPILED_RESOURCE_CACHE: dict[Path, dict[str, dict[str, Any]]] = {}


def _compiled_resource_inventory(executable: Path) -> dict[str, dict[str, Any]]:
    """Read exact per-symbol ptxas/SASS resources from the linked binary."""
    cached = _COMPILED_RESOURCE_CACHE.get(executable)
    if cached is not None:
        return cached
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise ValueError("cuobjdump is required for compiled-resource budgets")
    resource_output = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(executable)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    resource_pattern = re.compile(
        r"^ Function (.*?):\n"
        r"  REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        re.MULTILINE,
    )
    inventory: dict[str, dict[str, Any]] = {}
    for match in resource_pattern.finditer(resource_output):
        symbol, registers, stack, shared, local = match.groups()
        values = {
            "compiled_symbol": symbol,
            "registers_per_thread": int(registers),
            "stack_frame_bytes": int(stack),
            "static_shared_bytes_per_block": int(shared),
            "local_bytes_per_thread": int(local),
        }
        previous = inventory.get(symbol)
        if previous is not None and previous != values:
            raise ValueError(f"conflicting resource records for {symbol}")
        inventory[symbol] = values

    sass_output = subprocess.run(
        [cuobjdump, "--dump-sass", str(executable)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    current_symbol: str | None = None
    sass: dict[str, dict[str, int]] = {}
    for line in sass_output.splitlines():
        function = re.search(r"Function\s*:\s*(\S+)", line)
        if function:
            current_symbol = function.group(1)
            sass.setdefault(current_symbol, {
                "sass_code_bytes": 0,
                "sass_instruction_count": 0,
                "sass_local_load_instructions": 0,
                "sass_local_store_instructions": 0,
            })
            continue
        if current_symbol is None:
            continue
        instruction = re.search(
            r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9.]*)",
            line,
        )
        if instruction is None:
            continue
        address = int(instruction.group(1), 16)
        opcode = instruction.group(2)
        values = sass[current_symbol]
        values["sass_code_bytes"] = max(
            values["sass_code_bytes"], address + 16
        )
        values["sass_instruction_count"] += 1
        if opcode.startswith("LDL"):
            values["sass_local_load_instructions"] += 1
        if opcode.startswith("STL"):
            values["sass_local_store_instructions"] += 1

    for symbol, values in inventory.items():
        if symbol not in sass or sass[symbol]["sass_instruction_count"] == 0:
            raise ValueError(f"missing SASS inventory for {symbol}")
        values.update(sass[symbol])
    _COMPILED_RESOURCE_CACHE[executable] = inventory
    return inventory


def _attach_compiled_resources(
    executable: Path, diagnostics: list[dict[str, Any]]
) -> None:
    inventory = _compiled_resource_inventory(executable)
    for diagnostic in diagnostics:
        symbol = diagnostic.get("compiled_symbol")
        if symbol not in inventory:
            raise ValueError(
                f"{executable.name}: compiled resource symbol not found: {symbol}"
            )
        compiled = copy.deepcopy(inventory[symbol])
        runtime = diagnostic["resources"]
        for field in ("registers_per_thread", "static_shared_bytes_per_block"):
            if compiled[field] != runtime[field]:
                raise ValueError(
                    f"{executable.name}: runtime/compiled {field} differs for "
                    f"{symbol}"
                )
        compiled_local_allocation = (
            compiled["stack_frame_bytes"] + compiled["local_bytes_per_thread"]
        )
        if compiled_local_allocation != runtime["local_bytes_per_thread"]:
            raise ValueError(
                f"{executable.name}: runtime local bytes differ from compiled "
                f"stack plus local bytes for {symbol}"
            )
        diagnostic["compiled_resources"] = compiled


def run_commands(
    command_list: list[tuple[str, list[str]]]
) -> list[dict[str, Any]]:
    """Capture measurements and the exact launch resources preceding each."""
    measurements: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS"] = "1"
    for command_id, command in command_list:
        executable = Path(command[0])
        if not executable.is_file():
            raise FileNotFoundError(f"missing performance executable: {executable}")
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        pending_resources: list[dict[str, Any]] = []
        emitted = 0
        for line_number, line in enumerate(completed.stdout.splitlines(), 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{command_id}:{line_number}: non-JSON benchmark output: {line}"
                ) from error
            if document.get("type") == "cuda_kernel_launch_diagnostics":
                pending_resources.append(document)
                continue
            if document.get("schema") != "ai_factory_cuda_performance_baseline":
                raise ValueError(
                    f"{command_id}:{line_number}: unknown benchmark record"
                )
            if not pending_resources:
                raise ValueError(
                    f"{command_id}:{line_number}: measurement has no launch resources"
                )
            _attach_compiled_resources(executable, pending_resources)
            document["command_id"] = command_id
            executable_payload = executable.read_bytes()
            document["binary"] = {
                "executable_bytes": len(executable_payload),
                "executable_sha256": hashlib.sha256(
                    executable_payload
                ).hexdigest(),
                "command_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "command_id": command_id,
                            "executable": executable.name,
                            "arguments": command[1:],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
            document["resources"] = pending_resources
            pending_resources = []
            measurements.append(document)
            emitted += 1
        if pending_resources:
            raise ValueError(
                f"{command_id}: launch diagnostics were not attached to a measurement"
            )
        if emitted == 0:
            raise ValueError(f"{command_id}: command emitted no measurement")
    return measurements


def select_manifest(
    baseline: dict[str, Any],
    raw_measurements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for measurement in baseline["measurements"]:
        key = measurement_key(measurement)
        if key in expected:
            raise ValueError(f"baseline manifest contains duplicate key {key}")
        expected[key] = measurement

    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for measurement in raw_measurements:
        key = measurement_key(measurement)
        if key not in expected:
            raise ValueError(f"benchmark command emitted undeclared key {key}")
        if key in selected:
            raise ValueError(f"benchmark commands emitted duplicate key {key}")
        reference = expected[key]
        if measurement["command_id"] != reference["command_id"]:
            raise ValueError(
                f"benchmark key {key} emitted by {measurement['command_id']}, "
                f"expected {reference['command_id']}"
            )
        measurement["measurement_id"] = reference["id"]
        selected[key] = measurement
    missing = expected.keys() - selected.keys()
    if missing:
        raise ValueError(f"benchmark commands omitted manifest keys: {sorted(missing)}")
    return [selected[key] for key in expected]


def select_initialization_manifest(
    baseline: dict[str, Any],
    raw_measurements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Allow protocol-field migration while preserving every stable identity."""
    expected = {
        reference["id"]: reference for reference in baseline["measurements"]
    }
    if len(expected) != len(baseline["measurements"]):
        raise ValueError("baseline manifest contains duplicate measurement ids")
    command_counts: dict[str, int] = {}
    for measurement in raw_measurements:
        command_id = measurement["command_id"]
        command_counts[command_id] = command_counts.get(command_id, 0) + 1
    selected: dict[str, dict[str, Any]] = {}
    for measurement in raw_measurements:
        measurement_id = _measurement_id(
            measurement["command_id"],
            measurement["variant"],
            command_counts,
        )
        reference = expected.get(measurement_id)
        if reference is None:
            raise ValueError(
                "protocol initialization emitted undeclared measurement id "
                f"{measurement_id}"
            )
        if measurement_id in selected:
            raise ValueError(
                "protocol initialization emitted duplicate measurement id "
                f"{measurement_id}"
            )
        for field in ("command_id", "finding", "benchmark", "variant"):
            if measurement[field] != reference[field]:
                raise ValueError(
                    "protocol initialization changed stable identity field "
                    f"{field} for {measurement_id}"
                )
        measurement["measurement_id"] = measurement_id
        selected[measurement_id] = measurement
    missing = expected.keys() - selected.keys()
    if missing:
        raise ValueError(
            "protocol initialization omitted measurement ids: "
            f"{sorted(missing)}"
        )
    return [selected[measurement_id] for measurement_id in expected]


def _measurement_id(
    command_id: str,
    variant: str,
    command_counts: dict[str, int],
) -> str:
    if command_counts[command_id] == 1:
        return command_id
    suffix = re.sub(r"[^a-z0-9]+", "_", variant.lower()).strip("_")
    return f"{command_id}__{suffix}"


def _numerical_budgets(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    budgets: dict[str, dict[str, Any]] = {}
    informational = {
        "samples_per_second",
        "median_ms_per_price",
        "median_wall_per_launch_ms",
        "median_kernel_per_launch_ms",
    }
    maximums = {
        "maximum_absolute_error",
        "maximum_relative_error",
        "maximum_price_difference",
    }
    for field, value in values.items():
        if field in informational:
            budgets[field] = {"rule": "informational"}
        elif field in maximums:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"maximum numerical field {field} is not numeric")
            budgets[field] = {
                "rule": "maximum",
                "value": 0.0 if value == 0 else abs(float(value)) * 1.05,
            }
        elif isinstance(value, (bool, str, int)):
            budgets[field] = {"rule": "exact"}
        elif isinstance(value, float):
            budgets[field] = {
                "rule": "relative",
                "absolute_tolerance": 1.0e-7,
                "relative_tolerance": 1.0e-5,
            }
        else:
            raise ValueError(f"no numerical budget rule for {field}")
    return budgets


def _with_resource_budgets(
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = copy.deepcopy(diagnostics)
    for diagnostic in result:
        compiled = diagnostic["compiled_resources"]
        diagnostic["compiled_resource_budgets"] = {
            field: {
                "rule": "maximum",
                "value": compiled[field],
                "reason": "compiled kernel resources may not regress",
            }
            for field in (
                "registers_per_thread",
                "stack_frame_bytes",
                "static_shared_bytes_per_block",
                "local_bytes_per_thread",
                "sass_local_load_instructions",
                "sass_local_store_instructions",
            )
        }
        diagnostic["compiled_resource_budgets"].update({
            "sass_instruction_count": {
                "rule": "maximum",
                "value": max(
                    compiled["sass_instruction_count"] + 16,
                    int(compiled["sass_instruction_count"] * 1.05),
                ),
                "reason": "allow only bounded specialization code growth",
            },
            "sass_code_bytes": {
                "rule": "maximum",
                "value": max(
                    compiled["sass_code_bytes"] + 256,
                    int(compiled["sass_code_bytes"] * 1.05),
                ),
                "reason": "allow only bounded specialization code growth",
            },
        })
    return result


def _binary_budgets(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    size = observation["executable_bytes"]
    return {
        "executable_bytes": {
            "rule": "maximum",
            "value": max(size + 64 * 1024, int(size * 1.05)),
            "reason": "bound complete benchmark binary growth",
        },
        "executable_sha256": {
            "rule": "informational",
            "reason": "identify the exact profiled binary",
        },
        "command_sha256": {
            "rule": "exact",
            "reason": "the profiled command is part of the workload identity",
        },
    }


def _device_memory_budgets(
    observation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    budgets: dict[str, dict[str, Any]] = {}
    for field, value in observation.items():
        if field == "total_bytes":
            budgets[field] = {
                "rule": "exact",
                "reason": "the architecture profile fixes total device memory",
            }
        elif field == "free_margin_bytes":
            budgets[field] = {
                "rule": "minimum",
                "value": max(0, value - 64 * 1024 * 1024),
                "reason": "preserve the measured allocation safety margin",
            }
        elif field in {
            "observed_resident_bytes",
            "driver_context_and_pool_bytes",
        }:
            budgets[field] = {
                "rule": "maximum",
                "value": value + 64 * 1024 * 1024,
                "reason": "allow bounded CUDA context and plan-cache variation",
            }
        else:
            budgets[field] = {
                "rule": "maximum",
                "value": value,
                "reason": "owned device allocation categories may not grow",
            }
    return budgets


def initialize_measurements(
    baseline: dict[str, Any], raw_measurements: list[dict[str, Any]]
) -> None:
    """Initialize a new architecture manifest from exhaustive command output."""
    command_counts: dict[str, int] = {}
    for measurement in raw_measurements:
        command_id = measurement["command_id"]
        command_counts[command_id] = command_counts.get(command_id, 0) + 1
    references: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    kernel_regression_budget = {
        "rule": "regression",
        "maximum_regression": 0.05,
        "maximum_coefficient_of_variation": 0.05,
    }
    host_regression_budget = {
        "rule": "regression",
        "maximum_regression": 0.05,
        "maximum_coefficient_of_variation": 0.10,
    }
    for measurement in raw_measurements:
        reference = copy.deepcopy(measurement)
        reference.pop("measurement_id", None)
        reference.pop("protocol_version", None)
        reference.pop("environment", None)
        measurement_id = _measurement_id(
            reference["command_id"], reference["variant"], command_counts
        )
        if measurement_id in used_ids:
            raise ValueError(f"generated duplicate measurement id {measurement_id}")
        used_ids.add(measurement_id)
        reference["id"] = measurement_id
        reference["comparison_policy"] = (
            "informational"
            if reference["benchmark"] == "closed_form_launch_overhead"
            else "blocking"
        )
        reference["timing_budgets"] = {
            "kernel": copy.deepcopy(kernel_regression_budget),
            "public_api": copy.deepcopy(host_regression_budget),
            "pipeline": {"rule": "record_only"},
        }
        if "publication_wall" in reference:
            reference["timing_budgets"]["publication_wall"] = {
                "rule": "regression",
                "maximum_regression": 0.05,
                "maximum_coefficient_of_variation": 0.10,
            }
        reference["numerical_budgets"] = _numerical_budgets(
            reference["numerical_check"]
        )
        reference["resources"] = _with_resource_budgets(
            reference["resources"]
        )
        reference["device_memory_budgets"] = _device_memory_budgets(
            reference["device_memory"]
        )
        reference["binary_budgets"] = _binary_budgets(reference["binary"])
        references.append(reference)
    baseline["measurements"] = references


def initialize_baseline(
    baseline: dict[str, Any],
    candidate: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Initialize an incompatible protocol without losing numerical contracts."""
    prior = {
        reference["id"]: reference
        for reference in baseline.get("measurements", [])
    }
    initialized = copy.deepcopy(baseline)
    initialize_measurements(initialized, candidate)
    initialized_ids = [
        reference["id"] for reference in initialized["measurements"]
    ]
    if prior and list(prior) != initialized_ids:
        raise ValueError(
            "protocol initialization changed declared measurement identities"
        )
    for reference in initialized["measurements"]:
        predecessor = prior.get(reference["id"])
        if predecessor is None:
            continue
        predecessor_budgets = predecessor.get("numerical_budgets")
        if set((predecessor_budgets or {}).keys()) != set(
            reference["numerical_check"].keys()
        ):
            raise ValueError(
                "protocol initialization changed numerical contract fields for "
                f"{reference['id']}"
            )
        reference["numerical_budgets"] = copy.deepcopy(predecessor_budgets)
        reference["comparison_policy"] = predecessor["comparison_policy"]
    initialized["generated_at"] = datetime.date.today().isoformat()
    initialized["rebaseline_reason"] = reason
    return initialized


TIMING_SECTIONS = (
    "kernel",
    "public_api",
    "pipeline",
    "publication_wall",
)


def _aggregate_timing(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate every campaign without selecting a favourable attempt."""
    return {
        "minimum_ms": min(row["minimum_ms"] for row in rows),
        "median_ms": statistics.median(row["median_ms"] for row in rows),
        "p95_ms": max(row["p95_ms"] for row in rows),
        "mean_ms": statistics.mean(row["mean_ms"] for row in rows),
        "standard_deviation_ms": max(
            row["standard_deviation_ms"] for row in rows
        ),
        "coefficient_of_variation": statistics.median(
            row["coefficient_of_variation"] for row in rows
        ),
    }


def _aggregate_numerical(
    rows: list[dict[str, Any]],
    budgets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if any(set(row) != set(budgets) for row in rows):
        raise ValueError("campaign numerical fields differ from their budgets")
    result: dict[str, Any] = {}
    for field, budget in budgets.items():
        values = [row[field] for row in rows]
        rule = budget["rule"]
        if rule == "exact":
            if any(value != values[0] for value in values[1:]):
                raise ValueError(
                    f"campaigns disagree on exact numerical field {field}"
                )
            result[field] = values[0]
        elif rule == "maximum":
            result[field] = max(values)
        elif rule in {"relative", "informational"}:
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                raise ValueError(
                    f"campaign numerical field {field} is not aggregatable"
                )
            result[field] = statistics.median(values)
        else:
            raise ValueError(f"unknown numerical aggregation rule {rule}")
    return result


def aggregate_campaigns(
    attempts: list[list[dict[str, Any]]], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the pre-declared aggregate of complete ordered campaigns."""
    if not attempts:
        raise ValueError("at least one performance attempt is required")
    references = baseline["measurements"]
    if any(len(attempt) != len(references) for attempt in attempts):
        raise ValueError("performance attempts have inconsistent sizes")
    result: list[dict[str, Any]] = []
    for measurement_index, reference in enumerate(references):
        rows = [attempt[measurement_index] for attempt in attempts]
        key = measurement_key(rows[0])
        if any(measurement_key(row) != key for row in rows[1:]):
            raise ValueError("performance attempts have inconsistent ordering")
        for field in (
            "measurement_id",
            "command_id",
            "protocol_version",
            "protocol",
            "environment",
            "configuration",
            "binary",
            "resources",
        ):
            if any(row.get(field) != rows[0].get(field) for row in rows[1:]):
                raise ValueError(
                    f"campaigns disagree on non-timing field {field}: {key}"
                )
        candidate = copy.deepcopy(rows[0])
        for section in TIMING_SECTIONS:
            present = [section in row for row in rows]
            if any(present) and not all(present):
                raise ValueError(
                    f"campaigns disagree on timing section {section}: {key}"
                )
            if all(present):
                candidate[section] = _aggregate_timing(
                    [row[section] for row in rows]
                )
        if any(
            set(row["device_memory"]) != set(rows[0]["device_memory"])
            for row in rows[1:]
        ):
            raise ValueError(f"campaigns disagree on device-memory fields: {key}")
        candidate["device_memory"] = {
            field: (
                rows[0]["device_memory"][field]
                if field == "total_bytes"
                else min(
                    row["device_memory"][field] for row in rows
                )
                if field == "free_margin_bytes"
                else max(row["device_memory"][field] for row in rows)
            )
            for field in rows[0]["device_memory"]
        }
        candidate["numerical_check"] = _aggregate_numerical(
            [row["numerical_check"] for row in rows],
            reference["numerical_budgets"],
        )
        candidate["campaign_aggregation"] = {
            "method": baseline["decision_policy"]["campaign_aggregation"],
            "attempt_count": len(rows),
        }
        result.append(candidate)
    return result


def write_raw_campaigns(
    output: Path,
    campaigns: list[dict[str, Any]],
    directory: Path | None = None,
) -> Path:
    """Persist every attempted campaign, incrementally when given a directory."""
    if directory is None:
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        directory = output.parent / f"{output.name}.campaigns" / run_id
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=False)
    files: list[dict[str, Any]] = []
    for index, campaign in enumerate(campaigns, 1):
        attempt = campaign.get("measurements")
        entry = {
            "campaign": index,
            "status": campaign["status"],
            "reason": campaign.get("reason"),
            "preflight": campaign.get("preflight"),
        }
        if attempt is not None:
            path = directory / f"campaign_{index:02d}.ndjson"
            payload = "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in attempt
            )
            path.write_text(payload)
            entry.update({
                "path": path.name,
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "measurement_count": len(attempt),
            })
        files.append(entry)
    (directory / "manifest.json").write_text(json.dumps({
        "schema": "ai_factory_performance_raw_campaigns",
        "campaign_count": len(campaigns),
        "eligible_campaign_count": sum(
            campaign["status"] == "eligible" for campaign in campaigns
        ),
        "aggregation": "none",
        "files": files,
    }, indent=2) + "\n")
    return directory


def load_raw_campaigns(directory: Path) -> list[dict[str, Any]]:
    """Load a resumable journal only after verifying every retained payload."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "ai_factory_performance_raw_campaigns":
        raise ValueError("resume directory has an incompatible campaign schema")
    entries = manifest.get("files")
    if (
        not isinstance(entries, list)
        or manifest.get("campaign_count") != len(entries)
    ):
        raise ValueError("resume campaign manifest count is inconsistent")
    records: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, 1):
        if entry.get("campaign") != index:
            raise ValueError("resume campaigns are not contiguous and ordered")
        status = entry.get("status")
        if status not in {
            "eligible",
            "rejected_environment",
            "execution_error",
        }:
            raise ValueError(f"resume campaign has invalid status {status}")
        measurements = None
        relative_path = entry.get("path")
        if relative_path is not None:
            if (
                not isinstance(relative_path, str)
                or Path(relative_path).name != relative_path
            ):
                raise ValueError("resume campaign payload path is unsafe")
            path = directory / relative_path
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
                raise ValueError(
                    f"resume campaign payload hash mismatch: {relative_path}"
                )
            try:
                measurements = [
                    json.loads(line)
                    for line in payload.decode().splitlines()
                    if line.strip()
                ]
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"resume campaign payload is invalid: {relative_path}"
                ) from error
            if len(measurements) != entry.get("measurement_count"):
                raise ValueError(
                    f"resume campaign measurement count mismatch: {relative_path}"
                )
        elif status == "eligible":
            raise ValueError("eligible resume campaign has no measurement payload")
        records.append({
            "status": status,
            "reason": entry.get("reason"),
            "preflight": entry.get("preflight"),
            "measurements": measurements,
        })
    eligible_count = sum(record["status"] == "eligible" for record in records)
    if manifest.get("eligible_campaign_count") != eligible_count:
        raise ValueError("resume eligible campaign count is inconsistent")
    return records


def write_audit_reports(
    directory: Path,
    baseline: dict[str, Any],
    candidate: list[dict[str, Any]],
) -> None:
    """Materialize the four authoritative report scopes from one manifest."""
    entries: list[dict[str, Any]] = []
    for report_name, command_ids in baseline["audit_reports"].items():
        command_set = set(command_ids)
        measurements = [
            row for row in candidate if row["command_id"] in command_set
        ]
        measurement_ids = {row["measurement_id"] for row in measurements}
        decisions = [
            decision
            for decision in baseline["decisions"]
            if measurement_ids.intersection(
                decision["selected"] + decision["rejected"]
            )
        ]
        report = {
            "schema": "ai_factory_performance_audit_report",
            "protocol_version": baseline["protocol_version"],
            "report": report_name,
            "environment": baseline["environment"],
            "architecture_profile": baseline["architecture_profile"],
            "decision_policy": baseline["decision_policy"],
            "commands": [
                command
                for command in baseline["commands"]
                if command["id"] in command_set
            ],
            "measurements": measurements,
            "decisions": decisions,
        }
        path = directory / f"report_{report_name}.json"
        payload = json.dumps(report, indent=2) + "\n"
        path.write_text(payload)
        entries.append({
            "report": report_name,
            "path": path.name,
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "measurement_count": len(measurements),
        })
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audit_reports"] = entries
    manifest["provenance"] = {
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                baseline, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "candidate_sha256": hashlib.sha256(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in candidate
            ).encode()
        ).hexdigest(),
        "binary_sha256": {
            row["command_id"]: row["binary"]["executable_sha256"]
            for row in candidate
        },
        "command_sha256": {
            row["command_id"]: row["binary"]["command_sha256"]
            for row in candidate
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def rebaseline(
    baseline: dict[str, Any],
    candidate: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Replace observations while preserving declared workloads and budgets."""
    updated = copy.deepcopy(baseline)
    updated["generated_at"] = datetime.date.today().isoformat()
    updated["rebaseline_reason"] = reason
    observations = {measurement_key(row): row for row in candidate}
    observation_fields = (
        "kernel",
        "public_api",
        "pipeline",
        "publication_wall",
        "numerical_check",
        "device_memory",
        "binary",
        "resources",
    )
    for reference in updated["measurements"]:
        observation = observations[measurement_key(reference)]
        for field in observation_fields:
            if field in observation:
                reference[field] = (
                    _with_resource_budgets(observation[field])
                    if field == "resources"
                    else observation[field]
                )
            else:
                reference.pop(field, None)
        reference["device_memory_budgets"] = _device_memory_budgets(
            observation["device_memory"]
        )
        reference["binary_budgets"] = _binary_budgets(observation["binary"])
    return updated


def _json_changes(
    before: Any,
    after: Any,
    path: str = "$",
) -> list[dict[str, Any]]:
    """Return an exhaustive, deterministic leaf-level JSON diff."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(before.keys() | after.keys()):
            child = f"{path}.{key}"
            if key not in before:
                changes.append({"path": child, "change": "added", "after": after[key]})
            elif key not in after:
                changes.append({
                    "path": child,
                    "change": "removed",
                    "before": before[key],
                })
            else:
                changes.extend(_json_changes(before[key], after[key], child))
        return changes
    if isinstance(before, list) and isinstance(after, list):
        changes = []
        for index in range(max(len(before), len(after))):
            child = f"{path}[{index}]"
            if index >= len(before):
                changes.append({
                    "path": child,
                    "change": "added",
                    "after": after[index],
                })
            elif index >= len(after):
                changes.append({
                    "path": child,
                    "change": "removed",
                    "before": before[index],
                })
            else:
                changes.extend(_json_changes(before[index], after[index], child))
        return changes
    if before == after and type(before) is type(after):
        return []
    return [{
        "path": path,
        "change": "modified",
        "before": before,
        "after": after,
    }]


def write_rebaseline_diff(
    predecessor_path: Path,
    successor: dict[str, Any],
    output_path: Path,
    reason: str,
    approval: str,
    mode: str,
) -> str:
    """Persist hashes, environments, and every predecessor/successor delta."""
    predecessor_payload = predecessor_path.read_bytes()
    predecessor = json.loads(predecessor_payload)
    before_ids = [row.get("id") for row in predecessor.get("measurements", [])]
    after_ids = [row.get("id") for row in successor.get("measurements", [])]
    if before_ids != after_ids and not (
        mode == "protocol_initialization" and not before_ids
    ):
        raise ValueError(
            "rebaseline predecessor and successor measurement ids differ"
        )
    successor_payload = json.dumps(successor, indent=2) + "\n"
    changes = _json_changes(predecessor, successor)
    document = {
        "schema": "ai_factory_performance_rebaseline_diff",
        "version": 1,
        "mode": mode,
        "reason": reason,
        "approval": approval,
        "predecessor": {
            "path": str(predecessor_path),
            "sha256": hashlib.sha256(predecessor_payload).hexdigest(),
            "environment": predecessor.get("environment"),
            "protocol_version": predecessor.get("protocol_version"),
            "rebaseline_reason": predecessor.get("rebaseline_reason"),
        },
        "successor": {
            "sha256": hashlib.sha256(successor_payload.encode()).hexdigest(),
            "environment": successor.get("environment"),
            "protocol_version": successor.get("protocol_version"),
            "rebaseline_reason": successor.get("rebaseline_reason"),
        },
        "measurement_ids": after_ids,
        "change_count": len(changes),
        "changes": changes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, output_path)
    return successor_payload


def _write_json_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def _write_candidate(path: Path, candidate: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in candidate
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rebaseline-output", type=Path)
    parser.add_argument("--rebaseline-reason")
    parser.add_argument("--rebaseline-diff-output", type=Path)
    parser.add_argument("--rebaseline-approval")
    parser.add_argument("--predecessor-baseline", type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--resume-campaigns", type=Path)
    arguments = parser.parse_args()

    try:
        rebaseline_arguments = (
            arguments.rebaseline_output,
            arguments.rebaseline_reason,
            arguments.rebaseline_diff_output,
            arguments.rebaseline_approval,
            arguments.predecessor_baseline,
        )
        if any(value is not None for value in rebaseline_arguments) and not all(
            value is not None for value in rebaseline_arguments
        ):
            raise ValueError(
                "rebaseline output, reason, diff output, approval, and retained "
                "predecessor are all required"
            )
        if arguments.initialize and arguments.rebaseline_output is None:
            raise ValueError("--initialize requires --rebaseline-output")
        if arguments.rebaseline_output is not None:
            if not arguments.rebaseline_reason.strip():
                raise ValueError("rebaseline reason must be non-empty")
            if not arguments.rebaseline_approval.strip():
                raise ValueError("rebaseline approval must be non-empty")
            if arguments.predecessor_baseline.resolve() == (
                arguments.rebaseline_output.resolve()
            ):
                raise ValueError(
                    "rebaseline predecessor must be retained at a distinct path"
                )
            if arguments.rebaseline_diff_output.resolve() in {
                arguments.predecessor_baseline.resolve(),
                arguments.rebaseline_output.resolve(),
            }:
                raise ValueError(
                    "rebaseline diff must have its own versioned path"
                )
            if (
                not arguments.initialize
                and arguments.predecessor_baseline.read_bytes()
                != arguments.baseline.read_bytes()
            ):
                raise ValueError(
                    "retained predecessor must exactly match the baseline "
                    "used to run the campaign"
                )
        baseline = json.loads(arguments.baseline.read_text())
        attempt_count = baseline["decision_policy"]["campaign_attempts"]
        maximum_attempt_count = baseline["decision_policy"][
            "maximum_campaign_attempts"
        ]
        if not isinstance(attempt_count, int) or attempt_count < 1:
            raise ValueError("manifest campaign attempt count must be positive")
        if (
            not isinstance(maximum_attempt_count, int)
            or maximum_attempt_count < attempt_count
        ):
            raise ValueError(
                "manifest maximum campaign attempts must cover eligible attempts"
            )
        retry_seconds = baseline["decision_policy"]["preflight"].get(
            "retry_cooldown_seconds", 0
        )
        if (
            not isinstance(retry_seconds, int)
            or isinstance(retry_seconds, bool)
            or retry_seconds < 0
            or retry_seconds > 60
        ):
            raise ValueError(
                "manifest preflight retry cooldown must be an integer from "
                "zero to 60 seconds"
            )
        validate_build_configuration(baseline, arguments.build_dir)
        attempts: list[list[dict[str, Any]]] = []
        campaign_records: list[dict[str, Any]] = []
        raw_campaign_directory: Path | None = None
        if arguments.resume_campaigns is not None:
            raw_campaign_directory = arguments.resume_campaigns
            campaign_records = load_raw_campaigns(raw_campaign_directory)
            if len(campaign_records) >= maximum_attempt_count:
                raise ValueError(
                    "resume journal already exhausted the maximum campaign attempts"
                )
            if any(
                record["status"] == "execution_error"
                for record in campaign_records
            ):
                raise ValueError(
                    "resume journal contains an execution error; start a new "
                    "campaign series after correcting it"
                )
            retained = next(
                (
                    record["measurements"]
                    for record in campaign_records
                    if record["measurements"] is not None
                ),
                None,
            )
            if arguments.initialize and not baseline.get("measurements"):
                if retained is not None:
                    bootstrap = copy.deepcopy(retained)
                    for measurement in bootstrap:
                        measurement.pop("measurement_id", None)
                    initialize_measurements(baseline, bootstrap)
            if baseline.get("measurements"):
                for record in campaign_records:
                    if record["measurements"] is not None:
                        record["measurements"] = (
                            select_initialization_manifest(
                                baseline, record["measurements"]
                            )
                            if arguments.initialize
                            else select_manifest(
                                baseline, record["measurements"]
                            )
                        )
                    if record["status"] == "eligible":
                        attempts.append(record["measurements"])
            if len(attempts) > attempt_count:
                raise ValueError(
                    "resume journal contains too many eligible campaigns"
                )
        for attempt in range(len(campaign_records), maximum_attempt_count):
            if len(attempts) == attempt_count:
                break
            if not campaign_series_can_complete(
                len(attempts), attempt, attempt_count, maximum_attempt_count
            ):
                print(
                    "Stopping campaign series: too few declared attempts remain "
                    "to reach the required eligible count.",
                    flush=True,
                )
                break
            print(
                f"Performance campaign {attempt + 1}/{maximum_attempt_count} "
                f"({len(attempts)}/{attempt_count} eligible)",
                flush=True,
            )
            stabilization_evidence: list[dict[str, Any]] = []
            try:
                before = stabilize_thermal_environment(
                    baseline, arguments.build_dir, stabilization_evidence
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                subprocess.CalledProcessError,
            ) as error:
                campaign_records.append({
                    "status": "rejected_environment",
                    "reason": str(error),
                    "preflight": {
                        "thermal_stabilization": stabilization_evidence
                    },
                    "measurements": None,
                })
                raw_campaign_directory = write_raw_campaigns(
                    arguments.output,
                    campaign_records,
                    raw_campaign_directory,
                )
                print(f"Rejected campaign before execution: {error}", flush=True)
                if retry_seconds and attempt + 1 < maximum_attempt_count:
                    print(
                        f"Cooling down for {retry_seconds} second(s) before "
                        "the next declared attempt.",
                        flush=True,
                    )
                    time.sleep(retry_seconds)
                continue
            try:
                raw = run_commands(commands(baseline, arguments.build_dir))
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                json.JSONDecodeError,
                subprocess.CalledProcessError,
            ) as error:
                campaign_records.append({
                    "status": "execution_error",
                    "reason": str(error),
                    "preflight": {
                        "thermal_stabilization": stabilization_evidence,
                        "before": before,
                    },
                    "measurements": None,
                })
                write_raw_campaigns(
                    arguments.output,
                    campaign_records,
                    raw_campaign_directory,
                )
                raise
            if arguments.initialize and not baseline.get("measurements"):
                initialize_measurements(baseline, raw)
            selected = (
                select_initialization_manifest(baseline, raw)
                if arguments.initialize and baseline.get("measurements")
                else select_manifest(baseline, raw)
            )
            after = collect_preflight()
            try:
                validate_campaign_preflight(baseline, before, after)
            except ValueError as error:
                campaign_records.append({
                    "status": "rejected_environment",
                    "reason": str(error),
                    "preflight": {
                        "thermal_stabilization": stabilization_evidence,
                        "before": before,
                        "after": after,
                    },
                    "measurements": selected,
                })
                raw_campaign_directory = write_raw_campaigns(
                    arguments.output,
                    campaign_records,
                    raw_campaign_directory,
                )
                print(f"Rejected campaign: {error}", flush=True)
                continue
            attempts.append(selected)
            campaign_records.append({
                "status": "eligible",
                "reason": None,
                "preflight": {
                    "thermal_stabilization": stabilization_evidence,
                    "before": before,
                    "after": after,
                },
                "measurements": selected,
            })
            raw_campaign_directory = write_raw_campaigns(
                arguments.output,
                campaign_records,
                raw_campaign_directory,
            )
        raw_campaign_directory = write_raw_campaigns(
            arguments.output,
            campaign_records,
            raw_campaign_directory,
        )
        if len(attempts) != attempt_count:
            raise ValueError(
                f"only {len(attempts)} of {attempt_count} required campaigns "
                "satisfied environmental stability"
            )
        candidate = aggregate_campaigns(attempts, baseline)
        _write_candidate(arguments.output, candidate)
        write_audit_reports(raw_campaign_directory, baseline, candidate)
        reference = baseline
        if arguments.initialize:
            # Protocol initialization validates completeness against the exact
            # candidate-derived schema. It is explicitly not a regression pass
            # against the incompatible predecessor; the exhaustive diff below
            # preserves that distinction.
            reference = initialize_baseline(
                baseline, candidate, arguments.rebaseline_reason or ""
            )
        failures, inconclusive, informational = compare(reference, candidate)
        if arguments.rebaseline_output is not None and not failures and not inconclusive:
            updated = (
                reference
                if arguments.initialize
                else rebaseline(
                    baseline, candidate, arguments.rebaseline_reason or ""
                )
            )
            predecessor_payload = arguments.predecessor_baseline.read_bytes()
            updated["lineage"] = {
                "mode": (
                    "protocol_initialization"
                    if arguments.initialize
                    else "regression_checked_rebaseline"
                ),
                "predecessor_path": str(arguments.predecessor_baseline),
                "predecessor_sha256": hashlib.sha256(
                    predecessor_payload
                ).hexdigest(),
                "diff_path": str(arguments.rebaseline_diff_output),
                "approval": arguments.rebaseline_approval,
            }
            successor_payload = write_rebaseline_diff(
                arguments.predecessor_baseline,
                updated,
                arguments.rebaseline_diff_output,
                arguments.rebaseline_reason or "",
                arguments.rebaseline_approval or "",
                (
                    "protocol_initialization"
                    if arguments.initialize
                    else "regression_checked_rebaseline"
                ),
            )
            _write_json_atomic(arguments.rebaseline_output, successor_payload)
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
        f"Captured {len(candidate)} exhaustive manifest measurements from "
        f"{attempt_count} campaign(s)."
    )
    print(f"Raw campaigns: {raw_campaign_directory}")
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
    if arguments.initialize:
        print(
            f"INITIALIZED: {arguments.rebaseline_output} "
            f"({arguments.rebaseline_reason})"
        )
    elif arguments.rebaseline_output is not None:
        print(
            f"REBASELINED: {arguments.rebaseline_output} "
            f"({arguments.rebaseline_reason})"
        )
    else:
        print("PASS: complete blocking performance manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
