#!/usr/bin/env python3
"""Qualify frozen exercise/policy replay across early-exercise models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "black_scholes",
    "heston",
    "bates",
    "cir",
    "g2",
    "g2_plus_plus_svensson",
)
REPLAYS = ("frozen_exercise", "frozen_policy")
EXECUTIONS = ("mono", "node_graph")
FUNCTION = re.compile(r"^ Function (.+):$")
RESOURCE = re.compile(r"\b(REG|STACK|LOCAL|SHARED):(\d+)")
PROFILE_INVOCATIONS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.splitlines()).rstrip() + "\n"



def json_records(text: str) -> list[dict]:
    records: list[dict] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def parse_resource_usage(text: str) -> dict[str, dict[str, int]]:
    resources: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = FUNCTION.match(line)
        if match is not None:
            current = match.group(1)
            resources.setdefault(current, {})
            continue
        if current is None:
            continue
        for name, value in RESOURCE.findall(line):
            resources[current][name.lower()] = int(value)
    return resources


def phase(logical_name: str) -> str:
    if (
        "frozen_replay" in logical_name
        or "frozen_sensitiv" in logical_name
        or "finalize_frozen" in logical_name
    ):
        return "replay"
    return "capture"


def maximum(values: list[int | float], default: int | float = 0):
    return max(values) if values else default


def resource_summary(
    diagnostics: list[dict],
    compiled: dict[str, dict[str, int]],
) -> dict:
    symbols = {record["compiled_symbol"] for record in diagnostics}
    missing = sorted(symbol for symbol in symbols if symbol not in compiled)

    def phase_records(selected_phase: str) -> list[dict]:
        return [
            record
            for record in diagnostics
            if phase(record["kernel"]) == selected_phase
        ]

    def summarize(records: list[dict]) -> dict:
        record_symbols = {record["compiled_symbol"] for record in records}
        return {
            "maximum_registers_per_thread": maximum([
                record["resources"]["registers_per_thread"]
                for record in records
            ]),
            "maximum_local_bytes_per_thread": maximum([
                record["resources"]["local_bytes_per_thread"]
                for record in records
            ]),
            "maximum_stack_bytes_per_thread": maximum([
                compiled.get(symbol, {}).get("stack", 0)
                for symbol in record_symbols
            ]),
            "minimum_theoretical_occupancy": min(
                (
                    record["occupancy"]["theoretical"]
                    for record in records
                ),
                default=0.0,
            ),
        }

    return {
        "launched_kernel_count": len(symbols),
        "missing_cuobjdump_symbols": missing,
        "overall": summarize(diagnostics),
        "capture": summarize(phase_records("capture")),
        "replay": summarize(phase_records("replay")),
    }


def parse_nsys_csv(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Total Time" in line and "Instances" in line and "Name" in line
        ),
        None,
    )
    if header_index is None:
        raise RuntimeError("Nsight Systems kernel summary has no CSV header.")
    return list(csv.DictReader(io.StringIO("\n".join(lines[header_index:]))))


def profile_phases(
    nsys: str,
    command: list[str],
    diagnostics: list[dict],
    profile_csv: Path,
) -> dict:
    symbol_phase = {
        record["compiled_symbol"]: phase(record["kernel"])
        for record in diagnostics
    }
    stats_text: str | None = None
    failures: list[str] = []
    for attempt in range(1, 4):
        with tempfile.TemporaryDirectory(
            prefix="ai-factory-frozen-replay-"
        ) as directory:
            report_stem = Path(directory) / "profile"
            environment = os.environ.copy()
            environment["AI_FACTORY_PERFORMANCE_PROFILE_PROBE"] = "1"
            profiled = subprocess.run(
                [
                    nsys,
                    "profile",
                    "--force-overwrite=true",
                    "--trace=cuda",
                    "--sample=none",
                    "--output", str(report_stem),
                    *command,
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            if profiled.returncode != 0:
                failures.append(
                    f"attempt {attempt}: profile exit "
                    f"{profiled.returncode}: {profiled.stderr}"
                )
                continue
            stats = subprocess.run(
                [
                    nsys,
                    "stats",
                    "--report", "cuda_gpu_kern_sum:mangled",
                    "--format", "csv",
                    "--output", "-",
                    str(report_stem.with_suffix(".nsys-rep")),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if stats.returncode != 0:
                failures.append(
                    f"attempt {attempt}: stats exit "
                    f"{stats.returncode}: {stats.stderr}"
                )
                continue
            try:
                parse_nsys_csv(stats.stdout)
            except RuntimeError as error:
                failures.append(f"attempt {attempt}: {error}")
                continue
            stats_text = normalized_text(stats.stdout)
            break
    if stats_text is None:
        raise RuntimeError(
            "Nsight Systems produced no usable CUDA kernel summary after "
            "three attempts:\n" + "\n".join(failures)
        )
    profile_csv.write_text(stats_text)
    totals = {"capture": 0.0, "replay": 0.0, "unattributed": 0.0}
    unattributed: list[str] = []
    for row in parse_nsys_csv(stats_text):
        name = row["Name"]
        selected_phase = symbol_phase.get(name)
        if selected_phase is None:
            selected_phase = "unattributed"
            unattributed.append(name)
        totals[selected_phase] += float(row["Total Time (ns)"])
    return {
        "capture_ms_per_call":
            totals["capture"] / (1.0e6 * PROFILE_INVOCATIONS),
        "replay_ms_per_call":
            totals["replay"] / (1.0e6 * PROFILE_INVOCATIONS),
        "unattributed_ms_per_call":
            totals["unattributed"] / (1.0e6 * PROFILE_INVOCATIONS),
        "unattributed_symbols": sorted(set(unattributed)),
        "profile_invocations": PROFILE_INVOCATIONS,
        "raw_csv": profile_csv.name,
    }


def run_case(
    binary: Path,
    model: str,
    replay: str,
    execution: str,
    paths: int,
    compiled: dict[str, dict[str, int]],
    output_dir: Path,
    profile: bool,
    nsys: str | None,
) -> dict:
    case_id = f"{model}__{replay}__{execution}"
    command = [str(binary), model, replay, execution, str(paths)]
    environment = os.environ.copy()
    environment["AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS"] = "1"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    (output_dir / f"{case_id}.stdout.jsonl").write_text(result.stdout)
    (output_dir / f"{case_id}.stderr.jsonl").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"{case_id} failed with exit code {result.returncode}."
        )
    measurements = [
        record
        for record in json_records(result.stdout)
        if record.get("finding") == "FROZEN-REPLAY-QUALIFICATION"
    ]
    if len(measurements) != 1:
        raise RuntimeError(f"{case_id} emitted {len(measurements)} measurements.")
    diagnostics = [
        record
        for record in json_records(result.stderr)
        if record.get("type") == "cuda_kernel_launch_diagnostics"
    ]
    if not diagnostics:
        raise RuntimeError(f"{case_id} emitted no CUDA diagnostics.")
    phases = None
    if profile:
        if nsys is None:
            raise RuntimeError("Nsight Systems is unavailable.")
        phases = profile_phases(
            nsys,
            command,
            diagnostics,
            output_dir / f"{case_id}.nsys-kernels.csv",
        )
    return {
        "id": case_id,
        "command": command,
        "measurement": measurements[0],
        "resources": resource_summary(diagnostics, compiled),
        "phase_timings": phases,
    }


def ratio(value: float, baseline: float) -> float:
    return value / baseline if baseline else 0.0


def comparison_rows(cases: list[dict]) -> list[dict]:
    by_model: dict[str, dict[tuple[str, str], dict]] = {}
    for case in cases:
        configuration = case["measurement"]["configuration"]
        by_model.setdefault(configuration["model"], {})[
            (configuration["replay"], configuration["execution"])
        ] = case
    rows: list[dict] = []
    for model, variants in by_model.items():
        baseline = variants[("frozen_exercise", "mono")]
        baseline_api = baseline["measurement"]["public_api"]["median_ms"]
        baseline_memory = baseline["measurement"]["device_memory"][
            "tracked_peak_bytes"
        ]
        for (replay, execution), case in variants.items():
            measurement = case["measurement"]
            rows.append({
                "model": model,
                "replay": replay,
                "execution": execution,
                "public_api_median_ms":
                    measurement["public_api"]["median_ms"],
                "relative_to_frozen_exercise_mono":
                    ratio(
                        measurement["public_api"]["median_ms"],
                        baseline_api,
                    ),
                "tracked_peak_bytes":
                    measurement["device_memory"]["tracked_peak_bytes"],
                "memory_relative_to_frozen_exercise_mono":
                    ratio(
                        measurement["device_memory"]["tracked_peak_bytes"],
                        baseline_memory,
                    ),
            })
    return rows


def markdown(cases: list[dict], comparisons: list[dict]) -> str:
    lines = [
        "# Frozen replay performance qualification",
        "",
        (
            "API/kernel medians use the statistical protocol embedded in "
            "report.json. Capture/replay times are five-call Nsight Systems "
            "kernel aggregates."
        ),
        (
            "Peak MiB is tracked application memory: persistent inputs, "
            "outputs, caller workspace and transient LSM workspace; it "
            "excludes the CUDA driver context."
        ),
        "",
        "| Model | Replay | Execution | API median ms | Kernel median ms "
        "| Capture ms | Replay ms | Peak MiB | Regs | Stack B | Local B "
        "| Min occupancy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        measurement = case["measurement"]
        configuration = measurement["configuration"]
        resource = case["resources"]["overall"]
        phases = case["phase_timings"] or {}
        lines.append(
            f"| {configuration['model']} | {configuration['replay']} "
            f"| {configuration['execution']} "
            f"| {measurement['public_api']['median_ms']:.3f} "
            f"| {measurement['kernel']['median_ms']:.3f} "
            f"| {phases.get('capture_ms_per_call', float('nan')):.3f} "
            f"| {phases.get('replay_ms_per_call', float('nan')):.3f} "
            f"| {measurement['device_memory']['tracked_peak_bytes'] / 2**20:.2f} "
            f"| {resource['maximum_registers_per_thread']} "
            f"| {resource['maximum_stack_bytes_per_thread']} "
            f"| {resource['maximum_local_bytes_per_thread']} "
            f"| {resource['minimum_theoretical_occupancy']:.3f} |"
        )
    lines.extend([
        "",
        "Ratios use frozen-exercise/mono as the within-model baseline.",
        "",
        "| Model | Replay | Execution | API ratio | Memory ratio |",
        "|---|---|---:|---:|---:|",
    ])
    for row in comparisons:
        lines.append(
            f"| {row['model']} | {row['replay']} | {row['execution']} "
            f"| {row['relative_to_frozen_exercise_mono']:.3f} "
            f"| {row['memory_relative_to_frozen_exercise_mono']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--measurement-source",
        type=Path,
        help=(
            "Reuse measurements from a compatible report while collecting "
            "fresh profiles and resources."
        ),
    )
    arguments = parser.parse_args()
    if arguments.paths <= 0:
        raise ValueError("--paths must be positive.")
    arguments.output_dir.mkdir(parents=True, exist_ok=False)

    binary = (
        arguments.build_dir
        / "benchmark_price_gradients_early_exercise_replay"
    )
    if not binary.is_file():
        raise FileNotFoundError(f"Missing benchmark binary: {binary}")
    binary_hash = sha256(binary)
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is unavailable.")
    resource_result = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(binary)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    resource_text = normalized_text(resource_result.stdout)
    (arguments.output_dir / "cuobjdump-resource-usage.txt").write_text(
        resource_text
    )
    compiled = parse_resource_usage(resource_text)
    nsys = shutil.which("nsys") if arguments.profile else None

    cases = [
        run_case(
            binary,
            model,
            replay,
            execution,
            arguments.paths,
            compiled,
            arguments.output_dir,
            arguments.profile,
            nsys,
        )
        for model in MODELS
        for replay in REPLAYS
        for execution in EXECUTIONS
    ]
    measurement_source = None
    if arguments.measurement_source is not None:
        source_path = arguments.measurement_source.resolve()
        source_document = json.loads(source_path.read_text())
        if source_document.get("benchmark_binary_sha256") != binary_hash:
            raise RuntimeError(
                "The measurement source was produced by another binary."
            )
        if source_document.get("paths_per_price") != arguments.paths:
            raise RuntimeError(
                "The measurement source used another path count."
            )
        source_cases = {
            case["id"]: case for case in source_document.get("cases", [])
        }
        expected_ids = {case["id"] for case in cases}
        if set(source_cases) != expected_ids:
            raise RuntimeError(
                "The measurement source does not contain the same cases."
            )
        for case in cases:
            case["measurement"] = source_cases[case["id"]]["measurement"]
            raw_name = f"{case['id']}.stdout.jsonl"
            raw_source = source_path.parent / raw_name
            if raw_source.is_file():
                shutil.copy2(raw_source, arguments.output_dir / raw_name)
        copied_source = arguments.output_dir / "timing-source-report.json"
        shutil.copy2(source_path, copied_source)
        measurement_source = {
            "source": copied_source.name,
            "sha256": sha256(copied_source),
        }
    comparisons = comparison_rows(cases)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-uall"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    report = {
        "schema_version": 1,
        "benchmark_binary": str(binary.relative_to(ROOT)),
        "benchmark_binary_sha256": binary_hash,
        "paths_per_price": arguments.paths,
        "profiled_with_nsys": arguments.profile,
        "measurement_source": measurement_source,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip(),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "cases": cases,
        "comparisons": comparisons,
    }
    (arguments.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    summary = markdown(cases, comparisons)
    (arguments.output_dir / "summary.md").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
