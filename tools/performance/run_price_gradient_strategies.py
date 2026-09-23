#!/usr/bin/env python3
"""Run the permanent price-gradient strategy matrix and preserve CUDA evidence."""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT / "tests" / "performance" / "price_gradients"
    / "strategy_manifest.json"
)
FUNCTION = re.compile(r"^\s*Function\s*:\s*(.+?)\s*$")
LOCAL_INSTRUCTION = re.compile(r"\b(?:LDL|STL)(?:\.|\b)")
STALL_METRIC = re.compile(
    r"^smsp__average_warps_issue_stalled_(.+)_per_issue_active\.ratio$"
)
PROFILE_METRICS = {
    "duration_us": "gpu__time_duration.avg",
    "achieved_occupancy_pct":
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct":
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "uniform_branch_targets_pct":
        "smsp__sass_average_branch_targets_threads_uniform.pct",
    "global_load_bytes_per_sector":
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
    "global_store_bytes_per_sector":
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    "local_spilling_requests": "derived__local_spilling_requests",
    "shared_spilling_requests": "derived__shared_spilling_requests",
    "shared_bank_conflicts":
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "excessive_shared_wavefronts":
        "derived__memory_l1_wavefronts_shared_excessive",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported price-gradient performance manifest.")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The price-gradient performance manifest has no cases.")
    return document


def parse_json_lines(text: str) -> list[dict]:
    records: list[dict] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def parse_ncu_raw_csv(text: str) -> dict:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        raise RuntimeError("Nsight Compute raw CSV has no metric row.")
    header = rows[0]
    values = rows[2]
    if len(values) != len(header):
        raise RuntimeError("Nsight Compute raw CSV has inconsistent columns.")
    by_name = dict(zip(header, values, strict=True))

    def number(metric: str) -> float | None:
        raw = by_name.get(metric, "")
        if raw == "":
            return None
        return float(raw)

    stalls = []
    for metric, raw in by_name.items():
        match = STALL_METRIC.match(metric)
        if match is None or raw == "":
            continue
        stalls.append({"reason": match.group(1), "ratio": float(raw)})
    stalls.sort(key=lambda item: item["ratio"], reverse=True)
    return {
        "grid_size": by_name.get("Grid Size"),
        "block_size": by_name.get("Block Size"),
        **{name: number(metric) for name, metric in PROFILE_METRICS.items()},
        "leading_stalls": stalls[:5],
    }


def sass_local_instructions(
    text: str,
    launched_symbols: set[str] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    current = "<unattributed>"
    for line in text.splitlines():
        match = FUNCTION.match(line)
        if match:
            current = match.group(1)
            counts.setdefault(current, 0)
        elif LOCAL_INSTRUCTION.search(line):
            counts[current] = counts.get(current, 0) + 1
    if launched_symbols is not None:
        missing = launched_symbols.difference(counts)
        if missing:
            raise RuntimeError(
                "cuobjdump omitted launched symbols: " + ", ".join(sorted(missing))
            )
        counts = {name: counts[name] for name in launched_symbols}
    return {name: count for name, count in counts.items() if count != 0}


def representative_diagnostics(
    required_kernels: list[str], diagnostics: list[dict]
) -> list[dict]:
    """Choose the heaviest observed launch for each logical strategy phase."""
    selected: list[dict] = []
    for kernel in required_kernels:
        candidates = [record for record in diagnostics if record.get("kernel") == kernel]
        if not candidates:
            raise RuntimeError(f"No diagnostics available for required kernel {kernel}.")
        selected.append(
            max(
                enumerate(candidates),
                key=lambda item: (
                    item[1]["resources"]["registers_per_thread"],
                    item[1]["resources"]["local_bytes_per_thread"],
                    item[1]["launch"]["dynamic_shared_bytes_per_block"],
                    item[1]["launch"]["grid_block_count"],
                    item[0],
                ),
            )[1]
        )
    return selected


def profile_representatives(
    ncu: str,
    case_id: str,
    command: list[str],
    environment: dict[str, str],
    output_dir: Path,
    diagnostics: list[dict],
    required_kernels: list[str],
) -> list[dict]:
    reports: list[dict] = []
    representatives = representative_diagnostics(required_kernels, diagnostics)
    profile_environment = environment.copy()
    profile_environment["AI_FACTORY_PERFORMANCE_PROFILE_PROBE"] = "1"
    for index, diagnostic in enumerate(representatives):
        logical_name = diagnostic["kernel"]
        compiled_symbol = diagnostic["compiled_symbol"]
        report_stem = output_dir / f"{case_id}.phase-{index:02d}"
        profile_result = subprocess.run(
            [
                ncu,
                "--force-overwrite",
                "--replay-mode", "kernel",
                "--kernel-name-base", "mangled",
                "--kernel-name", compiled_symbol,
                "--launch-count", "1",
                "--section", "LaunchStats",
                "--section", "Occupancy",
                "--section", "SpeedOfLight",
                "--section", "MemoryWorkloadAnalysis",
                "--section", "MemoryWorkloadAnalysis_Tables",
                "--section", "ComputeWorkloadAnalysis",
                "--section", "SchedulerStats",
                "--section", "WarpStateStats",
                "--section", "SourceCounters",
                "--export", str(report_stem),
                *command,
            ],
            cwd=ROOT,
            env=profile_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        log_prefix = output_dir / f"{case_id}.phase-{index:02d}"
        Path(f"{log_prefix}.ncu.stdout.log").write_text(profile_result.stdout)
        Path(f"{log_prefix}.ncu.stderr.log").write_text(profile_result.stderr)
        if profile_result.returncode != 0:
            raise RuntimeError(
                f"Nsight Compute failed for {case_id} phase {logical_name}."
            )

        report_path = Path(f"{report_stem}.ncu-rep")
        csv_result = subprocess.run(
            [
                ncu,
                "--import", str(report_path),
                "--csv",
                "--page", "raw",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        csv_path = output_dir / f"{case_id}.phase-{index:02d}.ncu.csv"
        csv_path.write_text(csv_result.stdout)
        if csv_result.returncode != 0:
            raise RuntimeError(
                f"Nsight Compute report export failed for {case_id} phase "
                f"{logical_name}."
            )
        reports.append(
            {
                "kernel": logical_name,
                "variant": diagnostic.get("variant"),
                "compiled_symbol": compiled_symbol,
                "report": report_path.name,
                "metrics_csv": csv_path.name,
                "metrics": parse_ncu_raw_csv(csv_result.stdout),
            }
        )
    return reports


def run_case(case: dict, build_dir: Path, output_dir: Path, profile: bool) -> dict:
    case_id = case["id"]
    binary = build_dir / case["binary"]
    if not binary.is_file():
        raise FileNotFoundError(f"Missing benchmark binary: {binary}")
    arguments = [str(value) for value in case.get("arguments", [])]
    command = [str(binary), *arguments]
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
    prefix = output_dir / case_id
    prefix.with_suffix(".stdout.jsonl").write_text(result.stdout)
    prefix.with_suffix(".stderr.jsonl").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"{case_id} failed with exit code {result.returncode}; "
            f"see {prefix.with_suffix('.stderr.jsonl')}"
        )
    diagnostics = [
        record for record in parse_json_lines(result.stderr)
        if record.get("type") == "cuda_kernel_launch_diagnostics"
    ]
    observed = {record.get("kernel") for record in diagnostics}
    missing = [
        name for name in case["required_kernels"] if name not in observed
    ]
    if missing:
        raise RuntimeError(f"{case_id} omitted required kernel diagnostics: {missing}")
    launched_symbols = {record["compiled_symbol"] for record in diagnostics}

    cuobjdump = shutil.which("cuobjdump")
    sass_path = prefix.with_suffix(".sass.txt")
    local_instructions: dict[str, int] | None = None
    if cuobjdump is not None:
        sass = subprocess.run(
            [cuobjdump, "--dump-sass", str(binary)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        sass_path.write_text(sass.stdout)
        local_instructions = sass_local_instructions(sass.stdout, launched_symbols)

    ncu_reports: list[dict] = []
    if profile:
        ncu = shutil.which("ncu")
        if ncu is None:
            raise RuntimeError("Nsight Compute is unavailable.")
        ncu_reports = profile_representatives(
            ncu,
            case_id,
            command,
            environment,
            output_dir,
            diagnostics,
            case["required_kernels"],
        )

    return {
        "id": case_id,
        "command": command,
        "binary_sha256": sha256(binary),
        "measurement_record_count": len(parse_json_lines(result.stdout)),
        "diagnostic_record_count": len(diagnostics),
        "required_kernels": case["required_kernels"],
        "maximum_registers_per_thread": max(
            record["resources"]["registers_per_thread"] for record in diagnostics
        ),
        "maximum_local_bytes_per_thread": max(
            record["resources"]["local_bytes_per_thread"] for record in diagnostics
        ),
        "sass_local_instruction_counts": local_instructions,
        "ncu_reports": ncu_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Also capture one kernel-replay Nsight Compute report for the "
            "heaviest observed launch of every required phase."
        ),
    )
    arguments = parser.parse_args()
    manifest = load_manifest(arguments.manifest)
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = [
        run_case(case, arguments.build_dir, arguments.output_dir, arguments.profile)
        for case in manifest["cases"]
    ]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "finding": manifest["finding"],
        "manifest": str(arguments.manifest.relative_to(ROOT)),
        "manifest_sha256": sha256(arguments.manifest),
        "git_head": head,
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "profiled_with_ncu": arguments.profile,
        "cases": summaries,
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
