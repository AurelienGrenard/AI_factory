#!/usr/bin/env python3
"""Capture one manifest-owned representative kernel with Nsight Compute."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from .run_baseline import (
        collect_preflight,
        stabilize_thermal_environment,
        validate_build_configuration,
        validate_campaign_preflight,
    )
except ImportError:
    from run_baseline import (
        collect_preflight,
        stabilize_thermal_environment,
        validate_build_configuration,
        validate_campaign_preflight,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("performance candidate is empty")
    return rows


def select_profile_target(
    baseline: dict[str, Any],
    candidate: list[dict[str, Any]],
    build_directory: Path,
    measurement_id: str,
    resource_index: int,
) -> dict[str, Any]:
    """Resolve a profile target only from one declared measurement."""
    matches = [
        row for row in candidate
        if row.get("measurement_id", row.get("id")) == measurement_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one candidate row for {measurement_id}, got {len(matches)}"
        )
    row = matches[0]
    resources = row.get("resources")
    if (
        not isinstance(resources, list)
        or not 0 <= resource_index < len(resources)
    ):
        raise ValueError("profile resource index is outside the candidate row")
    resource = resources[resource_index]
    symbol = resource.get("compiled_symbol")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("profile target has no exact compiled symbol")

    command_id = row.get("command_id")
    specifications = [
        command for command in baseline.get("commands", [])
        if command.get("id") == command_id
    ]
    if len(specifications) != 1:
        raise ValueError(f"expected one manifest command for {command_id}")
    specification = specifications[0]
    executable_name = specification.get("executable")
    arguments = specification.get("arguments")
    if (
        not isinstance(executable_name, str)
        or Path(executable_name).name != executable_name
        or not isinstance(arguments, list)
        or not all(isinstance(argument, str) for argument in arguments)
    ):
        raise ValueError("invalid profile command specification")
    executable = build_directory / executable_name
    if not executable.is_file():
        raise ValueError(f"profile executable does not exist: {executable}")
    recorded_hash = row.get("binary", {}).get("executable_sha256")
    executable_hash = _sha256(executable)
    if recorded_hash != executable_hash:
        raise ValueError("profile executable differs from the measured candidate")

    scopes = [
        scope for scope, command_ids in baseline.get("audit_reports", {}).items()
        if command_id in command_ids
    ]
    if len(scopes) != 1:
        raise ValueError(f"profile command {command_id} has no unique audit scope")
    return {
        "measurement_id": measurement_id,
        "scope": scopes[0],
        "command_id": command_id,
        "command": [str(executable), *arguments],
        "executable_sha256": executable_hash,
        "resource_index": resource_index,
        "kernel": resource.get("kernel"),
        "variant": resource.get("variant"),
        "compiled_symbol": symbol,
        "launch": resource.get("launch"),
        "compiled_resources": resource.get("compiled_resources"),
    }


def ncu_command(
    ncu: str,
    target: dict[str, Any],
    report_stem: Path,
    section_set: str,
) -> list[str]:
    """Build the deterministic single-launch profiler command."""
    return [
        ncu,
        "--target-processes", "application-only",
        "--kernel-name-base", "mangled",
        "--kernel-name", target["compiled_symbol"],
        "--launch-count", "1",
        "--kill", "1",
        "--set", section_set,
        "--apply-rules", "on",
        "--export", str(report_stem),
        "--force-overwrite",
        *target["command"],
    ]


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--measurement-id", required=True)
    parser.add_argument("--resource-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--set", dest="section_set", default="detailed")
    arguments = parser.parse_args()

    try:
        ncu = shutil.which("ncu")
        if ncu is None:
            raise ValueError("ncu is unavailable")
        baseline_payload = arguments.baseline.read_bytes()
        candidate_payload = arguments.candidate.read_bytes()
        baseline = json.loads(baseline_payload)
        candidate = _candidate_rows(arguments.candidate)
        validate_build_configuration(baseline, arguments.build_dir)
        target = select_profile_target(
            baseline,
            candidate,
            arguments.build_dir,
            arguments.measurement_id,
            arguments.resource_index,
        )
        stabilization_evidence: list[dict[str, Any]] = []
        before = stabilize_thermal_environment(
            baseline, arguments.build_dir, stabilization_evidence
        )
        validate_campaign_preflight(baseline, before, before)
        version = subprocess.run(
            [ncu, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip()

        with tempfile.TemporaryDirectory(prefix="ai_factory_ncu_") as temporary:
            report_stem = Path(temporary) / "profile"
            profile_command = ncu_command(
                ncu, target, report_stem, arguments.section_set
            )
            profiled = subprocess.run(
                profile_command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            report_path = report_stem.with_suffix(".ncu-rep")
            if not report_path.is_file():
                raise ValueError("ncu did not produce a report")
            exported = subprocess.run(
                [
                    ncu,
                    "--import", str(report_path),
                    "--csv", "--page", "raw", "--print-fp",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            csv_payload = exported.stdout
            if not csv_payload.strip():
                raise ValueError("ncu exported an empty raw profile")
            report_sha256 = _sha256(report_path)

        after = collect_preflight()
        validate_campaign_preflight(baseline, before, after)
        profile_name = arguments.measurement_id + ".ncu.csv"
        profile_path = arguments.output_dir / profile_name
        _write_atomic(profile_path, csv_payload)
        metadata = {
            "schema": "ai_factory_nsight_compute_profile",
            "version": 1,
            "generated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "section_set": arguments.section_set,
            "ncu_version": version,
            "baseline": {
                "path": str(arguments.baseline),
                "sha256": hashlib.sha256(baseline_payload).hexdigest(),
            },
            "candidate": {
                "path": str(arguments.candidate),
                "sha256": hashlib.sha256(candidate_payload).hexdigest(),
            },
            "target": target,
            "preflight": {
                "thermal_stabilization": stabilization_evidence,
                "before": before,
                "after": after,
            },
            "invocation": profile_command,
            "profiler_stdout": profiled.stdout,
            "profiler_stderr": profiled.stderr,
            "transient_ncu_report_sha256": report_sha256,
            "profile": {
                "path": profile_name,
                "sha256": hashlib.sha256(csv_payload).hexdigest(),
                "bytes": len(csv_payload),
            },
        }
        metadata_path = arguments.output_dir / (
            arguments.measurement_id + ".profile.json"
        )
        _write_atomic(
            metadata_path,
            (json.dumps(metadata, indent=2) + "\n").encode(),
        )
    except subprocess.CalledProcessError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if error.stdout:
            print(error.stdout.rstrip(), file=sys.stderr)
        if error.stderr:
            print(error.stderr.rstrip(), file=sys.stderr)
        return 2
    except (
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"PROFILED: {arguments.measurement_id} -> {profile_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
