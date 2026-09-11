#!/usr/bin/env python3
"""Summarize preserved LSM probe journals, replay checks and Nsight phase CSVs.

This does not run GPU work or certify independent numerical accuracy. Geometry
and price-chunk comparisons require identical inputs and binary fingerprints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(path: Path) -> list[dict]:
    result = []
    for line in path.read_text().splitlines():
        start = line.find('{"event":')
        if start >= 0:
            result.append(json.JSONDecoder().raw_decode(line[start:])[0])
    return result


def output_bytes(row: dict) -> bytes:
    values = row["prices"] + row["errors"]
    return struct.pack(f"<{len(values)}f", *values)


def summarize(directories: list[Path]) -> dict:
    runs, profiles, provenance, failures, comparisons = [], [], {}, [], {}
    all_finite, replay_failures, repetitions = True, 0, 0
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        input_hash = digest(directory / "inputs.sha256.json")
        provenance[str(directory)] = {
            "revision": manifest["revision"],
            "binary_sha256": manifest["binary_sha256"],
            "manifest_sha256": digest(directory / "manifest.json"),
            "inputs_manifest_sha256": input_hash,
            "requested_jobs": len(manifest["jobs"]),
            "profiled": manifest["nsys_profile"],
        }
        outcomes = [json.loads(line) for line in (directory / "results.ndjson").read_text().splitlines()]
        provenance[str(directory)]["completed_jobs"] = sum("summary" in r for r in outcomes)
        executed = {r["id"] for r in outcomes}
        provenance[str(directory)]["unexecuted_job_ids"] = [
            job["id"] for job in manifest["jobs"] if job["id"] not in executed]
        for outcome in outcomes:
            if outcome["returncode"] or "summary" not in outcome:
                failures.append({"directory": str(directory), "id": outcome["id"],
                                 "returncode": outcome["returncode"], "timed_out": outcome["timed_out"]})
                continue
            raw = directory / outcome["raw"]
            rows = [r for r in events(raw) if r["event"] == "measurement"]
            config = outcome["summary"]["config"]
            first = rows[0]
            replay_failures += sum(output_bytes(r) != output_bytes(first) for r in rows[1:])
            all_finite &= all(r["finite"] for r in rows)
            repetitions += len(rows)
            # Exclude geometry and host chunk size, but never different calendars,
            # source slices, path counts, product sides, curves or binaries.
            semantic = {k: v for k, v in config.items() if k in (
                "model", "source", "product", "curve", "offset", "rows", "paths",
                "days_per_year", "steps_per_year", "calendars_first_interval_payments_exercises",
                "calendars_maturity_interval_exercises")}
            key = (manifest["binary_sha256"], input_hash, json.dumps(semantic, sort_keys=True))
            comparisons.setdefault(key, []).append((str(raw), first))
            stderr = raw.with_suffix(".stderr")
            diagnostics = []
            for line in stderr.read_text().splitlines():
                if line.startswith('{"type":"cuda_kernel_launch_diagnostics"'):
                    d = json.loads(line)
                    diagnostics.append({"phase": d["kernel"].rsplit(".", 1)[-1],
                        "threads": d["launch"]["threads_per_block"],
                        "registers": d["resources"]["registers_per_thread"],
                        "local_bytes_per_thread": d["resources"]["local_bytes_per_thread"],
                        "shared_bytes": d["resources"]["static_shared_bytes_per_block"]
                            + d["launch"]["dynamic_shared_bytes_per_block"],
                        "theoretical_occupancy": d["occupancy"]["theoretical"]})
            thermal = raw.with_suffix(".gpu.xml")
            if outcome.get("cooldown_samples", 0):
                thermal = raw.with_suffix(f".cooldown{outcome['cooldown_samples']}.gpu.xml")
            gpu = ET.parse(thermal).getroot().find("gpu")
            before_c = int(gpu.findtext("temperature/gpu_temp").split()[0])
            before_w = float(gpu.findtext("gpu_power_readings/current_power_limit").split()[0])
            runs.append({"suite": directory.name, "id": outcome["id"],
                "profiled": manifest["nsys_profile"],
                "config": {k: v for k, v in config.items() if not k.startswith("calendars_")},
                "gpu": outcome["summary"]["gpu"], "raw_host": outcome["summary"]["raw_host"],
                "workspace_bytes": first["workspace_bytes"], "batches": first["batch_count"],
                "kernel_launch_count": first["kernel_launch_count"],
                "no_candidates": first["no_candidates"],
                "insufficient_candidates": first["insufficient_candidates"],
                "output_fp32_sha256": hashlib.sha256(output_bytes(first)).hexdigest(),
                "start_temperature_c": outcome.get("start_temperature_c", before_c),
                "cooldown_samples": outcome.get("cooldown_samples", 0),
                "power_limit_start_w": outcome.get("power_limit_start_w", before_w),
                "power_limit_end_w": outcome.get("power_limit_end_w"),
                "power_envelope_changed": outcome.get("power_envelope_changed"),
                "timing_eligible": False,
                "timing_ineligibility_reasons": outcome.get("timing_ineligibility_reasons", []),
                "raw": str(raw), "raw_sha256": digest(raw), "diagnostics": diagnostics})
        for path in sorted(directory.glob("*_cuda_gpu_kern_sum_base.csv")):
            with path.open() as stream:
                values = list(csv.DictReader(stream))
            total = sum(int(r["Total Time (ns)"]) for r in values)
            profiles.append({"raw": str(path), "raw_sha256": digest(path),
                "total_kernel_ms": total / 1e6,
                "phases": {r["Name"]: {"ms": int(r["Total Time (ns)"]) / 1e6,
                    "percent": 100 * int(r["Total Time (ns)"]) / total,
                    "instances": int(r["Instances"])} for r in values}})
    compared = []
    for _, group in comparisons.items():
        if len(group) < 2:
            continue
        reference = group[0][1]
        compared.append({"reference": group[0][0], "runs": len(group),
            "bitwise_identical_prices_and_errors": all(output_bytes(r) == output_bytes(reference) for _, r in group),
            "max_absolute_price_difference": max(abs(a - b) for _, r in group
                for a, b in zip(reference["prices"], r["prices"], strict=True)),
            "max_absolute_error_difference": max(abs(a - b) for _, r in group
                for a, b in zip(reference["errors"], r["errors"], strict=True))})
    return {"scope": "Exploratory cross-model LSM diagnosis, not protocol-v3 acceptance",
        "dirty_worktree": True, "processes": len(runs), "measured_repetitions": repetitions,
        "unprofiled_processes": sum(not r["profiled"] for r in runs),
        "unprofiled_repetitions": sum(r["config"]["repetitions"] for r in runs if not r["profiled"]),
        "all_finite": all_finite, "replay_failures": replay_failures, "failed_processes": failures,
        "provenance": provenance, "comparison_groups": compared, "profiles": profiles, "runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.directories)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("processes", "measured_repetitions", "all_finite",
        "replay_failures", "failed_processes", "comparison_groups")}, indent=2))
    return int(not report["all_finite"] or report["replay_failures"] or bool(report["failed_processes"]))


if __name__ == "__main__":
    raise SystemExit(main())
