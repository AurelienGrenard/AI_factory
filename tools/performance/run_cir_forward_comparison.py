#!/usr/bin/env python3
"""Run explicit CIR method-comparison jobs sequentially, preserving raw evidence.

Exploratory only: no baseline or production publication. Temperature is telemetry,
not an execution veto. Each process has a watchdog and no automatic retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess

from run_baseline import collect_preflight
from run_fixed_income_lsm_probe import bounded

ROOT = Path(__file__).resolve().parents[2]
INPUTS = (
    ROOT / "datasets/model/fixed_income/cir/parameters/cir_01.json",
    ROOT / "datasets/product/bermudan_swaption/bermudan_swaptions_01.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_closure(paths):
    """Include untracked runtime headers as well as the benchmark entry points."""
    found = set()
    pending = list(paths)
    while pending:
        path = pending.pop().resolve()
        if path in found or not path.is_file():
            continue
        found.add(path)
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), re.M):
            for candidate in (path.parent / include, ROOT / include, ROOT / 'src' / include):
                if candidate.is_file():
                    pending.append(candidate)
                    break
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/ai_factory_cir_forward_measure_probe")
    args = parser.parse_args()
    if not 0 < args.timeout <= 3600:
        parser.error("Watchdog must be between zero and 3600 seconds per job")
    jobs = json.loads(args.jobs.read_text())
    if not isinstance(jobs, list) or not jobs or len({j['id'] for j in jobs}) != len(jobs):
        parser.error("Jobs must be nonempty and have unique IDs")
    for job in jobs:
        if job.get("indices") == "all":
            job["indices"] = list(range(1000))
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    binary_hash = digest(binary)
    sources = source_closure((ROOT / 'tests/performance/cir_forward_measure').glob('*.cu'))
    manifest = {
        "scope": "CIR numerical-method exploration, not protocol-v3 tuning acceptance",
        "jobs": jobs, "binary": str(binary), "binary_sha256": binary_hash,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in INPUTS},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "watchdog_seconds_per_job": args.timeout, "thermal_policy": "telemetry_only",
        "row_comparison": {"sigma_multiplier": 5.0, "rounding_absolute_floor": 2e-6,
            "meaning": "screen for discrepancies; not a financial certification or a bound on LSM bias"},
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dest / "git-status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    (dest / "git-diff.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    sources += [Path(__file__).resolve()]
    for source in sources:
        if source.is_file():
            snapshot = dest / 'sources' / source.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(source.read_bytes())
    with (dest / "journal.ndjson").open("w") as journal:
        for index, job in enumerate(jobs):
            if digest(binary) != binary_hash or any(digest(ROOT / p) != h for p, h in manifest['input_sha256'].items()):
                raise RuntimeError("Binary or inputs changed")
            before = collect_preflight()
            (dest / f"{index:03d}.before.json").write_text(json.dumps(before, indent=2))
            if before.get("concurrent_compute_processes"):
                raise RuntimeError("Another GPU compute process is running")
            if before.get("power_source") != "external_power":
                raise RuntimeError(f"External power required, observed {before.get('power_source')}")
            if before.get("throttle", {}).get("hardware_power_brake") == "Active":
                raise RuntimeError("GPU hardware power brake")
            stem = f"{index:03d}"
            job_path = dest / f"{stem}.jobs.json"
            job_path.write_text(json.dumps([job], indent=2))
            print(f"START {index + 1}/{len(jobs)} {job['id']}", flush=True)
            last_print = 0.0
            def monitor(elapsed):
                nonlocal last_print
                snapshot = collect_preflight()
                with (dest / f"{stem}.telemetry.ndjson").open("a") as telemetry:
                    telemetry.write(json.dumps({"elapsed_seconds": elapsed, **snapshot}) + "\n")
                if elapsed - last_print >= 30:
                    print(f"RUNNING {job['id']} {elapsed:.0f}s", flush=True)
                    last_print = elapsed
                if snapshot.get("throttle", {}).get("hardware_power_brake") == "Active":
                    return "GPU hardware power brake"
                return None
            outcome = bounded([str(binary), "--jobs", str(job_path)], dest / f"{stem}.ndjson",
                dest / f"{stem}.stderr", args.timeout, monitor=monitor)
            outcome.update(id=job['id'], index=index)
            journal.write(json.dumps(outcome) + "\n")
            journal.flush()
            (dest / f"{stem}.after.json").write_text(json.dumps(collect_preflight(), indent=2))
            print(f"END {job['id']} rc={outcome['returncode']} {outcome['process_wall_seconds']:.2f}s", flush=True)
            if outcome['returncode'] != 0 or outcome['timed_out'] or outcome['stop_reason']:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
