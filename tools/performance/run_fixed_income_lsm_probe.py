#!/usr/bin/env python3
"""Exploratory, sequential LSM probes with per-process watchdog and raw evidence.

This is not a protocol-v3 acceptance/rebaseline campaign. It writes only its
new results directory and never invokes catalogue generators or changes tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

try:
    from .experiment_environment import power_comparability_issue, hardware_power_brake_issue
except ImportError:
    from experiment_environment import power_comparability_issue, hardware_power_brake_issue

ROOT = Path(__file__).resolve().parents[2]


def cases(suite: str) -> list[dict]:
    jobs = []
    def add(label: str, **options):
        jobs.append({"id": label, "options": options})
    if suite == "geometry":
        for rows in (1, 16):
            for threads in (64, 128, 256, 512):
                for blocks in (16, 64, 256, 1024):
                    add(f"cir_r{rows}_t{threads}_b{blocks}", model="cir", rows=rows,
                        paths=1048576, threads=threads, blocks=blocks,
                        first=21, interval=21, payments=10, exercises=10)
        random.Random(20260905).shuffle(jobs)
    elif suite == "scaling":
        for model in ("cir", "ornstein_uhlenbeck", "g2"):
            for paths in (16384, 65536, 262144, 1048576):
                for blocks in (64, 1024):
                    add(f"{model}_n{paths}_b{blocks}", model=model, rows=4,
                        paths=paths, threads=128, blocks=blocks)
    elif suite == "calendar":
        for model in ("cir", "ornstein_uhlenbeck"):
            for exercises, interval in ((2, 210), (3, 105), (6, 42), (11, 21)):
                add(f"{model}_dates{exercises}", model=model, paths=1048576,
                    blocks=256, first=42, interval=interval, exercises=exercises, payments=200)
            for payments in (10, 40, 100, 200):
                add(f"{model}_coupons{payments}", model=model, paths=1048576,
                    blocks=256, first=21, interval=21, exercises=8, payments=payments)
            for first in (126, 1260, 7560):
                add(f"{model}_first{first}", model=model, paths=1048576,
                    blocks=256, first=first, interval=126, exercises=10, payments=10)
    elif suite == "catalog":
        for offset in range(0, 1000, 25):
            add(f"cir_catalog_{offset:04d}", model="cir", source="catalog",
                offset=offset, rows=25, paths=4096, blocks=64, warmups=0, repetitions=1)
    elif suite == "batch":
        for rows in (1, 4, 16, 64, 128):
            for blocks in (64, 256, 1024):
                add(f"cir_rows{rows}_b{blocks}", model="cir", rows=rows,
                    paths=1048576, blocks=blocks, first=21, interval=21,
                    exercises=4, payments=4, memory_limit_mib=8192)
    elif suite == "other-models":
        for model in ("ornstein_uhlenbeck", "vasicek", "g2", "hull_white", "g2_plus_plus"):
            curves = ("nelson_siegel", "svensson") if model in ("hull_white", "g2_plus_plus") else ("nelson_siegel",)
            for curve in curves:
                for payments in (10, 200):
                    add(f"{model}_{curve}_p{payments}", model=model, curve=curve,
                        rows=16, paths=1048576, first=21, interval=21,
                        exercises=10, payments=payments, blocks=64)
        for model in ("black_scholes", "heston", "bates", "variance_gamma"):
            for interval in (21, 1):
                add(f"{model}_interval{interval}", model=model, rows=16,
                    paths=1048576, maturity=252, interval=interval, blocks=64,
                    memory_limit_mib=4096, chunk_rows=1 if interval == 1 else 16)
    else:
        raise ValueError(suite)
    return jobs


def bounded(command: list[str], stdout: Path, stderr: Path, timeout: float, monitor=None) -> dict:
    start = time.monotonic()
    with stdout.open("w") as out, stderr.open("w") as err:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                start_new_session=True)
        timed_out = False
        stop_reason = None
        rc = None
        while rc is None:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                timed_out = True
                break
            try:
                rc = proc.wait(timeout=min(5, remaining) if monitor else remaining)
            except subprocess.TimeoutExpired:
                if monitor:
                    try:
                        stop_reason = monitor(time.monotonic() - start)
                    except Exception as exc:
                        stop_reason = f"Probe monitoring failed: {exc}"
                    if stop_reason:
                        break
        if rc is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                rc = proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait(timeout=3)
    return {"command": command, "returncode": rc, "timed_out": timed_out,
            "stop_reason": stop_reason,
            "process_wall_seconds": time.monotonic() - start}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("geometry", "scaling", "calendar", "catalog", "batch", "other-models"))
    parser.add_argument("--jobs", type=Path, help="Explicit job list instead of a suite")
    parser.add_argument("--binary", type=Path, default=ROOT / "build-dev/ai_factory_fixed_income_lsm_probe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--profile", action="store_true", help="Nsight Systems phase trace; not a timing campaign")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--long-cir-catalog", action="store_true",
                        help="Explicit long watchdog with live telemetry, only for all 1000 CIR rows")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Resume an explicit job list into a new evidence directory")
    args = parser.parse_args()
    if args.diagnostics:
        os.environ["AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS"] = "1"
    if not 0 < args.timeout <= (3600 if args.long_cir_catalog else 60):
        parser.error("Timeout exceeds the selected probe safety mode")
    jobs = json.loads(args.jobs.read_text()) if args.jobs else cases(args.suite)
    if not 0 <= args.start_index < len(jobs):
        parser.error("Start index is outside the job list")
    jobs = jobs[args.start_index:]
    if args.long_cir_catalog:
        for job in jobs:
            opts = job["options"]
            if (opts.get("model") != "cir" or opts.get("source") != "catalog"
                    or opts.get("rows") != 1000 or opts.get("offset", 0) != 0
                    or opts.get("warmups") != 0 or opts.get("repetitions") != 1):
                parser.error("Long mode requires the full CIR catalog and exactly one measured launch")
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    provenance = {"purpose": "exploratory bottleneck diagnosis, not baseline acceptance",
        "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "jobs": jobs, "timeout_seconds": args.timeout,
        "nsys_profile": args.profile, "diagnostics": args.diagnostics,
        "start_index": args.start_index,
        "long_cir_catalog": args.long_cir_catalog,
        "power_limit_policy": "record_and_disqualify_timing_only",
        "power_envelope_maximum_relative_change": 0.20,
        "thermal_policy": "telemetry_only"}
    (dest / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (dest / "git-status.txt").write_text(subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True))
    (dest / "git-diff.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    for name in ("tests/performance/fixed_income_lsm_probe.cu",
                 "tools/performance/run_fixed_income_lsm_probe.py",
                 "tools/performance/experiment_environment.py",
                 "tools/performance/run_baseline.py"):
        (dest / Path(name).name).write_bytes((ROOT / name).read_bytes())
    hashes = {}
    patterns = ("datasets/model/fixed_income/*/parameters/*_01.json",
                "datasets/model/equity/markovian/*/parameters/*_01.json",
                "datasets/curve/*/*_01.json",
                "datasets/product/bermudan_swaption/*_01.json",
                "datasets/product/american_option/*_01.json")
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (dest / "inputs.sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    bounded(["nvidia-smi", "-q", "-x"], dest / "gpu-before.xml", dest / "gpu-before.err", 10)
    reference_power = float(ET.parse(dest / "gpu-before.xml").getroot().findtext(
        "gpu/gpu_power_readings/current_power_limit").split()[0])
    failed = False
    with (dest / "results.ndjson").open("w") as journal:
        for index, job in enumerate(jobs):
            stem = f"{index:03d}_{job['id']}"
            if hashlib.sha256(binary.read_bytes()).hexdigest() != provenance["binary_sha256"]:
                raise RuntimeError("Probe binary changed during the campaign")
            bounded(["nvidia-smi", "-q", "-x"], dest / (stem + ".gpu.xml"),
                    dest / (stem + ".gpu.err"), 10)
            gpu = ET.parse(dest / (stem + ".gpu.xml")).getroot().find("gpu")
            temperature = int(gpu.findtext("temperature/gpu_temp").split()[0])
            power_start = float(gpu.findtext("gpu_power_readings/current_power_limit").split()[0])
            power_issues = set()
            start_issue = power_comparability_issue(power_start, reference_power,
                                                     relative_tolerance=.20)
            if start_issue:
                power_issues.add(start_issue)
            opts = {"warmups": 1, "repetitions": 3, **job["options"]}
            if args.profile:
                opts.update(warmups=0 if args.long_cir_catalog else 1, repetitions=1)
            command = [str(binary)]
            for key, value in opts.items():
                command += ["--" + key.replace("_", "-"), str(value)]
            if args.profile:
                command = ["nsys", "profile", "--trace=cuda", "--sample=none",
                    "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop", "--output=" + str(dest / stem), *command]
            power_excursion_count = 0
            def monitor(elapsed):
                nonlocal power_excursion_count
                snapshot = subprocess.run(["nvidia-smi", "-q", "-x"],
                    capture_output=True, text=True, timeout=5)
                if snapshot.returncode:
                    return "GPU telemetry failed"
                g = ET.fromstring(snapshot.stdout).find("gpu")
                def number(field):
                    return float(g.findtext(field).split()[0])
                sample = {"elapsed_seconds": round(elapsed, 1),
                    "temperature_c": number("temperature/gpu_temp"),
                    "power_limit_w": number("gpu_power_readings/current_power_limit"),
                    "power_draw_w": number("gpu_power_readings/instant_power_draw"),
                    "sm_clock_mhz": number("clocks/sm_clock"),
                    "memory_clock_mhz": number("clocks/mem_clock"),
                    "memory_used_mib": number("fb_memory_usage/used"),
                    "gpu_util_percent": number("utilization/gpu_util"),
                    "pstate": g.findtext("performance_state"),
                    "clock_event_reasons": {node.tag: node.text for node in g.find("clocks_event_reasons")}}
                issue = power_comparability_issue(sample["power_limit_w"], reference_power,
                                                   relative_tolerance=.20)
                sample["timing_comparability_issue"] = issue
                if issue:
                    power_excursion_count += 1
                    power_issues.add(issue)
                with (dest / (stem + ".telemetry.ndjson")).open("a") as stream:
                    stream.write(json.dumps(sample) + "\n")
                print(json.dumps({"id": job["id"], "running": True,
                    **{k: v for k, v in sample.items() if k != "clock_event_reasons"}}), flush=True)
                return hardware_power_brake_issue(sample["clock_event_reasons"])

            outcome = bounded(command, dest / (stem + ".ndjson"),
                              dest / (stem + ".stderr"), args.timeout,
                              monitor=monitor)
            after_path = dest / (stem + ".after.gpu.xml")
            bounded(["nvidia-smi", "-q", "-x"], after_path, after_path.with_suffix(".err"), 10)
            power_end = float(ET.parse(after_path).getroot().findtext(
                "gpu/gpu_power_readings/current_power_limit").split()[0])
            end_issue = power_comparability_issue(power_end, reference_power,
                                                   relative_tolerance=.20)
            if end_issue:
                power_issues.add(end_issue)
            power_changed = bool(power_issues)
            # Nsight's progress bar can precede a flushed application JSON on
            # the same line. Preserve raw output, decode only the event object.
            rows = []
            for line in (dest / (stem + ".ndjson")).read_text().splitlines():
                start = line.find('{"event":')
                if start >= 0:
                    rows.append(json.JSONDecoder().raw_decode(line[start:])[0])
            outcome.update({"id": job["id"], "raw": stem + ".ndjson"})
            outcome.update({"start_temperature_c": temperature,
                            "power_limit_start_w": power_start, "power_limit_end_w": power_end,
                            "power_envelope_changed": power_changed,
                            "power_excursion_samples": power_excursion_count,
                            "timing_eligible": False,
                            "timing_ineligibility_reasons": ["Exploratory LSM probe", *sorted(power_issues)]})
            summary = next((r for r in rows if r.get("event") == "summary"), None)
            if summary:
                outcome["summary"] = summary
            journal.write(json.dumps(outcome) + "\n")
            journal.flush()
            message = {"index": index, "id": job["id"], "rc": outcome["returncode"],
                       "timeout": outcome["timed_out"], "power_envelope_changed": power_changed}
            if summary:
                message.update({"gpu_ms": round(summary["gpu"]["median_ms"], 3),
                                "cv": round(summary["gpu"]["coefficient_of_variation"], 3)})
            print(json.dumps(message), flush=True)
            if outcome["returncode"] or not summary:
                failed = True
                break
    bounded(["nvidia-smi", "-q", "-x"], dest / "gpu-after.xml", dest / "gpu-after.err", 10)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
