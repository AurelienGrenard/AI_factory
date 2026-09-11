"""Run sequential, resumable pricing-scaling probes without touching published datasets.

Calibration and screening are diagnostic, never protocol-v3 acceptance. Each
process has a watchdog, continuous telemetry and immutable inputs/binary hashes.
No failed job is automatically retried and no baseline is replaced.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET

try:
    from .pricing_scaling_manifest import ROOT, PATH_COUNTS, PRICE_COUNTS
    from .pricing_scaling_inputs import prepare_inputs, INPUT_PROFILE, CATALOGUE_INPUT_PROFILE
    from .run_baseline import collect_preflight, _attach_compiled_resources
    from .experiment_environment import (power_comparability_issue, record_timing_issue,
        hardware_power_brake_issue, validate_experiment_preflight)
except ImportError:
    from pricing_scaling_manifest import ROOT, PATH_COUNTS, PRICE_COUNTS
    from pricing_scaling_inputs import prepare_inputs, INPUT_PROFILE, CATALOGUE_INPUT_PROFILE
    from run_baseline import collect_preflight, _attach_compiled_resources
    from experiment_environment import (power_comparability_issue, record_timing_issue,
        hardware_power_brake_issue, validate_experiment_preflight)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def production_jobs(case: dict, build: Path, price_counts) -> tuple[list[dict], list[dict]]:
    """Ask the compiled recipe planner; never copy its launch table into Python."""
    if case.get("scope") != "catalogue_recipe":
        raise ValueError(f"No production recipe for {case['id']}")
    tool = build / "inspect_pricing_launch_plan"
    pair = "/".join(filter(None, (case["model"], case.get("curve"), case["product"])))
    jobs, evidence = [], []
    for rows in price_counts:
        completed = subprocess.run([str(tool), pair, str(rows)], cwd=ROOT,
                                   check=True, capture_output=True, text=True, timeout=30)
        plan = json.loads(completed.stdout)
        expected_paths = 0 if case["family"] == "closed_form" else 1 << 20
        if plan["price_count"] != rows or plan["paths_per_price"] != expected_paths:
            raise ValueError("Compiled recipe planner returned a different workload")
        jobs.append({
            "id": f"r{rows}_n{expected_paths}_production_plan", "rows": rows, "offset": 0,
            "paths": expected_paths,
            "threads": plan.get("threads_per_block", plan.get("pricing_path_threads")),
            "batch_rows": plan["prices_per_launch"] or rows,
            "block_limit": plan.get("block_count", rows),
            "blocks_per_price": plan.get("blocks_per_price", 1),
            "path_chunk": plan.get("path_chunk_size", 1),
            "warmups": 2, "repetitions": 3,
            "operations_per_sample": 1024 if expected_paths == 0 and rows <= 1000 else 1,
        })
        evidence.append({"pair": pair, "plan": plan, "inspector_sha256": digest(tool)})
    return jobs, evidence


def snapshot(destination: Path) -> dict:
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "-z"], cwd=ROOT)
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only", "-z"], cwd=ROOT)
    names = sorted(set(os.fsdecode(n) for n in (untracked + changed).split(b"\0") if n))
    files = {name: digest(ROOT / name) for name in names if (ROOT / name).is_file()}
    (destination / "source.diff.patch").write_bytes(diff)
    (destination / "git-status.porcelain").write_bytes(status)
    with tarfile.open(destination / "changed-and-untracked.tar.gz", "w:gz") as archive:
        for name in files:
            archive.add(ROOT / name, arcname=name, recursive=False)
    result = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "porcelain_sha256": hashlib.sha256(status).hexdigest(),
        "changed_and_untracked_files": files,
        "archive_sha256": digest(destination / "changed-and-untracked.tar.gz"),
    }
    save_json(destination / "snapshot.json", result)
    return result


def compiled_thread_limit(case: dict, build: Path, evidence: Path) -> dict | None:
    """Use the exact ordinary-MC binary's measured resource ceiling, not a model guess."""
    if case["family"] not in ("mc_terminal", "mc_barrier") or case["mathdx"]:
        return None
    directory = evidence / case["id"]
    paths = {name: directory / (name + ".json")
             for name in ("provenance", "resources", "outcome")}
    documents = {name: json.loads(path.read_text()) for name, path in paths.items()}
    binary = build / ("ai_factory_pricing_scaling_" + case["id"])
    resources = documents["resources"]
    if (documents["outcome"]["status"] != "measured"
        or documents["provenance"]["binary_sha256"] != digest(binary)
        or len({r["compiled_symbol"] for r in resources}) != 1):
        raise ValueError("Thread limits require a complete calibration of this exact MC binary")
    limit = min(r["resources"]["maximum_threads_per_block"] for r in resources)
    if not isinstance(limit, int) or not 32 <= limit <= 1024:
        raise ValueError("Invalid compiled block-size ceiling")
    return {"maximum_threads": limit // 32 * 32,
            "device": resources[0]["device"],
            "source_hashes": {str(path): digest(path) for path in paths.values()}}


def jobs_for(case: dict, stage: str, price_counts=PRICE_COUNTS, path_counts=PATH_COUNTS,
             sm_count: int | None = None, thread_limit: int | None = None) -> list[dict]:
    closed = case["family"] == "closed_form"
    lsm = case["family"] == "lsm"
    threads = 128 if lsm else 256 if closed or case["factor_count"] else 512
    thread_candidates = (128, 256, 512)
    if thread_limit is not None:
        if lsm or closed or case["mathdx"] or not 32 <= thread_limit <= 1024 or thread_limit % 32:
            raise ValueError("A calibrated thread ceiling applies only to ordinary MC")
        threads = min(threads, thread_limit)
        thread_candidates = tuple(sorted({min(t, thread_limit)
                                          for t in (64, 128, 256, 512, thread_limit)}))
    if sm_count is None:
        profile = json.loads((ROOT / "tests/performance/baseline_sm89_v3.json").read_text())
        sm_count = profile["environment"]["sm_count"]
    def job(rows, paths, threads=threads, batch_rows=None, block_limit=4096, path_chunk=65536):
        return {"id": f"r{rows}_n{paths}_t{threads}_b{batch_rows or rows}_g{block_limit}_c{path_chunk}",
                "rows": rows, "paths": paths, "threads": threads,
                "batch_rows": batch_rows or rows, "block_limit": block_limit,
                "path_chunk": path_chunk, "warmups": 1, "repetitions": 3,
                **({"blocks_per_price": 64} if lsm else {})}
    if stage == "calibration":
        return [{**job(min(price_counts), min(path_counts)), "warmups": 1, "repetitions": 1}]
    if stage == "single_price":
        return [job(1, n) for n in ((65536,) if closed else path_counts)]
    if stage == "scaling":
        return [{**job(r, n), "operations_per_sample": 1024 if closed else 1}
                for r in price_counts for n in ((65536,) if closed else path_counts)]
    if stage in ("geometry", "screening"):
        result = []
        sizes = [(r, n) for r in price_counts for n in ((65536,) if closed else path_counts)]
        if lsm:
            result = [{**job(r, n, threads=t, batch_rows=b),
                       "id": f"r{r}_n{n}_t{t}_b{b}_blocks{blocks}", "blocks_per_price": blocks}
                      for r, n in sizes for b in sorted({min(r, 100), min(r, 1000), r})
                      for t, blocks in ((128, 32), (128, 64), (128, 128), (256, 64))]
        elif case["mathdx"]:
            result = [job(r, n, path_chunk=c) for r, n in sizes for c in (16384, 65536)]
        else:
            result = [{**job(r, n, threads=t, block_limit=grid),
                       "operations_per_sample": 1024 if closed else 1}
                      for r, n in sizes for t in thread_candidates
                      for grid in ((4096,) if closed else sorted({r, min(r, 2 * sm_count)}))]
            if not closed:
                result.extend(job(r, n, batch_rows=min(r, 100), block_limit=r)
                              for r, n in sizes if r > 100)
                # Current generated recipes batch at 4096 rows with a 4096-block cap.
                result.extend(job(r, n, batch_rows=4096, block_limit=4096)
                              for r, n in sizes if r > 4096)
        if stage == "screening":
            result = [{**j, "repetitions": 1} for j in result]
        return result
    raise ValueError(f"Unknown stage {stage}")


def telemetry() -> tuple[str, dict]:
    xml = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=10)
    gpu = ET.fromstring(xml).find("gpu")
    def value(field):
        text = gpu.findtext(field)
        if not text or text == "N/A":
            raise ValueError(f"Missing GPU telemetry: {field}")
        return float(text.split()[0])
    return xml, {
        "temperature_c": value("temperature/gpu_temp"),
        "power_limit_w": value("gpu_power_readings/current_power_limit"),
        "memory_clock_mhz": value("clocks/mem_clock"),
        "sm_clock_mhz": value("clocks/sm_clock"),
        "memory_used_mib": value("fb_memory_usage/used"),
        "utilization_percent": value("utilization/gpu_util"),
        "throttle": {node.tag: node.text for node in gpu.find("clocks_event_reasons")},
        "compute_process_ids": [int(node.findtext("pid"))
                                for node in gpu.findall("processes/process_info")
                                if node.findtext("type") in ("C", "C+G")],
    }


def read_job_progress(stream, previous: dict | None) -> dict | None:
    """Read completed start records without parsing large price-result arrays."""
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line:
            return previous
        if not line.endswith("\n"):
            stream.seek(position)
            return previous
        if line.startswith('{"event":"job_start"'):
            row = json.loads(line)
            previous = {key: row[key] for key in ("id", "rows", "paths")}


def run_process(command: list[str], destination: Path, timeout: float,
                reference_power: float, environment_policy: str = "strict",
                minimum_power: float = 140, maximum_power: float | None = None) -> dict:
    started = time.monotonic()
    stop_reason = None
    next_progress_seconds = 30
    active_job = None
    outcome = {"environment_policy": environment_policy, "timing_eligible": True,
               "timing_ineligibility_reasons": []}
    if environment_policy == "observe":
        record_timing_issue(outcome, "Observation-only campaign")
    record_timing_issue(outcome, power_comparability_issue(
        reference_power, reference_power, minimum_power, maximum_power))
    with (destination / "stdout.ndjson").open("w") as out, \
         (destination / "stderr.log").open("w") as err, \
         (destination / "stdout.ndjson").open("r") as progress:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                   env={**os.environ, "AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS": "1"},
                                   start_new_session=True)
        try:
            with (destination / "telemetry.ndjson").open("w") as log:
                while process.poll() is None:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        elapsed = time.monotonic() - started
                        xml, sample = telemetry()
                        sample["elapsed_seconds"] = elapsed
                        active_job = read_job_progress(progress, active_job)
                        sample["latest_flushed_job_start"] = active_job
                        issue = power_comparability_issue(sample["power_limit_w"],
                            reference_power, minimum_power, maximum_power)
                        sample["timing_comparability_issue"] = issue
                        record_timing_issue(outcome, issue)
                        log.write(json.dumps(sample) + "\n")
                        log.flush()
                        stop_reason = hardware_power_brake_issue(sample["throttle"])
                        if set(sample["compute_process_ids"]) - {process.pid}:
                            stop_reason = "Concurrent GPU compute process detected"
                        if elapsed > timeout:
                            stop_reason = "Declared process watchdog expired"
                        if elapsed >= next_progress_seconds or stop_reason:
                            print(json.dumps({"case": destination.name, "job": active_job,
                                              "running_seconds": round(elapsed),
                                              "gpu_util": sample["utilization_percent"],
                                              "sm_clock_mhz": sample["sm_clock_mhz"],
                                              "timing_eligible": outcome["timing_eligible"],
                                              "stop_reason": stop_reason}), flush=True)
                            next_progress_seconds = elapsed + 30
                        if stop_reason:
                            (destination / "gpu-stop.xml").write_text(xml)
                            break
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
    if stop_reason or process.returncode:
        record_timing_issue(outcome, stop_reason or "Probe process failed")
    return {**outcome, "returncode": process.returncode, "stop_reason": stop_reason,
            "process_wall_seconds": time.monotonic() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration", "scaling", "geometry", "screening", "single_price", "production"), default="calibration")
    parser.add_argument("--price-counts", type=int, nargs="+", choices=PRICE_COUNTS, default=PRICE_COUNTS)
    parser.add_argument("--path-counts", type=int, nargs="+", choices=PATH_COUNTS, default=PATH_COUNTS)
    parser.add_argument("--input-profile", choices=("stratified-tile", "catalogue"), default="stratified-tile",
                        help="catalogue retains the complete original 1,000-row datasets; no repeated tile")
    parser.add_argument("--cases", nargs="*", help="Exact manifest case IDs; omitted means every available case")
    parser.add_argument("--families", nargs="+", choices=("mc_terminal", "mc_barrier", "closed_form", "lsm"),
                        help="Family filter; default excludes LSM, which requires an explicit separate campaign")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--environment-policy", choices=("strict", "observe"), default="strict",
                        help="strict assesses timing eligibility; observe never qualifies tuning; neither mode stops for power-limit or temperature variation")
    parser.add_argument("--jobs", type=Path, help="Predeclared per-case jobs; overrides stage generation")
    parser.add_argument("--resource-evidence", type=Path,
                        help="Complete calibration of the same binaries; bounds ordinary-MC thread candidates")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600:
        parser.error("Process timeout must be in [1, 3600] seconds")
    build = args.build_dir.resolve()
    manifest = json.loads((build / "pricing-scaling/manifest.json").read_text())
    families = args.families or (["mc_terminal", "mc_barrier", "closed_form"] if args.cases is None
                                else ["mc_terminal", "mc_barrier", "closed_form", "lsm"])
    selected = [c for c in manifest["cases"] if c["family"] in families
                and (args.cases is None or c["id"] in args.cases)]
    if not selected or (args.cases is not None and set(args.cases) != {c["id"] for c in selected}):
        parser.error("Unknown or empty selection of scaling cases")
    if args.cases is not None:
        if len(args.cases) != len(set(args.cases)):
            parser.error("Duplicate scaling case IDs")
        by_id = {c["id"]: c for c in selected}
        selected = [by_id[name] for name in args.cases]
    explicit = json.loads(args.jobs.read_text()) if args.jobs else None
    recipe_plans = {}
    if args.stage == "production":
        if args.jobs or args.path_counts != [1 << 20]:
            parser.error("Production stage requires --path-counts 1048576 and no --jobs override")
        explicit = {}
        for case in selected:
            explicit[case["id"]], recipe_plans[case["id"]] = production_jobs(case, build, args.price_counts)
    limits = {c["id"]: compiled_thread_limit(c, build, args.resource_evidence.resolve())
              if args.resource_evidence else None for c in selected}
    plans = {c["id"]: explicit[c["id"]] if explicit else jobs_for(
        c, args.stage, args.price_counts, args.path_counts,
        thread_limit=limits[c["id"]]["maximum_threads"] if limits[c["id"]] else None)
        for c in selected}
    input_profile = CATALOGUE_INPUT_PROFILE if args.input_profile == "catalogue" else INPUT_PROFILE
    if input_profile == CATALOGUE_INPUT_PROFILE and any(
        j["rows"] != 1000 or j.get("offset", 0) != 0 for jobs in plans.values() for j in jobs
    ):
        parser.error("The catalogue profile requires complete, offset-zero 1,000-price jobs")
    if any(limits[c["id"]] and any(j["threads"] > limits[c["id"]]["maximum_threads"]
                                   for j in plans[c["id"]]) for c in selected):
        parser.error("A declared job exceeds its calibrated compiled thread ceiling")
    if manifest.get("schema") != "ai_factory_pricing_scaling_v2":
        parser.error("Reconfigure/rebuild the scaling probes for the 10,000-price input contract")
    destination = args.output.resolve()
    # Refuse tracked/publication trees even if a caller supplies an unfortunate path.
    allowed = destination.is_relative_to(build) or destination.is_relative_to(Path("/tmp"))
    if not allowed or destination == build:
        parser.error("Evidence must use a new subdirectory of the build directory or /tmp")
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "plan.json", {"manifest": manifest, "selected_jobs": plans,
        "stage": args.stage, "timeout_seconds": args.timeout,
        "environment_policy": args.environment_policy,
        "power_limit_policy": "record_and_disqualify_timing_only",
        "input_profile": input_profile,
        "compiled_thread_limits": limits,
        "production_launch_plans": recipe_plans,
        "explicit_jobs_source": {"path": str(args.jobs.resolve()), "sha256": digest(args.jobs)}
                                if args.jobs else None,
        "price_counts": list(args.price_counts), "path_counts": list(args.path_counts),
        "selection_rule": "per-shape screening then independent repeated confirmation; no global geometry or best-of-repetitions selection",
        "timing_claim": "diagnostic scaling, not protocol-v3 rebaseline",
        "price_tolerance": {"absolute": 1e-6, "relative": 1e-6},
        "standard_error_tolerance": {"absolute": 1e-8, "relative": 1e-5},
        "scaling_alert_normalized_cost_ratio": 1.20,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    if args.plan_only:
        print(f"Planned {len(plans)} cases; no GPU execution.")
        return 0
    snapshot(destination)
    profile = json.loads((ROOT / "tests/performance/baseline_sm89_v3.json").read_text())
    failed = False
    with (destination / "journal.ndjson").open("w") as journal:
        for case in selected:
            case_dir = destination / case["id"]
            case_dir.mkdir()
            jobs = plans[case["id"]]
            binary = build / ("ai_factory_pricing_scaling_" + case["id"])
            if not binary.is_file():
                journal.write(json.dumps({"case": case["id"], "status": "unavailable", "reason": "binary not built"}) + "\n")
                journal.flush()
                failed = True
                continue
            sources = [case["model_dataset"], case["product_dataset"]]
            if case["curve"]:
                sources.append(f'datasets/curve/{case["curve"]}/{case["curve"]}_01.json')
            if any(not (ROOT / source).is_file() for source in sources):
                journal.write(json.dumps({"case": case["id"], "status": "unavailable", "reason": "catalogue inputs missing"}) + "\n")
                journal.flush()
                failed = True
                continue
            provenance = {"binary": str(binary), "binary_sha256": digest(binary),
                          "inputs": {s: digest(ROOT / s) for s in sources}}
            roles = {"model": ROOT / case["model_dataset"], "product": ROOT / case["product_dataset"]}
            if case["curve"]:
                roles["curve"] = ROOT / sources[-1]
            prepared = prepare_inputs(roles, case_dir / "inputs",
                                      [j["rows"] + j.get("offset", 0) for j in jobs], input_profile)
            maximum_rows = max(j["rows"] + j.get("offset", 0) for j in jobs)
            save_json(case_dir / "inputs.json", prepared["by_count"][str(maximum_rows)])
            provenance["fixture"] = prepared
            for job in jobs:
                if job["rows"] >= 100 and job.get("offset", 0) == 0 and args.stage != "calibration":
                    publication = case_dir / "publication"
                    publication.mkdir(exist_ok=True)
                    job["publication_directory"] = str(publication)
                    job.update({"publication_" + role: path
                                for role, path in prepared["by_count"][str(job["rows"])].items()})
            save_json(case_dir / "jobs.json", jobs)
            save_json(case_dir / "provenance.json", provenance)
            outcome = {}
            try:
                before = collect_preflight()
                save_json(case_dir / "before.json", before)
                validate_experiment_preflight(profile, before)
                if limits[case["id"]] and before["gpu"] != limits[case["id"]]["device"]["name"]:
                    raise ValueError("Resource calibration belongs to a different GPU model")
                reference_power = before["power_limits_w"]["current"]
                command = [str(binary), "--jobs", str(case_dir / "jobs.json"),
                           "--inputs", str(case_dir / "inputs.json")]
                outcome = run_process(command, case_dir, args.timeout, reference_power,
                                      args.environment_policy,
                                      profile["decision_policy"]["preflight"]["minimum_current_power_limit_w"],
                                      before["power_limits_w"]["maximum"])
                after = collect_preflight()
                save_json(case_dir / "after.json", after)
                validate_experiment_preflight(profile, after)
                record_timing_issue(outcome, power_comparability_issue(
                    after["power_limits_w"]["current"], reference_power,
                    profile["decision_policy"]["preflight"]["minimum_current_power_limit_w"],
                    before["power_limits_w"]["maximum"]))
                if digest(binary) != provenance["binary_sha256"] or any(
                    digest(ROOT / s) != value for s, value in provenance["inputs"].items()
                ) or any(
                    digest(Path(path)) != value for path, value in prepared["paths_sha256"].items()
                ):
                    raise ValueError("Binary or catalogue inputs changed during the case")
                rows = [json.loads(line) for line in (case_dir / "stdout.ndjson").read_text().splitlines()]
                results = [r for r in rows if r.get("event") == "scaling_result"]
                resources = []
                for line in (case_dir / "stderr.log").read_text().splitlines():
                    if line.startswith("{"):
                        row = json.loads(line)
                        if "compiled_symbol" in row:
                            resources.append(row)
                _attach_compiled_resources(binary, resources)
                save_json(case_dir / "resources.json", resources)
                complete = (outcome["returncode"] == 0 and not outcome["stop_reason"]
                            and [r["id"] for r in results] == [j["id"] for j in jobs])
                outcome.update(case=case["id"], status="measured" if complete else "failed",
                               measured_jobs=len(results), expected_jobs=len(jobs),
                               binary_sha256=provenance["binary_sha256"])
                save_json(case_dir / "outcome.json", outcome)
                journal.write(json.dumps(outcome) + "\n")
                journal.flush()
                print(json.dumps(outcome), flush=True)
                if not complete:
                    failed = True
                if outcome["stop_reason"]:
                    break
            except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
                outcome.update(case=case["id"], status="inconclusive", reason=str(error),
                               environment_policy=args.environment_policy,
                               timing_eligible=False)
                save_json(case_dir / "outcome.json", outcome)
                journal.write(json.dumps(outcome) + "\n")
                journal.flush()
                print(json.dumps(outcome), flush=True)
                failed = True
                break
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
