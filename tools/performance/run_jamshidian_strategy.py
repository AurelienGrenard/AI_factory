#!/usr/bin/env python3
"""Bounded Jamshidian scalar/cooperative experiments; never changes production tuning.

Catalogue rows are repeated, not independently generated. Each phase writes a
fresh evidence directory; later phases select candidates from earlier results.
Temperature is telemetry only. This is not a protocol-v3 rebaseline campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

try:
    from .experiment_environment import power_comparability_issue
    from .run_baseline import _attach_compiled_resources, collect_preflight
    from .run_fixed_income_lsm_probe import bounded
except ImportError:
    from experiment_environment import power_comparability_issue
    from run_baseline import _attach_compiled_resources, collect_preflight
    from run_fixed_income_lsm_probe import bounded

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/codegen/pricing_bindings"))
from capability_manifest import PRODUCT_BINDING_SPECS  # noqa: E402

THREADS = (64, 128, 256, 512)
SIZES = (100, 1000, 16384, 65536, 262144, 1048576)


def compositions() -> list[dict]:
    return [{"model": b.model, "curve": b.curve or ""}
            for b in PRODUCT_BINDING_SPECS
            if b.asset_class == "fixed_income" and b.product == "european_swaption"
            and b.engine == "fixed_income_closed_form"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def geometry(rows: int, strategy: str, threads: int, grid_policy: str) -> dict:
    full = rows if strategy == "cooperative" else math.ceil(rows / threads)
    blocks = full if grid_policy == "full" else min(full, int(grid_policy))
    return {"rows": rows, "strategy": strategy, "threads": threads,
            "blocks": blocks, "grid_policy": grid_policy}


def broad_jobs(sizes: tuple[int, ...], pilot: bool = False,
               grid_policies: tuple[str, ...] = ("full", "64", "256", "1024", "4096")) -> list[dict]:
    jobs = []
    for rows in sizes:
        for strategy in ("scalar", "cooperative"):
            for threads in THREADS:
                seen = set()
                for policy in (("full",) if pilot else grid_policies):
                    job = geometry(rows, strategy, threads, policy)
                    if job["blocks"] not in seen:
                        jobs.append(job)
                        seen.add(job["blocks"])
    return jobs


def load_results(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths
            for line in (path / "results.ndjson").read_text().splitlines() if line.strip()]


def matching(records: list[dict], composition: dict, profile: str, side: str,
             allow_incomplete: bool = False) -> list[dict]:
    statuses = ("passed", "reference_incomplete") if allow_incomplete else ("passed",)
    return [r for r in records if all(r.get(k) == v for k, v in composition.items())
            and r.get("profile") == profile and r.get("side") == side
            and r.get("status") in statuses and r.get("timing_eligible", False)]


def selected_jobs(records: list[dict], sizes: tuple[int, ...], candidates: int) -> list[dict]:
    jobs = []
    for rows in sizes:
        for strategy in ("scalar", "cooperative"):
            options = [r for r in records if r["job"]["strategy"] == strategy]
            if not options:
                raise ValueError(f"No eligible {strategy} candidate")
            nearest = min({r["job"]["rows"] for r in options},
                          key=lambda n: abs(math.log(n / rows)))
            ranked = sorted((r for r in options if r["job"]["rows"] == nearest),
                            key=lambda r: r["kernel"]["median_ms"])
            seen = set()
            for result in ranked:
                old = result["job"]
                key = (old["threads"], old["grid_policy"])
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(geometry(rows, strategy, *key))
                if len(seen) == candidates:
                    break
    return jobs


def estimate_ms(job: dict, records: list[dict]) -> float:
    same = [r for r in records if r["job"]["strategy"] == job["strategy"]]
    if not same:
        return 10.0
    closest = min(same, key=lambda r: (
        abs(math.log(r["job"]["rows"] / job["rows"])),
        r["job"]["threads"] != job["threads"],
        r["job"]["grid_policy"] != job["grid_policy"]))
    return closest["kernel"]["median_ms"] * max(1, job["rows"] / closest["job"]["rows"])


def own_benchmark_process(line: str, binary: Path, known_pids: set[int] | None = None) -> bool:
    """WSL can report '[Not Found]'; identify our child by PID and executable."""
    try:
        pid = int(line.split(",", 1)[0].strip())
        directory = Path(f"/proc/{pid}")
        status = dict(row.split(":", 1) for row in (directory / "status").read_text().splitlines())
        if int(status["PPid"]) != os.getpid():
            return False
        if status["State"].lstrip().startswith("Z"):
            return known_pids is not None and pid in known_pids
        owned = (directory / "exe").resolve(strict=True) == binary
        if owned and known_pids is not None:
            known_pids.add(pid)
        return owned
    except (OSError, ValueError, KeyError):
        # The child can finish while nvidia-smi is collecting its snapshot.
        # Only a previously verified, now absent PID is accepted as our exit.
        if known_pids is not None:
            try:
                pid = int(line.split(",", 1)[0].strip())
                return pid in known_pids and not Path(f"/proc/{pid}").exists()
            except ValueError:
                pass
        return False


def execution_issue(snapshot: dict, binary: Path, running: bool = False,
                    known_pids: set[int] | None = None) -> str | None:
    if snapshot["power_source"] != "external_power":
        return "External power is required"
    if snapshot["throttle"].get("hardware_power_brake") == "Active":
        return "Hardware power brake active"
    foreign = [line for line in snapshot["concurrent_compute_processes"]
               if not (running and own_benchmark_process(line, binary, known_pids))]
    return f"Other GPU computation: {foreign}" if foreign else None


def run_plan(binary: Path, directory: Path, plan: dict, reference: dict,
             hashes: dict[str, str]) -> list[dict]:
    directory.mkdir()
    for name, expected in hashes.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Input or executable changed: {name}")
    save(directory / "plan.json", plan)
    telemetry, power_issues = [], set()
    known_pids: set[int] = set()

    def capture(running=False):
        snapshot = collect_preflight()
        telemetry.append(snapshot)
        save(directory / "telemetry.json", telemetry)
        issue = power_comparability_issue(snapshot["power_limits_w"].get("current"),
            reference["power_limits_w"].get("current"), relative_tolerance=.20)
        if issue:
            power_issues.add(issue)
        return execution_issue(snapshot, binary, running, known_pids)

    problem = capture()
    if problem:
        raise RuntimeError(problem)
    outcome = bounded([str(binary), str(directory / "plan.json")],
        directory / "stdout.ndjson", directory / "stderr.ndjson", 1800,
        monitor=lambda elapsed: capture(running=True))
    problem = capture()
    outcome.update(timing_eligible=not power_issues and not problem,
                   power_issues=sorted(power_issues), final_execution_issue=problem)
    save(directory / "outcome.json", outcome)
    if outcome["returncode"] or outcome["timed_out"] or outcome["stop_reason"] or problem:
        raise RuntimeError(f"Benchmark stopped: see {directory}")
    diagnostics = [json.loads(line) for line in (directory / "stderr.ndjson").read_text().splitlines()
                   if line.startswith('{"type":"cuda_kernel_launch_diagnostics"')]
    _attach_compiled_resources(binary, diagnostics)
    resources = {d["kernel"]: d for d in diagnostics}
    records = []
    for line in (directory / "stdout.ndjson").read_text().splitlines():
        result = json.loads(line)
        if result["type"] != "result":
            continue
        result.update({k: plan[k] for k in ("model", "curve", "profile", "side", "stage", "repeat")})
        result.update(timing_eligible=outcome["timing_eligible"],
                      power_issues=outcome["power_issues"], evidence=str(directory.relative_to(ROOT)))
        result["diagnostics"] = resources[result["job"]["id"]]
        records.append(result)
    if len(records) != len(plan["jobs"]):
        raise RuntimeError("Missing benchmark results")
    for name, expected in hashes.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Input or executable changed during measurement: {name}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("pilot", "screen", "profiles", "large", "confirm"))
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/ai_factory_jamshidian_strategy_benchmark")
    parser.add_argument("--repeat", type=int, default=0, help="Independent process campaign identifier")
    parser.add_argument("--side", choices=("payer", "receiver"), default="payer")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Explicit composition resume index, into a fresh directory")
    args = parser.parse_args()
    binary, dest = args.binary.resolve(), args.output.resolve()
    records = load_results(args.source)
    inventory = compositions()
    if not 0 <= args.start_index < len(inventory):
        parser.error("Resume index is outside the composition inventory")
    plans = []
    for composition in inventory[args.start_index:]:
        for profile in (("short", "long") if args.stage == "profiles" else ("catalogue",)):
            # Incomplete references may guide further timing probes only. Their
            # numerical ineligibility remains explicit; never production tuning.
            previous = matching(records, composition, profile, args.side, allow_incomplete=True)
            if not previous:
                previous = matching(records, composition, "catalogue", "payer", allow_incomplete=True)
            if args.stage == "pilot":
                jobs = broad_jobs((1000,), pilot=True)
            elif args.stage == "screen":
                jobs = broad_jobs((100, 1000, 16384, 65536))
            elif args.stage == "profiles":
                jobs = broad_jobs((16384,))
            elif args.stage == "large":
                # Re-sweep threads at large N: small-batch winners need not scale.
                # Tiny persistent grids were already covered in the broad screen.
                jobs = broad_jobs((262144, 1048576), grid_policies=("full", "1024", "4096"))
            else:
                jobs = selected_jobs(previous, SIZES, candidates=1)
            for index, job in enumerate(jobs):
                estimate = estimate_ms(job, previous)
                job.update(id=f"job_{index:03d}", warmups=5 if args.stage == "confirm" else 1,
                    repetitions=21 if args.stage == "confirm" else 3,
                    operations=min(128, max(1, math.ceil(10 / max(.001, estimate)))),
                    predicted_ms=estimate)
            random.Random(20260908 + args.repeat).shuffle(jobs)
            plans.append({**composition, "profile": profile, "side": args.side,
                "stage": args.stage, "repeat": args.repeat, "jobs": jobs})
    predicted_seconds = sum(j["predicted_ms"] * j["operations"] * (j["warmups"] + j["repetitions"])
                            for p in plans for j in p["jobs"]) / 1000
    dest.mkdir(parents=True, exist_ok=False)
    names = {"datasets/product/european_swaption/european_swaptions_01.json", str(binary.relative_to(ROOT))}
    for c in inventory:
        names.add(f"datasets/model/fixed_income/{c['model']}/parameters/{c['model']}_01.json")
        if c["curve"]:
            names.add(f"datasets/curve/{c['curve']}/{c['curve']}_01.json")
    hashes = {name: sha256(ROOT / name) for name in sorted(names)}
    save(dest / "inputs.sha256.json", hashes)
    # Includes untracked implementation files from the deliberately dirty worktree.
    sources = [p for base in ("src", "tools/codegen", "tools/performance", "tests/performance")
               for p in (ROOT / base).rglob("*") if p.is_file() and p.suffix in (".cu", ".cuh", ".hpp", ".cpp", ".py")]
    save(dest / "sources.sha256.json", {str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)})
    (dest / "git-status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))
    (dest / "git-diff.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    reference = collect_preflight()
    save(dest / "manifest.json", {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inventory": inventory, "plans": plans, "reference_environment": reference,
        "predicted_gpu_seconds": predicted_seconds, "thermal_policy": "telemetry_only",
        "purpose": "exploratory strategy selection; not protocol-v3 acceptance",
        "fixture": "cyclic repetitions of aligned catalogue rows; not independent new prices",
        "source_campaigns": [str(p) for p in args.source]})
    print(f"{args.stage}: {len(plans)} plans, {sum(len(p['jobs']) for p in plans)} configurations; "
          f"estimated timed GPU work {predicted_seconds:.1f}s (not a wall-time guarantee)", flush=True)
    with (dest / "results.ndjson").open("w") as journal:
        for index, plan in enumerate(plans):
            directory = dest / f"{index:02d}_{plan['model']}_{plan['curve']}_{plan['profile']}"
            for result in run_plan(binary, directory, plan, reference, hashes):
                journal.write(json.dumps(result) + "\n")
            journal.flush()
            print(f"Completed {index+1}/{len(plans)}: {plan['model']} {plan['curve']} {plan['profile']}", flush=True)
    print(f"Evidence: {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
