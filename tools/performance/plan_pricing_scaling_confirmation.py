"""Predeclare fresh, longer comparisons of per-shape MC candidates and native defaults.

This host-only planner reads an immutable completed geometry campaign. It neither
runs a GPU job nor installs a production setting. A run uses the resulting JSON
with run_pricing_scaling.py --jobs; no unsuccessful measurement is retried here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from .summarize_pricing_scaling import summarize
except ImportError:
    from summarize_pricing_scaling import summarize


JOB_FIELDS = ("rows", "offset", "paths", "threads", "batch_rows", "block_limit",
              "blocks_per_price", "path_chunk")


def execution_key(job: dict) -> tuple:
    """Canonicalize equivalent ordinary-MC grid caps below the requested batch."""
    batch = min(job["rows"], job["batch_rows"])
    return (job["rows"], job.get("offset", 0), job["paths"], job["threads"],
            batch, min(batch, job["block_limit"]))


def confirmation_jobs(case: dict, points: list[dict], source_jobs: list[dict],
                      sample_ms: float, repetitions: int, warmups: int) -> tuple[list[dict], list[dict]]:
    if case["family"] not in ("mc_terminal", "mc_barrier") or case["mathdx"]:
        raise ValueError("This confirmation planner currently covers ordinary MC, not FFT/LSM/closed form")
    if not math.isfinite(sample_ms) or not 1 <= sample_ms <= 5000:
        raise ValueError("Minimum sample duration must be in [1, 5000] ms")
    if not 3 <= repetitions <= 21 or not 1 <= warmups <= 5:
        raise ValueError("Confirmation needs 3..21 repetitions and 1..5 excluded warmups")
    by_id = {j["id"]: j for j in source_jobs}
    jobs, decisions = [], []
    for index, point in enumerate(sorted(points, key=lambda p: (p["rows"], p["paths_per_price"]))):
        if point["rows"] < 100 or point["paths_per_price"] == 0:
            continue
        if not math.isfinite(point["gpu_median_ms"]) or point["gpu_median_ms"] <= 0:
            raise ValueError("A confirmation requires a finite positive source timing")
        candidate = {key: value for key, value in by_id[point["candidate_id"]].items()
                     if key in JOB_FIELDS}
        native = {**candidate, "threads": 256 if case["factor_count"] else 512,
                  "batch_rows": min(candidate["rows"], 4096), "block_limit": 4096}
        operations = min(4096, max(1, math.ceil(sample_ms / point["gpu_median_ms"])))
        alternatives = [("candidate", candidate)]
        if execution_key(native) != execution_key(candidate):
            alternatives.append(("native_reference", native))
        # Alternate order between neighboring shapes; do not always measure the
        # proposed setting after its reference or select a best raw repetition.
        if index % 2:
            alternatives.reverse()
        for role, configuration in alternatives:
            jobs.append({**configuration,
                         "id": f"r{point['rows']}_n{point['paths_per_price']}_{role}",
                         "warmups": warmups, "repetitions": repetitions,
                         "operations_per_sample": operations})
        decisions.append({"rows": point["rows"], "paths_per_price": point["paths_per_price"],
                          "screening_id": point["candidate_id"],
                          "screening_median_ms": point["gpu_median_ms"],
                          "same_effective_geometry_as_native": len(alternatives) == 1,
                          "operations_per_sample": operations,
                          "predicted_candidate_sample_ms": operations * point["gpu_median_ms"],
                          "production_tuning_accepted": False})
    if not jobs:
        raise ValueError("No completed ordinary-MC shapes to confirm")
    return jobs, decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-sample-ms", type=float, default=1000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    source = args.campaign.resolve()
    destination = args.output.resolve()
    if not (destination.is_relative_to(source.parent) or destination.is_relative_to(Path("/tmp"))):
        parser.error("Confirmation plans belong beside build evidence or below /tmp")
    destination.mkdir(parents=True, exist_ok=False)
    plan = json.loads((source / "plan.json").read_text())
    summary = summarize([source])
    specs = {c["id"]: c for c in plan["manifest"]["cases"]}
    results = {c["case"]: c for c in summary["cases"]}
    jobs, decisions = {}, {}
    for case in args.cases:
        if case not in plan["selected_jobs"]:
            parser.error(f"Case was not measured by the source campaign: {case}")
        outcome = json.loads((source / case / "outcome.json").read_text())
        if outcome["status"] != "measured":
            parser.error(f"Case did not complete; inspect its failure before planning: {case}")
        if any(not p["prices_pass"] or not p["standard_errors_pass"]
               for p in results[case]["numerical_parity"]):
            parser.error(f"Investigate the numerical disagreement before confirming timings: {case}")
        jobs[case], decisions[case] = confirmation_jobs(
            specs[case], results[case]["geometry_envelope"], plan["selected_jobs"][case],
            args.minimum_sample_ms, args.repetitions, args.warmups)
    (destination / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    metadata = {"source_campaign": str(source),
                "source_plan_sha256": hashlib.sha256((source / "plan.json").read_bytes()).hexdigest(),
                "minimum_sample_ms": args.minimum_sample_ms,
                "repetitions": args.repetitions, "warmups": args.warmups,
                "decisions": decisions,
                "limits": "predicted durations are scheduling estimates, not new timings; no automatic retry or production acceptance"}
    (destination / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"cases": len(jobs), "jobs": sum(map(len, jobs.values())),
                      "jobs_file": str(destination / "jobs.json")}))


if __name__ == "__main__":
    main()
