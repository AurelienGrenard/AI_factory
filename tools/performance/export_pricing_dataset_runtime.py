"""Export portable, exact-shape dataset-runtime evidence for the presentation notebook.

This CPU-only reader never launches a generator, substitutes a smaller workload,
or fills a missing measurement with an extrapolation. Raw campaign files remain
the evidence owner; the compact export includes their hashes and timing samples.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

try:
    from .pricing_scaling_inputs import CATALOGUE_INPUT_PROFILE
except ImportError:
    from pricing_scaling_inputs import CATALOGUE_INPUT_PROFILE

PRICE_COUNT = 1000
PATH_COUNT = 1 << 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # A live process may be halfway through its final output line.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith("\n") and line.strip()]


def nonnegative(value: float, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {label}: {value!r}")
    return float(value)


def timing_breakdown(row: dict) -> dict:
    """Keep the GPU clock separate; summing it with the host API double-counts work."""
    phases = {
        "preparation_ms": nonnegative(row["preparation_once_ms"], "preparation"),
        "raw_host_api_ms": nonnegative(row["raw_host_clock"]["median_ms"], "raw host API"),
        "output_copy_ms": nonnegative(row["output_copy"]["median_ms"], "output copy"),
        "local_publication_ms": nonnegative(row["publication"]["wall_ms"], "publication"),
    }
    if not row["publication"]["native_writer"]:
        raise ValueError("Dataset runtime requires the native local JSON/YAML writer")
    return {**phases,
            "generation_phase_sum_ms": sum(phases.values()),
            "gpu_median_ms": nonnegative(row["gpu"]["median_ms"], "GPU"),
            "public_api_envelope_ms": nonnegative(row["public_api"]["median_ms"], "API envelope")}


def case_result(directory: Path, case: dict, job: dict) -> dict:
    name = case["id"]
    closed = case["family"] == "closed_form"
    if job["rows"] != PRICE_COUNT or job.get("offset", 0) != 0 or (
        not closed and job["paths"] != PATH_COUNT
    ):
        raise ValueError(f"{name}: expected exactly 1,000 prices and 2^20 paths when applicable")
    folder = directory / name
    outcome_path = folder / "outcome.json"
    journal = [r for r in json_lines(directory / "journal.ndjson") if r.get("case") == name]
    outcome = (json.loads(outcome_path.read_text()) if outcome_path.exists()
               else journal[-1] if journal else {})
    result = {"case": name, "model": case["model"], "product": case["product"],
              "side": case["side"], "family": case["family"], "generator": case["generator"],
              "engine": case["engine"], "factor_count": case.get("factor_count"),
              "curve": case.get("curve"),
              "rows": PRICE_COUNT, "paths_per_price": 0 if closed else PATH_COUNT,
              "status": outcome.get("status", "running" if folder.exists() else "pending"),
              "reason": outcome.get("reason") or outcome.get("stop_reason"),
              "configuration": job, "timings": None,
              "environment_eligible_for_tuning": outcome.get("timing_eligible", False),
              "timing_ineligibility_reasons": outcome.get("timing_ineligibility_reasons", [])}
    if result["status"] != "measured":
        return result
    raw = folder / "stdout.ndjson"
    rows = [r for r in json_lines(raw) if r.get("event") == "scaling_result"]
    if len(rows) != 1 or rows[0]["id"] != job["id"]:
        raise ValueError(f"{name}: expected one completed predeclared dataset measurement")
    row = rows[0]
    config = row["configuration"]
    if (config["rows"], config["offset"], config["paths_per_price"]) != (
        PRICE_COUNT, 0, 0 if closed else PATH_COUNT
    ):
        raise ValueError(f"{name}: output workload differs from the requested dataset")
    provenance = json.loads((folder / "provenance.json").read_text())
    fixture = provenance["fixture"]
    if fixture["profile"] != CATALOGUE_INPUT_PROFILE or fixture["source_indices_zero_based"] != list(range(1000)):
        raise ValueError(f"{name}: not the full ordered catalogue")
    for filename, digest in fixture["paths_sha256"].items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f"{name}: input artifact changed after measurement")
    if len(row["prices"]) != PRICE_COUNT or (not closed and len(row["standard_errors"]) != PRICE_COUNT):
        raise ValueError(f"{name}: incomplete numerical output")
    if not row["deterministic_replay"] or not all(math.isfinite(v) for v in row["prices"]):
        raise ValueError(f"{name}: failed finite deterministic replay")
    if not all(math.isfinite(v) and v >= 0 for v in row["standard_errors"]):
        raise ValueError(f"{name}: invalid standard errors")
    samples = {k: row[k] for k in ("gpu_samples_ms", "raw_host_samples_ms", "api_samples_ms")}
    if any(len(v) != config["repetitions"] for v in samples.values()):
        raise ValueError(f"{name}: incomplete timing samples")
    for values in samples.values():
        for value in values:
            nonnegative(value, "timing sample")
    resources = json.loads((folder / "resources.json").read_text())
    unique = {}
    for resource in resources:
        key = (resource["kernel"], resource["variant"], resource["launch"]["threads_per_block"])
        blocks = resource["launch"]["grid_block_count"]
        entry = unique.setdefault(key, {"kernel": resource["kernel"], "variant": resource["variant"],
            "threads": resource["launch"]["threads_per_block"],
            "grid_blocks_min": blocks, "grid_blocks_max": blocks,
            "registers": resource["resources"]["registers_per_thread"],
            "local_bytes": resource["resources"]["local_bytes_per_thread"],
            "shared_bytes": resource["resources"]["static_shared_bytes_per_block"]
                            + resource["launch"]["dynamic_shared_bytes_per_block"],
            "theoretical_occupancy": resource["occupancy"]["theoretical"]})
        entry["grid_blocks_min"] = min(entry["grid_blocks_min"], blocks)
        entry["grid_blocks_max"] = max(entry["grid_blocks_max"], blocks)
    telemetry = json_lines(folder / "telemetry.ndjson")
    ranges = {}
    for field in ("temperature_c", "sm_clock_mhz", "power_limit_w"):
        values = [r[field] for r in telemetry if field in r]
        ranges[field] = [min(values), max(values)] if values else None
    return {**result, "configuration": config, "timings": timing_breakdown(row),
            "samples": samples, "gpu_statistics": row["gpu"],
            "raw_host_statistics": row["raw_host_clock"],
            "operations_per_sample": row["operations_per_sample"],
            "fixed_time_step": row["fixed_time_step"],
            "environment": row["environment"], "telemetry_ranges": ranges,
            "memory": row["device_memory"], "resources": list(unique.values()),
            "lsm_batches": row.get("lsm_batches_last_repetition"),
            "deterministic_replay": True, "independent_price_certification": False,
            "binary_sha256": provenance["binary_sha256"], "source_inputs_sha256": provenance["inputs"],
            "evidence_sha256": {p.name: sha256(p) for p in (raw, outcome_path, folder / "resources.json", folder / "provenance.json")}}


def export_campaign(directory: Path) -> dict:
    plan = json.loads((directory / "plan.json").read_text())
    if plan.get("input_profile") != CATALOGUE_INPUT_PROFILE:
        raise ValueError("A repeated tile cannot stand in for full catalogue runtime")
    if not plan["selected_jobs"]:
        raise ValueError("A dataset-runtime campaign must contain at least one case")
    specs = {c["id"]: c for c in plan["manifest"]["cases"]}
    cases = []
    for name, jobs in plan["selected_jobs"].items():
        if len(jobs) != 1:
            raise ValueError("Export a predeclared single profile per pair, not a best-of-geometries selection")
        cases.append(case_result(directory, specs[name], jobs[0]))
    snapshot = json.loads((directory / "snapshot.json").read_text())
    return {"schema": "ai_factory_pricing_dataset_runtime_v1",
            "exported_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "campaign": str(directory), "started_utc": plan["started_utc"],
            "revision": snapshot["revision"], "source_diff_sha256": snapshot["diff_sha256"],
            "source_archive_sha256": snapshot["archive_sha256"],
            "input_profile": plan["input_profile"], "price_count": PRICE_COUNT, "mc_paths_per_price": PATH_COUNT,
            "production_launch_plans": plan.get("production_launch_plans", {}),
            "complete": all(c["status"] == "measured" for c in cases),
            "timing_scope": "generation_phase_sum = preparation + median raw host API + median D2H + one native local publication; not a cold-process stopwatch, no upload or independent validation",
            "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_campaign(args.campaign)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "complete": result["complete"],
                      "measured": sum(c["status"] == "measured" for c in result["cases"]),
                      "expected": len(result["cases"])}))


if __name__ == "__main__":
    main()
