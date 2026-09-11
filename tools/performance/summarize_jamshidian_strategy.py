#!/usr/bin/env python3
"""Compact Jamshidian evidence by workload and exact launch geometry.

Screening minima stay exploratory. A repeated candidate is numerically and
statistically qualified only when all three independent confirmations pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

from run_jamshidian_strategy import load_results, save, sha256

KEYS = ("model", "curve", "profile", "side", "stage")
GEOMETRY = ("rows", "strategy", "threads", "blocks", "grid_policy")


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        if "kernel" in record:
            key = tuple(record[k] for k in KEYS) + tuple(record["job"][k] for k in GEOMETRY)
            groups[key].append(record)
    result = []
    for key, runs in sorted(groups.items()):
        entry = dict(zip(KEYS + GEOMETRY, key))
        samples = [v for run in runs for v in run["gpu_samples_ms"]]
        independent = len({run["repeat"] for run in runs})
        numeric_ok = all(r["status"] == "passed" and r["numerically_eligible"] for r in runs)
        environment_ok = all(r["timing_eligible"] for r in runs)
        within_cv = max(r["kernel"]["coefficient_of_variation"] for r in runs)
        medians = [r["kernel"]["median_ms"] for r in runs]
        between_cv = statistics.pstdev(medians) / statistics.mean(medians)
        qualified = (entry["stage"] == "confirm" and independent >= 3
                     and numeric_ok and environment_ok and within_cv <= .05
                     and between_cv <= .05 and all(
                         r["warmups"] >= 5 and r["repetitions"] >= 21
                         and len(r["gpu_samples_ms"]) >= 21 for r in runs))
        resources = runs[0]["diagnostics"]
        entry.update(runs=len(runs), independent_campaigns=independent,
            gpu_median_ms=statistics.median(medians), gpu_min_ms=min(samples),
            gpu_max_ms=max(samples), public_api_median_ms=statistics.median(
                r["public_api"]["median_ms"] for r in runs),
            maximum_within_run_cv=within_cv, between_run_cv=between_cv,
            numerically_eligible=numeric_ok, timing_eligible=environment_ok,
            confirmed_candidate=qualified,
            reference_invalid_rows=max(r["numerics"]["reference_invalid_rows"] for r in runs),
            additional_failed_rows=max(r["numerics"]["failed_rows"] for r in runs),
            maximum_price_error=max(r["numerics"]["maximum_absolute_error"] for r in runs),
            registers=resources["resources"]["registers_per_thread"],
            stack_bytes=resources["compiled_resources"]["stack_frame_bytes"],
            local_bytes=resources["compiled_resources"]["local_bytes_per_thread"],
            local_load_instructions=resources["compiled_resources"]["sass_local_load_instructions"],
            local_store_instructions=resources["compiled_resources"]["sass_local_store_instructions"],
            dynamic_shared_bytes=resources["launch"]["dynamic_shared_bytes_per_block"],
            theoretical_occupancy=resources["occupancy"]["theoretical"],
            evidence=[r["evidence"] for r in runs])
        result.append(entry)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = load_results(args.campaigns)
    rows = summarize(records)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "summary.json", {"purpose": "strategy experiment, not a regression rebaseline",
        "record_count": len(records), "statuses": dict(Counter(r["status"] for r in records)),
        "campaign_hashes": {str(p): sha256(p / "results.ndjson") for p in args.campaigns},
        "configuration_count": len(rows), "configuration_file": "configurations.ndjson",
        "raw_measurements_file": "measurements.ndjson", "table_file": "summary.csv"})
    for name, values in (("configurations.ndjson", rows), ("measurements.ndjson", records)):
        with (args.output / name).open("w") as stream:
            for value in values:
                stream.write(json.dumps(value, separators=(",", ":")) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != "evidence"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(records)} records, {len(rows)} distinct configurations: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
