"""Report scaling coverage, normalized costs and numerical parity without inventing passes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .pricing_scaling_manifest import PATH_COUNTS, PRICE_COUNTS
except ImportError:
    from pricing_scaling_manifest import PATH_COUNTS, PRICE_COUNTS


def within_tolerance(reference: list[float], candidate: list[float], absolute: float, relative: float) -> bool:
    return len(reference) == len(candidate) and all(
        math.isfinite(a) and math.isfinite(b) and abs(a - b) <= absolute + relative * abs(a)
        for a, b in zip(reference, candidate)
    )


def environment_observations(directory: Path) -> dict:
    """Describe the measured operating regime without turning it into a pass."""
    path = directory / "telemetry.ndjson"
    samples = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    result = {"sample_count": len(samples)}
    for field in ("temperature_c", "sm_clock_mhz", "power_limit_w", "utilization_percent"):
        values = [s[field] for s in samples if field in s]
        result[field] = {"minimum": min(values), "maximum": max(values)} if values else None
    result["software_thermal_samples"] = sum(
        s.get("throttle", {}).get("clocks_event_reason_sw_thermal_slowdown") == "Active"
        for s in samples
    )
    return result


def timing_qualified(row: dict) -> bool:
    return (row["outcome"] == "measured" and row["timing_eligible"]
            and not row.get("numerical_conflict", False)
            and row["environment_policy"] == "strict"
            and row["configuration"]["warmups"] >= 1
            and row["configuration"]["repetitions"] >= 3
            and row["gpu"]["coefficient_of_variation"] <= .05)


def geometry_envelope(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Compare workload-specific geometries, never a minimum over raw repetitions.

    The median-ranked envelope is diagnostic, not an automatic production
    tuning decision. Screening-only points always require a fresh confirmation.
    """
    groups = {}
    for row in rows:
        config = row["configuration"]
        groups.setdefault((config["rows"], config["paths_per_price"], config.get("offset", 0)), []).append(row)
    envelope = []
    chosen = []
    for (prices, paths, offset), candidates in sorted(groups.items()):
        eligible = [r for r in candidates if timing_qualified(r)]
        selected = min(eligible or candidates, key=lambda r: r["gpu"]["median_ms"])
        chosen.append(selected)
        envelope.append({"rows": prices, "paths_per_price": paths, "offset": offset,
                         "candidate_id": selected["id"], "run": selected["run"],
                         "configuration": selected["configuration"],
                         "gpu_median_ms": selected["gpu"]["median_ms"],
                         "candidate_count": len(candidates),
                         "repeated_timing_qualified": bool(eligible),
                         "production_tuning_accepted": False})
    comparisons = []
    for axis in ("paths_per_price", "rows"):
        fixed = "rows" if axis == "paths_per_price" else "paths_per_price"
        for fixed_value in sorted({r["configuration"][fixed] for r in chosen}):
            points = sorted((r for r in chosen if r["configuration"][fixed] == fixed_value
                             and r["configuration"][axis] > 0
                             and r["configuration"]["rows"] >= 100),
                            key=lambda r: r["configuration"][axis])
            for a, b in zip(points, points[1:]):
                work_ratio = b["configuration"][axis] / a["configuration"][axis]
                time_ratio = b["gpu"]["median_ms"] / a["gpu"]["median_ms"]
                comparable = (timing_qualified(a) and timing_qualified(b)
                              and a["run"] == b["run"]
                              and bool(a.get("fixture_signature"))
                              and a.get("fixture_signature") == b.get("fixture_signature")
                              and bool(a.get("binary_sha256"))
                              and a.get("binary_sha256") == b.get("binary_sha256"))
                comparisons.append({"axis": axis, "fixed_axis": fixed, "fixed_value": fixed_value,
                    "from": a["configuration"][axis], "to": b["configuration"][axis],
                    "reference_id": a["id"], "candidate_id": b["id"],
                    "reference_run": a["run"], "candidate_run": b["run"],
                    "work_ratio": work_ratio, "gpu_time_ratio": time_ratio,
                    "normalized_cost_ratio": time_ratio / work_ratio,
                    "same_launch_configuration_required": False,
                    "assessment": "inconclusive" if not comparable else
                    "investigate_superlinear" if time_ratio / work_ratio > 1.20 else
                    "no_superlinear_alert_on_this_pair"})
    return envelope, comparisons


def single_price_information(envelope: list[dict]) -> list[dict]:
    """Keep P*T(1) visible without treating one row as a saturation baseline."""
    isolated = {p["paths_per_price"]: p for p in envelope if p["rows"] == 1}
    return [{"rows": point["rows"], "paths_per_price": point["paths_per_price"],
             "single_price_run": isolated[point["paths_per_price"]]["run"],
             "batch_run": point["run"],
             "gpu_time_over_prices_times_single_price": point["gpu_median_ms"] / (
                 point["rows"] * isolated[point["paths_per_price"]]["gpu_median_ms"]),
             "assessment": "informational_only",
             "limitation": "one selected row, not the tile-average work or a saturated reference"}
            for point in envelope if point["rows"] >= 100
            and point["paths_per_price"] in isolated]


def summarize(directories: list[Path]) -> dict:
    records = []
    specifications = {}
    runs = []
    environments = []
    for directory in directories:
        plan = json.loads((directory / "plan.json").read_text())
        specifications.update((c["id"], {**c,
            "price_counts": plan["manifest"].get("price_counts", PRICE_COUNTS),
            "path_counts": plan["manifest"].get("path_counts", PATH_COUNTS),
            "input_profile": plan.get("input_profile", "legacy_catalogue_order")})
            for c in plan["manifest"]["cases"])
        runs.append({"directory": str(directory), "stage": plan["stage"]})
        for raw in sorted(directory.glob("*/stdout.ndjson")):
            outcome_path = raw.parent / "outcome.json"
            outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {"status": "interrupted"}
            provenance_path = raw.parent / "provenance.json"
            provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
            environments.append({"run": str(directory), "case": raw.parent.name,
                                 "environment_policy": plan.get("environment_policy", "strict"),
                                 **environment_observations(raw.parent)})
            for line in raw.read_text().splitlines():
                row = json.loads(line)
                if row.get("event") == "scaling_result":
                    row["run"] = str(directory)
                    row["outcome"] = outcome["status"]
                    row["timing_eligible"] = outcome.get("timing_eligible", True)
                    row["timing_ineligibility_reasons"] = outcome.get("timing_ineligibility_reasons", [])
                    row["environment_policy"] = plan.get("environment_policy", "strict")
                    row["input_profile"] = plan.get("input_profile", "legacy_catalogue_order")
                    row["fixture_signature"] = provenance.get("fixture", {}).get("fixture_signature")
                    row["binary_sha256"] = provenance.get("binary_sha256")
                    row["numerical_conflict"] = False
                    row["configuration"].setdefault("warmups", 0)
                    row["stage"] = plan["stage"]
                    records.append(row)
    results = []
    for key, spec in sorted(specifications.items()):
        rows = [r for r in records if r["case"] == key and r["input_profile"] == spec["input_profile"]]
        expected = {(r, p) for r in spec["price_counts"]
                    for p in ((0,) if spec["family"] == "closed_form" else spec["path_counts"])}
        observed = {(r["configuration"]["rows"], r["configuration"]["paths_per_price"])
                    for r in rows if r["outcome"] == "measured"}
        parity = []
        for index, reference in enumerate(rows):
            a = reference["configuration"]
            for candidate in rows[index + 1:]:
                b = candidate["configuration"]
                if reference.get("fixture_signature") != candidate.get("fixture_signature"):
                    continue
                if (a["rows"], a["offset"], a["paths_per_price"], a["seed"]) != (
                    b["rows"], b["offset"], b["paths_per_price"], b["seed"]
                ):
                    continue
                price_ok = within_tolerance(reference["prices"], candidate["prices"], 1e-6, 1e-6)
                error_ok = within_tolerance(reference["standard_errors"], candidate["standard_errors"], 1e-8, 1e-5)
                if not price_ok or not error_ok:
                    # Neither member is established correct by a disagreement.
                    # Keep the evidence, but exclude both from qualified geometry selection.
                    reference["numerical_conflict"] = True
                    candidate["numerical_conflict"] = True
                parity.append({"reference": reference["id"], "candidate": candidate["id"],
                               "reference_run": reference["run"], "candidate_run": candidate["run"],
                               "prices_pass": price_ok, "standard_errors_pass": error_ok,
                               "bitwise_values_equal": reference["prices"] == candidate["prices"]
                               and reference["standard_errors"] == candidate["standard_errors"]})
        qualified = {(r["configuration"]["rows"], r["configuration"]["paths_per_price"])
                     for r in rows if timing_qualified(r)}
        envelope, scaling = geometry_envelope(rows)
        per_campaign = []
        for campaign in dict.fromkeys(r["run"] for r in rows):
            campaign_envelope, campaign_scaling = geometry_envelope(
                [r for r in rows if r["run"] == campaign])
            per_campaign.append({"run": campaign, "geometry_envelope": campaign_envelope,
                                 "comparisons": campaign_scaling})
        results.append({"case": key, "family": spec["family"],
                        "input_profile": spec["input_profile"], "geometry_envelope": envelope,
                        "per_campaign": per_campaign,
                        "single_price_information": single_price_information(envelope),
                        "missing_shapes": sorted(expected - observed),
                        "unqualified_shapes": sorted(expected - qualified),
                        "measured_shapes": sorted(observed), "comparisons": scaling,
                        "numerical_parity": parity,
                        "measurements": [{"id": r["id"], "run": r["run"],
                            "configuration": r["configuration"], "gpu": r["gpu"],
                            "public_api": r["public_api"], "preparation_once_ms": r["preparation_once_ms"],
                            "raw_host_clock": r.get("raw_host_clock"),
                            "raw_host_samples_ms": r.get("raw_host_samples_ms"),
                            "operations_per_sample": r.get("operations_per_sample", 1),
                            "output_copy": r.get("output_copy"),
                            "publication": r["publication"], "memory": r["device_memory"],
                            "lsm_batches_last_repetition": r.get("lsm_batches_last_repetition"),
                            "timing_scope": r.get("timing_scope"),
                            "timing_eligible": r["timing_eligible"],
                            "numerical_conflict": r["numerical_conflict"],
                            "timing_ineligibility_reasons": r["timing_ineligibility_reasons"],
                            "environment_policy": r["environment_policy"],
                            "outcome": r["outcome"]} for r in rows]})
    return {"schema": "ai_factory_pricing_scaling_summary_v1", "runs": runs,
            "environments": environments,
            "qualification": "exploratory; not protocol-v3 acceptance or extrapolated 1M-price runtime",
            "cases": results, "missing_case_count": sum(bool(c["missing_shapes"]) for c in results),
            "unqualified_case_count": sum(bool(c["unqualified_shapes"]) for c in results),
            "numerical_failures": sum(not p["prices_pass"] or not p["standard_errors_pass"]
                                      for c in results for p in c["numerical_parity"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new summary path; previous evidence is immutable")
    result = summarize(args.directories)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "runs", "environments")}))
