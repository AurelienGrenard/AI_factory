"""Apply a predeclared bump policy to cached independent-reference analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def _sha256(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _budget(criteria, prefix, reference):
    return criteria[f"{prefix}_absolute"] + criteria[f"{prefix}_relative"] * abs(reference)


def qualify(policy, analysis):
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported bump policy schema")
    if analysis.get("schema_version") != 2:
        raise ValueError("unsupported bump analysis schema")
    if analysis.get("policy_id") != policy["policy_id"]:
        raise ValueError("analysis and policy identities differ")
    criteria = policy["criteria"]
    expected_seeds = set(policy["execution"]["seeds"])
    expected_keys = {
        (model, coordinate, case["id"], factor, refinement, seed)
        for model in policy["scope"]["models"]
        for coordinate in policy["base_bumps"][model]
        for case in policy["cases"][model]
        for factor in policy["execution"]["bump_factors"]
        for refinement in policy["execution"]["refinements"][model]
        for seed in policy["execution"]["seeds"]
    }
    observed_keys = [(row["model"], row["coordinate"], row["case_id"], row["factor"],
                      row["refinement"], row["seed"]) for row in analysis["rows"]]
    if len(observed_keys) != len(set(observed_keys)):
        raise ValueError("duplicate analyzed model/coordinate/case/factor/refinement/seed observation")
    if set(observed_keys) != expected_keys:
        missing, extra = expected_keys - set(observed_keys), set(observed_keys) - expected_keys
        raise ValueError(f"analyzed campaign surface mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    expected_classes = {(model, case["id"]): case["classes"]
                        for model in policy["scope"]["models"] for case in policy["cases"][model]}
    execution = policy["execution"]
    for row in analysis["rows"]:
        if row["case_classes"] != expected_classes[(row["model"], row["case_id"])]:
            raise ValueError("analyzed case classes differ from policy")
        if (row["paths"] != execution["paths_per_price"]
                or row.get("threads_per_block") != execution["threads_per_block"]
                or row.get("sensitivity_batch_size") != execution["sensitivity_batch_size"]):
            raise ValueError("analyzed launch configuration differs from policy")
    groups = {}
    for row in analysis["rows"]:
        key = (row["model"], row["coordinate"], row["case_id"], row["factor"], row["refinement"])
        groups.setdefault(key, []).append(row)
    cases = []
    for (model, coordinate, case_id, factor, refinement), samples in sorted(groups.items()):
        seeds = {sample["seed"] for sample in samples}
        if len(samples) != len(expected_seeds) or seeds != expected_seeds:
            raise ValueError(f"incomplete seed set for {model}/{coordinate}/{case_id}/{factor}/{refinement}")
        invariant_fields = (
            "case_classes", "reference_stencil", "reference_price_convergence",
            "reference_gradient_convergence", "derivative", "reference_derivative_convergence",
            "reference_derivative_stencil", "reference_convergence_applicable", "stencil_bias", "dt",
        )
        for field in invariant_fields:
            if any(sample[field] != samples[0][field] for sample in samples[1:]):
                raise ValueError(f"non-invariant {field} across seeds")
        if "fp32_error" in samples[0] and any(
                sample["fp32_error"] != samples[0]["fp32_error"] for sample in samples[1:]):
            raise ValueError("non-invariant FP32 diagnostic across seeds")
        estimate = statistics.mean(sample["mc"] for sample in samples)
        combined_se = math.sqrt(sum(sample["mc_se"] ** 2 for sample in samples)) / len(samples)
        reference_stencil = samples[0]["reference_stencil"]
        derivative = samples[0]["derivative"]
        if derivative is None:
            raise ValueError("qualification requires an independent infinitesimal derivative")
        residual = estimate - reference_stencil
        total_error = estimate - derivative
        stencil_bias = reference_stencil - derivative
        convergence_applicable = samples[0].get("reference_convergence_applicable", True)
        reference_pass = (
            (not convergence_applicable or (
                samples[0]["reference_price_convergence"] <= criteria["reference_price_convergence_absolute"]
                and samples[0]["reference_gradient_convergence"] <= criteria["reference_gradient_convergence_absolute"]
            ))
            and samples[0]["reference_derivative_convergence"] <= criteria["reference_derivative_convergence_absolute"]
        )
        noise_limit = criteria["monte_carlo_standard_errors"] * combined_se
        noise_pass = abs(residual) <= noise_limit
        total_limit = _budget(criteria, "total_error", derivative)
        total_pass = abs(total_error) <= total_limit
        stencil_limit = _budget(criteria, "stencil_bias", derivative)
        stencil_pass = abs(stencil_bias) <= stencil_limit
        fp32_error = samples[0].get("fp32_error")
        fp32_limit = None if fp32_error is None else _budget(criteria, "fp32_error", reference_stencil)
        fp32_pass = True if fp32_error is None else abs(fp32_error) <= fp32_limit
        case_pass = reference_pass and noise_pass and total_pass and stencil_pass and fp32_pass
        cases.append({
            "model": model, "coordinate": coordinate, "case_id": case_id,
            "case_classes": samples[0]["case_classes"], "factor": factor,
            "refinement": refinement, "seeds": sorted(seeds), "estimate": estimate,
            "combined_standard_error": combined_se, "reference_stencil": reference_stencil,
            "reference_derivative": derivative, "residual_to_stencil": residual,
            "total_error_to_derivative": total_error, "stencil_bias": stencil_bias,
            "fp32_error": fp32_error,
            "limits": {"mc_noise": noise_limit, "total_error": total_limit,
                       "stencil_bias": stencil_limit, "fp32_error": fp32_limit},
            "checks": {"reference": reference_pass, "mc_noise": noise_pass,
                       "total_error": total_pass, "stencil_bias": stencil_pass,
                       "fp32": fp32_pass},
            "pass": case_pass,
            "failure_causes": [name for name, passed in (
                ("independent_reference", reference_pass), ("monte_carlo_residual", noise_pass),
                ("total_error", total_pass), ("finite_difference_stencil_bias", stencil_pass),
                ("fp32_closed_form", fp32_pass)) if not passed],
        })
    paired_grids = {}
    for case in cases:
        key = (case["model"], case["coordinate"], case["case_id"], case["factor"])
        paired_grids.setdefault(key, {})[case["refinement"]] = case
    for refinements in paired_grids.values():
        if set(refinements) != {1, 2}:
            continue
        coarse, fine = refinements[1], refinements[2]
        shift = coarse["estimate"] - fine["estimate"]
        shift_se = math.hypot(coarse["combined_standard_error"], fine["combined_standard_error"])
        diagnostic = {
            "coarse_minus_fine": shift,
            "conservative_standard_error": shift_se,
            "significant_beyond_mc": abs(shift) > criteria["monte_carlo_standard_errors"] * shift_se,
            "fine_grid_closer_to_derivative": abs(fine["total_error_to_derivative"])
                < abs(coarse["total_error_to_derivative"]),
        }
        coarse["time_discretization_diagnostic"] = diagnostic
        fine["time_discretization_diagnostic"] = diagnostic
    required_classes = set(policy["domain_policy"]["required_case_classes"])
    candidates = []
    recommendations = []
    factors = policy["execution"]["bump_factors"]
    for model in policy["scope"]["models"]:
        for coordinate in policy["base_bumps"][model]:
            passing = []
            for factor in factors:
                selected = [case for case in cases if case["model"] == model
                            and case["coordinate"] == coordinate and case["factor"] == factor]
                covered = {item for case in selected for item in case["case_classes"]}
                complete = bool(selected) and required_classes <= covered
                passed = complete and all(case["pass"] for case in selected)
                candidates.append({"model": model, "coordinate": coordinate, "factor": factor,
                                   "case_count": len(selected), "covered_case_classes": sorted(covered),
                                   "complete": complete, "pass": passed,
                                   "failed_case_count": sum(not case["pass"] for case in selected)})
                if passed:
                    passing.append(factor)
            recommendations.append({"model": model, "coordinate": coordinate,
                                    "selected_factor": passing[0] if passing else None,
                                    "passing_factors": passing,
                                    "status": "qualified_candidate" if passing else "unqualified"})
    execution_eligible = bool(analysis.get("execution_eligible"))
    if not execution_eligible:
        for recommendation in recommendations:
            recommendation["status"] = "ineligible_execution"
            recommendation["selected_factor"] = None
    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "policy_status": policy["status"],
        "scope": "bump policy evidence; not catalogue dataset certification",
        "execution_eligible": execution_eligible,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": sum(case["pass"] for case in cases),
            "failed_case_count": sum(not case["pass"] for case in cases),
            "qualified_coordinate_count": sum(r["status"] == "qualified_candidate" for r in recommendations),
            "coordinate_count": len(recommendations),
            "failure_cause_counts": {
                cause: sum(cause in case["failure_causes"] for case in cases)
                for cause in ("independent_reference", "monte_carlo_residual", "total_error",
                              "finite_difference_stencil_bias", "fp32_closed_form")
            },
            "significant_time_grid_shift_count": sum(
                case.get("time_discretization_diagnostic", {}).get("significant_beyond_mc", False)
                for case in cases if case["refinement"] == 1),
        },
        "recommendations": recommendations,
        "candidates": candidates,
        "cases": cases,
        "domain_policy": policy["domain_policy"],
    }


def main(policy_path, analysis_path, output_path):
    policy = json.loads(policy_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    policy_fingerprint = _sha256(policy_path)
    if analysis.get("fingerprints", {}).get("policy") != policy_fingerprint:
        raise ValueError("analysis does not fingerprint the supplied policy")
    native_fingerprint = analysis.get("fingerprints", {}).get("native")
    if not isinstance(native_fingerprint, str) or not native_fingerprint.startswith("sha256:"):
        raise ValueError("analysis lacks its native-campaign fingerprint")
    report = qualify(policy, analysis)
    report["fingerprints"] = {"policy": policy_fingerprint, "native": native_fingerprint,
                              "analysis": _sha256(analysis_path)}
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.policy, args.analysis, args.output)
