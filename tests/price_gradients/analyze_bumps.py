"""Build independent price, derivative, stencil, FP32 and MC diagnostics.

This is an explicit reference-regeneration command. It may import QuantLib
for CEV and the SciPy-backed Riccati/Fourier implementation for Heston. Routine
qualification tests never invoke either external reference engine.
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from validation.volterra.rough_heston import (
    RoughHestonParameters, ExponentialKernel, lifted_heston_european_option_price,
)


def bs_values(s):
    m = s["model"]
    spot, r, q = m["model.spot"], m["model.risk_free_rate"], m["model.dividend_yield"]
    sigma, t, strike = m["model.volatility"], s["maturity"], s["strike"]
    d1 = (math.log(spot / strike) + (r - q + .5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    cdf = lambda x: .5 * math.erfc(-x / math.sqrt(2))
    dq, dr = math.exp(-q * t), math.exp(-r * t)
    density = math.exp(-.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return spot * dq * cdf(d1) - strike * dr * cdf(d2), {
        "model.spot": dq * cdf(d1),
        "model.volatility": spot * dq * density * math.sqrt(t),
        "product.strike": -dr * cdf(d2),
        "model.risk_free_rate": strike * t * dr * cdf(d2),
        "model.dividend_yield": -spot * t * dq * cdf(d1),
        "product.maturity_years": spot * dq * density * sigma / (2 * math.sqrt(t))
        + r * strike * dr * cdf(d2) - q * spot * dq * cdf(d1),
    }


@functools.lru_cache(maxsize=None)
def reference(serialized, model, fine):
    s = json.loads(serialized)
    if model == "black_scholes":
        return bs_values(s)[0]
    m = s["model"]
    if model == "heston":
        p = RoughHestonParameters(
            m["model.spot"], m["model.risk_free_rate"], m["model.dividend_yield"],
            m["model.initial_variance"], m["model.kappa"], m["model.kappa"] * m["model.theta"],
            m["model.gamma"], .5, m["model.rho"],
        )
        return lifted_heston_european_option_price(
            p, ExponentialKernel((0.,), (1.,), (p.initial_variance,)), s["strike"], s["maturity"],
            "call", 640. if fine else 320., 6401 if fine else 3201,
        )
    if model == "cev":
        import QuantLib as ql
        from validation.quantlib.model.equity.cev.european_option import continuous_time_price
        qm = {"spot": m["model.spot"], "risk_free_rate": m["model.risk_free_rate"],
              "dividend_yield": m["model.dividend_yield"], "sigma": m["model.sigma"],
              "beta": m["model.beta"]}
        return continuous_time_price(
            qm, {"strike": s["strike"], "maturity": s["maturity"]}, ql.Option.Call)
    raise ValueError(f"unsupported independent reference model: {model}")


def stencil_value(stencil, prices):
    y0, y1, y2 = prices
    if stencil["kind"] == 0:
        return (y2 - y1) / stencil["width"]
    a, b = stencil["first"] - stencil["central"], stencil["second"] - stencil["central"]
    return b / (a * (b - a)) * (y1 - y0) - a / (b * (b - a)) * (y2 - y0)


def coordinate_value(scenario, coordinate):
    if coordinate.startswith("model."):
        return scenario["model"][coordinate]
    if coordinate == "product.strike":
        return scenario["strike"]
    if coordinate == "product.maturity_years":
        return scenario["maturity"]
    raise ValueError(f"unknown coordinate {coordinate}")


def with_coordinate(scenario, coordinate, value):
    bumped = copy.deepcopy(scenario)
    if coordinate.startswith("model."):
        bumped["model"][coordinate] = value
    elif coordinate == "product.strike":
        bumped["strike"] = value
    elif coordinate == "product.maturity_years":
        bumped["maturity"] = value
    else:
        raise ValueError(f"unknown coordinate {coordinate}")
    return bumped


def reference_key(scenario):
    """Serialize only inputs consumed by the independent terminal-price engines."""
    return json.dumps({"model": scenario["model"], "strike": scenario["strike"],
                       "maturity": scenario["maturity"]}, sort_keys=True)


def valid_scenario(model, scenario):
    m = scenario["model"]
    if not (scenario["strike"] > 0 and scenario["maturity"] > 0 and m["model.spot"] > 0):
        return False
    if model == "black_scholes":
        return m["model.volatility"] > 0
    if model == "heston":
        return (m["model.initial_variance"] >= 0 and m["model.kappa"] > 0
                and m["model.theta"] > 0 and m["model.gamma"] > 0
                and -1 <= m["model.rho"] <= 1)
    if model == "cev":
        return m["model.sigma"] > 0 and .5 <= m["model.beta"] < 1
    return False


def finite_difference_reference(model, scenario, coordinate, bump):
    """Return a fine independent derivative and the h-to-h/2 convergence gap."""
    if model == "black_scholes":
        return bs_values(scenario)[1][coordinate], 0.0, "analytic"
    central = coordinate_value(scenario, coordinate)
    requested = float(bump["reference_displacement"])
    h = requested * abs(central) if bump["scale"] == "relative" else requested
    if not h > 0:
        raise ValueError(f"non-positive reference bump for {model} {coordinate}")

    def derivative(step):
        lower = with_coordinate(scenario, coordinate, central - step)
        upper = with_coordinate(scenario, coordinate, central + step)
        if valid_scenario(model, lower) and valid_scenario(model, upper):
            return ((reference(reference_key(upper), model, True)
                     - reference(reference_key(lower), model, True)) / (2 * step), "centered")
        upper2 = with_coordinate(scenario, coordinate, central + 2 * step)
        if valid_scenario(model, upper) and valid_scenario(model, upper2):
            f0 = reference(reference_key(scenario), model, True)
            f1 = reference(reference_key(upper), model, True)
            f2 = reference(reference_key(upper2), model, True)
            return (-3 * f0 + 4 * f1 - f2) / (2 * step), "forward_order2"
        lower2 = with_coordinate(scenario, coordinate, central - 2 * step)
        if valid_scenario(model, lower) and valid_scenario(model, lower2):
            f0 = reference(reference_key(scenario), model, True)
            f1 = reference(reference_key(lower), model, True)
            f2 = reference(reference_key(lower2), model, True)
            return (3 * f0 - 4 * f1 + f2) / (2 * step), "backward_order2"
        raise ValueError(f"no independent derivative stencil for {model} {coordinate}")

    coarse, coarse_kind = derivative(h)
    fine, fine_kind = derivative(h / 2)
    if coarse_kind != fine_kind:
        raise ValueError(f"reference derivative changed stencil for {model} {coordinate}")
    return fine, abs(fine - coarse), fine_kind


def _validate_surface(native_records, policy):
    if not native_records:
        raise ValueError("empty native bump campaign")
    expected_models = set(policy["scope"]["models"])
    observed_models = {record["model"] for record in native_records}
    if observed_models != expected_models:
        raise ValueError(f"model surface mismatch: expected {expected_models}, got {observed_models}")
    expected_keys = {
        (model, refinement, factor, seed)
        for model in policy["scope"]["models"]
        for refinement in policy["execution"]["refinements"][model]
        for factor in policy["execution"]["bump_factors"]
        for seed in policy["execution"]["seeds"]
    }
    observed_keys = [(record["model"], record["refinement"], record["factor"], record["seed"])
                     for record in native_records]
    if len(observed_keys) != len(set(observed_keys)):
        raise ValueError("duplicate native model/refinement/factor/seed execution")
    if set(observed_keys) != expected_keys:
        missing, extra = expected_keys - set(observed_keys), set(observed_keys) - expected_keys
        raise ValueError(f"native execution surface mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for record in native_records:
        model = record["model"]
        if record.get("schema_version") != policy["execution"]["native_schema_version"]:
            raise ValueError("native campaign schema mismatch")
        if record.get("policy_id") != policy["policy_id"]:
            raise ValueError("native policy identity mismatch")
        if (len(record["coordinates"]) != len(set(record["coordinates"]))
                or set(record["coordinates"]) != set(policy["base_bumps"][model])):
            raise ValueError(f"coordinate surface mismatch for {model}")
        expected_cases = policy["cases"][model]
        if record.get("case_ids") != [case["id"] for case in expected_cases]:
            raise ValueError(f"case identity mismatch for {model}")
        if record.get("case_classes") != [case["classes"] for case in expected_cases]:
            raise ValueError(f"case class mismatch for {model}")
        rows, coordinates = len(expected_cases), len(policy["base_bumps"][model])
        if record.get("rows") != rows or record.get("k") != coordinates:
            raise ValueError(f"native row/coordinate cardinality mismatch for {model}")
        if len(record["scenarios"]) != rows * (1 + 2 * coordinates):
            raise ValueError(f"native scenario cardinality mismatch for {model}")
        if len(record["stencils"]) != rows * coordinates:
            raise ValueError(f"native stencil cardinality mismatch for {model}")
        for field, count in (("price", rows), ("price_se", rows),
                             ("gradient", rows * coordinates),
                             ("gradient_se", rows * coordinates)):
            if len(record[field]) != count:
                raise ValueError(f"native {field} cardinality mismatch for {model}")
        execution = policy["execution"]
        if (record["paths"] != execution["paths_per_price"]
                or record.get("threads_per_block") != execution["threads_per_block"]
                or record.get("sensitivity_batch_size") != execution["sensitivity_batch_size"]):
            raise ValueError(f"native launch configuration mismatch for {model}")
        expected_dt = 1.0 / (504.0 * record["refinement"])
        if not math.isclose(record["dt"], expected_dt, rel_tol=0.0, abs_tol=2e-10):
            raise ValueError(f"native time grid mismatch for {model}")
    if {record["factor"] for record in native_records} != set(policy["execution"]["bump_factors"]):
        raise ValueError("bump-factor surface mismatch")
    if {record["seed"] for record in native_records} != set(policy["execution"]["seeds"]):
        raise ValueError("seed surface mismatch")


def analyze(source, output, policy_path=None):
    native_records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    policy = json.loads(policy_path.read_text()) if policy_path else None
    if policy:
        _validate_surface(native_records, policy)
    records = []
    execution_eligible = True
    for native in native_records:
        k, model = native["k"], native["model"]
        n = 1 + 2 * k
        if policy and native["paths"] != policy["execution"]["paths_per_price"]:
            execution_eligible = False
        for row in range(native["rows"]):
            scenarios = native["scenarios"][row * n:(row + 1) * n]
            analytic_reference = model in ("black_scholes", "cev")
            fine = [reference(reference_key(s), model, True) for s in scenarios]
            coarse = fine if analytic_reference else [reference(reference_key(s), model, False) for s in scenarios]
            for i, name in enumerate(native["coordinates"]):
                index = row * k + i
                stencil = native["stencils"][index]
                points = [0, 2 * i + 1, 2 * i + 2]
                exact = stencil_value(stencil, [fine[p] for p in points])
                coarse_gradient = stencil_value(stencil, [coarse[p] for p in points])
                convergence = abs(exact - coarse_gradient)
                price_gap = max(abs(fine[p] - coarse[p]) for p in points)
                derivative = derivative_convergence = derivative_stencil = None
                if policy:
                    derivative, derivative_convergence, derivative_stencil = finite_difference_reference(
                        model, scenarios[0], name, policy["base_bumps"][model][name])
                elif model == "black_scholes":
                    derivative = bs_values(scenarios[0])[1][name]
                    derivative_convergence, derivative_stencil = 0.0, "analytic"
                case_ids = native.get("case_ids", [str(j) for j in range(native["rows"])])
                case_classes = native.get("case_classes", [[] for _ in range(native["rows"])])
                item = {key: native[key] for key in ("model", "factor", "refinement", "seed", "paths",
                                                     "dt", "threads_per_block", "sensitivity_batch_size")}
                item.update(
                    row=row, coordinate=name, stencil=stencil, case_id=case_ids[row],
                    case_classes=case_classes[row], mc=native["gradient"][index],
                    mc_se=native["gradient_se"][index], reference_stencil=exact,
                    reference_price=fine[0], native_price=native["price"][row],
                    price_se=native["price_se"][row], reference_price_convergence=price_gap,
                    reference_gradient_convergence=convergence, derivative=derivative,
                    reference_qualified=price_gap <= 2e-7 and convergence <= 1e-3,
                    reference_derivative_convergence=derivative_convergence,
                    reference_derivative_stencil=derivative_stencil,
                    reference_convergence_applicable=not analytic_reference,
                    stencil_bias=None if derivative is None else exact - derivative,
                    residual=native["gradient"][index] - exact,
                )
                if model == "black_scholes":
                    item.update(cf=native["cf_gradient"][index], fp32_error=native["cf_gradient"][index] - exact)
                records.append(item)
        print(model, native["factor"], native["refinement"], native["seed"], "analyzed", flush=True)
    payload = {
        "schema_version": 2,
        "scope": "bounded bump qualification input; no dataset certification",
        "policy_id": policy["policy_id"] if policy else None,
        "execution_eligible": execution_eligible,
        "reference": policy["reference"] if policy else
            "BS analytic FP64; Heston constant-kernel Riccati/Fourier, cutoff 320/640 and 3201/6401 nodes",
        "criteria": policy["criteria"] if policy else {
            "reference_price_convergence_absolute": 2e-7,
            "reference_gradient_convergence_absolute": 1e-3,
        },
        "fingerprints": {
            "native": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
            "policy": None if policy_path is None else
                "sha256:" + hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        },
        "rows": records,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    analyze(args.source, args.output, args.policy)
