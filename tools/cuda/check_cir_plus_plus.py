#!/usr/bin/env python3
"""Explicit independent CIR++ diagnostics; never certifies or modifies datasets.

Read JSON lines emitted by test_cir_plus_plus_cuda. Use native QuantLib CIR
bonds/options and the published deterministic-shift identity; Bermudans use
the independent risk-neutral finite-difference solver at two resolutions.
Premia CirPP1D candidates are not declared unavailable: publication still
requires their method audit and the full persistent-reference hierarchy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import QuantLib as ql

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from validation.quantlib.model.fixed_income.cir.reference import quantlib_model
from validation.quantlib.swaption import swaption_price
from validation.quantlib.term_structure import nelson_siegel_discount, svensson_discount
from tools.performance.cir_forward_reference import pde_price


class ShiftedCirReference:
    """Compose the established independent CIR pricer and an FP64 input curve."""

    def __init__(self, factor, curve, curve_name):
        self.factor = factor
        self.base = quantlib_model(factor, None, {})
        evaluator = nelson_siegel_discount if curve_name == "nelson_siegel" else svensson_discount
        self.discount = lambda time: evaluator(curve, time)

    def shift_discount(self, time):
        return self.discount(time) / self.base.discountBond(0., time, self.factor["initial_state"])

    def discountBond(self, time, maturity, state):
        return (self.shift_discount(maturity) / self.shift_discount(time)
                * self.base.discountBond(time, maturity, state))

    def conditional_option(self, kind, strike, time, expiry, maturity, state):
        conditional = quantlib_model(dict(self.factor, initial_state=state), None, {})
        adjusted = strike * self.shift_discount(expiry) / self.shift_discount(maturity)
        return (self.shift_discount(maturity) / self.shift_discount(time)
                * conditional.discountBondOption(kind, adjusted, expiry-time, maturity-time))

    def discountBondOption(self, kind, strike, expiry, maturity):
        return self.conditional_option(kind, strike, 0., expiry, maturity, self.factor["initial_state"])


def check_analytics(row):
    ref = ShiftedCirReference(row["model"],row["curve_parameters"],row["curve"])
    product = dict(notional=1.,strike=.035,accrual_fraction=.5,exercise_time=2.,
                   payment_interval=.5,payment_count=8)
    expected = {
        "conditional_bond":ref.discountBond(2.,9.,.08),
        "call":ref.conditional_option(ql.Option.Call,.8,2.,5.,9.,.08),
        "put":ref.conditional_option(ql.Option.Put,.8,2.,5.,9.,.08),
        "payer":swaption_price(ref,product,"payer"),
        "receiver":swaption_price(ref,product,"receiver"),
    }
    errors = {name:abs(row[name]-value) for name,value in expected.items()}
    for name,value in expected.items():
        allowance = 5e-7 + 5e-5*abs(value)
        if not math.isfinite(row[name]) or errors[name] > allowance:
            raise AssertionError(f"{row['curve']} {name}: CUDA={row[name]}, reference={value}, "
                                 f"error={errors[name]}, allowance={allowance}, model={row['model']}")
    return max(errors.values())


def check_bermudan(row):
    curve = dict(beta0=.03,beta1=-.01,beta2=.02,tau=2.)
    if row["curve"] == "svensson":
        curve = dict(beta0=.03,beta1=-.01,beta2=.02,beta3=.01,tau1=2.,tau2=5.)
    ref = ShiftedCirReference(row["model"],curve,row["curve"])
    product = dict(notional=1.,strike=.035,accrual_fraction=.5,
                   first_exercise_time=126,payment_interval=126,payment_count=6,exercise_count=4)

    def transformed_obstacle(states, exercise):
        time = .5 + .5*exercise
        coupon_bond = np.zeros_like(states)
        for payment in range(1,7-exercise):
            maturity = time+.5*payment
            a = ref.discountBond(time,maturity,0.)
            b = math.log(a/ref.discountBond(time,maturity,1.))
            weight = .035*.5 + (1. if payment==6-exercise else 0.)
            coupon_bond += weight*a*np.exp(-b*states)
        swap = 1.-coupon_bond
        return ref.shift_discount(time)*np.maximum(swap if row["side"]=="payer" else -swap,0.)

    values = [pde_price(row["model"],product,row["side"],nodes=n,steps_per_year=s,
                        payoff=transformed_obstacle)
              for n,s in ((1024,256),(2048,512))]
    discretization = abs(values[1]-values[0])
    if discretization > 2e-6:
        raise AssertionError(f"CIR++ PDE did not converge: {values}, {row}")
    allowance = 5*row["standard_error"] + 2*discretization + 5e-7
    # LSM has an exercise-policy approximation; keep its signed PDE gap visible.
    gap = row["price"]-values[1]
    if gap > allowance or gap < -allowance-5e-4*max(values[1],1e-3):
        raise AssertionError(f"CIR++ LSM/PDE discrepancy: gap={gap}, allowance={allowance}, {row}")
    return {"curve":row["curve"],"row":row["row"],"side":row["side"],
            "paths":row["paths"],"cuda":row["price"],"pde":values[1],
            "pde_refinement_delta":discretization,"signed_gap":gap,"allowance":allowance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path,help="JSON-lines output of test_cir_plus_plus_cuda")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    analytics = [row for row in rows if row["kind"]=="analytics"]
    bermudans = [row for row in rows if row["kind"]=="bermudan"]
    if len(analytics)!=48 or len(bermudans)!=12:
        raise ValueError("Expected all 48 analytics cases and 12 Bermudan cases.")
    maximum = max(check_analytics(row) for row in analytics)
    results = [check_bermudan(row) for row in bermudans]
    print(json.dumps({"quantlib_version":ql.__version__,"analytical_comparisons":5*len(analytics),
                      "maximum_analytical_absolute_error":maximum,"bermudan_comparisons":results},indent=2))


if __name__ == "__main__":
    main()
