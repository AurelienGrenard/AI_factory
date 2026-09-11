#!/usr/bin/env python3
"""Check GPU CIR samples or discount chains against independent expectations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import QuantLib as ql

from tools.performance.cir_forward_reference import bond, forward_transition
from validation.quantlib.model.fixed_income.cir.reference import quantlib_model


def check_discount_chains(path, models):
    """A single long Q step checks rate moments only, not its coarse trapezoid."""
    checks = []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cases = [row for row in rows if row.get('kind') == 'result']
    for row in cases:
        model = models[row['source_index']]['parameters']
        k, theta, sigma, initial = (model[key] for key in
            ('mean_reversion', 'long_term_mean', 'volatility', 'initial_state'))
        decay = math.exp(-k*row['horizon'])
        mean = theta + (initial-theta)*decay
        variance = (initial*sigma*sigma*decay*(1-decay)/k
                    + theta*sigma*sigma*(1-decay)**2/(2*k))
        expected = {'rate': mean, 'rate_squared': mean*mean+variance}
        if row['steps'] > 1:
            for precision in ('fp32', 'fp64'):
                expected['discount_'+precision] = float(bond(model, initial, row['horizon']))
                expected['bond_'+precision] = float(bond(model, initial, row['horizon']+2))
        for field, target in expected.items():
            measured = row[field]
            difference = measured['mean']-target
            tolerance = 5*measured['standard_error']+2e-6
            checks.append({'source_index':row['source_index'], 'steps':row['steps'],
                'field':field, 'expected':target, **measured, 'difference':difference,
                'z_score':difference/measured['standard_error'] if measured['standard_error'] else None,
                'tolerance':tolerance, 'pass':math.isfinite(difference) and abs(difference)<=tolerance})
    return cases, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--discount-chains', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    models = json.loads((root/'datasets/model/fixed_income/cir/parameters/cir_01.json').read_text())['models']
    if args.discount_chains:
        cases, checks = check_discount_chains(args.samples, models)
        payload = {'scope':'diagnose production Q chains, not forward-method acceptance',
                   'cases':len(cases), 'checks':len(checks),
                   'failed':sum(not c['pass'] for c in checks),
                   'source_sha256':hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                   'paired_discount_differences':[{'source_index':r['source_index'],
                       'steps':r['steps'], **r['discount_difference']} for r in cases],
                   'details':checks}
        args.output.write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps({k:payload[k] for k in ('scope','cases','checks','failed')}))
        for check in checks:
            if not check['pass']: print(json.dumps(check))
        return int(payload['failed'] != 0)
    samples = json.loads(args.samples.read_text())
    checks = []
    for row in samples['samples']:
        model = models[row['source_index']]['parameters']
        dt, remaining = row['interval'], row['remaining']
        scale, degree, loading = forward_transition(model, dt, remaining)
        mean = scale*(degree+loading*model['initial_state'])
        second = mean*mean+2*scale*scale*(degree+2*loading*model['initial_state'])
        reference = quantlib_model(model, None, {})
        expected = {'rate':mean, 'rate_squared':second,
            'bond':float(bond(model, model['initial_state'], dt+remaining+2)),
            'call':reference.discountBondOption(ql.Option.Call, float(np.float32(.8)), dt, dt+remaining+2)}
        for field, target in expected.items():
            measured = row[field]
            tolerance = 5*measured['standard_error']+2e-6*max(1, abs(target))
            difference = measured['mean']-target
            checks.append({'source_index':row['source_index'], 'interval':dt, 'remaining':remaining,
                'field':field, 'expected':target, **measured, 'difference':difference,
                'tolerance':tolerance, 'pass':bool(math.isfinite(difference) and abs(difference)<=tolerance)})
    payload={'cases':len(samples['samples']), 'checks':len(checks),
             'failed':sum(not c['pass'] for c in checks), 'details':checks}
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='details'}))
    for check in checks:
        if not check['pass']: print(json.dumps(check))
    return int(payload['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
