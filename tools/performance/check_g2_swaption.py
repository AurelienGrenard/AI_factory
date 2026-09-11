#!/usr/bin/env python3
"""Check G2 CUDA rows against native QuantLib and a technical-exception MC fallback.

This opt-in numerical check never regenerates a persistent validation cache and
does not promote QuantLib above the production Premia hierarchy.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path

from validation.quantlib.g2_swaption import monte_carlo_prices, swaption_price


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    documents = {}

    def parameters(path, kind, index):
        path = root / path
        if path not in documents:
            documents[path] = json.loads(path.read_text())
        return documents[path][kind][index]['parameters'].copy()

    checks = []
    seen = set()
    mc_references = {}
    for line in args.samples.read_text().splitlines():
        row = json.loads(line)
        model_name, curve_name, index = row['model'], row['curve'], row['source_index']
        identity = (model_name, curve_name, row['side'], index)
        if identity in seen:
            raise ValueError(f'Duplicate diagnostic row: {identity}')
        seen.add(identity)
        model = parameters(f'datasets/model/fixed_income/{model_name}/parameters/{model_name}_01.json', 'models', index)
        curve = parameters(f'datasets/curve/{curve_name}/{curve_name}_01.json', 'curves', index) if curve_name else None
        product = parameters('datasets/product/european_swaption/european_swaptions_01.json', 'products', index)
        product['exercise_time'] /= 252
        product['payment_interval'] /= 252
        product['payment_count'] = row['payment_count']
        module = f'validation.quantlib.model.fixed_income.{model_name}' + (f'.{curve_name}' if curve_name else '') + '.reference'
        reference = importlib.import_module(module).quantlib_model(model, curve, product)
        error = None
        reference_method = 'QuantLib.G2SwaptionEngine'
        reference_standard_error = 0.0
        try:
            coarse = swaption_price(reference, product, row['side'], integration_points=128)
            fine = swaption_price(reference, product, row['side'], integration_points=256)
            mesh_difference = abs(fine-coarse)
            difference = row['price']-fine
            budget = 5e-7 + 5e-5*abs(fine) + 6*row['standard_error']
            passed = (math.isfinite(fine) and fine >= 0 and math.isfinite(difference)
                and math.isfinite(row['standard_error']) and row['standard_error'] >= 0
                and mesh_difference <= 1e-7*max(1, abs(fine)) and abs(difference) <= budget)
        except RuntimeError as exception:
            error = str(exception)
            # Only a technical engine exception reaches this diagnostic fallback.
            # Finite discrepancies and unconverged quadratures remain failures.
            key = (model_name, curve_name, index, row['payment_count'])
            if key not in mc_references:
                mc_references[key] = monte_carlo_prices(reference, product)
            fine, reference_standard_error = mc_references[key][row['side']]
            reference_method = 'CPU terminal-forward MC + QuantLib.discountBond'
            coarse = mesh_difference = None
            difference = row['price']-fine
            budget = 5e-7 + 5e-5*abs(fine) + 6*math.hypot(row['standard_error'],reference_standard_error)
            passed = (math.isfinite(fine) and fine >= 0 and math.isfinite(difference)
                and math.isfinite(row['standard_error']) and row['standard_error'] >= 0
                and math.isfinite(reference_standard_error) and abs(difference) <= budget)
        checks.append({**row, 'quantlib_128': coarse,
            'quantlib_256': fine if error is None else None, 'reference_price': fine,
            'reference_method': reference_method, 'reference_standard_error': reference_standard_error,
            'reference_mesh_difference': mesh_difference, 'difference': difference,
            'budget': budget, 'reference_exception': error, 'pass': passed})
    expected = {(model,curve,side,index)
        for model,curve in [('g2',''),('g2_plus_plus','nelson_siegel'),('g2_plus_plus','svensson')]
        for side in ('payer','receiver') for index in (0,100,499,899,900,950,975,999)}
    if seen != expected:
        raise ValueError(f'Incomplete or unexpected diagnostic coverage: missing={expected-seen}, extra={seen-expected}')
    result = {'scope': 'independent diagnostic, not persistent catalogue certification',
        'checks': len(checks), 'failed': sum(not c['pass'] for c in checks),
        'source_sha256': hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        'input_sha256': {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in documents},
        'details': checks}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('scope','checks','failed')}))
    for check in checks:
        if not check['pass']:
            print(json.dumps(check))
    return int(result['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
