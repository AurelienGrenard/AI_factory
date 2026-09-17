#!/usr/bin/env python3
"""Compute row-preserving independent CPU evidence for the CIR experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from validation.quantlib.model.fixed_income.cir import forward_reference

held_out_lsm = forward_reference.held_out_lsm
pde_price = forward_reference.pde_price

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--side', choices=('payer', 'receiver'), default='payer')
    parser.add_argument('--nodes', type=int, default=768)
    parser.add_argument('--steps-per-year', type=int, default=128)
    parser.add_argument('--holdout-paths', type=int, default=0)
    args = parser.parse_args()
    model_path = ROOT / 'datasets/model/fixed_income/cir/parameters/cir_01.json'
    product_path = ROOT / 'datasets/product/bermudan_swaption/bermudan_swaptions_01.json'
    models = json.loads(model_path.read_text())['models']
    products = json.loads(product_path.read_text())['products']
    indices = args.indices if args.indices is not None else list(range(1000))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'method': 'risk-neutral CIR PDE, no forward-measure sampler or LSM',
        'side': args.side, 'indices': indices, 'fine_nodes': args.nodes,
        'fine_steps_per_year': args.steps_per_year, 'holdout_paths': args.holdout_paths,
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (model_path, product_path)}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'reference_source.py').write_bytes(Path(forward_reference.__file__).read_bytes())
    with (args.output / 'rows.ndjson').open('w') as stream:
        for index in indices:
            start = time.monotonic()
            model, product = models[index]['parameters'], products[index]['parameters']
            coarse = pde_price(model, product, args.side, args.nodes//2, args.steps_per_year//2)
            fine = pde_price(model, product, args.side, args.nodes, args.steps_per_year)
            row = {'source_index': index, 'model_id': models[index]['id'],
                'product_id': products[index]['id'], 'coarse_price': coarse,
                'fine_price': fine, 'mesh_difference': abs(fine-coarse)}
            if args.holdout_paths:
                row['holdout'] = held_out_lsm(model, product, args.side, args.holdout_paths, seed=902026+index)
            row['seconds'] = time.monotonic()-start
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            if len(indices) < 30 or (index+1)%50 == 0:
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
