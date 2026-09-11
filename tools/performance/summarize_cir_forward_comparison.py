#!/usr/bin/env python3
"""Compare every original CIR row without averaging away method discrepancies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def result(path: Path):
    results = [row for line in path.read_text().splitlines()
               if (row := json.loads(line)).get('event') == 'result']
    if len(results) != 1:
        raise ValueError(f'Expected exactly one complete result in {path}')
    return results[0]


def compare(old, new, pde=None):
    for field in ('indices', 'paths'):
        if old['job'][field] != new['job'][field]:
            raise ValueError(f'Mismatched {field}')
    if old['job'].get('side', 'payer') != new['job'].get('side', 'payer'):
        raise ValueError('Mismatched swaption sides')
    old_rows = {r['source_index']: r for r in old['rows']}
    new_rows = {r['source_index']: r for r in new['rows']}
    if (len(old_rows) != len(old['rows']) or len(new_rows) != len(new['rows'])
        or set(old_rows) != set(new_rows) or set(old_rows) != set(old['job']['indices'])):
        raise ValueError('Duplicate, missing or unexpected source rows')
    if pde is not None:
        if set(pde) != set(old_rows):
            raise ValueError('PDE must cover exactly the compared rows')
        if any(not math.isfinite(r[k]) or r[k] < 0
               for r in pde.values() for k in ('fine_price', 'mesh_difference')):
            raise ValueError('Invalid PDE result')
    rows = []
    for index in sorted(old_rows):
        a, b = old_rows[index], new_rows[index]
        if any(not math.isfinite(r[k]) or r[k] < 0
               for r in (a, b) for k in ('price', 'standard_error')):
            raise ValueError('Invalid numerical result')
        delta = b['price']-a['price']
        combined = math.hypot(a['standard_error'], b['standard_error'])
        budget = 5*combined+2e-6
        row = {'source_index': index, 'row_id': f'{index+1:06}',
            'regime': 'core' if index < 900 else 'stress',
            'old_price': a['price'], 'forward_price': b['price'],
            'old_standard_error': a['standard_error'], 'forward_standard_error': b['standard_error'],
            'difference': delta, 'absolute_difference': abs(delta),
            'relative_difference': delta/a['price'] if a['price'] else None,
            'combined_standard_error': combined,
            'z_score': delta/combined if combined else None,
            'screen_budget': budget, 'screen_pass': abs(delta) <= budget}
        if pde is not None:
            reference = pde[index]
            row.update(pde_price=reference['fine_price'], pde_mesh_difference=reference['mesh_difference'])
            for label, value in [('old', a), ('forward', b)]:
                row[label+'_minus_pde'] = value['price']-reference['fine_price']
                row[label+'_pde_screen_pass'] = abs(row[label+'_minus_pde']) <= (
                    5*value['standard_error']+2e-6+2*reference['mesh_difference'])
        rows.append(row)
    return rows


def job_summary(job):
    """Row indices are already preserved in the CSV; avoid three copied lists."""
    return {**{key: value for key, value in job.items() if key != 'indices'},
            'indices_source': 'rows.csv:source_index',
            'threads': job.get('threads', 128), 'blocks': job.get('blocks', 64),
            'chunk_rows': job.get('chunk_rows', 32)}


def provenance(path):
    manifest_path = path.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    return {'manifest':str(manifest_path),
            'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            **{key:manifest[key] for key in ('revision','binary_sha256','input_sha256')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--forward', type=Path, required=True)
    parser.add_argument('--pde', type=Path)
    parser.add_argument('--repeat', type=Path, help='Independent forward-method seed')
    parser.add_argument('--csv-only-rows', action='store_true', help='Avoid duplicate JSON row data')
    parser.add_argument('--overwrite', action='store_true', help='Refresh only this tool\'s generated output files')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, new = result(args.old), result(args.forward)
    pde = None
    if args.pde:
        reference_rows = [json.loads(line) for line in args.pde.read_text().splitlines()]
        pde = {r['source_index']: r for r in reference_rows}
        if len(pde) != len(reference_rows):
            raise ValueError('Duplicate PDE rows')
    rows = compare(old, new, pde)
    repetition = None
    if args.repeat:
        repetition = result(args.repeat)
        if repetition['job']['seed'] == new['job']['seed']:
            raise ValueError('The repetition must have an independent seed')
        repeat_comparison = compare(new, repetition, pde)
        for row, repeated in zip(rows, repeat_comparison):
            row.update(forward_repeat_price=repeated['forward_price'],
                       forward_repeat_standard_error=repeated['forward_standard_error'],
                       forward_seed_screen_pass=repeated['screen_pass'])
            if pde is not None:
                row['forward_repeat_pde_screen_pass'] = repeated['forward_pde_screen_pass']
    origins = {label:provenance(path) for label,path in
               [('old',args.old),('forward',args.forward),('forward_repeat',args.repeat)] if path}
    if any(origin['input_sha256'] != origins['old']['input_sha256'] for origin in origins.values()):
        raise ValueError('Compared runs do not have identical input fingerprints')
    reference_origin = None
    if args.pde:
        reference_manifest_path = args.pde.parent/'manifest.json'
        reference_manifest = json.loads(reference_manifest_path.read_text())
        if (reference_manifest['input_sha256'] != origins['old']['input_sha256']
                or reference_manifest['side'] != old['job'].get('side','payer')):
            raise ValueError('PDE inputs or side do not match the GPU runs')
        reference_origin = {key:reference_manifest[key] for key in
            ('method','side','fine_nodes','fine_steps_per_year','input_sha256')}
        reference_origin.update(manifest=str(reference_manifest_path),
            manifest_sha256=hashlib.sha256(reference_manifest_path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256((args.pde.parent/'reference_source.py').read_bytes()).hexdigest())
    summary = {'scope': 'row-wise statistical discrepancy screen, NOT independent certification',
        'old_source': str(args.old), 'forward_source': str(args.forward),
        'pde_source': str(args.pde) if args.pde else None,
        'paths_per_price': old['job']['paths'], 'row_count': len(rows),
        'side': old['job'].get('side', 'payer'),
        'provenance': origins,
        'pde_provenance': reference_origin,
        'jobs': {'old': job_summary(old['job']), 'forward': job_summary(new['job']),
                 'forward_repeat': job_summary(repetition['job']) if repetition else None},
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (args.old, args.forward, args.pde, args.repeat) if path},
        'sigma_multiplier': 5, 'absolute_rounding_floor': 2e-6,
        'old_gpu_seconds': old['gpu_seconds'], 'forward_gpu_seconds': new['gpu_seconds'],
        'old_raw_host_seconds': old['raw_host_seconds'], 'forward_raw_host_seconds': new['raw_host_seconds'],
        'old_workspace_bytes': old['workspace_bytes'], 'forward_workspace_bytes': new['workspace_bytes'],
        'timing_status': 'exploratory, not geometry qualification; CPU reference work may overlap',
        'failed_rows': [r['row_id'] for r in rows if not r['screen_pass']],
        'maximum_absolute_difference': max(r['absolute_difference'] for r in rows),
        'regimes': {regime: {'rows': sum(r['regime']==regime for r in rows),
                            'failed': sum(r['regime']==regime and not r['screen_pass'] for r in rows)}
                    for regime in ('core','stress')}}
    if pde:
        summary['pde_failed_rows'] = {method: [r['row_id'] for r in rows
            if method+'_pde_screen_pass' in r and not r[method+'_pde_screen_pass']]
            for method in ('old','forward')}
    if repetition:
        summary['forward_seed_failed_rows'] = [r['row_id'] for r in rows if not r['forward_seed_screen_pass']]
        summary['forward_repeat_gpu_seconds'] = repetition['gpu_seconds']
        summary['forward_repeat_raw_host_seconds'] = repetition['raw_host_seconds']
        if pde:
            summary['pde_failed_rows']['forward_repeat'] = [r['row_id'] for r in rows
                if not r['forward_repeat_pde_screen_pass']]
    args.output.mkdir(parents=True, exist_ok=args.overwrite)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if not args.csv_only_rows:
        (args.output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    with (args.output/'rows.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for r in rows for key in r)))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({key:summary[key] for key in
        ('side','row_count','failed_rows','old_gpu_seconds','forward_gpu_seconds')}, indent=2))


if __name__ == '__main__':
    main()
