"""Summarize the --rates fixture without discarding outliers or changing precision."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(native_path, analysis_path, output_path):
    native=[json.loads(line) for line in native_path.read_text().splitlines()]
    expected={(model,bp,refinement,seed) for model in ('black_scholes','heston')
        for bp in (1.,2.,5.,10.) for refinement in ((1,) if model=='black_scholes' else (1,2))
        for seed in (719,2719,4719)}
    assert len(native)==len(expected)
    assert {(x['model'],x['factor'],x['refinement'],x['seed']) for x in native}==expected
    central={}
    for x in native:
        assert x['coordinates']==['model.risk_free_rate'] and x['k']==1
        assert x['requested_bump_basis_points']==x['factor']
        assert x['rows']==(6 if x['model']=='black_scholes' else 5)
        key=(x['model'],x['refinement'],x['seed'])
        values=(x['price'],x['price_se'])
        assert central.setdefault(key,values)==values, 'Bump changed central price/moments'
    analysis=json.loads(analysis_path.read_text())
    groups={}
    for x in analysis['rows']:
        groups.setdefault((x['model'],x['row'],x['factor'],x['refinement']),[]).append(x)
    assert len(groups)==64
    cases=[]
    for (model,row,bp,refinement),samples in sorted(groups.items()):
        assert len(samples)==3 and {x['seed'] for x in samples}=={719,2719,4719}
        reference=samples[0]['reference_stencil']
        assert all(x['reference_stencil']==reference for x in samples)
        mean=statistics.mean(x['mc'] for x in samples)
        se=math.sqrt(sum(x['mc_se']**2 for x in samples))/3
        residual=mean-reference
        case={'model':model,'row':row,'basis_points':bp,'refinement':refinement,
            'reference_qualified':all(x['reference_qualified'] for x in samples),
            'mc':mean,'reference':reference,'combined_se':se,'residual':residual,
            'relative_error':None if reference==0 else residual/abs(reference),
            'absolute_z':None if se==0 else abs(residual)/se}
        if model=='black_scholes':
            case.update(cf=samples[0]['cf'],derivative=samples[0]['derivative'],
                stencil_bias=samples[0]['stencil_bias'],fp32_error=samples[0]['fp32_error'])
        cases.append(case)
    summary=[]
    for model in ('black_scholes','heston'):
        for bp in (1,2,5,10):
            selected=[x for x in cases if x['model']==model and x['basis_points']==bp]
            summary.append({'model':model,'basis_points':bp,'cases':len(selected),
                'unqualified_references':sum(not x['reference_qualified'] for x in selected),
                'maximum_absolute_relative_error':max(abs(x['relative_error']) for x in selected),
                'maximum_absolute_z':max(x['absolute_z'] for x in selected),
                'cases_above_6se':sum(x['absolute_z']>6 for x in selected)})
    output_path.write_text(json.dumps({'central_prices_invariant':True,'scope':'bounded rate fixture',
        'summary':summary,'cases':cases},indent=2,allow_nan=False)+'\n')
    for row in summary: print(json.dumps(row))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native',type=Path);parser.add_argument('analysis',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();summarize(args.native,args.analysis,args.output)
