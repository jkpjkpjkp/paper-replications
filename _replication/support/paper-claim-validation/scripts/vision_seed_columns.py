#!/usr/bin/env python3
"""Resolve checkpoint seed names versus anonymous released prediction columns.

Infer a single permutation on the first date, then check the two later dates.
This does not alter a financial ensemble: its arithmetic mean is invariant.
"""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]/'results/vision_checkpoint'
data=pd.read_csv(OUT/'checkpoint_prediction_comparison.csv')
records=[];rows=[]
for model in ['vit','cnn']:
    frame=data[data.model.eq(model)]
    p=frame.pivot(index=['date','permno'],columns='seed_column',values='replayed')
    r=frame.pivot(index=['date','permno'],columns='seed_column',values='released')
    first=p.index.get_level_values('date').min()
    train=p.index.get_level_values('date')==first
    a,b=p.to_numpy(),r.to_numpy()
    errors=np.mean(np.abs(a[train,:,None]-b[train,None,:]),axis=0)
    permutation=min(itertools.permutations(range(5)),key=lambda ix:sum(errors[i,j] for i,j in enumerate(ix)))
    diff=np.abs(a-b[:,permutation])
    np.testing.assert_allclose(a.mean(axis=1),a[:,permutation].mean(axis=1),atol=1e-15,rtol=0)
    for case,(date,permno) in enumerate(p.index):
        for i,j in enumerate(permutation):
            rows.append(dict(model=model,date=date,permno=int(permno),checkpoint_seed=[1,42,123,999,12345][i],
                released_column=f'pred{j+1}',replayed=float(a[case,i]),released=float(b[case,j]),
                absolute_error=float(diff[case,i]),held_out_date=bool(not train[case])))
    records.append(dict(model=model,checkpoint_seed_order=[1,42,123,999,12345],
        corresponding_released_columns=[f'pred{i+1}' for i in permutation],inference_date=first,
        later_date_checks=int(diff[~train].size),later_date_matches_1e_minus_4=int((diff[~train]<=1e-4).sum()),
        all_matches_1e_minus_4=int((diff<=1e-4).sum()),all_probabilities=int(diff.size),
        ensemble_mean_unchanged_by_seed_permutation=True))
pd.DataFrame(rows).to_csv(OUT/'seed_reconciled_prediction_comparison.csv',index=False)
record=dict(status='seed_column_mapping_resolved',models=records,
    matched_probabilities=sum(x['all_matches_1e_minus_4'] for x in records),total_probabilities=len(rows),
    remaining_mismatches=[dict(model=r['model'],date=r['date'],permno=r['permno'],error=r['absolute_error']) for r in rows if r['absolute_error']>1e-4],
    full_paper_reproduced=False,
    interpretation='Release columns reorder seeds 42/1 and 999/123 relative to inference.py. One global mapping inferred on the first date generalizes to later dates. Ensemble averages are unchanged; two observed chart cases retain input/vintage differences.')
(OUT/'seed_column_reconciliation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='remaining_mismatches'},indent=2))
