"""Summarize pre-specified diagnostics without selecting a replacement result."""
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduction/discrepancy40'
sys.path.insert(0,str(ROOT/'reproduction'))
from audit_results import TARGETS

def metrics_table(folder):
    rows=[]
    for p in sorted((folder/'results').glob('dp40_*test_metrics.csv')):
        row=pd.read_csv(p).iloc[0].to_dict()
        row['run']=int(re.search(r'_(\d+)_test_metrics.csv$',p.name)[1])
        rows.append(row)
    return pd.DataFrame(rows).set_index('run')

def epochs(log):
    out=[]
    for part in log.split('>>>>>>>start training :')[1:]:
        e=[]
        for block in re.split(r'(?=\[Epoch \d+\])',part)[1:]:
            m=re.search(r'\[Epoch (\d+)\] mean_loss=([\d.]+).*?Vali_CVaR=([\d.]+) \| Test_CVaR=([\d.]+)',block,re.S)
            if m:e.append(dict(epoch=int(m[1]),train=float(m[2]),val=float(m[3]),test=float(m[4]),saved='Saving model' in block))
        out.append(e)
    return out

def compare_states(a,b):
    x=torch.load(a,map_location='cpu',weights_only=True)
    y=torch.load(b,map_location='cpu',weights_only=True)
    assert x.keys()==y.keys()
    return dict(tensors=len(x),bitwise_equal=all(torch.equal(x[k],y[k]) for k in x),
                max_abs_difference=max(float((x[k]-y[k]).abs().max()) for k in x))

primary=ROOT/'reproduction/fresh_local'
primary_metrics=metrics_table(primary)
primary_epochs=epochs((ROOT/'reproduction/logs/fresh_local_dp40.log').read_text())
result={}
all_rows=[]
for case,n in [('repeat_released',3),('longer_patience_first_run',1),('tf32_released',3)]:
    folder=OUT/case
    assert (folder/'finished.txt').exists(),case
    table=metrics_table(folder)
    assert len(table)==n
    case_epochs=epochs((folder/'training.log').read_text())
    comparisons=[]
    for run in table.index:
        name=next((folder/'checkpoints').glob(f'dp40_*_{run}')).name
        states=compare_states(primary/'checkpoints'/name/'checkpoint.pth',folder/'checkpoints'/name/'checkpoint.pth')
        a=pd.read_csv(primary/'results'/f'{name}_test_positions.csv')
        b=pd.read_csv(folder/'results'/f'{name}_test_positions.csv')
        assert a.Date.equals(b.Date)
        comparisons.append(dict(run=int(run),checkpoint=states,
            max_weight_difference=float(np.max(np.abs(a.filter(like='Weight_').values-b.filter(like='Weight_').values))),
            max_metric_difference=float((primary_metrics.loc[run]-table.loc[run]).abs().max()),
            common_training_prefix_identical_at_log_precision=case_epochs[run][:len(primary_epochs[run])]==primary_epochs[run],
            best_epoch=max(e['epoch'] for e in case_epochs[run] if e['saved']),
            stopped_epoch=case_epochs[run][-1]['epoch']))
    values={}
    matches=0
    for metric,(target_mean,target_sd) in TARGETS[40].items():
        mean=float(table[metric].mean());sd=float(table[metric].std()) if n>1 else None
        values[metric]=dict(mean=mean,sample_sd=sd,paper_mean=target_mean,paper_sd=target_sd)
        if n>1:matches+=int(abs(mean-target_mean)<=.00005)+int(abs(sd-target_sd)<=.00005)
    elapsed=(dt.datetime.fromisoformat((folder/'finished.txt').read_text().strip())-dt.datetime.fromisoformat((folder/'started.txt').read_text().strip())).total_seconds()
    result[case]=dict(runs=n,metrics=values,comparisons_to_primary_same_index=comparisons,
                      printed_matches=matches if n>1 else None,comparison_cells=8 if n>1 else None,
                      epochs=case_epochs,elapsed_seconds=elapsed)
    for run,row in table.iterrows():all_rows.append(dict(case=case,run=int(run),**row.to_dict()))
pd.DataFrame(all_rows).to_csv(OUT/'diagnostic_metrics.csv',index=False)
(OUT/'diagnostic_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
