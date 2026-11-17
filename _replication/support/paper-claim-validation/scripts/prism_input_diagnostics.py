#!/usr/bin/env python3
"""Measure what temporal normalization can see in repeated monthly CZ inputs.

This is a posthoc input diagnostic, not an ablation or performance estimate.
The raw-feature loading branch is retained in PRISM and is not claimed to vanish.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'prism-vq-reproduction'
HERE=PROJECT/'experiments/cz_vw'
OUT=ROOT/'paper-claim-validation/results'
sys.path[:0]=[str(HERE),str(PROJECT/'scripts')]
from panel import Panel as CZPanel
from runtime import Panel as NativePanel
from module.layers.src.RevIN import RevIN


def summarize(panel,name):
    records=[]
    for t in panel.plans['test']:
        x,_,_,ids=panel.batch(t,'test')
        spread=x.amax(dim=1)-x.amin(dim=1)
        records.append(dict(dataset=name,date=str(panel.dates[t]),stocks=len(ids),
                            temporal_constant_channel_fraction=float((spread==0).float().mean())))
    return records


def main():
    torch.set_num_threads(2)
    cz=CZPanel();native=NativePanel('sp500')
    rows=summarize(cz,'CZ202')+summarize(native,'Alpha158')
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'prism_temporal_constant_channels.csv',index=False)
    # Two different stock-level vectors, repeated over time: RevIN removes
    # every level difference before the latent encoder. Use exact binary values
    # so the identity has no floating-mean rounding ambiguity.
    x=torch.stack([torch.ones(20,202),2*torch.ones(20,202)])
    norm=RevIN(202)
    with torch.no_grad():z=norm(x,mode='norm')
    torch.testing.assert_close(z[0],z[1],rtol=0,atol=0)
    counts=[]
    for month in np.unique(cz.month_index[cz.plans['test']]):
        ds=[t for t in cz.plans['test'] if cz.month_index[t]==month]
        counts.append(dict(month=int(month),single_source_month_days=sum(
            len(np.unique(cz.month_index[t-19:t+1]))==1 for t in ds),test_days=len(ds)))
    record=dict(kind='posthoc_input_diagnostic',financial_performance_test=False,
                mean_temporal_constant_channel_fraction=frame.groupby('dataset').temporal_constant_channel_fraction.mean().to_dict(),
                different_constant_stock_levels_normalize_identically=True,
                single_source_month_test_days=sum(c['single_source_month_days'] for c in counts),
                total_CZ_test_days=sum(c['test_days'] for c in counts),
                interpretation='Native temporal RevIN removes characteristic levels from the latent encoder when histories are constant. Monthly input frequency changes what that branch can learn.',
                limitation='The separate loading branch still consumes raw features; this does not prove the cause of negative alpha or establish a profitable alternative.')
    (OUT/'prism_input_diagnostic.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
