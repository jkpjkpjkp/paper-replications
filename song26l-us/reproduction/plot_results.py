#!/usr/bin/env python3
"""Export the primary comparison as standalone scientific plots."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
report=json.loads((OUT/'metrics.json').read_text())
status=json.loads((OUT/'status.json').read_text())
if status['state']!='complete':raise RuntimeError('Plot only completed primary results')
with (OUT/'daily_metrics.csv').open() as handle:rows=list(csv.DictReader(handle))
dates=sorted({r['date'] for r in rows});methods=['standard','selected']
daily={method:np.array([np.mean([float(r['ic']) for r in rows if r['method']==method and r['date']==date]) for date in dates]) for method in methods}
weights=[];horizons=[]
for seed in status['protocol']['seeds']:
    selection=json.loads((OUT/f'seed_{seed}/horizon_selection.json').read_text())
    weights.append(selection['weights']);horizons.append(selection['selected_horizon'])
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for method,label,color in [('standard','Target-horizon baseline','#59687b'),('selected','BLO-selected horizon','#176f8a')]:
    axes[0,0].plot(np.arange(len(dates))+1,np.cumsum(daily[method])/(np.arange(len(dates))+1),label=label,color=color)
axes[0,0].set(xlabel='Test trading session',ylabel='Cumulative mean daily IC',title='Same next-day-close evaluation target');axes[0,0].legend(frameon=False)
weights=np.array(weights)
axes[0,1].plot(np.arange(1,391),weights.mean(0),color='#176f8a')
axes[0,1].fill_between(np.arange(1,391),weights.min(0),weights.max(0),color='#176f8a',alpha=.15,label='Seed range')
axes[0,1].axhline(1/390,color='#59687b',linestyle='--',linewidth=1)
axes[0,1].set(xlabel='Training-label minute in next session',ylabel='Learned horizon weight',title='Training-only horizon selection');axes[0,1].legend(frameon=False)
axes[1,0].bar(np.arange(len(horizons)),horizons,color='#176f8a');axes[1,0].axhline(390,color='#59687b',linestyle='--',linewidth=1)
axes[1,0].set(xticks=np.arange(len(horizons)),xlabel='Random seed',ylabel='Selected minute',ylim=(0,405),title='Selected horizon before final retraining')
delta=report['paired_IC_difference'];lo,hi=delta['date_block_bootstrap_95pct_CI']
axes[1,1].hlines(0,lo,hi,color='#176f8a',linewidth=2)
axes[1,1].plot(delta['estimate'],0,'o',color='#176f8a')
axes[1,1].axvline(0,color='#59687b',linestyle='--',linewidth=1)
axes[1,1].set(xlabel='Selected minus standard mean IC',yticks=[],title='Paired 20-session block bootstrap, 95% interval')
for ax in axes.flat:ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
fig.suptitle('S&P 500 independent test: disclosed features, five LSTM seeds',fontsize=14)
fig.savefig(OUT/'us_lstm_comparison.png',dpi=180)
plt.close(fig)
