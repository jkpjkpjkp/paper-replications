"""Additional baseline sensitivity screen; fixed grids, all four datasets.

This follows exploratory candidate screening and is not an independent test.
Discovery-only parameter selection is frozen before reading holdout metrics.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ocp import run,controller,metrics


def main():
    start=time.time()
    manifest=json.loads((HERE/'manifest.json').read_text())
    alpha_grid=[.0025,.005,.0075,.01,.015,.02,.025,.03,.035,.04,.045,.05,.06,.075,.1,.125,.15,.2,.25,.3]
    sf_grid=[.001,.01,.1,.25,1,2.5,10,25,100,1000,10000,100000]
    p_grid=[.001,.005,.01,.025,.05,.1,.25,.5,1,2,5]
    windows=[7,25,100,300]
    # Reuse the already saved parameter-free runs; no new selection.
    old=pd.read_csv(HERE/'all_runs.csv')
    old=old[old.method.isin(['UP','UP-paper','KT','DtACI'])].copy()
    old['window']=0
    rows=[]
    for name in manifest:
        z=np.load(HERE/f'{name}_scores.npz')
        s=z['scores'];b=int(z['burnin']);scale=float(z['scale'])
        for method in ['SFOGD','P']:
            for window in ([0] if method=='SFOGD' else windows):
                for lr in (sf_grid if method=='SFOGD' else p_grid):
                    for alpha in alpha_grid:
                        q=(run(s[b:],'SFOGD',alpha,lr) if method=='SFOGD'
                           else controller(s,alpha,lr,window)[b:])
                        for period,slc in [('all',slice(51,None)),('discovery',slice(51,len(q)*3//5)),('confirmation',slice(len(q)*3//5,None))]:
                            m=metrics(s[b:][slc],q[slc])
                            for key in ['mean_width','median_width','q75_width','q90_width','q95_width']:m[key]*=scale
                            rows.append(dict(dataset=name,method=method,alpha=alpha,lr=lr,window=window,period=period,**m))
        print(name,'broad grids finished',round(time.time()-start,1),flush=True)
    frame=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
    frame.to_csv(HERE/'broad_all_runs.csv',index=False)
    selected=[]
    for (dataset,method),part in frame[frame.period=='discovery'].groupby(['dataset','method']):
        feasible=part[(part.coverage>=.95)&np.isfinite(part.mean_width)]
        if len(feasible):
            r=feasible.loc[feasible.mean_width.idxmin()]
            selected.append(dict(dataset=dataset,method=method,alpha=float(r.alpha),lr=float(r.lr),window=int(r.window),
                                 discovery_coverage=float(r.coverage),discovery_width=float(r.mean_width)))
    # Freeze file is written before the confirmation lookup below.
    (HERE/'broad_discovery_selected.json').write_text(json.dumps(selected,indent=2))
    held=[]
    for r in selected:
        test=frame[(frame.dataset==r['dataset'])&(frame.method==r['method'])&(frame.period=='confirmation')&
                   (frame.alpha==r['alpha'])&(frame.lr==r['lr'])&(frame.window==r['window'])].iloc[0]
        held.append(dict(**r,confirmation_coverage=float(test.coverage),confirmation_width=float(test.mean_width),
                         confirmation_longest_error_streak=int(test.longest_error_streak),confirmation_n=int(test.n)))
    pd.DataFrame(held).to_csv(HERE/'broad_discovery_selected_holdout.csv',index=False)
    front=[]
    for (dataset,period,method),part in frame.groupby(['dataset','period','method']):
        for target in [.9,.95,.975]:
            feasible=part[(part.coverage>=target)&np.isfinite(part.mean_width)]
            if len(feasible):
                r=feasible.loc[feasible.mean_width.idxmin()]
                front.append(dict(dataset=dataset,period=period,method=method,target=target,coverage=r.coverage,
                                  width=r.mean_width,alpha=r.alpha,lr=r.lr,window=r.window,streak=r.longest_error_streak))
    pd.DataFrame(front).to_csv(HERE/'broad_oracle_frontiers.csv',index=False)
    print(pd.DataFrame(front).query("period=='all' and target==0.95").to_string(index=False))
    print('COVID frozen-discovery parameters and holdout',pd.DataFrame(held).query("dataset=='covid_us'").to_string(index=False))


if __name__=='__main__':main()
