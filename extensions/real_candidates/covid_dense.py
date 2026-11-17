"""Dense common-coverage sensitivity, both common and as-original init.

Exploratory follow-up; all configurations retained. No test-picked settings are
represented as deployed tuning decisions. Selection-transfer settings are
chosen solely on the chronological discovery segment.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ocp import run,controller,metrics


def main():
    z=np.load(HERE/'covid_us_scores.npz')
    s=z['scores'];b=int(z['burnin']);scale=float(z['scale'])
    alpha_grid=np.unique(np.r_[np.arange(.0025,.06001,.0005),[.075,.1,.125,.15,.2,.25,.3]])
    sf_grid=[.001,.01,.1,.25,1,2.5,10,25,100,1000,10000,100000]
    p_grid=[.001,.005,.01,.025,.05,.1,.25,.5,1,2,5]
    windows=[7,25,100,300]
    split=(len(s)-b)*3//5
    rows=[]
    for init in ['as-original','common']:
        for method in ['UP','UP-paper','KT','DtACI','SFOGD','P']:
            for window in (windows if method=='P' else [0]):
                for lr in (sf_grid if method=='SFOGD' else p_grid if method=='P' else [0]):
                    for alpha in alpha_grid:
                        if method=='P':
                            q=controller(s,alpha,lr,window)[b:] if init=='as-original' else controller(s[b:],alpha,lr,window)
                        else:q=run(s[b:],method,alpha,lr)
                        for period,slc in [('all',slice(51,None)),('discovery',slice(51,split)),('confirmation',slice(split,None))]:
                            m=metrics(s[b:][slc],q[slc])
                            for key in ['mean_width','median_width','q75_width','q90_width','q95_width']:m[key]*=scale
                            rows.append(dict(initialization=init,dataset='covid_us',method=method,alpha=alpha,lr=lr,window=window,period=period,**m))
            print(init,method,'done',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'covid_dense_runs.csv',index=False)
    chosen=[]
    for (init,method),d in frame[frame.period=='discovery'].groupby(['initialization','method']):
        f=d[(d.coverage>=.95)&np.isfinite(d.mean_width)]
        if not len(f):continue
        r=f.loc[f.mean_width.idxmin()]
        chosen.append(dict(initialization=init,method=method,alpha=float(r.alpha),lr=float(r.lr),window=int(r.window),
                           discovery_coverage=float(r.coverage),discovery_width=float(r.mean_width)))
    (HERE/'covid_dense_frozen_discovery.json').write_text(json.dumps(chosen,indent=2))
    transfer=[]
    for r in chosen:
        test=frame[(frame.initialization==r['initialization'])&(frame.method==r['method'])&(frame.period=='confirmation')&
                   (frame.alpha==r['alpha'])&(frame.lr==r['lr'])&(frame.window==r['window'])].iloc[0]
        transfer.append(dict(**r,confirmation_coverage=float(test.coverage),confirmation_width=float(test.mean_width),
                             confirmation_streak=int(test.longest_error_streak),confirmation_n=int(test.n)))
    pd.DataFrame(transfer).to_csv(HERE/'covid_dense_transfer.csv',index=False)
    front=[]
    for (init,period,method),d in frame.groupby(['initialization','period','method']):
        for target in [.945,.95,.955]:
            f=d[(d.coverage>=target)&np.isfinite(d.mean_width)]
            if not len(f):continue
            r=f.loc[f.mean_width.idxmin()]
            front.append(dict(initialization=init,period=period,method=method,target=target,
                              coverage=float(r.coverage),width=float(r.mean_width),alpha=float(r.alpha),lr=float(r.lr),window=int(r.window),streak=int(r.longest_error_streak)))
    result=pd.DataFrame(front)
    result.to_csv(HERE/'covid_dense_frontiers.csv',index=False)
    print(result[result.period=='all'].to_string(index=False))
    print('TRANSFER',pd.DataFrame(transfer).to_string(index=False))


if __name__=='__main__':main()
