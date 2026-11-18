"""Independent accounting and paper comparisons from saved dated portfolios."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import digest

PAPER={
 'set_sequence':{'sharpe':4.82,'annual_return_pct':13.0,'annual_volatility_pct':2.69,'beta':.028,'daily_turnover':.91,'short_fraction':.48,'seed_sd':.12},
 'longconv':{'sharpe':3.64,'annual_return_pct':12.8,'annual_volatility_pct':3.51,'beta':.033,'daily_turnover':.97,'short_fraction':.48,'seed_sd':.14}}
METRICS=['sharpe','annual_return_pct','annual_volatility_pct','beta','daily_turnover','short_fraction']


def statistics(d):
    r=d['excess_return'].to_numpy(dtype=float);m=d['market'].to_numpy(dtype=float)
    sd=r.std(ddof=0)
    return {'sharpe':float(np.sqrt(252)*r.mean()/sd),'annual_return_pct':float(25200*r.mean()),
        'annual_volatility_pct':float(100*np.sqrt(252)*sd),'beta':float(np.mean((r-r.mean())*(m-m.mean()))/m.var()),
        'daily_turnover':float(d.turnover.mean()),'short_fraction':float(d.short_fraction.mean()),
        'posthoc_net_sharpe':float(np.sqrt(252)*d.net_return.mean()/d.net_return.std(ddof=0)),
        'n_days':len(d),'first_date':str(d.index.min().date()),'last_date':str(d.index.max().date())}


def evaluate_track(data,results,kind,seed,require_complete=True):
    path=results/kind/f'seed-{seed}'
    years=[year for year in range(2002,2022) if (path/str(year)/'complete.json').exists()]
    if require_complete and years!=list(range(2002,2022)):
        return None
    if not years:return None
    records=[];year_metrics=[];prev={};audit=[]
    for year in years:
        folder=path/str(year)
        complete=json.loads((folder/'complete.json').read_text())
        assert digest(folder/'predictions.npz')==complete['prediction_sha256']
        assert digest(folder/'model.pt')==complete['checkpoint_sha256']
        prediction=np.load(folder/'predictions.npz')
        w=prediction['weights'].astype(float);y=prediction['returns'].astype(float);ids=prediction['permnos'];dates=prediction['dates']
        labels=np.load(data/'panels'/str(year)/'labels.npz')
        mask=~labels['train']
        np.testing.assert_array_equal(y,labels['y'][mask]);np.testing.assert_array_equal(dates,labels['dates'][mask])
        np.testing.assert_array_equal(ids,labels['permnos']);np.testing.assert_array_equal(prediction['input_dates'],labels['input_dates'][mask])
        assert (prediction['input_dates']<dates).all()
        np.testing.assert_allclose(abs(w).sum(1),1,atol=1e-5,rtol=0)
        assert len(ids)==len(set(ids))==500 and np.isfinite(w).all()
        # Independent vectorized dot product and scalar-loop spot checks.
        portfolio=np.einsum('ij,ij->i',w,y)
        for i in [0,len(w)//2,len(w)-1]:
            assert abs(portfolio[i]-sum(float(a)*float(b) for a,b in zip(w[i],y[i])))<1e-12
        turnover=np.empty(len(w))
        current_ids=set(ids.tolist())
        turnover[0]=sum(abs(float(a)-prev.get(int(pid),0.)) for pid,a in zip(ids,w[0]))+sum(abs(a) for pid,a in prev.items() if pid not in current_ids)
        turnover[1:]=abs(np.diff(w,axis=0)).sum(1)
        short=np.maximum(-w,0).sum(1)
        cost=.0005*turnover+.0001*short
        d=pd.DataFrame({'excess_return':portfolio,'market':prediction['market'],'turnover':turnover,
            'short_fraction':(w<0).mean(1),'short_exposure':short,'net_exposure':w.sum(1),'posthoc_cost':cost,
            'net_return':portfolio-cost,'equal_weight_return':y.mean(1)},index=pd.to_datetime(dates))
        records.append(d);prev=dict(zip(ids.astype(int).tolist(),w[-1].tolist()))
        year_metrics.append({'year':year,**statistics(d)})
        audit.append({'year':year,'predictions_sha256':complete['prediction_sha256'],'checkpoint_sha256':complete['checkpoint_sha256'],
            'max_l1_error':float(np.max(abs(abs(w).sum(1)-1))),'n_firm_days':int(w.size)})
    daily=pd.concat(records)
    assert not daily.index.duplicated().any() and daily.index.is_monotonic_increasing
    daily.index.name='date';daily.to_parquet(path/'daily_returns.parquet')
    pd.DataFrame(year_metrics).to_csv(path/'yearly_metrics.csv',index=False)
    metrics={'model':kind,'seed':seed,'complete_2002_2021':years==list(range(2002,2022)),**statistics(daily),
        'period_2002_2016':statistics(daily[daily.index.year<=2016]) if min(years)<=2016 else None,
        'audit':audit,'note':'Net metrics subtract costs from gross-trained weights; not Table 9 net-objective training.'}
    (path/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return metrics


def evaluate(data,results,partial=False):
    tracks=[]
    for kind in PAPER:
        for seed in range(5):
            result=evaluate_track(data,results,kind,seed,require_complete=not partial)
            if result:tracks.append(result)
    comparisons=[]
    for kind,reference in PAPER.items():
        group=[r for r in tracks if r['model']==kind and r['complete_2002_2021']]
        if not group:continue
        for metric in METRICS:
            values=np.array([r[metric] for r in group])
            mean=float(values.mean());error=mean-reference[metric]
            comparisons.append({'model':kind,'metric':metric,'paper':reference[metric],'local_mean':mean,
                'local_seed_sd':float(values.std(ddof=1)) if len(values)>1 else None,'completed_seeds':len(values),
                'absolute_error':abs(error),'signed_error':error,'relative_error_pct':100*error/abs(reference[metric]),
                'paper_seed_sd':reference['seed_sd'] if metric=='sharpe' else None,
                'within_paper_sd':abs(error)<=reference['seed_sd'] if metric=='sharpe' and len(values)==5 else None})
    frame=pd.DataFrame(comparisons)
    frame.to_csv(results/'paper_comparison.csv',index=False)
    document={'completed_tracks':len(tracks),'complete_full_period_tracks':sum(t['complete_2002_2021'] for t in tracks),
        'expected_full_period_tracks':10,'comparisons':comparisons,'tracks':[{k:v for k,v in t.items() if k!='audit'} for t in tracks]}
    (results/'evaluation.json').write_text(json.dumps(document,indent=2)+'\n')
    if len(frame):print(frame.to_string(index=False),flush=True)
    else:print(json.dumps(document,indent=2),flush=True)
    return document


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();evaluate(a.data,a.results,a.partial)
