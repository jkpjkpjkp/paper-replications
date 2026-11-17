#!/usr/bin/env python3
"""Independent scalar construction, tied-rank, HAC and absolute-dollar checks."""
from pathlib import Path
import json, sys, math, hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from grid import models, weights, compile_specs, screen_day
from screen import hac_t
sys.path.insert(0,str(ROOT/'factor_overlay'))
from portfolio import Ledger


def rank_reference(x):
    """Count strictly smaller/equal observations; no scipy/pandas ranking."""
    finite=[float(v) for v in x if np.isfinite(v)]
    if not finite:
        return np.full(len(x),np.nan)
    unique,count=np.unique(finite,return_counts=True)
    preceding=0
    mapping={}
    for v,n in zip(unique,count):
        mapping[v]=(preceding+(n+1)/2)/len(finite)
        preceding+=n
    return np.array([mapping.get(float(v),np.nan) for v in x])


def scalar_weights(z,specs,minimum):
    n=len(z['age'])
    chars={c:rank_reference(z[f]) for c,f in [('size','lag_cap'),('adv','lag_adv'),('vol','sigma63')]}
    tr=rank_reference(np.cos(2*np.pi*(z['age'].astype(float)-13)/65))
    membership={'none':np.ones(n,bool),'H':tr>.9,'L':tr<=.1}
    for name,lo,hi in [('P1',6,20),('P2',21,35),('P3',36,50)]:
        membership[name]=np.array([lo<=v<=hi for v in z['age']])
    result=np.zeros((len(specs),n))
    for m,s in enumerate(specs):
        halves=['low','high'] if s['group']=='balanced' else [s['group']]
        proposed={}
        failed=False
        for half in halves:
            for leg,sign in [('long',1),('short',-1)]:
                picked=[]
                for i in range(n):
                    if s['factor_index']>=0:
                        q=z['rank'][i,s['factor_index']]
                        cutoff=float(np.float32((.9 if sign==1 else .1) if s['family']=='D10' else (.7 if sign==1 else .3)))
                        if not ((q>cutoff) if sign==1 else (q<=cutoff)):
                            continue
                    if s['characteristic']!='none':
                        q=chars[s['characteristic']][i]
                        if not ((q<=.5) if half=='low' else (q>.5)):
                            continue
                    if not membership[s[leg+'_timing']][i]:
                        continue
                    a=1. if s['weighting']=='EW' else float(z['lag_cap'][i])
                    if not np.isfinite(a) or a<=0:
                        continue
                    picked.append((i,a))
                if len(picked)<minimum:
                    failed=True
                    break
                total=math.fsum(a for _,a in picked)
                for i,a in picked:
                    proposed[i]=proposed.get(i,0.)+sign*a/total/len(halves)
            if failed:
                break
        if not failed:
            for i,v in proposed.items():
                result[m,i]=v
    return result


def actual_days(specs):
    rows=[]
    for year,date in [(2005,'2005-07-01'),(2020,'2020-03-16')]:
        with np.load(ROOT/f'factor_overlay/data/{year}.npz') as f:
            ix=f['date']==np.datetime64(date)
            z={k:f[k][ix] for k in f.files}
        assert len(z['age'])>0
        assert np.all(z['known_idx']<z['idx']) and np.all(z['event_idx']<z['idx'])
        for minimum in [5,20]:
            ref=scalar_weights(z,specs,minimum)
            got=weights(z,specs,minimum)
            difference=float(np.max(abs(ref-got)))
            assert difference<2e-14,(date,minimum,difference)
            screen=screen_day(z,compile_specs(specs),minimum)
            direct=ref@np.where(np.isfinite(z['ret']),z['ret'],0.)
            return_error=float(np.max(abs(direct-screen['returns'])))
            assert return_error<1e-12
            active=(abs(ref).sum(1)>1e-14)
            assert np.array_equal(active,screen['active'])
            assert np.array_equal((ref>1e-14).sum(1),screen['n_long'])
            assert np.array_equal((ref< -1e-14).sum(1),screen['n_short'])
            assert np.allclose(ref.sum(1),0.,atol=2e-14)
            assert np.allclose(np.maximum(ref,0.).sum(1)[active],1.,atol=2e-14)
            rows.append(dict(date=date,min_names=minimum,eligible_stocks=len(z['age']),active_models=int(active.sum()),max_weight_error=difference,max_return_error=return_error))
        # Portfolio decisions must not change when return outcomes are replaced.
        changed={**z,'ret':np.full(len(z['age']),123.)}
        assert np.array_equal(weights(z,specs),weights(changed,specs))
    return rows


def tie_and_boundary_checks(specs):
    # Factor tails preserve .1/.3 float32 endpoints and exclude upper equality.
    n=120
    z=dict(age=np.tile(np.arange(1,61),2),rank=np.tile(np.array([.1,.3,.7,.9,.99,.01],np.float32),(n,1)),lag_cap=np.ones(n),lag_adv=np.ones(n),sigma63=np.ones(n),ret=np.zeros(n))
    for minimum in [5,20]:
        assert np.allclose(weights(z,specs,minimum),scalar_weights(z,specs,minimum),atol=1e-14)
    # All identical factor values cannot be divided into opposite tails.
    z['rank'][:]=.505
    assert not np.any(weights(z,[s for s in specs if s['factor_index']>=0]))
    # All timing scores identical: average percentile near1/2, neither decile.
    z['age'][:]=13
    decile=[s for s in specs if s['id']=='TIMING__decile']
    assert not np.any(weights(z,decile))
    # If low and high characteristic cells cannot both fund, balanced is cash.
    z['rank'][:,0]=np.r_[np.full(60,.99),np.full(60,.01)]
    z['age']=np.tile(np.arange(1,61),2)
    balanced=[s for s in specs if s['factor']=='MS' and s['group']=='balanced']
    assert not np.any(weights(z,balanced)) # all characteristic values tie
    return {'equal_values_unsplit':True,'float32_cutpoints':True,'balanced_underfunded_cash':True,'timing_equal_values_no_decile':True}


def ledger_checks():
    rng=np.random.default_rng(981771)
    largest=0.
    cases=0
    for cost in [0.,.0001,.0005,.001,.0025]:
        ledger=Ledger(3,cost)
        nav=np.ones(3)
        previous=[{} for _ in range(3)]
        borrow_rate=.01 if cost else 0.
        for day in range(14):
            ids=np.array([1,2,3,4] if day<6 else [1,2,4])
            ret=rng.normal(.0003,.013,len(ids))
            retx=ret-rng.uniform(0.,.001,len(ids))
            if day==4:
                ret[0]=np.nan;retx[0]=np.nan
            market={int(k):(float(r) if np.isfinite(r) else 0.,float(rx) if np.isfinite(rx) else (float(r) if np.isfinite(r) else 0.)) for k,r,rx in zip(ids,ret,retx)}
            target=np.zeros((3,len(ids)))
            target[0,0]=1;target[0,1]=-1
            target[1,0]=.5;target[1,-1]=.5;target[1,1]=-1
            if day%4!=0:
                target[2,-1]=1;target[2,0]=-1
            rebalance=day%3==0
            terminal=day==13
            observed=ledger.step(ids,ret,retx,ids if rebalance else None,target if rebalance else None,terminal=terminal)
            for j in range(3):
                oldnav=nav[j]
                old=previous[j]
                if rebalance:
                    tw=dict(zip(map(int,ids),target[j]))
                    keys=set(tw)|set(old)
                    lower,upper=0.,oldnav
                    for _ in range(80):
                        mid=(lower+upper)/2
                        residual=mid+cost*sum(abs(mid*tw.get(k,0.)-old.get(k,0.)) for k in keys)-oldnav
                        if residual>0:upper=mid
                        else:lower=mid
                    current={k:(lower+upper)/2*v for k,v in tw.items()}
                else:
                    current={k:v for k,v in old.items() if k in market}
                turnover=sum(abs(current.get(k,0.)-old.get(k,0.)) for k in set(current)|set(old))
                fee=cost*turnover
                borrow=borrow_rate/252*sum(-v for v in current.values() if v<0)
                pnl=sum(v*market[k][0] for k,v in current.items())
                nextpos={k:v*(1+market[k][1]) for k,v in current.items()}
                if terminal:
                    closing=sum(abs(v) for v in nextpos.values())
                    turnover+=closing;fee+=cost*closing;nextpos={}
                nav[j]=oldnav+pnl-fee-borrow
                previous[j]=nextpos
                expected=np.array([nav[j]/oldnav-1,turnover/oldnav,fee/oldnav,borrow/oldnav,pnl/oldnav])
                actual=np.array([observed[k][j] for k in ['returns','turnover','fees','borrow','gross_return']])
                largest=max(largest,float(np.max(abs(expected-actual))))
                assert np.allclose(expected,actual,rtol=1e-10,atol=1e-12)
                assert np.isclose(nav[j],ledger.nav[j],rtol=1e-11,atol=1e-12)
                cases+=1
        assert ledger.hold.size==0
    return {'independent_account_days':cases,'max_absolute_error':largest,'covers':['all5costs','ownNAV','postfeeallocation','held-share drift','dividendcash','missingreturn','forcedexit','cashbook','terminalliquidation']}


def inference_checks():
    rng=np.random.default_rng(219)
    r=rng.normal(.0005,.02,(600,7))
    r[1:]+=.45*r[:-1]
    calculated=hac_t(r,20)
    reference=np.array([sm.OLS(r[:,j],np.ones((len(r),1))).fit(cov_type='HAC',cov_kwds={'maxlags':20}).tvalues[0] for j in range(r.shape[1])])
    error=float(np.max(abs(calculated-reference)))
    assert error<1e-11
    return {'HAC20_statsmodels_max_error':error}


def frozen_checks():
    p=HERE/'results/frozen.json'
    if not p.exists():
        return {'status':'not_yet_frozen'}
    f=json.loads(p.read_text())
    t=pd.read_csv(HERE/'results/training_gross_summary.csv').set_index('id')
    assert f['training_end']=='2004-12-31' and not f['later_returns_read']
    for selection in f['selections']:
        candidates=t[(t.factor==selection['factor'])&(t.family==selection['family'])&t.selection_eligible]
        if not selection['selected_id']:
            assert len(candidates)==0
            continue
        chosen=t.loc[selection['selected_id']]
        assert chosen.selection_eligible
        assert chosen.annual_mean>0 and chosen.paired_annual_mean>0
        assert chosen.active_fraction>=.5 and chosen.baseline_active_fraction>=.8
        winner=candidates.reset_index().sort_values([selection['criterion'],'id'],ascending=[False,True]).iloc[0].id
        assert winner==selection['selected_id']
    return {'selected_models':len(f['selected_ids']),'selection_constraints_passed':True,'frozen_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def source_checks():
    import duckdb
    c=duckdb.connect();c.execute('SET threads=2')
    factor_names=['MS','CompEquIss','InvestPPEInv','Cash','RIO_Disp','RDIPO']
    doc=pd.read_csv('/home/jkp/Téléchargements/Chen_Zimmermann_Factor_Zoo_2026_10/SignalDoc.csv').set_index('Acronym')
    rows=[]
    for year,date,month in [(2005,'2005-07-01',200506),(2020,'2020-03-16',202002)]:
        with np.load(ROOT/f'factor_overlay/data/{year}.npz') as f:
            ix=f['date']==np.datetime64(date)
            z={k:f[k][ix] for k in f.files}
        quoted=','.join('"'+k+'"' for k in factor_names)
        raw=c.execute(f"SELECT permno,{quoted} FROM read_parquet('{ROOT}/factor_timing/data/Predictors.parquet') WHERE yyyymm={month}").df().set_index('permno')
        raw=raw.replace([np.inf,-np.inf],np.nan)
        signed=raw*doc.loc[factor_names,'Sign'].to_numpy(float)
        # Reconstruct each source-universe rank with an independent rank routine.
        ranked=pd.DataFrame({name:rank_reference(signed[name].to_numpy()) for name in factor_names},index=raw.index)
        expected=ranked.reindex(z['permno']).to_numpy().astype(np.float32)
        assert np.array_equal(np.isnan(expected),np.isnan(z['rank']))
        error=float(np.nanmax(abs(expected-z['rank'])))
        assert error==0.
        assert np.all(z['month']==month)
        market=c.execute(f"SELECT permno,CAST(dlyret AS DOUBLE) AS ret,CAST(dlyretx AS DOUBLE) AS retx FROM read_parquet('{ROOT.parent}/crsp-daily/data/crsp_a_stock/dsf_v2/{year}.parquet') WHERE dlycaldt=DATE '{date}'").df().set_index('permno').reindex(z['permno'])
        for k in ['ret','retx']:
            assert np.allclose(market[k].to_numpy(),z[k],rtol=0,atol=0,equal_nan=True)
        rows.append(dict(date=date,source_month=month,eligible_stocks=len(z['age']),max_rank_error=error,raw_CRSP_ret_and_retx_exact=True))
    c.close()
    return rows


def vw_concentration(specs):
    frozen=HERE/'results/frozen.json'
    if not frozen.exists():
        return []
    selected=json.loads(frozen.read_text())['selected_ids']
    chosen=[s for s in specs if s['id'] in selected and s['weighting']=='VW']
    results=[]
    for s in chosen:
        maximum=0.;daily=[];detail={}
        for year in range(2005,2026):
            with np.load(ROOT/f'factor_overlay/data/{year}.npz') as f:
                z={k:f[k] for k in f.files}
            starts=np.r_[0,np.flatnonzero(np.diff(z['idx']))+1]
            ends=np.r_[starts[1:],len(z['idx'])]
            for a,b in zip(starts,ends):
                zz={k:v[a:b] for k,v in z.items()}
                w=weights(zz,[s])[0]
                if np.any(w):
                    m=float(np.max(abs(w)));daily.append(m)
                    if m>maximum:
                        maximum=m;i=int(np.argmax(abs(w)))
                        detail=dict(date=str(zz['date'][i]),permno=int(zz['permno'][i]),signed_weight=float(w[i]),nlong=int((w>0).sum()),nshort=int((w<0).sum()))
        results.append(dict(id=s['id'],start=2005,end=2025,active_days=len(daily),max_abs_security_NAV_weight=maximum,median_daily_max_weight=float(np.median(daily)),p95_daily_max_weight=float(np.quantile(daily,.95)),maximum_detail=detail))
    (HERE/'results/selected_vw_concentration.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def main():
    specs=models()
    assert len(specs)==2762
    lookup={s['id']:s for s in specs}
    for s in specs:
        if s['baseline_id']:
            baseline=lookup[s['baseline_id']]
            for key in ['factor','family','characteristic','group','weighting']:
                assert baseline[key]==s[key]
            assert baseline['long_timing']==baseline['short_timing']=='none'
    result={'models':len(specs),'actual_days':actual_days(specs),'tie_boundaries':tie_and_boundary_checks(specs),'ledger':ledger_checks(),'inference':inference_checks(),'frozen':frozen_checks(),'source_spot_checks':source_checks(),'selected_VW_concentration':vw_concentration(specs)}
    result['code_sha256']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['grid.py','screen.py','evaluate.py','full_grid_report.py','validate.py','design.json','README.txt']}
    result['passed']=True
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
