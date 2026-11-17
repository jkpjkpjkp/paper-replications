#!/usr/bin/env python3
"""Independent small-sample ledger and event-label validation for horizons.py.

Only writes independent_smoke* / independent_validation.json outputs.
The reference uses absolute NAV, scalar bisection, dictionaries and explicit
per-stock return compounding, rather than production vectorized NAV equations.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import horizons as h


def load_npz(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def reference_one(model, cost, year=1985, min_names=1):
    cal=h.base.calendar()
    cal=cal[cal.date.dt.year.eq(year)]
    indices=cal.idx.to_numpy(int)
    entries=load_npz(h.DATA/f'entries_{year}.npz')
    marks=load_npz(h.DATA/f'market_{year}.npz')
    factor_i=h.base.metadata()['factors'].index(model['factor'])
    starts={}
    for k in np.flatnonzero(entries['n']==model['n']):
        rank=float(entries['rank'][k,factor_i])
        # Persisted rank arrays and production comparisons are float32. Use
        # the same representable tercile cutoffs, particularly for exact ties.
        side=1 if rank>float(np.float32(2/3)) else -1 if rank<float(np.float32(1/3)) else 0
        if side:
            starts.setdefault(int(entries['entry_idx'][k]),[]).append(
                (int(entries['permno'][k]),side,int(entries['exit_idx'][k,model['m']-1])))
    market={}
    for k,t in enumerate(marks['idx']):
        market.setdefault(int(t),{})[int(marks['permno'][k])]=(float(marks['ret'][k]),float(marks['retx'][k]))
    spells={}
    held={}
    nav=1.
    out={k:[] for k in ['returns','turnover','fees','active','n_long','n_short']}
    forced=0
    for t in indices:
        today=market.get(int(t),{})
        spells={p:z for p,z in spells.items() if z[1]>=t and p in today}
        for p,sign,expiry in starts.get(int(t),[]):
            if p not in spells:
                spells[p]=(sign,expiry)
        longs=[p for p,(sign,_) in spells.items() if sign>0]
        shorts=[p for p,(sign,_) in spells.items() if sign<0]
        active=len(longs)>=min_names and len(shorts)>=min_names
        if not active:
            forced+=bool(held)
            spells={};longs=[];shorts=[]
        weights={p:1/len(longs) for p in longs}
        weights.update({p:-1/len(shorts) for p in shorts})
        stockset=set(weights)|set(held)
        old_nav=nav

        # Solve post-cost NAV + cost * dollars traded = pre-trade NAV.
        def residual(post_nav):
            turn=sum(abs(post_nav*weights.get(p,0)-held.get(p,0)) for p in stockset)
            return post_nav+cost*turn-old_nav
        lo=0.;hi=old_nav
        for _ in range(65):
            mid=(lo+hi)/2
            if residual(mid)>0:hi=mid
            else:lo=mid
        after_fee=(lo+hi)/2
        targets={p:after_fee*w for p,w in weights.items()}
        traded=sum(abs(targets.get(p,0)-held.get(p,0)) for p in stockset)
        fees=cost*traded
        pnl=sum(v*today[p][0] for p,v in targets.items())
        borrow=(.01/252 if cost else 0)*sum(-v for v in targets.values() if v<0)
        nav=old_nav-fees+pnl-borrow
        held={p:v*(1+today[p][1]) for p,v in targets.items()}
        if t==indices[-1]:
            terminal=sum(abs(v) for v in held.values())
            traded+=terminal
            fees+=cost*terminal
            nav-=cost*terminal
            held={}
        out['returns'].append(nav/old_nav-1)
        out['turnover'].append(traded/old_nav)
        out['fees'].append(fees/old_nav)
        out['active'].append(float(active))
        out['n_long'].append(len(longs))
        out['n_short'].append(len(shorts))
    return {k:np.asarray(v) for k,v in out.items()}


def prefix_check():
    e=load_npz(h.DATA/'entries_1985.npz')
    # Some late-1985 opportunities exit in 1986; use direct day-by-day marks.
    market={}
    for y in [1985,1986]:
        a=load_npz(h.DATA/f'market_{y}.npz')
        for p,t,r in zip(a['permno'],a['idx'],a['ret']):
            market[(int(p),int(t))]=float(r)
    checks=0;max_error=0.
    rng=np.random.default_rng(19851102)
    rows=rng.choice(len(e['permno']),size=min(60,len(e['permno'])),replace=False)
    for row in rows:
        first=int(e['entry_idx'][row]);forecast=int(e['forecast_idx'][row]);n=int(e['n'][row])
        assert first-1==forecast-n
        next_event=e['next_event_LABEL'][row];known=e['next_known_LABEL'][row]
        for m in [1,7,30]:
            expected_scheduled=min(forecast+40,max(next_event+m,known)) if np.isfinite(next_event) and np.isfinite(known) else forecast+40
            assert int(e['scheduled_exit_idx'][row,m-1])==expected_scheduled
            last=int(e['exit_idx'][row,m-1]);p=int(e['permno'][row])
            assert int(e['duration'][row,m-1])==last-first+1
            returns=[market[(p,t)] for t in range(first,last+1)]
            expected=float(np.prod(1+np.asarray(returns))-1)
            error=abs(expected-float(e['returns'][row,m-1]))
            max_error=max(max_error,error);checks+=1
    assert max_error<1e-10, max_error
    return dict(prefix_checks=checks,prefix_max_error=max_error)


def reference_multiyear(model,cost,start=2005,end=2025,audit_dates=('2016-03-29','2016-04-04')):
    """Absolute-dollar reference across annual cache boundaries, with ledgers."""
    cal=h.base.calendar();cal=cal[cal.date.dt.year.between(start,end)]
    date_by_idx={int(t):str(d.date()) for t,d in zip(cal.idx,cal.date)}
    last_idx=int(cal.idx.iloc[-1]);factor_i=h.base.metadata()['factors'].index(model['factor'])
    spells={};held={};nav=1.;ledger=[]
    out={k:[] for k in ['returns','turnover','fees','active','n_long','n_short']}
    for year in range(start,end+1):
        print('Independent absolute-dollar reference',model['id'],year,flush=True)
        entries=load_npz(h.DATA/f'entries_{year}.npz');marks=load_npz(h.DATA/f'market_{year}.npz')
        starts={}
        for k in np.flatnonzero(entries['n']==model['n']):
            rank=float(entries['rank'][k,factor_i]);side=1 if rank>float(np.float32(2/3)) else -1 if rank<float(np.float32(1/3)) else 0
            if side:starts.setdefault(int(entries['entry_idx'][k]),[]).append((int(entries['permno'][k]),side,int(entries['exit_idx'][k,model['m']-1])))
        del entries
        market={}
        for k,t in enumerate(marks['idx']):
            market.setdefault(int(t),{})[int(marks['permno'][k])]=(float(marks['ret'][k]),float(marks['retx'][k]))
        del marks
        for t in cal.loc[cal.date.dt.year.eq(year),'idx'].to_numpy(int):
            today=market.get(int(t),{})
            spells={p:z for p,z in spells.items() if z[1]>=t and p in today}
            for p,sign,expiry in starts.get(int(t),[]):
                if p not in spells:spells[p]=(sign,expiry)
            longs=[p for p,(sign,_) in spells.items() if sign>0];shorts=[p for p,(sign,_) in spells.items() if sign<0]
            active=bool(longs and shorts)
            if not active:spells={};longs=[];shorts=[]
            weights={p:1/len(longs) for p in longs};weights.update({p:-1/len(shorts) for p in shorts})
            stockset=sorted(set(weights)|set(held));wa=np.asarray([weights.get(p,0) for p in stockset]);ha=np.asarray([held.get(p,0) for p in stockset])
            old_nav=nav;lo=0.;hi=nav
            for _ in range(65):
                mid=(lo+hi)/2
                residual=mid+cost*float(np.abs(mid*wa-ha).sum())-old_nav
                if residual>0:hi=mid
                else:lo=mid
            after_fee=(lo+hi)/2;targets={p:after_fee*w for p,w in weights.items()}
            traded=float(np.abs(after_fee*wa-ha).sum());fees=cost*traded
            pnl=sum(v*today[p][0] for p,v in targets.items())
            borrow=(.01/252 if cost else 0)*sum(-v for v in targets.values() if v<0)
            nav=old_nav-fees+pnl-borrow
            assert nav>0,('Reference unexpectedly insolvent',model['id'],t)
            held={p:v*(1+today[p][1]) for p,v in targets.items()}
            if t==last_idx:
                terminal=sum(abs(v) for v in held.values());traded+=terminal;fees+=cost*terminal;nav-=cost*terminal;held={}
            day_return=nav/old_nav-1
            if date_by_idx[int(t)] in audit_dates:
                ledger.append(dict(model=model['id'],date=date_by_idx[int(t)],idx=int(t),nav_before=old_nav,nav_after=nav,return_net=day_return,return_gross=pnl/old_nav,fees_nav=fees/old_nav,borrow_nav=borrow/old_nav,turnover=traded/old_nav,long_count=len(longs),short_count=len(shorts),positions=[dict(permno=p,side='long' if v>0 else 'short',weight_pre_cost=weights[p],position_divided_by_nav=v/old_nav,stock_total_return=today[p][0],stock_exdiv_return=today[p][1],contribution=v/old_nav*today[p][0],planned_exit_idx=int(spells[p][1])) for p,v in sorted(targets.items())]))
            out['returns'].append(day_return);out['turnover'].append(traded/old_nav);out['fees'].append(fees/old_nav);out['active'].append(float(active));out['n_long'].append(len(longs));out['n_short'].append(len(shorts))
    return {k:np.asarray(v) for k,v in out.items()},ledger


def extreme_day_audit():
    final=load_npz(h.OUT/'final.npz');models=json.loads((h.OUT/'final_models.json').read_text());ix={m['id']:j for j,m in enumerate(models)}
    errors={};ledgers=[];ci=int(np.flatnonzero(np.isclose(final['costs'],.0005))[0])
    for spec in [h.model('AnalystRevision',30,1),h.model('AnalystRevision',1,1)]:
        reference,rows=reference_multiyear(spec,.0005)
        np.savez_compressed(h.OUT/f"independent_reference_{spec['id']}.npz",**reference)
        j=ix[spec['id']];errors[spec['id']]={}
        for field,values in reference.items():
            err=float(np.max(np.abs(values-final[field][:,ci,j])))
            errors[spec['id']][field]=err;assert err<1e-10,(spec['id'],field,err)
        ledgers.extend(rows)
    # Independently verify extreme constituent total returns against the original
    # derived stock-day source (rather than only the horizon market cache).
    c=h.connection();raw_checks=[];original_checks=[]
    for row in ledgers:
        if row['model']!='AnalystRevision__N01_M01':continue
        ids=','.join(str(p['permno']) for p in row['positions'])
        stocks=c.execute(f"SELECT permno,ret,retx,lag_price,lag_cap,lag_adv FROM read_parquet('{h.base.PANEL}/2016.parquet') WHERE idx={row['idx']} AND permno IN ({ids})").df()
        for p in row['positions']:
            observed=stocks.loc[stocks.permno.eq(p['permno'])].iloc[0]
            assert abs(float(observed.ret)-p['stock_total_return'])<1e-14
            raw_checks.append(dict(date=row['date'],permno=p['permno'],source_panel_return=float(observed.ret),lag_price=float(observed.lag_price),lag_cap=float(observed.lag_cap),lag_adv=float(observed.lag_adv)))
        source=h.ROOT.parent/'crsp-daily/data/crsp_a_stock/dsf_v2/2016.parquet'
        originals=c.execute(f"SELECT permno,CAST(dlyret AS DOUBLE) AS ret,CAST(dlyretx AS DOUBLE) AS retx,abs(CAST(dlyprc AS DOUBLE)) AS price FROM read_parquet('{source}') WHERE dlycaldt=DATE '{row['date']}' AND permno IN ({ids})").df()
        for p in row['positions']:
            observed=originals.loc[originals.permno.eq(p['permno'])].iloc[0]
            assert abs(float(observed.ret)-p['stock_total_return'])<1e-14
            original_checks.append(dict(date=row['date'],permno=p['permno'],side=p['side'],raw_crsp_total_return=float(observed.ret),raw_crsp_exdiv_return=float(observed.retx),raw_close_price=float(observed.price)))
    result=dict(start='2005-01-03',end='2025-12-31',reference='Independent absolute-dollar NAV/holdings dictionaries, scalar bisection',max_field_errors=errors,extreme_day_ledgers=ledgers,source_panel_constituent_checks=raw_checks,original_crsp_constituent_checks=original_checks)
    (h.OUT/'extreme_day_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    models=[h.model('Mom12m',1,1),dict(h.model('Mom12m',1,1),id='duplicate_baseline'),h.model('BM',10,7)]
    costs=(0.,.0005,.001)
    h.simulate(models,1985,1985,'independent_smoke',costs=costs)
    prod=load_npz(h.OUT/'independent_smoke.npz')
    errors={k:0. for k in ['returns','turnover','fees','active','n_long','n_short']}
    for ci,cost in enumerate(costs):
        for j,model in enumerate(models):
            ref=reference_one(model,cost)
            for field in errors:
                error=float(np.max(np.abs(ref[field]-prod[field][:,ci,j])))
                errors[field]=max(errors[field],error)
                assert error<1e-10,(field,model,cost,error)
    for field in errors:
        assert np.array_equal(prod[field][:,:,0],prod[field][:,:,1]),field
    result=dict(absolute_dollar_scalar_bisection_reference=True,baseline_duplicate_identical=True,models=models,cost_bps=[0,5,10],max_field_errors=errors,**prefix_check())
    (h.OUT/'independent_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
