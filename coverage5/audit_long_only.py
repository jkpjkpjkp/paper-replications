#!/usr/bin/env python3
"""Independent audit of the regular-65 long-only result and holding claims.

Reconstruct target membership from the frozen panel's eligibility/clock fields;
read ALL returns anew from raw CRSP; maintain absolute dollar positions/cash;
solve fees with bisection rather than the strategy's fixed-point execution code.
Realized earnings dates below are diagnostic labels only, never trading inputs.
"""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"results_regular65"
ROOT=HERE.parent.parent

def main():
    c=duckdb.connect();c.execute("SET threads=2");c.execute("SET memory_limit='1200MB'")
    original=pd.read_csv(OUT/"daily_returns.csv",parse_dates=["date"])
    original=original.loc[original.strategy.eq("selected_long_only")].set_index(["date","scenario"])
    cal=pd.read_csv(HERE.parent/"results/trading_calendar.csv",parse_dates=["date"])
    date_lookup=cal.set_index("idx").date.to_dict()
    labels=c.execute(f"""SELECT DISTINCT permno,event_idx FROM
      read_parquet('{HERE.parent}/results/linked_events.parquet') ORDER BY permno,event_idx""").df()
    event_lookup={int(k):g.event_idx.to_numpy(dtype=int) for k,g in labels.groupby("permno",sort=False)}
    scenarios={"gross":0.,"1bp":.0001,"5bps":.0005,"10bps":.001}
    portfolios={k:dict(nav=1.,positions={}) for k in scenarios}
    previous_ids=set();spells={};completed=[];records=[];contributions=[]
    # Seed only the 2014 boundary using its independently read raw price returns.
    prior_date=cal.loc[cal.date<pd.Timestamp("2015-01-01"),"date"].max()
    raw=ROOT/"crsp-daily/data/crsp_a_stock/dsf_v2"
    first=c.execute(f"""SELECT p.permno,CAST(r.dlyret AS DOUBLE) AS ret,
      CAST(r.dlyretx AS DOUBLE) AS retx FROM read_parquet('{OUT}/panel/2014.parquet') p
      JOIN read_parquet('{raw}/2014.parquet') r ON p.permno=r.permno AND p.date=r.dlycaldt
      WHERE p.date=DATE '{prior_date.date()}' AND p.eligible AND p.clock BETWEEN -5 AND -1""").df()
    if len(first)>=20:
        for scenario,pf in portfolios.items():
            row=original.loc[(prior_date,scenario)]
            rx=first.retx.fillna(first.ret).fillna(0)
            pf["positions"]={int(i):float((1-row.cost)/len(first)*(1+x)/(1+row.returns))
                for i,x in zip(first.permno,rx)}
    for year in range(2015,2026):
        print("Auditing raw CRSP and dollar ledger",year,flush=True)
        targets=c.execute(f"""SELECT p.permno,p.date,p.idx,p.clock,p.forecast_idx,p.known_idx,
           CAST(r.dlyret AS DOUBLE) AS ret,CAST(r.dlyretx AS DOUBLE) AS retx,r.ticker,
           p.ret AS cached_ret,p.retx AS cached_retx
          FROM read_parquet('{OUT}/panel/{year}.parquet') p
          LEFT JOIN read_parquet('{raw}/{year}.parquet') r
            ON p.permno=r.permno AND p.date=r.dlycaldt
          WHERE p.eligible AND p.clock BETWEEN -5 AND -1
          ORDER BY p.idx,p.permno""").df()
        assert np.allclose(targets.ret,targets.cached_ret,equal_nan=True)
        assert np.allclose(targets.retx,targets.cached_retx,equal_nan=True)
        grouped={int(i):g for i,g in targets.groupby("idx",sort=False)}
        days=cal.loc[cal.date.dt.year.eq(year)]
        for entry in days.itertuples():
            idx=int(entry.idx);dt=pd.Timestamp(entry.date)
            f=grouped.get(idx)
            active=f is not None and len(f)>=20
            if not active:f=targets.iloc[:0]
            ids=f.permno.to_numpy(dtype=int);current=set(map(int,ids));n=len(ids)
            # Completed stock holding spells count sessions, not reweighting orders.
            for p in previous_ids-current:
                s=spells.pop(p);s["last_idx"]=idx-1
                s["sessions"]=idx-s["first_idx"];s["right_censored"]=False
                completed.append(s)
            for row in f.itertuples():
                p=int(row.permno)
                if p not in previous_ids:
                    spells[p]=dict(permno=p,ticker=row.ticker,first_idx=idx,
                      initial_forecast_idx=int(row.forecast_idx),stock_gross_growth=1.)
                spells[p]["stock_gross_growth"]*=1+(0 if pd.isna(row.ret) else float(row.ret))
                assert -5<=row.clock<=-1
                assert row.known_idx<idx
            previous_ids=current
            ret=f.ret.fillna(0).to_numpy(dtype=float)
            rx=f.retx.fillna(f.ret).fillna(0).to_numpy(dtype=float)
            # Partition raw portfolio gross P&L by nearest realized release. This
            # is ex-post interpretation ONLY; future labels do not affect positions.
            category={k:0. for k in ["before_actual","on_actual","after_actual","no_date"]}
            count={k:0 for k in category}
            for row in f.itertuples():
                dates=event_lookup.get(int(row.permno),np.array([],int))
                if len(dates):
                    k=int(np.searchsorted(dates,idx));candidates=[]
                    if k<len(dates):candidates.append(int(dates[k]))
                    if k>0:candidates.append(int(dates[k-1]))
                    nearest=min(candidates,key=lambda x:(abs(x-idx),x>idx))
                    key="on_actual" if nearest==idx else ("before_actual" if nearest>idx else "after_actual")
                else:key="no_date"
                category[key]+=(0 if pd.isna(row.ret) else float(row.ret))/n
                count[key]+=1
            contributions.append(dict(date=dt,**{f"gross_{k}":v for k,v in category.items()},
                **{f"stockdays_{k}":v for k,v in count.items()}))
            for scenario,fee_rate in scenarios.items():
                pf=portfolios[scenario];before_nav=pf["nav"];old=pf["positions"]
                old_in=np.array([old.get(int(p),0.) for p in ids])
                exit_dollars=sum(abs(v) for p,v in old.items() if p not in current)
                if n:
                    lo=0.;hi=before_nav
                    for _ in range(48):
                        invest=(lo+hi)/2
                        turn=float(np.abs(invest/n-old_in).sum())+exit_dollars
                        if invest+fee_rate*turn>before_nav:hi=invest
                        else:lo=invest
                    invest=(lo+hi)/2
                    turn=float(np.abs(invest/n-old_in).sum())+exit_dollars
                    fee=fee_rate*turn
                    holdings=np.full(n,invest/n)
                    new_nav=before_nav-fee+float(holdings@ret)
                    positions={int(p):float(v*(1+x)) for p,v,x in zip(ids,holdings,rx)}
                    exposure=invest/before_nav
                else:
                    turn=sum(abs(v) for v in old.values());fee=fee_rate*turn
                    new_nav=before_nav-fee;positions={};exposure=0.
                # Cash is explicitly reconciled, including distributions.
                cash=new_nav-sum(positions.values())
                assert cash>-1e-10
                returns=new_nav/before_nav-1
                expected=original.loc[(dt,scenario)]
                assert abs(returns-expected.returns)<1e-11,(dt,scenario,returns,expected.returns)
                assert abs(turn/before_nav-expected.turnover)<1e-9
                pf.update(nav=new_nav,positions=positions)
                records.append(dict(date=dt,scenario=scenario,returns=returns,nav=new_nav,
                    fee=fee/before_nav,turnover=turn/before_nav,exposure=exposure,
                    n_stocks=n,active=active,cash_end=cash))
    last_idx=int(cal.idx.max())
    for p,s in spells.items():
        s["last_idx"]=last_idx;s["sessions"]=last_idx-s["first_idx"]+1
        s["right_censored"]=True;completed.append(s)
    spells=pd.DataFrame(completed)
    spells["first_return_date"]=spells.first_idx.map(date_lookup)
    spells["last_return_date"]=spells.last_idx.map(date_lookup)
    spells["entry_close_date"]=(spells.first_idx-1).map(date_lookup)
    spells["exit_close_date"]=spells.last_idx.map(date_lookup)
    spells["initial_forecast_date"]=spells.initial_forecast_idx.map(date_lookup)
    spells.to_csv(OUT/"audit_long_only_spells.csv",index=False)
    ledger=pd.DataFrame(records);ledger.to_csv(OUT/"audit_long_only_ledger.csv",index=False)
    parts=pd.DataFrame(contributions);parts.to_csv(OUT/"audit_actual_release_contributions.csv",index=False)
    z=ledger.loc[ledger.scenario.eq("1bp")].copy();z["year"]=z.date.dt.year
    annual=z.groupby("year").agg(market_sessions=("active","size"),invested_sessions=("active","sum"),
        returns=("returns",lambda x:np.prod(1+x)-1),mean_exposure=("exposure","mean"))
    annual.to_csv(OUT/"audit_long_only_years.csv")
    completed=spells.loc[~spells.right_censored]
    s={"scope":"2015-2025 regular-65 long-only candidate; previously frozen definition, no retuning",
      "independent_raw_return_and_absolute_dollar_ledger_match":True,
      "market_sessions":len(z),"invested_sessions":int(z.active.sum()),
      "invested_fraction":float(z.active.mean()),"mean_invested_sessions_per_year":float(annual.invested_sessions.mean()),
      "mean_stocks_when_invested":float(z.loc[z.active,"n_stocks"].mean()),
      "maximum_target_exposure":float(z.exposure.max()),
      "growth_of_one_dollar_1bp":float(z.nav.iloc[-1]),
      "cagr_252_session_year_1bp":float(z.nav.iloc[-1]**(252/len(z))-1),
      "cagr_exact_11_calendar_years_1bp":float(z.nav.iloc[-1]**(1/11)-1),
      "annual_arithmetic_mean_1bp":float(z.returns.mean()*252),
      "completed_stock_spells":len(completed),"mean_spell_sessions":float(completed.sessions.mean()),
      "median_spell_sessions":float(completed.sessions.median()),"max_spell_sessions":int(completed.sessions.max()),
      "spell_length_counts":{str(k):int(v) for k,v in completed.sessions.value_counts().sort_index().items()},
      "raw_equal_spell_weight_mean_stock_return":float((completed.stock_gross_growth-1).mean()),
      "calendar_2020_return_1bp":float(annual.loc[2020,"returns"]),
      "cagr_excluding_2020_1bp":float((z.nav.iloc[-1]/(1+annual.loc[2020,"returns"]))**(.1)-1),
      "forecast_day_zero_held_stockdays":0,
      "actual_release_labels_diagnostic_only":True,
      "actual_release_stockdays":{k:int(parts[f"stockdays_{k}"].sum()) for k in ["before_actual","on_actual","after_actual","no_date"]},
      "actual_release_annual_gross_contributions":{k:float(parts[f"gross_{k}"].mean()*252) for k in ["before_actual","on_actual","after_actual","no_date"]}}
    (OUT/"audit_long_only_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))

if __name__=="__main__":main()
