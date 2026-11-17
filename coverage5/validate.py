#!/usr/bin/env python3
"""Independent raw coverage, chronology, portfolio ledger and inference checks."""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
ROOT=HERE.parent.parent

def main():
    c=duckdb.connect();c.execute("SET threads=2");c.execute("SET memory_limit='1200MB'")
    checks={}
    paths=f"{OUT}/panel/[12]*.parquet"
    n,bad,dups=c.execute(f"""SELECT count(*),count(*) FILTER(WHERE
      (eligible AND (n180<5 OR lag_price<=0 OR lag_cap<=0)) OR n90>n180
      OR (known_idx IS NOT NULL AND known_idx>=idx)
      OR (clock IS DISTINCT FROM CASE WHEN forecast_idx-idx>=0 AND forecast_idx-idx<age
         THEN idx-forecast_idx ELSE age END)
      OR (median_clock IS DISTINCT FROM CASE WHEN regular AND predicted_idx-idx>=0
         AND predicted_idx-idx<age THEN idx-predicted_idx ELSE age END)
      OR ret < -1),count(*)-count(DISTINCT (permno,date))
      FROM read_parquet('{paths}')""").fetchall()[0]
    assert bad==dups==0
    checks["panel_rows"]=int(n);checks["coverage_chronology_clock_keys_return_checks"]=True
    cols=c.execute(f"DESCRIBE SELECT * FROM read_parquet('{paths}')").df().column_name.tolist()
    assert not any("actual_date" in x or "DIAGNOSTIC" in x or x in ["value","actual","revdats"] for x in cols)
    assert c.execute(f"SELECT count(*) FROM read_parquet('{OUT}/coverage_states.parquet') WHERE n90<0 OR n180<n90").fetchall()[0][0]==0
    checks["no_forecast_values_or_future_event_labels_in_trading_panel"]=True
    # Reconstruct analyst counts directly from raw detail records; no interval code.
    cal=pd.read_csv(HERE.parent/"results/trading_calendar.csv",parse_dates=["date"])
    direct=[]
    for day in ["1985-06-14","2008-10-10","2019-06-14","2025-12-31"]:
        prior=cal.loc[cal.date<pd.Timestamp(day),"date"].max()
        y=day[:4]
        samples=c.execute(f"""SELECT permno,lag_cusip,n180,n90 FROM read_parquet('{OUT}/panel/{y}.parquet')
          WHERE date=DATE '{day}' AND lag_cusip IS NOT NULL AND n180 IN (4,5,6,10)
          QUALIFY row_number() OVER(PARTITION BY n180 ORDER BY permno)=1""").df()
        c.register("samples",samples)
        q=f"""WITH raw AS (
          SELECT trim(cusip) AS cusip,analys,greatest(anndats,actdats) AS available
          FROM read_parquet('{ROOT}/ibes/all-parquet/ibes/det_epsus/*.parquet')
          WHERE usfirm=1 AND measure='EPS' AND fpi IN ('1','2') AND analys>0
            AND anndats IS NOT NULL AND actdats IS NOT NULL
            AND length(trim(cusip))=8),
          a AS (
          SELECT cusip,count(DISTINCT analys) FILTER(WHERE available>=DATE '{prior.date()}'-INTERVAL '90 days') AS n90,
            count(DISTINCT analys) AS n180
          FROM raw SEMI JOIN samples s ON raw.cusip=s.lag_cusip
          WHERE available<DATE '{prior.date()}' AND available>=DATE '{prior.date()}'-INTERVAL '180 days'
          GROUP BY cusip)
          SELECT s.*,coalesce(a.n90,0) AS direct90,coalesce(a.n180,0) AS direct180
          FROM samples s LEFT JOIN a ON s.lag_cusip=a.cusip"""
        z=c.execute(q).df();c.unregister("samples")
        assert (z.n90==z.direct90).all() and (z.n180==z.direct180).all(),z
        direct.extend([dict(date=day,permno=int(x.permno),n180=int(x.n180),n90=int(x.n90)) for x in z.itertuples()])
    checks["independent_raw_distinct_analyst_counts"]=direct
    # Independent forecast history check. These event rows are chronological states;
    # future actual labels are neither needed nor selected.
    ev=c.execute(f"""SELECT permno,anndats,known_idx,regular,forecast_gap,seasonal_predicted_date,
       predicted_date,seasonal_gap FROM read_parquet('{HERE.parent}/results/linked_events.parquet')
       ORDER BY permno,known_idx""").df()
    fourth=ev.groupby("permno").anndats.shift(3)
    gaps=ev.groupby("permno").anndats.diff().dt.days
    median=gaps.groupby(ev.permno).transform(lambda s:s.rolling(3,min_periods=3).median())
    valid=ev.regular
    assert ((fourth+pd.Timedelta(days=364))[valid]==ev.loc[valid,"seasonal_predicted_date"]).all()
    assert np.allclose(median[valid],ev.loc[valid,"forecast_gap"])
    assert (ev.loc[valid,"predicted_date"]==ev.loc[valid,"anndats"]+pd.to_timedelta(median[valid].round(),unit="D")).all()
    checks["forecast_dates_recomputed_from_past_event_history"]=int(valid.sum())
    d=pd.read_csv(OUT/"daily_returns.csv",parse_dates=["date"])
    assert not d.duplicated(["date","strategy","scenario"]).any()
    assert np.allclose(d.returns,d.before_fee-d.cost-d.borrow,atol=1e-12)
    fees={"gross":0.,"1bp":.0001,"5bps":.0005,"10bps":.001,"25bps":.0025}
    for name,z in d.groupby("scenario"):
        assert np.allclose(z.cost,fees[name]*z.turnover,atol=1e-12)
        rate=.02 if name=="25bps" else (.0 if name=="gross" else .01)
        expected=np.where(z.n_short>0,rate/252*(1-z.cost),0)
        assert np.allclose(z.borrow,expected,atol=1e-12)
    checks["daily_fees_borrow_and_pnl_identities"]=True
    # Independently mark yesterday's after-cost equal weights, then solve today's
    # cost sizing by bisection rather than using the implementation's fixed point.
    choice=json.loads((OUT/"selected_windows.json").read_text())
    ledger=[]
    for day in ["2008-10-10","2019-06-14","2025-12-31"]:
        previous=cal.loc[cal.date<pd.Timestamp(day),"date"].max()
        tables=[];targets=[]
        for dt in [previous,pd.Timestamp(day)]:
            f=c.execute(f"SELECT * FROM read_parquet('{OUT}/panel/{dt.year}.parquet') WHERE date=DATE '{dt.date()}'").df().set_index("permno")
            w=pd.Series(0.,index=f.index)
            a,b=map(int,choice["long"].split(":"));lm=f.eligible&f.clock.between(a,b)
            a,b=map(int,choice["short"].split(":"));ss=f.eligible&f.clock.between(a,b)
            if min(lm.sum(),ss.sum())>=20:
                w.loc[lm]=1/lm.sum();w.loc[ss]=-1/ss.sum()
            tables.append(f);targets.append(w.loc[w!=0])
        for scenario in ["gross","10bps","25bps"]:
            prevrow=d.loc[(d.date==previous)&(d.strategy=="nearest_selected")&(d.scenario==scenario)].iloc[0]
            curr=d.loc[(d.date==day)&(d.strategy=="nearest_selected")&(d.scenario==scenario)].iloc[0]
            union=targets[0].index.union(targets[1].index)
            priorrx=tables[0].retx.fillna(tables[0].ret).fillna(0).reindex(union).fillna(0)
            v=targets[0].reindex(union).fillna(0)*(1-prevrow.cost)*(1+priorrx)/(1+prevrow.returns)
            w=targets[1].reindex(union).fillna(0)
            lower=0.;upper=1.;fee_rate=fees[scenario]
            for _ in range(60):
                x=(lower+upper)/2
                if x+fee_rate*np.abs(x*w-v).sum()>1:upper=x
                else:lower=x
            x=(lower+upper)/2;turn=float(np.abs(x*w-v).sum());fee=fee_rate*turn
            ret=tables[1].ret.reindex(union).fillna(0)
            gross=float((x*w*ret).sum())
            borrow=(0 if scenario=="gross" else (.02 if scenario=="25bps" else .01))/252*x*float(-w.clip(upper=0).sum())
            actual=gross-fee-borrow
            assert abs(actual-curr.returns)<1e-11 and abs(turn-curr.turnover)<1e-10
            ledger.append(dict(date=day,scenario=scenario,returns=actual,turnover=turn))
    checks["independent_two_day_bisection_portfolio_ledgers"]=ledger
    bs=pd.read_csv(OUT/"bin_statistics.csv")
    train=bs.loc[(bs.period=="development")&(bs.days>=1000)].set_index("bin")
    assert choice["long"]==train.excess_bps.idxmax() and choice["short"]==train.excess_bps.idxmin()
    checks["window_selection_development_only"]=choice
    perf=pd.read_csv(OUT/"performance.csv")
    h=d.loc[(d.date>="2015-01-01")&(d.strategy=="nearest_selected")&(d.scenario=="10bps")].sort_values("date")
    fit=sm.OLS(h.returns.to_numpy(),np.ones((len(h),1))).fit(cov_type="HAC",cov_kwds={"maxlags":20})
    row=perf.loc[(perf.period=="holdout")&(perf.strategy=="nearest_selected")&(perf.scenario=="10bps")].iloc[0]
    assert abs(float(fit.params[0])*252-row.annual_mean)<1e-12
    assert abs(float(fit.tvalues[0])-row.t)<1e-10
    peers=perf.loc[(perf.period=="holdout")&(perf.scenario=="10bps")&perf.p_holm_6_ls.notna()]
    assert np.allclose(multipletests(peers.p,method="holm")[1],peers.p_holm_6_ls)
    checks["independent_holdout_inference_and_six_test_adjustment"]=True
    checks["missing_held_return_observations_primary_gross"]=int(d.loc[(d.strategy=="nearest_selected")&(d.scenario=="gross"),"missing_held_returns"].sum())
    checks["absent_name_exits_primary_gross"]=int(d.loc[(d.strategy=="nearest_selected")&(d.scenario=="gross"),"absent_exits"].sum())
    (OUT/"validation_checks.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=="__main__":main()
