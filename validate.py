#!/usr/bin/env python3
"""Economic ledger, chronology, and independent inference checks for the run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def main():
    con=duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1GB'")
    checks={}
    panel=str(OUT/"panel/[12][0-9][0-9][0-9].parquet")
    q=f"""SELECT count(*) AS n,
      count(*) FILTER(WHERE known_idx IS NOT NULL AND known_idx>=idx) AS premature,
      count(*) FILTER(WHERE event_idx>known_idx) AS backwards,
      count(*) FILTER(WHERE lag_price<5 OR lag_cap<100000 OR lag_adv<1000000) AS screen_failure,
      count(*) FILTER(WHERE ret IS NULL) AS missing_returns,
      count(*) FILTER(WHERE ret < -1) AS impossible_returns
      FROM read_parquet('{panel}')"""
    z=con.execute(q).fetchall()[0]
    checks["panel"]={k:int(v) for k,v in zip(["stock_days","premature_information_use","backwards_event_clock","lagged_screen_failures","missing_returns","impossible_returns"],z)}
    assert z[1]==z[2]==z[3]==z[5]==0
    fields=[r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{panel}')").fetchall()]
    assert not any("DIAGNOSTIC_ONLY" in s for s in fields)
    assert {"abnormal_ew","abnormal_vw","nearest_predicted_signed_clock"}.issubset(fields)
    bad=con.execute(f"""SELECT count(*) FROM read_parquet('{panel}')
        WHERE abs(abnormal_ew-(ret-ew_market))>1e-12
           OR abs(abnormal_vw-(ret-vw_market))>1e-12
           OR nearest_predicted_signed_clock IS DISTINCT FROM
             CASE WHEN to_predicted>=0 AND to_predicted<age THEN -to_predicted ELSE age END""").fetchall()[0][0]
    assert bad==0
    checks["abnormal_return_and_nearest_clock_identities"]=True
    d=pd.read_csv(OUT/"daily_strategies.csv",parse_dates=["date"])
    assert not d.duplicated(["date","strategy"]).any()
    checks["unique_strategy_dates"]=True
    assert np.allclose(d.gross,d.long_return-d.short_underlying_return,atol=1e-12)
    assert np.allclose(d.net_10bps_1pct_borrow,
                       d.gross-.001*d.turnover-np.where(d.active,.01/252,0),atol=1e-12)
    assert np.allclose(d.net_1bp_1pct_borrow,
                       d.gross-.0001*d.turnover-np.where(d.active,.01/252,0),atol=1e-12)
    assert np.allclose(d.net_25bps_2pct_borrow,
                       d.gross-.0025*d.turnover-np.where(d.active,.02/252,0),atol=1e-12)
    checks["portfolio_ledger_and_cost_identities"]=True
    # Freeze selected windows using only the development output, never holdout.
    b=pd.read_csv(OUT/"timing_bin_statistics.csv")
    train=b.loc[b.period=="development"].set_index("bin")
    checks["selected_on_development_only"]={"long":train.abnormal_bps_daily.idxmax(),
                                              "short":train.abnormal_bps_daily.idxmin()}
    states=con.execute(f"SELECT permno,known_idx,event_idx,availability_date FROM read_parquet('{OUT}/linked_events.parquet') ORDER BY permno,known_idx").df()
    changes=states.groupby("permno").event_idx.diff().dropna()
    assert (changes>0).all()
    assert (states.known_idx>=states.event_idx).all()
    calendar=pd.read_csv(OUT/"trading_calendar.csv",parse_dates=["date"])
    available=states.loc[states.known_idx<len(calendar)]
    effective_dates=calendar.date.to_numpy()[available.known_idx.to_numpy(dtype=int)]
    assert (effective_dates>available.availability_date.to_numpy()).all()
    checks["information_states_monotone"]=True
    # An independent ledger check reads two adjacent days from the source-derived
    # panel, recomputes the marked holdings and verifies all turnover components.
    sample_date="2019-06-14"
    prior_date="2019-06-13"
    prior={}
    prior_pnl=0.
    for date in [prior_date,sample_date]:
        p=con.execute(f"SELECT * FROM read_parquet('{OUT}/panel/2019.parquet') WHERE date=DATE '{date}'").df()
        long=p.regular.fillna(False)&p.age.between(6,20)
        short=p.regular.fillna(False)&p.age.between(36,50)
        assert min(long.sum(),short.sum())>=20
        weights={int(k):1/long.sum() for k in p.loc[long,"permno"]}
        weights.update({int(k):-1/short.sum() for k in p.loc[short,"permno"]})
        ret=p.set_index("permno").ret.fillna(0).to_dict()
        retx=p.set_index("permno").retx.fillna(p.set_index("permno").ret).fillna(0).to_dict()
        pnl=sum(w*ret[k] for k,w in weights.items())
        actual=d.loc[(d.date==date)&(d.strategy=="cycle")].iloc[0]
        assert abs(pnl-actual.gross)<1e-12
        if date==sample_date:
            turnover=sum(abs(weights.get(k,0)-prior.get(k,0)/(1+prior_pnl))
                         for k in set(weights)|set(prior))
            assert abs(turnover-actual.turnover)<1e-12
            checks["independent_two_day_ledger"]={"date":date,"gross_return":pnl,"turnover":turnover}
        prior={k:w*(1+retx[k]) for k,w in weights.items()}
        prior_pnl=pnl
    # Recompute holdout uncertainty with longer lags and monthly aggregation.
    robustness=[]
    for name,g in d.loc[d.date>="2015-01-01"].groupby("strategy"):
        for col in ["gross","net_1bp_1pct_borrow","net_10bps_1pct_borrow","net_25bps_2pct_borrow"]:
            for lags in [5,20,60]:
                fit=sm.OLS(g[col],np.ones((len(g),1)),missing="raise").fit(cov_type="HAC",cov_kwds={"maxlags":lags})
                robustness.append(dict(strategy=name,returns=col,frequency="daily",hac_lags=lags,
                    annual_mean=float(g[col].mean()*252),t=float(fit.tvalues.iloc[0]),p=float(fit.pvalues.iloc[0])))
            monthly=g.set_index("date")[col].resample("ME").apply(lambda x:np.prod(1+x)-1)
            fit=sm.OLS(monthly,np.ones((len(monthly),1)),missing="raise").fit(cov_type="HAC",cov_kwds={"maxlags":3})
            robustness.append(dict(strategy=name,returns=col,frequency="monthly",hac_lags=3,
                annual_mean=float(monthly.mean()*12),t=float(fit.tvalues.iloc[0]),p=float(fit.pvalues.iloc[0])))
    pd.DataFrame(robustness).to_csv(OUT/"inference_robustness.csv",index=False)
    (OUT/"validation_checks.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
