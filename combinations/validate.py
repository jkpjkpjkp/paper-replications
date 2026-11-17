#!/usr/bin/env python3
"""Independent signal-window, portfolio-cost, and paired-inference checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm
from combine import execute_step,target_weights

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parent
OUT=HERE/"results"


def main():
    checks={}
    # Closed-form self-financing check: long $1 / short $1 on new capital.
    c=.001;b=.01
    w=np.array([[1.,-1.]])
    r=np.array([.02,-.01]);rx=np.array([.018,-.012])
    pnl,turn,borrowing,after,exposure,fee=execute_step(np.zeros((1,2)),w,np.ones((1,2),bool),np.array([True]),r,rx,c,b)
    x=1/(1+2*c)
    expected=x*.03-2*c*x-b/252*x
    assert abs(pnl[0]-expected)<1e-13
    assert abs(turn[0]-2*x)<1e-13
    assert np.allclose(after[0],x*w[0]*(1+rx)/(1+expected),atol=1e-13)
    # Carrying the same shares must not incur rebalancing costs; a required
    # exit must be charged even when it is not a scheduled rebalance date.
    carry=execute_step(after,w,np.ones((1,2),bool),np.array([False]),r,rx,c,b)
    assert carry[1][0]==0
    assert abs(carry[0][0]-(after[0]@r-b/252*(-min(after[0,1],0))))<1e-13
    exit_=execute_step(after,w,np.array([[True,False]]),np.array([False]),r,rx,c,b)
    assert abs(exit_[1][0]-abs(after[0,1]))<1e-13
    checks["closed_form_after_cost_sizing_and_carry_exit_tests"]=True
    con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='1GB'")
    cal=pd.read_csv(PARENT/"results/trading_calendar.csv",parse_dates=["date"])
    con.register("calendar_frame",cal)
    con.execute("CREATE TEMP TABLE calendar AS SELECT * FROM calendar_frame")
    con.unregister("calendar_frame")
    window_checks=[]
    # Check year boundaries and a volatile date by directly multiplying source
    # returns across market-calendar ranges, rather than calling feature code.
    for date in ["1985-01-02","2008-10-10","2019-06-14","2025-12-31"]:
        year=int(date[:4]);path=OUT/f"features/{year}.parquet"
        frame=con.execute(f"SELECT * FROM read_parquet('{path}') WHERE date=DATE '{date}' ORDER BY permno").df()
        assert len(frame)>100
        weights,eligible,nlong,nshort=target_weights(frame)
        for row in range(8):
            base=(row//4)*4
            assert np.all((weights[row]==0)|(np.sign(weights[row])==np.sign(weights[base])))
            assert np.all((weights[row]==0)|(weights[base]!=0))
            if np.any(weights[row]):
                assert abs(weights[row][weights[row]>0].sum()-1)<1e-12
                assert abs(weights[row][weights[row]<0].sum()+1)<1e-12
        sample=frame.loc[frame.momentum12_2.notna()&frame.reversal5.notna()].iloc[len(frame)//3]
        permno=int(sample.permno);idx=int(sample.idx)
        raw=ROOT/"crsp-daily/data/crsp_a_stock/dsf_v2"
        files=f"['{raw}/{year-1}.parquet','{raw}/{year}.parquet']"
        hist=con.execute(f"""SELECT c.idx,CAST(r.dlyret AS DOUBLE) AS r
          FROM read_parquet({files}) r JOIN calendar c ON r.dlycaldt=c.date
          WHERE r.permno={permno} AND c.idx BETWEEN {idx-252} AND {idx-1}
          ORDER BY c.idx""").df()
        rv=hist.loc[hist.idx.between(idx-5,idx-1),"r"].dropna()
        mv=hist.loc[hist.idx.between(idx-252,idx-22),"r"].dropna()
        assert len(rv)==5 and len(mv)>=200
        assert np.isclose(np.prod(1+rv)-1,sample.reversal5,atol=1e-11)
        assert np.isclose(np.prod(1+mv)-1,sample.momentum12_2,atol=1e-11)
        window_checks.append({"date":date,"permno":permno,"momentum_observations":len(mv),"reversal_observations":len(rv)})
    checks["independent_raw_signal_windows"]=window_checks
    checks["overlay_position_containment_and_equal_leg_exposure"]=True
    d=pd.read_csv(OUT/"daily_returns.csv",parse_dates=["date"])
    assert not d.duplicated(["date","strategy","scenario"]).any()
    for label,c in [("gross",0),("1bp",.0001),("5bps",.0005),("10bps",.001),("25bps",.0025)]:
        g=d.loc[d.scenario==label]
        assert np.allclose(g.cost,c*g.turnover,atol=1e-12)
        assert (1+g.returns>0).all()
    checks["fee_equals_cost_per_dollar_times_all_traded_dollars"]=True
    # Every overlay shares its baseline's scheduled rebalance dates exactly.
    for (baseline,scenario),z in d.groupby(["baseline","scenario"]):
        schedules=z.pivot(index="date",columns="overlay",values="scheduled_rebalance")
        assert schedules.eq(schedules.baseline,axis=0).all().all()
        expected=cal.loc[cal.date.between("1985-01-01","2025-12-31"),"date"]
        assert len(schedules)==len(expected)
        if baseline=="momentum":assert schedules.baseline.sum()==41*12
        else:assert schedules.baseline.all()
    checks["same_baseline_and_overlay_scheduled_trade_dates"]=True
    paired=pd.read_csv(OUT/"paired_incremental_results.csv")
    assert np.allclose(paired.annual_increment,paired.annual_before_fee_increment
        -paired.annual_trading_cost_increment-paired.annual_borrow_increment,atol=1e-11)
    checks["increment_equals_return_gain_minus_incremental_trading_and_borrow_costs"]=True
    delta_checks=[]
    for row in paired.loc[(paired.period=="holdout")&(paired.scenario=="10bps")].itertuples():
        z=d.loc[(d.date>="2015-01-01")&(d.baseline==row.baseline)&(d.scenario=="10bps")]
        w=z.pivot(index="date",columns="overlay",values="returns")
        delta=w[row.overlay]-w.baseline
        fit=sm.OLS(delta.to_numpy(),np.ones((len(delta),1)),missing="raise").fit(cov_type="HAC",cov_kwds={"maxlags":20})
        assert abs(delta.mean()*252-row.annual_increment)<1e-11
        assert abs(fit.tvalues[0]-row.incremental_t_hac20)<1e-9
        delta_checks.append({"baseline":row.baseline,"overlay":row.overlay,"increment":row.annual_increment})
    checks["independently_recomputed_primary_paired_tests"]=delta_checks
    train=paired.loc[(paired.period=="development")&(paired.scenario=="10bps")]
    choices=json.loads((OUT/"development_choices.json").read_text())
    for baseline,z in train.groupby("baseline"):
        assert choices[baseline]==z.loc[z.annual_increment.idxmax(),"overlay"]
    checks["selection_uses_development_only"]=True
    checks["strategy_date_scenarios"]=len(d)
    (OUT/"validation_checks.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
