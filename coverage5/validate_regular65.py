#!/usr/bin/env python3
"""Regular-65 profile validation, including independent raw history checks."""
import json
import validate as common
import pandas as pd
import numpy as np
import duckdb

def main():
    common.OUT=common.HERE/"results_regular65"
    common.main()
    out=common.OUT
    checks=json.loads((out/"validation_checks.json").read_text())
    c=duckdb.connect();c.execute("SET threads=2");c.execute("SET memory_limit='1200MB'")
    p=f"{out}/panel/[12]*.parquet"
    bad=c.execute(f"""SELECT count(*) FROM read_parquet('{p}') WHERE eligible AND (
      NOT regular OR age>80 OR age<0 OR g1 NOT BETWEEN 50 AND 80
      OR g2 NOT BETWEEN 50 AND 80 OR g3 NOT BETWEEN 50 AND 80
      OR period_sessions NOT BETWEEN 60 AND 70 OR forecast_idx<>event_idx+period_sessions
      OR clock IS NULL OR seasonal_clock IS DISTINCT FROM
        CASE WHEN seasonal_forecast_idx-idx>=0 AND seasonal_forecast_idx-idx<age
          THEN idx-seasonal_forecast_idx ELSE age END)""").fetchall()[0][0]
    assert bad==0
    states=c.execute(f"""SELECT DISTINCT permno,known_idx,g1,g2,g3,period_sessions
      FROM read_parquet('{p}') WHERE eligible""").df()
    ev=c.execute(f"""SELECT permno,known_idx,event_idx FROM
      read_parquet('{common.HERE.parent}/results/linked_events.parquet')
      ORDER BY permno,known_idx""").df()
    delta=ev.groupby("permno").event_idx.diff()
    ev["x1"]=delta
    ev["x2"]=delta.groupby(ev.permno).shift(1)
    ev["x3"]=delta.groupby(ev.permno).shift(2)
    ev["med"]=ev[["x1","x2","x3"]].median(axis=1)
    z=states.merge(ev,on=["permno","known_idx"],validate="one_to_one")
    assert len(z)==len(states)
    for actual,expected in [("g1","x1"),("g2","x2"),("g3","x3"),("period_sessions","med")]:
        assert np.allclose(z[actual],z[expected])
    assert z[["x1","x2","x3"]].ge(50).all().all()
    assert z[["x1","x2","x3"]].le(80).all().all()
    assert z.med.between(60,70).all()
    choice=json.loads((out/"selected_windows.json").read_text())
    assert all(-40<=int(v)<=35 for k in ["long","short"] for v in choice[k].split(":"))
    checks["regular_65_gate_and_alternative_clock"]=True
    checks["independent_three_past_announcement_gaps"]=len(states)
    checks["no_stale_81_to_130_session_selection"]=True
    (out/"validation_checks.json").write_text(json.dumps(checks,indent=2))
    print("Regular 65-session history checks passed for",len(states),"distinct eligible event states.")

if __name__=="__main__":main()
