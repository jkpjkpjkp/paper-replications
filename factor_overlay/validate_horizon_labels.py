#!/usr/bin/env python3
"""Read 50 labels anew from raw daily panels and audit feature preservation."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from data import OUT,PANEL


def main():
    original=dict(np.load(OUT/'train.npz'))
    held=dict(np.load(OUT/'train5.npz'))
    cal=pd.read_csv(OUT/'calendar.csv',parse_dates=['date'])
    ends=cal.assign(year=cal.date.dt.year).groupby('year').idx.max().to_dict()
    keep=np.array([int(i)+4<=ends[int(y)] for i,y in zip(original['idx'],original['year'])])
    for k in original:
        if k!='y':
            assert np.array_equal(original[k][keep],held[k],equal_nan=True),k
    assert np.array_equal(held['label_end_idx'],held['idx']+4)
    checks=pd.read_csv(OUT/'train5_independent_labels.csv')
    checks=checks.iloc[np.linspace(0,len(checks)-1,50,dtype=int)]
    c=duckdb.connect();c.execute('SET threads=2')
    records=[]
    for r in checks.itertuples():
        rows=c.execute(f"SELECT idx,ret,retx FROM read_parquet('{PANEL}/{r.year}.parquet') WHERE permno={r.permno} AND idx BETWEEN {r.idx} AND {r.idx+4} ORDER BY idx").fetchall()
        raw={int(idx):(ret,retx) for idx,ret,retx in rows}
        # Independent absolute-dollar cash + share-mark account.
        value=1.;cash=0.;missing=False
        for idx in range(r.idx,r.idx+5):
            if idx not in raw:
                cash+=value;value=0.;missing=True;break
            ret,retx=raw[idx]
            ret=0. if ret is None or not np.isfinite(ret) else float(ret)
            retx=ret if retx is None or not np.isfinite(retx) else float(retx)
            cash+=value*(ret-retx)
            value*=1+retx
        payoff=value+cash-1
        loc=np.flatnonzero((held['idx']==r.idx)&(held['permno']==r.permno))
        assert len(loc)
        signed=held['Xbase'][loc,6]
        error=float(np.max(abs(held['y'][loc]-signed*payoff/5)))
        assert error<1e-8,(r,error)
        assert abs(payoff-r.gross_five_day)<1e-12
        records.append(dict(year=r.year,idx=r.idx,permno=r.permno,raw_cash_account_payoff=payoff,
                            maximum_stored_label_error=error,forced_missing=missing))
    pd.DataFrame(records).to_csv(OUT/'train5_raw_spotchecks.csv',index=False)
    report=json.loads((OUT/'train5_audit.json').read_text())
    report['fresh_raw_panel_spotchecks']=len(records)
    report['fresh_raw_panel_max_label_error']=max(x['maximum_stored_label_error'] for x in records)
    report['all_original_features_and_weights_preserved_exactly']=True
    (OUT/'train5_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['fresh_raw_panel_spotchecks','fresh_raw_panel_max_label_error','all_original_features_and_weights_preserved_exactly']},indent=2))


if __name__=='__main__':
    main()
