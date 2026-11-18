"""Independently reconcile saved holdings against raw CRSP dates and returns."""
import argparse
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from build_panel import DEFAULT, BASE, save_json


def verify(out):
    c=duckdb.connect();c.execute('SET threads=4')
    checks={}
    for file in sorted((out/'portfolios').glob('*_holdings.parquet')):
        c.execute(f"CREATE OR REPLACE TEMP VIEW h AS SELECT * FROM '{file}'")
        bad=c.sql(f"""SELECT count(*) FROM h LEFT JOIN '{BASE}/crsp-monthly/parquet/*.parquet' r
            ON h.permno=r.permno AND year(r.mthcaldt)*12+month(r.mthcaldt)=h.month_id-1
            WHERE r.permno IS NULL OR r.mthdelflg<>'N' OR r.mthprcdt<>r.mthcaldt OR h.available>h.date""").fetchall()[0][0]
        wrong=c.sql(f"""SELECT count(*) FROM h LEFT JOIN '{BASE}/crsp-monthly/parquet/*.parquet' r
            ON h.permno=r.permno AND year(r.mthcaldt)*12+month(r.mthcaldt)=h.month_id
            WHERE (r.mthret IS NULL)<>(h.mthret IS NULL) OR abs(r.mthret-h.mthret)>1e-12""").fetchall()[0][0]
        missing=c.sql('SELECT count(*) FROM h WHERE mthret IS NULL').fetchall()[0][0]
        counts=c.sql('select count(*),count(distinct month_id),count(distinct (permno,month_id)) from h').fetchall()[0]
        assert counts==(12000,240,12000) and bad==0 and wrong==0,(file,counts,bad,wrong)
        checks[file.stem]={'rows':counts[0],'months':counts[1],'duplicate_holdings':0,
                          'invalid_formation_rows':bad,'incorrect_held_returns':wrong,
                          'missing_held_returns':missing}
    x=np.load(out/'x.npy',mmap_mode='r');y=np.load(out/'y.npy',mmap_mode='r')
    meta=pd.read_parquet(out/'samples.parquet')
    mask=meta.split.eq('test').to_numpy()&np.isfinite(y[:,-1,3])
    mse=float(np.mean((x[mask,-1,3].astype(float)-y[mask,-1,3])**2))
    report=json.loads((out/'comparison.json').read_text())
    assert abs(mse-report['models']['QFM']['standardized_ebit_mse'])<1e-10
    checks['independent_qfm_mse']=mse
    save_json(out/'verification.json',checks)
    print('Verified raw CRSP dates/returns for',len(checks)-1,'portfolios; QFM MSE',mse,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    verify(parser.parse_args().out)
