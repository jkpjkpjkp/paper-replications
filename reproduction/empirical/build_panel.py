"""Build the explicitly current/restated-data reconstruction; never alter raw inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE = Path('/home/jkp/data')
DEFAULT = Path('/mnt/data/home/jkp/replications/lookahead-restated-20261118')
FIN = ['revenue', 'cogs', 'sga', 'ebit', 'fcf', 'cash', 'receivables',
       'inventory', 'other_current_assets', 'ppe', 'other_assets', 'current_debt',
       'payables', 'tax_payable', 'other_current_liabilities', 'equity', 'assets', 'liabilities']
# The prose lists 18 items, while the paper says 17 outputs; the released code's
# generic field range does not resolve the mismatch. Keep all explicitly listed
# items and report this reconstruction choice, rather than invent an omission.
GICS = [1010,1510,2010,2020,2030,2510,2520,2530,2550,3010,3020,3030,
        2540,3510,3520,4010,4020,4030,4510,4520,4530,5010,5020,5510,6010,6020]


def save_json(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False)+'\n')
    tmp.replace(path)


def quarterly_flow(values, fiscal_year, quarter):
    previous = values.shift()
    consecutive = fiscal_year.eq(fiscal_year.shift()) & quarter.eq(quarter.shift()+1)
    return values.where(quarter.eq(1), (values-previous).where(consecutive))


def make_quarterly(q):
    q = q.sort_values(['gvkey','datadate','fyearq','fqtr']).copy()
    duplicates = int(q.duplicated(['gvkey','datadate']).sum())
    # Stable explicit resolution of unusual source duplicate fiscal assignments.
    q = q.drop_duplicates(['gvkey','datadate'], keep='last').reset_index(drop=True)
    parts = []
    for _, g in q.groupby('gvkey', sort=False):
        g = g.copy()
        quarter_id = g.fyearq*4 + g.fqtr
        valid_window = quarter_id.diff().eq(1).rolling(3,min_periods=3).sum().eq(3)
        for src, dst in [('saleq','revenue'),('cogsq','cogs'),('xsgaq','sga'),('oiadpq','ebit')]:
            g[dst] = g[src].rolling(4, min_periods=4).sum().where(valid_window)
        op = quarterly_flow(g.oancfy, g.fyearq, g.fqtr)
        cap = quarterly_flow(g.capxy, g.fyearq, g.fqtr)
        g['fcf'] = (op-cap).rolling(4,min_periods=4).sum().where(valid_window)
        for src,dst in zip(['cheq','rectq','invtq','acoq','ppentq','aoq','dlcq','apq',
                            'txpq','lcoq','seqq','atq','ltq'], FIN[5:]):
            g[dst] = g[src]
        lag = g.datadate + pd.DateOffset(months=4)
        valid_rdq = g.rdq.where(g.rdq.ge(g.datadate))
        g['available'] = pd.concat([lag,valid_rdq],axis=1).max(axis=1)
        parts.append(g[['gvkey','datadate','available','dlttq','pstkq']+FIN])
    return pd.concat(parts,ignore_index=True), duplicates


def build(out):
    out.mkdir(parents=True,exist_ok=True)
    c=duckdb.connect()
    c.execute("SET threads=4")
    c.execute("SET memory_limit='8GB'")
    cols=['gvkey','datadate','rdq','fyearq','fqtr','saleq','cogsq','xsgaq','oiadpq',
          'oancfy','capxy','cheq','rectq','invtq','acoq','ppentq','aoq','dlcq','apq',
          'txpq','lcoq','seqq','atq','ltq','dlttq','pstkq']
    q=c.sql(f"""SELECT {','.join(cols)} FROM '{BASE}/compustat/all-parquet/comp/fundq/*.parquet'
        WHERE indfmt='INDL' AND datafmt='STD' AND consol='C' AND popsrc='D'
        AND curcdq='USD' AND datadate BETWEEN DATE '1965-01-01' AND DATE '2020-12-31'
        AND fqtr BETWEEN 1 AND 4 AND fyearq IS NOT NULL""").df()
    print('Quarterly source rows',len(q),flush=True)
    quarterly, duplicates=make_quarterly(q)
    del q
    quarterly=quarterly.sort_values(['gvkey','available','datadate']).drop_duplicates(['gvkey','available'],keep='last')
    # A very late old report must not replace a newer fiscal period already seen.
    latest=quarterly.groupby('gvkey').datadate.cummax()
    quarterly=quarterly[quarterly.datadate.eq(latest)]
    quarterly.to_parquet(out/'quarterly_features.parquet',index=False)
    c.register('quarterly',quarterly)
    c.execute(f"""CREATE TEMP TABLE m AS SELECT r.*,l.gvkey,l.ccm_liid,
        last_day(r.mthcaldt) AS date, year(r.mthcaldt)*12+month(r.mthcaldt) AS month_id,
        abs(r.mthprc)*r.shrout/1000.0 AS market_cap,
        co.gind
        FROM '{BASE}/crsp-monthly/parquet/*.parquet' r
        JOIN '{BASE}/compustat/ccm/primary_links.parquet' l ON r.permno=l.permno
          AND r.mthcaldt>=coalesce(l.ccm_linkdt,DATE '1900-01-01')
          AND r.mthcaldt<=coalesce(l.ccm_linkenddt,DATE '2100-01-01')
        LEFT JOIN '{BASE}/compustat/all-parquet/comp/company/*.parquet' co USING(gvkey)
        WHERE r.mthcaldt BETWEEN DATE '1965-01-01' AND DATE '2020-12-31'
          AND r.usincflg='Y' AND r.securitytype='EQTY' AND r.securitysubtype='COM'
          AND r.primaryexch IN ('N','A','Q') AND NOT r.siccd BETWEEN 6000 AND 6999
        QUALIFY row_number() OVER(PARTITION BY gvkey,last_day(r.mthcaldt)
          ORDER BY abs(r.mthprc)*r.shrout DESC NULLS LAST,r.permno)=1""")
    # Latest AVAILABLE quarter only; do not join on the fiscal period end.
    c.execute("""CREATE TEMP TABLE panel AS SELECT m.*,
        q.* EXCLUDE(gvkey) FROM m ASOF LEFT JOIN quarterly q
        ON m.gvkey=q.gvkey AND m.date>=q.available""")
    short_path=BASE/'compustat/gui-downloads/short-interest-1973-2019/short-interest.parquet'
    # Reject missing identifiers and ambiguous duplicate observations rather than
    # silently adding together shares across issues. Use the CCM issue identifier.
    c.execute(f"""CREATE TEMP TABLE si AS SELECT gvkey,iid,datadate+INTERVAL 1 MONTH AS si_available,
        max(shortint) AS shortint FROM '{short_path}'
        WHERE gvkey IS NOT NULL AND datadate IS NOT NULL AND shortint>=0
        GROUP BY gvkey,iid,datadate HAVING count(DISTINCT shortint)=1""")
    p=c.sql("""SELECT p.*,s.si_available,s.shortint FROM panel p ASOF LEFT JOIN si s
        ON p.gvkey=s.gvkey AND p.ccm_liid=s.iid AND p.date>=s.si_available
        ORDER BY p.gvkey,p.date""").df()
    assert not p.duplicated(['gvkey','date']).any()
    p[FIN]=p.groupby('gvkey',sort=False)[FIN].ffill()
    # Forward-only momentum, keeping gaps missing instead of crossing them.
    groups=p.groupby('gvkey',sort=False)
    logs=np.log1p(p.mthret.where(p.mthret.gt(-1)))
    p['_logret']=logs
    for horizon in [1,3,6,9]:
        cumulative=groups['_logret'].transform(lambda s: s.rolling(horizon,min_periods=horizon).sum())
        continuous=p.month_id.sub(groups.month_id.shift(horizon-1)).eq(horizon-1)
        p[f'momentum_{horizon}']=np.expm1(cumulative.where(continuous))
        p[f'mom_rank_{horizon}']=p.groupby('date')[f'momentum_{horizon}'].rank(pct=True)
    p['short_fraction']=p.shortint/(p.shrout*1000)
    p.loc[(p.date-p.si_available).dt.days.gt(90),'short_fraction']=np.nan
    cpi=pd.read_csv(out/'CPIAUCSL.csv',parse_dates=[0]).rename(columns={'observation_date':'date'})
    cpi['month_id']=cpi.date.dt.year*12+cpi.date.dt.month
    base=float(cpi.loc[cpi.month_id.eq(2010*12+1),'CPIAUCSL'].iloc[0])
    cpi_map=cpi.set_index('month_id').CPIAUCSL
    p['real_cap']=p.market_cap*base/p.month_id.map(cpi_map)
    p['eligible']=p.real_cap.ge(100) & p.market_cap.gt(0) & p.ebit.notna() & p.tradingstatusflg.eq('A')
    p['ev']=p.market_cap+p.current_debt+p.dlttq+p.pstkq.fillna(0)-p.cash
    p.to_parquet(out/'monthly_panel.parquet',index=False)
    print('Monthly panel',len(p),'companies',p.gvkey.nunique(),flush=True)
    aux=[f'mom_rank_{h}' for h in [1,3,6,9]]+['short_fraction']
    gind=pd.to_numeric(p.gind,errors='coerce').fillna(0).astype(int)//100
    for g in GICS+[0]:
        name=f'industry_{g}'
        p[name]=(gind.eq(g) if g else ~gind.isin(GICS)).astype(float)
        aux.append(name)
    for i,(lo,hi) in enumerate([(0,300),(300,2000),(2000,10000),(10000,np.inf)]):
        name=f'size_{i}';p[name]=((p.real_cap>=lo)&(p.real_cap<hi)).astype(float);aux.append(name)
    # An input must have five annual observations AND 12 consecutive listing
    # months. No future membership requirement; labels are a separate mask.
    index=pd.MultiIndex.from_frame(p[['gvkey','month_id']])
    end=np.flatnonzero(p.eligible & p.month_id.between(1970*12+1,2019*12+12))
    ids=[]
    for delta in [48,36,24,12,0,12]:
        offset=-delta if len(ids)<5 else delta
        keys=pd.MultiIndex.from_arrays([p.gvkey.iloc[end],p.month_id.iloc[end]+offset])
        ids.append(index.get_indexer(keys))
    ids=np.stack(ids,axis=1)
    complete=(ids[:,:5]>=0).all(axis=1)
    end=end[complete];ids=ids[complete]
    # Require observed EBIT at all five input timestamps; avoid imputed histories.
    valid=np.isfinite(p.ebit.to_numpy()[ids[:,:5]]).all(axis=1)
    # Explicitly require a continuous 60-month monthly history.
    first=index.get_indexer(pd.MultiIndex.from_arrays([p.gvkey.iloc[end],p.month_id.iloc[end]-59]))
    valid &= (first>=0) & ((end-first)==59)
    end=end[valid];ids=ids[valid]
    np.save(out/'sample_rows.npy',ids)
    meta=p.iloc[end][['gvkey','permno','date','month_id','market_cap','ev','mthprc','mthvol','available']].reset_index(drop=True)
    meta['target_month_id']=meta.month_id+12
    meta['target_row']=ids[:,5]
    formation=meta.month_id.to_numpy()
    train_period=(formation>=1970*12+1)&((formation+12)<=1999*12+12)&(ids[:,5]>=0)
    companies=np.sort(meta.loc[train_period,'gvkey'].unique())
    rng=np.random.default_rng(521);rng.shuffle(companies)
    heldout=set(companies[:int(np.ceil(len(companies)*.3))])
    validation=meta.gvkey.isin(heldout).to_numpy()
    meta['split']=np.where(train_period,np.where(validation,'valid','train'),
        np.where(formation>=2000*12+1,'test','gap'))
    fin=p[FIN].to_numpy(dtype=np.float32)
    auxiliaries=p[aux].to_numpy(dtype=np.float32)
    cap=meta.market_cap.to_numpy(dtype=np.float32)
    train_idx=np.flatnonzero(meta.split.eq('train'))
    # Fit normalization on training inputs only; target scaling reuses it.
    count=np.zeros(len(FIN)+len(aux));sums=count.copy();squares=count.copy()
    for block in np.array_split(train_idx,max(1,len(train_idx)//10000+1)):
        a=np.concatenate([fin[ids[block,:5]]/cap[block,None,None],auxiliaries[ids[block,:5]]],axis=2).reshape(-1,len(count))
        count+=np.isfinite(a).sum(axis=0);sums+=np.nansum(a.astype(float),axis=0);squares+=np.nansum(a.astype(float)**2,axis=0)
    mean=np.divide(sums,count,out=np.zeros_like(sums),where=count>0)
    std=np.sqrt(np.maximum(np.divide(squares,count,out=np.ones_like(squares),where=count>0)-mean**2,1e-12))
    # One-hot features retain their exact coding.
    mean[len(FIN)+5:]=0;std[len(FIN)+5:]=1
    n=len(meta);x=np.lib.format.open_memmap(out/'x.npy',mode='w+',dtype='float32',shape=(n,5,len(count)))
    y=np.lib.format.open_memmap(out/'y.npy',mode='w+',dtype='float32',shape=(n,5,len(FIN)))
    for start in range(0,n,10000):
        stop=min(start+10000,n);ii=ids[start:stop];scale=cap[start:stop,None,None]
        xx=np.concatenate([fin[ii[:,:5]]/scale,auxiliaries[ii[:,:5]]],axis=2)
        x[start:stop]=np.nan_to_num((xx-mean)/std,nan=0,posinf=0,neginf=0)
        yy=fin[np.maximum(ii[:,1:],0)]/scale
        yy[ii[:,1:]<0]=np.nan
        y[start:stop]=(yy-mean[:len(FIN)])/std[:len(FIN)]
    x.flush();y.flush()
    meta.to_parquet(out/'samples.parquet',index=False)
    save_json(out/'scaling.json',{'financial_fields':FIN,'auxiliary_fields':aux,'mean':mean.tolist(),'std':std.tolist(),
        'observed_training_input_counts':count.tolist(),'heldout_companies':sorted(heldout),'seed':521})
    audit={'panel_rows':len(p),'panel_companies':int(p.gvkey.nunique()),'samples':n,
           'split_counts':meta.split.value_counts().to_dict(),'financial_outputs':len(FIN),
           'input_features':len(count),'duplicate_quarter_rows_resolved':duplicates,
           'input_missing_fraction':{k:float(p[k].isna().mean()) for k in FIN+aux[:5]},
           'train_last_target_month':int(meta.loc[meta.split.isin(['train','valid']),'target_month_id'].max()),
           'test_first_formation_month':int(meta.loc[meta.split.eq('test'),'month_id'].min()),
           'current_restated_data':True,'historical_vintages_verified':False,
           'cpi_source':'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL',
           'cpi_sha256':hashlib.sha256((out/'CPIAUCSL.csv').read_bytes()).hexdigest()}
    save_json(out/'panel_audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    build(parser.parse_args().out)
