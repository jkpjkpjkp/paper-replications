"""Build a point-in-time CRSP top-1000 panel, without modifying source data."""
from pathlib import Path
import json, hashlib
import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
con=duckdb.connect()
con.execute("SET threads=2")
con.execute("SET memory_limit='2GB'")
monthly=con.execute(f"""SELECT permno, yyyymm, CAST(mthret AS DOUBLE) ret,
 abs(CAST(mthprc AS DOUBLE)) price, CAST(mthcap AS DOUBLE) cap,
 CAST(mthvol AS DOUBLE)/NULLIF(shrout*1000.,0) turnover,
 securitytype, securitysubtype, sharetype, primaryexch, tradingstatusflg,
 mthdelflg, siccd FROM read_parquet('{BASE}/crsp-monthly/parquet/*.parquet')
 WHERE yyyymm BETWEEN 198901 AND 202512""").df()
monthly['month']=pd.to_datetime(monthly.yyyymm.astype(str),format='%Y%m').dt.to_period('M').astype('int64')
assert not monthly.duplicated(['permno','month']).any()
pieces=[]
for year in range(1989,2026):
    f=BASE/f'crsp-daily/parquet/{year}.parquet'
    d=con.execute(f"""SELECT permno, CAST(floor(yyyymmdd/100) AS BIGINT) yyyymm,
      count(dlyret) ndays, sum(pow(CAST(dlyret AS DOUBLE),2)) rv,
      sum(CASE WHEN dlyret<0 THEN pow(CAST(dlyret AS DOUBLE),2) ELSE 0 END) downside,
      max(CAST(dlyret AS DOUBLE)) maxret, skewness(CAST(dlyret AS DOUBLE)) skew
      FROM read_parquet('{f}') GROUP BY permno, floor(yyyymmdd/100)""").df()
    pieces.append(d)
    print('aggregated',year,len(d),flush=True)
rv=pd.concat(pieces,ignore_index=True)
df=monthly.merge(rv,on=['permno','yyyymm'],how='left',validate='1:1').sort_values(['permno','month'])
# Keep all securities during lag construction; eligibility uses only formation-month fields.
g=df.groupby('permno',sort=False)
for k in [1,2,3,6,12]:
    df[f'lag{k}']=g.ret.shift(k)
for k in [3,6,12]:
    df[f'mom{k}']=g.ret.transform(lambda s:s.rolling(k,min_periods=k).sum())
    df[f'rv{k}']=g.rv.transform(lambda s:s.rolling(k,min_periods=k).mean())
df['mom12skip']=df['mom12']-df.ret
df['continuous12']=(df.month-g.month.shift(11)==11)
df['yret']=g.ret.shift(-1)
df['yrv']=g.rv.shift(-1)
df['yn']=g.ndays.shift(-1)
df['ymonth']=g.month.shift(-1)
df['ydelist']=g.mthdelflg.shift(-1)
df['logcap']=np.log(df.cap.clip(lower=1))
df['logprice']=np.log(df.price.clip(lower=.01))
df['logturnover']=np.log1p(df.turnover.clip(lower=0))
for c in ['rv','rv3','rv6','rv12']:
    df['log_'+c]=np.log(df[c].clip(lower=1e-8))
df['downshare']=df.downside/df.rv.clip(lower=1e-8)
elig=(df.securitytype.eq('EQTY') & df.securitysubtype.eq('COM') & df.sharetype.eq('NS')
      & df.primaryexch.isin(['N','A','Q']) & df.tradingstatusflg.eq('A')
      & (df.price>=5) & (df.cap>0) & (df.ndays>=15) & df.continuous12)
df=df[elig].copy()
df['size_rank']=df.groupby('month').cap.rank(method='first',ascending=False)
df=df[df.size_rank<=1000].copy()
formed=len(df)
df['target_month']=df.month+1
df['return_observed']=df.ymonth.eq(df.target_month)&df.yret.notna()
df['variance_observed']=df.return_observed & df.yrv.gt(0) & df.yn.ge(15)
df.groupby('target_month').agg(eligible=('permno','size'),returns=('return_observed','sum'),joint=('variance_observed','sum')).to_csv(ROOT/'data/coverage.csv')
# Cross-sectional ranks use every eligible formation-month security, before target filtering.
raw=['ret','lag1','lag2','mom3','mom6','mom12skip','log_rv','log_rv3','log_rv6','log_rv12',
     'downshare','maxret','skew','logcap','logprice','logturnover']
features=raw.copy()
for c in raw:
    name='rank_'+c
    df[name]=df.groupby('month')[c].rank(pct=True)-.5
    features.append(name)
for c in ['ret','rv']:
    df['market_'+c]=df.groupby('month')[c].transform('mean')
    features.append('market_'+c)
df=df[df.target_month.between(pd.Period('1991-01','M').ordinal,pd.Period('2025-12','M').ordinal)].copy()
df['year']=df.target_month.map(lambda m:pd.Period(ordinal=int(m),freq='M').year)
df['date']=df.target_month.map(lambda m:str(pd.Period(ordinal=int(m),freq='M')))
keep=['permno','month','target_month','date','year','yret','yrv','yn','size_rank','cap','ydelist','return_observed','variance_observed']+features
df[keep].to_parquet(ROOT/'data/panel.parquet',index=False)
meta={'features':features,'formed_rows':formed,'rows':len(df),'joint_rows':int(df.variance_observed.sum()),
      'return_rows':int(df.return_observed.sum()),'unique_stocks':df.permno.nunique(),
      'period':[df.date.min(),df.date.max()], 'target_return':'CRSP CIZ monthly total simple return, untrimmed',
      'target_variance':'sum of squared CRSP daily simple total returns; >=15 valid days; no annualization',
      'universe':'largest 1000 eligible common stocks by formation-month capitalization, price >= $5, N/A/Q, >=12 contiguous historical months',
      'features_timing':'all features through formation month; label is exact next calendar month',
      'source_validation':json.loads((BASE/'crsp-daily/validation.json').read_text())}
(ROOT/'data/metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
