"""Prepare an auditable CRSP extension without looking ahead for tradability."""
import hashlib
import json
from pathlib import Path
import sys
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/recent'
SOURCE = ROOT.parent / 'crsp-daily/data/crsp_a_stock/dsf_v2'
NAMES = ROOT.parent / 'song26l-us/metadata/sp500_security_names.parquet'


def canonical(value):
    return ''.join(c for c in str(value).upper() if c.isalnum())


def name_key(row):
    ticker = str(row['ticker'])
    return canonical(ticker+'-'+str(row['shareclass'])) if ticker in ['BF','BRK'] and row['shareclass'] else canonical(ticker)


def prior_eligibility(observed,delisted,lookback=16):
    complete=pd.DataFrame(observed.astype(int)).rolling(lookback,min_periods=lookback).sum().to_numpy()==lookback
    eligible=np.zeros_like(observed,dtype=bool)
    eligible[1:]=complete[:-1]&~delisted[:-1]
    return eligible


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original = np.load(ROOT/'data/sp500.npz')
    tickers = original['tickers']
    names = pd.read_parquet(NAMES)
    names['namedt'], names['nameenddt'] = pd.to_datetime(names.namedt),pd.to_datetime(names.nameenddt)
    names['key'] = names.apply(name_key,axis=1)
    anchor = pd.Timestamp('2022-05-25')
    mapping = []
    for ticker in tickers:
        candidates = names[(names['key']==canonical(ticker)) & (names.namedt<=anchor)]
        active = candidates[candidates.nameenddt>=anchor]
        ids = active.permno.unique() if len(active) else candidates.permno.unique()
        if len(ids)!=1:
            raise ValueError(f'ambiguous or missing historical ID for {ticker}: {ids}')
        mapping.append({'ticker':str(ticker),'permno':int(ids[0]),'active_at_anchor':bool(len(active))})
    pd.DataFrame(mapping).to_csv(OUT/'security_mapping.csv',index=False)
    ids = np.array([x['permno'] for x in mapping])
    assert len(np.unique(ids))==len(tickers)
    con = duckdb.connect()
    con.register('selected_ids',pd.DataFrame({'permno':ids}))
    paths = [str(SOURCE/f'{year}.parquet') for year in range(2016,2026)]
    columns = ['dlyopen','dlyhigh','dlylow','dlyclose','dlyprc','dlyvol','dlyret','dlyretx','dlycumfacpr','dlycumfacshr']
    numeric = ','.join(f'CAST({col} AS DOUBLE) AS {col}' for col in columns)
    sql = f'SELECT permno,dlycaldt,{numeric},tradingstatusflg,dlydelflg FROM read_parquet(?) JOIN selected_ids USING(permno) ORDER BY dlycaldt,permno'
    frame = con.execute(sql,[paths]).fetchdf()
    frame['dlycaldt'] = pd.to_datetime(frame.dlycaldt)
    dates = pd.DatetimeIndex(sorted(frame.loc[frame.permno==10107,'dlycaldt'].unique()))
    assert dates[0]==pd.Timestamp('2016-01-04') and dates[-1]==pd.Timestamp('2025-12-31')
    def panel(column):
        return frame.pivot(index='dlycaldt',columns='permno',values=column).reindex(index=dates,columns=ids).to_numpy()
    price = np.stack([panel(c) for c in ['dlyopen','dlyhigh','dlylow','dlyclose']],axis=-1).astype(float)
    factors, share_factors, volume = panel('dlycumfacpr').astype(float),panel('dlycumfacshr').astype(float),panel('dlyvol').astype(float)
    price_ret, total_ret = panel('dlyretx').astype(float),panel('dlyret').astype(float)
    active = panel('tradingstatusflg')=='A'
    delisted = panel('dlydelflg')=='Y'
    observed = np.isfinite(price).all(-1)&(price>0).all(-1)&np.isfinite(volume)&(volume>0)&np.isfinite(factors)&(factors>0)&np.isfinite(share_factors)&(share_factors>0)&active
    raw = np.load(ROOT/'upstream/estimate/src/data/US/sp500/baseline_data_sp500.npy',allow_pickle=True).item()
    multipliers, volume_multipliers = [],[]
    for col,ticker in enumerate(tickers):
        old = raw[str(ticker)].copy()
        old.index = pd.to_datetime(old.index)
        old = old.reindex(dates)
        overlap = (dates<=anchor)&(dates>='2022-01-01')&observed[:,col]
        overlap &= np.isfinite(old['close'])&(old['close']>0)&np.isfinite(old['volume'])&(old['volume']>0)
        index = np.flatnonzero(overlap)[-60:]
        if len(index)<10:
            raise ValueError(f'insufficient pre-test vendor overlap for {ticker}')
        multipliers.append(np.median(old['close'].to_numpy()[index]/(price[index,col,3]/factors[index,col])))
        volume_multipliers.append(np.median(old['volume'].to_numpy()[index]/(volume[index,col]*share_factors[index,col])))
    multipliers,volume_multipliers = np.array(multipliers),np.array(volume_multipliers)
    adjusted_price = price/factors[...,None]*multipliers[None,:,None]
    adjusted_volume = volume*share_factors*volume_multipliers[None,:]
    values = np.concatenate([adjusted_price,np.log1p(np.maximum(adjusted_volume,0))[...,None]],axis=-1)
    # A history may use only information observed by the previous close.
    eligible=prior_eligibility(observed,delisted)
    missing = eligible & (~np.isfinite(price_ret)|~np.isfinite(total_ret))
    historical_missing_labels = []
    if missing.any():
        day,stock = np.where(missing)
        pd.DataFrame({'date':dates[day],'ticker':tickers[stock],'price_ret':price_ret[day,stock],'total_ret':total_ret[day,stock]}).to_csv(OUT/'missing_eligible_returns.csv',index=False)
        if np.any(missing[dates>anchor]):
            raise ValueError('an eligible post-anchor security-day has no realized return; an explicit execution treatment is required')
        # Missing historical regression labels can be excluded from training.
        # No test or validation security is selected/excluded using future returns.
        historical_missing_labels = [{'date':str(dates[d].date()),'ticker':str(tickers[s])} for d,s in zip(day,stock)]
        eligible[missing] = False
    # Only historical forward fill; initial missing cells use the original pre-2020 training mean.
    for feature in range(5):
        values[:,:,feature] = pd.DataFrame(np.where(observed,values[:,:,feature],np.nan)).ffill().to_numpy()
    values = np.where(np.isfinite(values),values,original['mean'][None])
    price_ret = np.where(np.isfinite(price_ret),price_ret,0).astype(np.float32)
    total_ret = np.where(np.isfinite(total_ret),total_ret,0.)
    dates_text = dates.strftime('%Y-%m-%d').to_numpy(dtype=str)
    shared = {'returns':price_ret,'pnl_returns':total_ret,'valid':eligible,
              'observed':observed,'incidence':original['incidence'],'dates':dates_text,
              'tickers':tickers,'sectors':original['sectors']}
    frozen_features = ((values-original['mean'][None])/np.maximum(original['std'][None],1e-6)).astype(np.float32)
    np.savez_compressed(OUT/'frozen.npz',features=frozen_features,**shared)
    folds = []
    for year in [2023,2024,2025]:
        begin = int(np.searchsorted(dates.values,np.datetime64(f'{year-5}-01-01')))
        val = int(np.searchsorted(dates.values,np.datetime64(f'{year-1}-01-01')))
        test = int(np.searchsorted(dates.values,np.datetime64(f'{year}-01-01')))
        end = int(np.searchsorted(dates.values,np.datetime64(f'{year+1}-01-01')))
        mean,std = values[begin:val].mean(0),values[begin:val].std(0)
        features = ((values[begin:end]-mean)/np.maximum(std,1e-6)).astype(np.float32)
        arrays = {key:value[begin:end] if key in ['returns','pnl_returns','valid','observed','dates'] else value for key,value in shared.items()}
        dest = OUT/f'fold-{year}.npz'
        np.savez_compressed(dest,features=features,mean=mean,std=std,**arrays)
        folds.append({'year':year,'data':str(dest),'train_end':val-begin,'val_end':test-begin,
                      'train_dates':[dates_text[begin],dates_text[val-1]],'validation_dates':[dates_text[val],dates_text[test-1]],
                      'test_dates':[dates_text[test],dates_text[end-1]],'test_days':end-test})
    audit = {'source':'local validated CRSP dsf_v2 CIZ; fixed original benchmark universe',
             'source_endpoint':dates_text[-1],'mapping_date':'2022-05-25','stocks':len(tickers),'days':len(dates),
             'eligibility':'16 complete prior observations, active prior close, not previously delisted; no current/future-return eligibility filter',
             'missing_post_anchor_eligible_returns':int(missing[dates>anchor].sum()),
             'historical_missing_training_labels_excluded':historical_missing_labels,
             'frozen_test_start':int(np.searchsorted(dates.values,np.datetime64('2022-05-26'))),
             'price_multipliers':multipliers.tolist(),'volume_multipliers':volume_multipliers.tolist(),
             'folds':folds,'source_receipts':[]}
    for year in range(2016,2026):
        receipt=json.loads((SOURCE/f'{year}.done.json').read_text())
        audit['source_receipts'].append({'year':year,'receipt':receipt})
    audit['names_sha256']=hashlib.sha256(NAMES.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['price_multipliers','volume_multipliers','source_receipts']},indent=2))


if __name__=='__main__':
    main()
