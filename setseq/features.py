"""Causal daily reconstruction of Tables 10/11; choices in ASSUMPTIONS.txt.

Raw CRSP/Compustat inputs are read only. Raw features retain missing values;
imputation and aggregate scaling happen separately for each training window.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "configs/equity_features.json").read_text())
NAMES = SPEC["unit_features_in_table10"]
MEDIANS = list(SPEC["cross_sectional_median_tcodes_in_table11"])
ALIASES = {"DP2A": "DPI2A", "ST_REV": "ST_Rev"}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rolling_sum(a, window):
    """Full-window sums; NaNs treated as zero (caller checks counts)."""
    c = np.concatenate([np.zeros_like(a[:1]), np.cumsum(np.nan_to_num(a), axis=0)])
    out = np.full_like(a, np.nan, dtype=float)
    out[window - 1:] = c[window:] - c[:-window]
    return out


def rank_features(values):
    """[time, stocks] -> centered fractional ranks; ties/missing explicit."""
    d = pd.DataFrame(values)
    n = d.count(axis=1)
    ranks = d.rank(axis=1, method="average").sub(1).div((n - 1).replace(0, np.nan), axis=0) - .5
    return ranks.fillna(0).to_numpy(dtype=np.float32)


def transform_medians(a, code):
    s = pd.Series(a)
    if code in (3, 4):
        s = np.log(s.where(s > 0))
    if code in (2, 3, 4):
        s = s.diff()
    if code == 4:
        s = s.diff()
    return s.to_numpy()


def training_scale(a, train_mask):
    train = a[train_mask]
    finite = np.isfinite(train)
    center = float(np.median(train[finite])) if finite.any() else 0.
    filled = np.where(np.isfinite(a), a, center)
    scale = float(np.std(filled[train_mask]))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.
    return np.clip((filled - center) / scale, -10, 10).astype(np.float32), center, scale


def load_market(data_root, out, permnos):
    cache = out / "market.parquet"
    if not cache.exists():
        db = duckdb.connect(config={"threads": 4, "memory_limit": "4GB"})
        ids = pd.DataFrame({"permno": permnos})
        db.register("ids", ids)
        columns = "d.permno,dlycaldt,dlyret,dlyretx,dlyprc,dlycap,shrout,dlyvol,dlybid,dlyask,dlycumfacpr,dlycumfacshr"
        chunks = []
        for y in range(1989, 2022):
            p = data_root / f"crsp-daily/parquet/{y}.parquet"
            d = db.execute(f"SELECT {columns} FROM read_parquet(?) d JOIN ids USING (permno)", [str(p)]).df()
            chunks.append(d)
            print(f"market {y}: {len(d):,} rows", flush=True)
        all_data = pd.concat(chunks, ignore_index=True)
        if all_data.duplicated(["permno", "dlycaldt"]).any():
            raise ValueError("Duplicate CRSP stock/date")
        all_data.to_parquet(cache, index=False)
    d = pd.read_parquet(cache)
    d["dlycaldt"] = pd.to_datetime(d.dlycaldt)
    dates = np.sort(d.dlycaldt.unique())
    frames = {}
    for name in d.columns:
        if name not in ("permno", "dlycaldt"):
            frames[name] = d.pivot(index="dlycaldt", columns="permno", values=name).reindex(index=dates, columns=permnos).astype(float)
    return pd.DatetimeIndex(dates), frames


def accounting_events(data_root, permnos):
    fields = "at lt seq ceq pstkrv pstkl pstk txditc txdb che ivao dlc dltt mib sale revt cogs xsga xrd xad gp ni ib dp capx wcapch act lct txp ppegt invt oiadp xint".split()
    db = duckdb.connect(config={"threads": 4, "memory_limit": "2GB"})
    db.register("ids", pd.DataFrame({"permno": permnos}))
    p = data_root / "compustat/ccm/annual_crsp_monthly.parquet"
    quoted_fields = ','.join('"'+n+'"' for n in fields)
    q = f"""SELECT gvkey, permno, datadate, fyear, ccm_linkdt, ccm_linkenddt, {quoted_fields}
        FROM read_parquet(?) JOIN ids USING (permno)
        WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
        AND curcd='USD' AND datadate BETWEEN DATE '1987-01-01' AND DATE '2021-12-31'
        ORDER BY gvkey,datadate"""
    a = db.execute(q, [str(p)]).df()
    a[fields] = a[fields].astype(float)
    if a.duplicated(["gvkey", "datadate"]).any():
        raise ValueError("Duplicate standard accounting record")
    ps = a.pstkrv.fillna(a.pstkl).fillna(a.pstk).fillna(0)
    sh = a.seq.fillna(a.ceq + ps).fillna(a['at'] - a['lt'])
    be = sh + a.txditc.fillna(0) - ps
    be = be.where(be > 0)
    lag = a.groupby("gvkey", sort=False)[fields + ["datadate"]].shift()
    valid_lag = (a.datadate - lag.datadate).dt.days.between(300, 430)
    lag_at = lag['at'].where(valid_lag & (lag['at'] > 0))
    assets = a['at'].where(a['at'] > 0)
    sales = a.sale.where(a.sale != 0)
    noa = a.dlc.fillna(0) + a.dltt.fillna(0) + a.mib.fillna(0) + a.pstk.fillna(0) + a.ceq - a.che - a.ivao.fillna(0)
    lag_noa = noa.groupby(a.gvkey, sort=False).shift().where(valid_lag)
    # The reference's working-capital formula subtracts DLC; retain it explicitly.
    wc = a.act - a.che - a.lct - a.dlc.fillna(0) - a.txp.fillna(0)
    lag_wc = wc.groupby(a.gvkey, sort=False).shift().where(valid_lag)
    f = pd.DataFrame(index=a.index)
    f['Investment'] = (a['at'] - lag['at']) / lag_at
    f['NOA'] = noa / lag_at
    f['DPI2A'] = (a.ppegt + a.invt - lag.ppegt - lag.invt) / lag_at
    f['PROF'] = a.gp.fillna(a.sale - a.cogs) / be
    f['CTO'] = a.sale / lag_at
    f['FC2Y'] = (a.xsga + a.xrd.fillna(0) + a.xad.fillna(0)) / sales
    f['OP'] = (a.revt.fillna(a.sale) - a.cogs - a.xint.fillna(0) - a.xsga.fillna(0)) / be
    f['PM'] = a.oiadp / sales
    f['RNA'] = a.oiadp / lag_noa.where(lag_noa.abs() > 1e-12)
    f['D2A'] = a.dp / assets
    f['OA'] = (wc - lag_wc - a.dp) / lag_at
    f['OL'] = (a.cogs + a.xsga.fillna(0)) / assets
    f['PCM'] = (a.sale - a.cogs) / sales
    f['C'] = a.che / assets
    # Legacy WCAPCH is almost entirely absent in modern standard Compustat.
    # Recover its change-in-working-capital role from consecutive balance sheets.
    wcap_change = a.wcapch.fillna((a.act-a.lct)-(lag.act-lag.lct)).where(valid_lag)
    f['CF'] = (a.ni + a.dp - wcap_change - a.capx) / be
    debt = a.dltt.fillna(0) + a.dlc.fillna(0)
    f['Lev'] = debt / (debt + sh).where((debt + sh) > 0)
    f['AT'] = assets
    # Numerators for features needing market equity (all monetary amounts USD millions).
    f['_be'] = be
    f['_cf'] = a.ib + a.dp + a.txdb.fillna(0)
    f['_ib'] = a.ib
    f['_ceq_txdb'] = a.ceq + a.txdb.fillna(0)
    f['permno'] = a.permno
    f['datadate'] = a.datadate
    f['available'] = a.datadate + pd.DateOffset(months=6) + pd.Timedelta(days=1)
    f['expires'] = a.datadate + pd.DateOffset(months=18)
    f['link_end'] = a.ccm_linkenddt
    f['link_start'] = a.ccm_linkdt
    f['gvkey'] = a.gvkey
    return f.sort_values(['permno', 'available', 'datadate'])


def build_raw(data_root, out, permnos):
    dates, m = load_market(data_root, out, permnos)
    ff = pd.read_parquet(data_root / 'factors-daily/ff3_daily.parquet').set_index('date').reindex(dates)
    if ff.isna().any().any():
        raise ValueError('Fama-French factors missing on a CRSP date')
    r = m['dlyret']
    logret = np.log1p(r.where(r > -1))
    raw = np.lib.format.open_memmap(out / 'raw_features.npy', mode='w+', dtype='float32', shape=(len(dates), len(permnos), len(NAMES)))
    raw[:] = np.nan
    def save(name, value):
        arr = np.asarray(value, dtype=np.float32)
        raw[:, :, NAMES.index(name)] = np.where(np.isfinite(arr), arr, np.nan)
        raw.flush()
        print(f'feature {name}: finite {np.isfinite(arr).mean():.3%}', flush=True)
    for name, window, shift in [('r2_1',21,0),('ST_Rev',21,0),('r12_2',231,21),('r12_7',126,126),('r36_13',504,252),('Ret_W1',5,0)]:
        save(name, np.expm1(logret.rolling(window, min_periods=window).sum().shift(shift)))
    save('Ret_D1', r)
    save('STD_W1', r.rolling(5).std(ddof=0))
    save('Variance', r.rolling(42).var(ddof=0))
    # Split-adjusted prices and volumes; cumulative factors cancel in ratios.
    price = m['dlyprc'].abs() / m['dlycumfacpr'].where(m['dlycumfacpr'] > 0)
    volume = m['dlyvol'] * m['dlycumfacshr']
    shares = m['shrout'] * 1000 * m['dlycumfacshr']
    save('Rel2High', price / price.rolling(252, min_periods=120).max())
    # Absolute volume must not depend on the final-vintage cumulative split factor.
    save('Vol', m['dlyvol'].rolling(5).sum())
    save('LTurnover', volume.rolling(21).sum() / shares.where(shares > 0))
    save('LME', m['dlycap'] / 1000)
    spread = (m['dlyask'] - m['dlybid']) / ((m['dlyask'] + m['dlybid']) / 2)
    spread = spread.where((m['dlybid'] > 0) & (m['dlyask'] >= m['dlybid']))
    save('Spread', spread.rolling(21, min_periods=10).mean())
    save('RF', np.broadcast_to(ff.rf.to_numpy()[:,None], r.shape))
    exlog = logret.sub(np.log1p(ff.rf), axis=0)
    marketlog = np.log1p(ff.mkt_rf + ff.rf) - np.log1p(ff.rf)
    corr = exlog.rolling(3).sum().rolling(1260, min_periods=750).corr(marketlog.rolling(3).sum())
    beta = corr.mul(exlog.rolling(252, min_periods=120).std()).div(marketlog.rolling(252, min_periods=120).std(), axis=0)
    save('Beta', beta)
    # OLS residual variance with intercept and three FF factors in 42-session windows.
    xx = np.column_stack([np.ones(len(dates)), ff[['mkt_rf','smb','hml']].to_numpy()])
    gram = rolling_sum(xx[:,:,None] * xx[:,None,:], 42)
    inverse = np.zeros_like(gram)
    inverse[41:] = np.linalg.pinv(gram[41:])
    yy = r.sub(ff.rf, axis=0).to_numpy()
    cross = np.stack([rolling_sum(yy * xx[:,i,None], 42) for i in range(4)], axis=2)
    b = np.einsum('tij,tmj->tmi', inverse, np.nan_to_num(cross))
    resid = (rolling_sum(yy**2, 42) - (b * cross).sum(axis=2)) / 42
    resid[rolling_sum(np.isfinite(yy).astype(float),42) < 42] = np.nan
    save('Resid_Var', np.maximum(resid,0))
    del cross, b, yy, resid, beta, corr
    # Volume response is standardized relative to the previous 60 observations.
    suv = np.full(r.shape, np.nan, dtype=np.float32)
    for j in range(len(permnos)):
        rr = r.iloc[:,j].to_numpy()
        vv = volume.iloc[:,j].to_numpy() / 1e6
        x = np.column_stack([np.ones(len(rr)), np.maximum(rr,0), np.maximum(-rr,0)])
        valid = np.isfinite(rr) & np.isfinite(vv)
        x[~valid] = np.nan
        g = rolling_sum(x[:,:,None] * x[:,None,:],60)
        xy = rolling_sum(x * vv[:,None],60)
        v2 = rolling_sum(vv**2,60)
        count = rolling_sum(valid.astype(float),60)
        # Fit through t-1, score t. Ridge only regularizes near-singular designs.
        g = np.roll(g,1,axis=0); xy=np.roll(xy,1,axis=0); v2=np.roll(v2,1); count=np.roll(count,1)
        ok = (count == 60) & np.isfinite(g).all(axis=(1,2)) & valid
        coef = np.zeros((len(rr),3))
        coef[ok] = np.linalg.solve(g[ok] + np.eye(3)*1e-10, xy[ok,:,None])[...,0]
        std = np.sqrt(np.maximum((v2 - (coef*xy).sum(1))/60,0))
        daily = np.full(len(rr),np.nan)
        good = ok & (std > 1e-8)
        daily[good] = (vv[good] - (coef[good]*x[good]).sum(1))/std[good]
        suv[:,j] = pd.Series(daily).rolling(21,min_periods=21).mean()
    save('SUV', suv)
    events = accounting_events(data_root,permnos)
    events.to_parquet(out/'accounting_events.parquet',index=False)
    account_names = [n for n in NAMES if not np.isfinite(raw[:,:,NAMES.index(n)]).any()]
    for j,pid in enumerate(permnos):
        a = events[events.permno == pid].copy()
        if not len(a):
            continue
        # Last available fiscal record wins; records expire and link bounds apply.
        a = a.drop_duplicates('available',keep='last')
        inds = np.searchsorted(a.available.to_numpy(), dates.to_numpy(), side='right') - 1
        safe = np.maximum(inds,0)
        valid = (inds>=0) & (dates.to_numpy() <= a.expires.to_numpy()[safe])
        ends = a.link_end.fillna(pd.Timestamp('2099-12-31')).to_numpy()[safe]
        starts = a.link_start.fillna(pd.Timestamp('1900-01-01')).to_numpy()[safe]
        valid &= (dates.to_numpy()<=ends) & (dates.to_numpy()>=starts)
        dec = pd.to_datetime((a.available.dt.year-1).astype(str)+'-12-31').to_numpy()
        decidx = np.searchsorted(dates.to_numpy(), dec, side='right')-1
        cap = (m['dlycap'].iloc[:,j].to_numpy()/1000)
        dec_cap = cap[np.maximum(decidx,0)]
        dec_cap[(decidx<0)|(dec_cap<=0)] = np.nan
        a['A2ME'] = a.AT / dec_cap
        a['BEME'] = a._be / dec_cap
        a['CF2P'] = a._cf / dec_cap
        a['E2P'] = a._ib / dec_cap
        for n in account_names:
            if n == 'Q':
                value = (a.AT.to_numpy()[safe] + cap - a._ceq_txdb.to_numpy()[safe]) / a.AT.to_numpy()[safe]
            else:
                value = a[n].to_numpy()[safe]
            raw[:,j,NAMES.index(n)] = np.where(valid & np.isfinite(value),value,np.nan)
    raw.flush()
    np.save(out/'dates.npy', dates.to_numpy(dtype='datetime64[D]'))
    np.save(out/'permnos.npy', permnos)
    np.save(out/'returns.npy', r.to_numpy(dtype=np.float32))
    ff.to_parquet(out/'factors.parquet')
    metadata = {'names':NAMES,'shape':list(raw.shape),'data_root':str(data_root),'raw_sha256':digest(out/'raw_features.npy'),
                'sources':[{ 'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in [data_root/'compustat/ccm/annual_crsp_monthly.parquet',data_root/'factors-daily/ff3_daily.parquet']],
                'feature_code_sha256':digest(Path(__file__))}
    (out/'raw_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')


def build_panel(out, year, cohorts):
    destination = out/'panels'/str(year)
    if (destination/'manifest.json').exists():
        print(f'panel {year}: reuse completed',flush=True); return
    destination.mkdir(parents=True,exist_ok=True)
    dates = np.load(out/'dates.npy')
    permnos = np.load(out/'permnos.npy')
    chosen = cohorts.loc[(cohorts.test_year==year)&(cohorts.cohort=='paper_test_complete')].sort_values('rank').permno.to_numpy()
    assert len(chosen)==500 and len(np.unique(chosen))==500
    cols = np.searchsorted(permnos, chosen)
    assert (permnos[cols] == chosen).all()
    # Include one previous predictor date; target date is always the next session.
    begin = np.searchsorted(dates, np.datetime64(f'{year-8}-01-01'))-1
    end = np.searchsorted(dates, np.datetime64(f'{year+1}-01-01'))
    input_dates, target_dates = dates[begin:end-1],dates[begin+1:end]
    train = target_dates < np.datetime64(f'{year}-01-01')
    raw = np.load(out/'raw_features.npy',mmap_mode='r')[begin:end-1][:,cols,:]
    x = np.lib.format.open_memmap(destination/'x.npy',mode='w+',dtype='float32',shape=(*raw.shape[:2],79))
    missing = {}
    for i,n in enumerate(NAMES):
        z = raw[:,:,i]
        missing[n] = {'train':float((~np.isfinite(z[train])).mean()),'test':float((~np.isfinite(z[~train])).mean())}
        if n == 'RF':
            x[:,:,i]=z
        else:
            x[:,:,i]=rank_features(z)
    scalers={}
    for j,n in enumerate(MEDIANS):
        i=NAMES.index(ALIASES.get(n,n))
        med = np.nanmedian(raw[:,:,i],axis=1)
        code = SPEC['cross_sectional_median_tcodes_in_table11'][n]
        tr=transform_medians(med,code)
        scaled,center,scale=training_scale(tr,train)
        x[:,:,40+j]=scaled[:,None]
        scalers[n]={'center':center,'scale':scale,'invalid_train':float((~np.isfinite(tr[train])).mean()),'invalid_test':float((~np.isfinite(tr[~train])).mean())}
    returns=np.load(out/'returns.npy',mmap_mode='r')[begin+1:end][:,cols]
    ff=pd.read_parquet(out/'factors.parquet').reindex(pd.to_datetime(target_dates))
    y=returns-ff.rf.to_numpy(dtype=np.float32)[:,None]
    assert np.isfinite(y).all() and np.isfinite(x).all()
    assert (input_dates<target_dates).all()
    x.flush()
    np.savez(destination/'labels.npz',y=y,dates=target_dates,input_dates=input_dates,permnos=chosen,market=ff.mkt_rf.to_numpy(dtype=np.float32),rf=ff.rf.to_numpy(dtype=np.float32),train=train)
    manifest={'year':year,'shape':list(x.shape),'n_train':int(train.sum()),'n_test':int((~train).sum()),'missing_raw':missing,'aggregate_scalers':scalers,
        'x_sha256':digest(destination/'x.npy'),'labels_sha256':digest(destination/'labels.npz'),'features':NAMES+['median_'+n for n in MEDIANS],
        'source_manifest_sha256':digest(out/'raw_manifest.json'),'feature_code_sha256':digest(Path(__file__))}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'panel {year}: {x.shape}, train={train.sum()}, test={(~train).sum()}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=ROOT.parent)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--years',type=int,nargs='+',default=list(range(2002,2022)))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cohorts=pd.read_csv(a.out/'cohorts/candidate_equity_cohorts.csv')
    ids=np.sort(cohorts[cohorts.cohort=='paper_test_complete'].permno.unique())
    if not (a.out/'raw_manifest.json').exists():
        build_raw(a.data_root,a.out,ids)
    for year in a.years:
        build_panel(a.out,year,cohorts)


if __name__=='__main__':
    main()
