"""Build a price-only diagnostic; never label these inputs the authors' data."""
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'csi/qlib/cn_data'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_field(symbol, field, calendar):
    p = DATA / f'features/{symbol.lower()}/{field}.day.bin'
    raw = np.fromfile(p, dtype='<f4')
    offset = int(raw[0])
    out = np.full(len(calendar), np.nan)
    out[offset:offset+len(raw)-1] = raw[1:]
    return out, {'path': str(p), 'sha256': sha(p)}

def main():
    text = (ROOT/'sources/paper.txt').read_text()
    section = text.split('Table B.9\n', 1)[1].split('Appendix A', 1)[0]
    codes = sorted(set(re.findall(r'\b(?:000|001|002|300|600|601|603)\d{3}\b', section)))
    assert len(codes) == 86, len(codes)
    symbols = [c+('.SH' if c.startswith('6') else '.SZ') for c in codes]
    cal = np.array((DATA/'calendars/day.txt').read_text().splitlines())
    use = np.flatnonzero((cal >= '2017-10-01') & (cal <= '2021-06-15'))
    dates = cal[use]
    arrays = {}; manifest = []
    for field in ['open','high','low','close','volume']:
        cols=[]
        for symbol in symbols:
            a,m = read_field(symbol, field, cal)
            cols.append(a[use]); manifest.append(m)
        arrays[field] = np.stack(cols, axis=1)
    c = pd.DataFrame(arrays['close']); o=arrays['open']; h=arrays['high']; l=arrays['low']; v=pd.DataFrame(arrays['volume'])
    feat = [o, h, l, c.values, np.log1p(v.values), (h-l)/c.values, (c.values-o)/o]
    names=['open','high','low','close','log_volume','range_close','intraday_return']
    ret=c.pct_change(fill_method=None)
    for w in [1,3,5,10,20]:
        feat.append(c.pct_change(w,fill_method=None).values); names.append(f'return_{w}')
    for w in [5,10,20,30]:
        feat.extend([(c/c.rolling(w).mean()-1).values, ret.rolling(w).std(ddof=0).values,
                     (v/v.rolling(w).mean()-1).values])
        names.extend([f'ma_relative_{w}',f'return_std_{w}',f'volume_relative_{w}'])
    features=np.stack(feat,axis=-1)
    # Forecast origin t; labels are next-close return and std(close[t-4:t+6]).
    yret=c.shift(-1).values/c.values-1
    yrisk=np.full_like(yret,np.nan)
    for t in range(4,len(dates)-5):
        block=c.values[t-4:t+6]
        valid=np.isfinite(block).all(axis=0)
        yrisk[t,valid]=np.std(block[:,valid],axis=0,ddof=0)
    target_dates=np.concatenate((dates[1:], ['9999-12-31']))
    # End the risk label window before each split cutoff, for BOTH methods.
    idx={}
    for split,start,end in [('train','2018-01-02','2020-05-25'),('valid','2020-05-26','2020-09-21'),('test','2020-09-22','2021-06-02')]:
        keep=(target_dates>=start)&(target_dates<=end)&(np.arange(len(dates))>=6)
        if split!='test':
            keep &= np.r_[dates[5:],np.repeat('9999-12-31',5)]<=end
        idx[split]=np.flatnonzero(keep)
    train_days=np.unique(np.concatenate([np.arange(t-6,t+1) for t in idx['train']]))
    tr=features[train_days].reshape(-1,len(names))
    features[~np.isfinite(features)] = np.nan
    tr=features[train_days].reshape(-1,len(names))
    median=np.nanmedian(tr,axis=0); lo=np.nanmin(tr,axis=0);hi=np.nanmax(tr,axis=0)
    filled=np.where(np.isfinite(features),features,median)
    x=((filled-lo)/np.maximum(hi-lo,1e-12)).astype('float32')
    mask=np.isfinite(yret)&np.isfinite(yrisk)
    out=ROOT/'data';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'price_only.npz',x=x,yret=yret.astype('float32'),yrisk=yrisk.astype('float32'),
                        mask=mask,dates=dates,target_dates=target_dates,symbols=symbols,
                        feature_names=names,**idx)
    pd.DataFrame({'symbol':symbols}).to_csv(out/'universe.csv',index=False)
    audit={'classification':'reduced_price_only_diagnostic_not_published_replication',
           'source':'../csi community Tushare/Qlib snapshot; not RESSET', 'stocks':len(symbols),
           'feature_names':names,'features':len(names),'missing_input_values':int(np.isnan(features).sum()),
           'splits':{k:{'origins':len(v),'first_target':target_dates[v[0]],'last_target':target_dates[v[-1]],
                        'labeled_stock_days':int(mask[v].sum())} for k,v in idx.items()},
           'paper_train_days':581,'local_calendar_train_days':int(((dates>='2018-01-02')&(dates<='2020-05-25')).sum()),
           'risk_definition':'population std of ten source close prices at origin t-4 through t+5, following prose, not misprinted Eq14',
           'split_embargo':'training/validation risk windows must end by split cutoff; matched samples',
           'normalization':'feature median and min/max fit on training inputs only; no target standardization',
           'original_missing':['39 original quantitative features','134520 original news titles','MSC trained checkpoint/embeddings','336 typed graph edges','author model code','exact walk-forward/refit schedule'],
           'input_manifest':manifest,'panel_sha256':sha(out/'price_only.npz')}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='input_manifest'},indent=2))

if __name__=='__main__':main()
