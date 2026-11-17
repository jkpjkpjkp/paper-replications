"""Local CRSP-to-Alpha158 pipeline; dynamic historical inference universe."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'
SOURCE = ROOT.parent/'crsp-daily/parquet'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def config():
    return json.loads((ROOT/'protocol.json').read_text())


def fill_window(x):
    x = np.asarray(x).copy()
    if np.isfinite(x).all():
        return x
    for t in range(1, x.shape[1]):
        missing = ~np.isfinite(x[:,t])
        x[:,t][missing] = x[:,t-1][missing]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        means = np.nanmean(x, axis=1, keepdims=True)
    return np.nan_to_num(np.where(np.isfinite(x), x, means), nan=0., posinf=0., neginf=0.)


class Panel:
    def __init__(self):
        self.p = config()
        self.folder = DATA
        for name in ['dates','symbols','features','labels','member','market','total_return','price_return','terminal','execution_tradable']:
            setattr(self, name, np.load(DATA/f'{name}.npy', mmap_mode='r'))
        self.eligible = self.member
        self.lookback = self.p['lookback']

    def split(self, split):
        a,b = self.p[split]
        ix = np.arange(self.lookback-1, len(self.dates)-2)
        return ix[(self.dates[ix] >= a) & (self.dates[ix+2] <= b) &
                  (self.eligible[:,ix].sum(0) >= 50)]

    def day(self, t):
        ids = np.flatnonzero(self.eligible[:,t])
        x = fill_window(self.features[ids,t-self.lookback+1:t+1])
        raw = np.array(self.labels[ids,t])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            normalized = (raw-np.nanmean(raw,axis=0))/np.maximum(np.nanstd(raw,axis=0,ddof=1),1e-8)
        return ids,x,normalized.astype(np.float32),raw


def arrow_read(path, columns, filters=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=columns, filters=filters)
    for j, field in enumerate(table.schema):
        if pa.types.is_decimal(field.type):
            table = table.set_column(j, field.name, table.column(j).cast(pa.float64()))
    return table.to_pandas()


def select_universe():
    DATA.mkdir(exist_ok=True)
    out = DATA/'membership.parquet'
    if out.exists():
        return pd.read_parquet(out)
    fields = ['permno','permco','dlycaldt','dlycap','dlyclose','dlyvol','dlyprcflg',
              'securitytype','securitysubtype','sharetype','issuertype','usincflg',
              'primaryexch','conditionaltype','tradingstatusflg','dlydelflg']
    filters = [('securitytype','=','EQTY'),('securitysubtype','=','COM'),
               ('sharetype','=','NS'),('issuertype','=','CORP'),('usincflg','=','Y')]
    parts=[]
    for year in range(2007,2026):
        frame = arrow_read(SOURCE/f'{year}.parquet',fields,filters)
        good = (frame.primaryexch.isin(['N','A','Q']) & frame.conditionaltype.eq('RW') &
                frame.tradingstatusflg.eq('A') & frame.dlydelflg.eq('N') &
                frame.dlyprcflg.eq('TR') & frame.dlyclose.ge(5) & frame.dlyvol.gt(0) & frame.dlycap.gt(0))
        frame = frame.loc[good].sort_values(['dlycaldt','dlycap','permno'],ascending=[True,False,True])
        frame = frame.drop_duplicates(['dlycaldt','permco']).groupby('dlycaldt',sort=True).head(500)
        assert frame.groupby('dlycaldt').size().min() == 500
        parts.append(frame[['permno','permco','dlycaldt']])
        write_json(DATA/'build_status.json',dict(stage='universe',year=year))
        print('universe',year,'rows',len(frame),flush=True)
    result = pd.concat(parts,ignore_index=True)
    result.to_parquet(out,index=False)
    return result


def raw_panel(members):
    cache = DATA/'raw.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    ids = sorted(members.permno.unique().tolist())
    cols = ['permno','permco','dlycaldt','dlyclose','dlyopen','dlyhigh','dlylow',
            'dlyvol','dlyret','dlyretx','dlycumfacpr','dlyprcflg','dlydelflg',
            'conditionaltype','tradingstatusflg']
    parts=[]
    for year in range(2007,2026):
        frame = arrow_read(SOURCE/f'{year}.parquet',cols,[('permno','in',ids)])
        parts.append(frame)
        write_json(DATA/'build_status.json',dict(stage='raw',year=year))
        print('raw',year,len(frame),flush=True)
    frame = pd.concat(parts,ignore_index=True)
    frame['dlycaldt'] = pd.to_datetime(frame.dlycaldt)
    assert not frame[['permno','dlycaldt']].duplicated().any()
    frame.to_parquet(cache,index=False)
    return frame


def make_arrays(frame, members):
    ids = sorted(members.permno.unique())
    dates = pd.DatetimeIndex(sorted(members.dlycaldt.unique()))
    def pivot(name):
        return frame.pivot(index='permno',columns='dlycaldt',values=name).reindex(index=ids,columns=dates)
    raw = {n:pivot(n).to_numpy(dtype=np.float64) for n in
           ['dlyclose','dlyopen','dlyhigh','dlylow','dlyvol','dlyret','dlyretx','dlycumfacpr']}
    factor = raw['dlycumfacpr']
    close = np.where(factor>0,raw['dlyclose']/factor,np.nan)
    # All same-day OHLC use the same split factor. Ratio features are invariant
    # to the arbitrary vendor base scale; direct CRSP return labels are retained.
    o,h,l = [np.where(factor>0,raw[n]/factor,np.nan) for n in ['dlyopen','dlyhigh','dlylow']]
    proxy = (o+h+l+close)/4
    raw_proxy = (raw['dlyopen']+raw['dlyhigh']+raw['dlylow']+raw['dlyclose'])/4
    vol = raw['dlyvol']
    member = members.assign(value=True).pivot(index='permno',columns='dlycaldt',values='value').reindex(index=ids,columns=dates).fillna(False).to_numpy(bool)
    flags = {n:pivot(n).to_numpy() for n in ['dlyprcflg','dlydelflg','conditionaltype','tradingstatusflg']}
    tradable = (flags['dlyprcflg']=='TR') & (flags['conditionaltype']=='RW') & (flags['tradingstatusflg']=='A') & (raw['dlyclose']>0) & (vol>0) & (flags['dlydelflg']=='N')
    terminal = pd.notna(flags['dlydelflg']) & (flags['dlydelflg']!='N')
    # Independent adjustment check. No future return decides membership.
    adjusted_ratio = close[:,1:]/close[:,:-1]-1
    xret = raw['dlyretx'][:,1:]
    check = member[:,1:] & np.isfinite(adjusted_ratio) & np.isfinite(xret)
    discrepancy = abs(adjusted_ratio-xret)
    large = check & (discrepancy > .002)
    # A material disagreement fails before fitting; detailed rows are retained.
    if large.sum():
        ii,jj = np.where(large)
        pd.DataFrame(dict(permno=np.array(ids)[ii],date=dates[1:][jj],
            adjusted_price_return=adjusted_ratio[large],crsp_retx=xret[large],difference=discrepancy[large])).to_csv(DATA/'price_adjustment_discrepancies.csv',index=False)
    write_json(DATA/'adjustment_check.json',dict(observations=int(check.sum()),
        over_20bp=int(large.sum()),max_abs_error=float(np.nanmax(discrepancy[check]))))
    if large.sum() > max(10,check.sum()*.0001):
        raise ValueError('Material split-adjustment mismatch; inspect before training')
    with np.errstate(divide='ignore',invalid='ignore'):
        logv = np.log(np.where(vol>0,vol,np.nan))
        prior = pd.DataFrame(logv.T).rolling(5,min_periods=5).mean().to_numpy().T
        labels = np.full((*close.shape,3),np.nan,np.float32)
        labels[:,:-2,0] = raw['dlyretx'][:,2:]
        labels[:,:-2,1] = logv[:,2:]-prior[:,1:-1]
        labels[:,:-2,2] = (h[:,2:]-l[:,2:])/proxy[:,2:]
    labels[~np.isfinite(labels)] = np.nan
    market = np.stack([close,vol,1/factor,raw['dlyclose'],raw['dlyhigh'],raw['dlylow'],raw_proxy],axis=-1)
    arrays = dict(dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),symbols=np.array([f'P{i}' for i in ids]),
        member=member,market=market.astype(np.float32),labels=labels,total_return=raw['dlyret'],
        price_return=raw['dlyretx'],terminal=terminal,execution_tradable=tradable)
    for name,a in arrays.items():
        np.save(DATA/f'{name}.npy',a)
    provider = DATA/'qlib'
    (provider/'calendars').mkdir(parents=True,exist_ok=True)
    (provider/'instruments').mkdir(exist_ok=True)
    (provider/'calendars/day.txt').write_text('\n'.join(arrays['dates'])+'\n')
    (provider/'instruments/all.txt').write_text(''.join(f'P{i}\t{arrays["dates"][0]}\t{arrays["dates"][-1]}\n' for i in ids))
    fields = dict(open=o,high=h,low=l,close=close,volume=vol,vwap=proxy)
    for j,symbol in enumerate(arrays['symbols']):
        folder = provider/'features'/symbol.lower()
        folder.mkdir(parents=True,exist_ok=True)
        for name,a in fields.items():
            np.r_[np.float32(0),a[j].astype(np.float32)].astype('<f4').tofile(folder/f'{name}.day.bin')
    return arrays


def init_features(provider, dates):
    global CALENDAR, EXPRESSIONS
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    qlib.init(provider_uri=provider,region='us',kernels=1,logging_level=40,expression_cache=None,dataset_cache=None)
    CALENDAR = pd.DatetimeIndex(dates)
    EXPRESSIONS = Alpha158DL.get_feature_config()[0]


def extract_features(item):
    symbol,path = item
    from qlib.data import D
    from qlib.data.cache import H
    frame = D.features([symbol],EXPRESSIONS,start_time=str(CALENDAR[0].date()),end_time=str(CALENDAR[-1].date()))
    frame = frame.xs(symbol,level='instrument').reindex(CALENDAR)
    a = frame.to_numpy(np.float32)
    a[~np.isfinite(a)] = np.nan
    np.save(path,a)
    H.clear()
    return symbol


def build():
    from concurrent.futures import ProcessPoolExecutor
    from qlib.contrib.data.loader import Alpha158DL
    p = config()
    if (DATA/'manifest.json').exists():
        manifest=json.loads((DATA/'manifest.json').read_text())
        assert manifest['protocol_sha256']==sha(ROOT/'protocol.json')
        print('Using completed US dataset',flush=True)
        return
    members=select_universe()
    frame=raw_panel(members)
    arrays=make_arrays(frame,members)
    del frame
    cache=DATA/'feature_cache'
    cache.mkdir(exist_ok=True)
    jobs=[(s,cache/f'{s}.npy') for s in arrays['symbols'] if not (cache/f'{s}.npy').exists()]
    with ProcessPoolExecutor(max_workers=2,initializer=init_features,initargs=(str(DATA/'qlib'),arrays['dates'])) as pool:
        for n,_ in enumerate(pool.map(extract_features,jobs),1):
            if n%50==0:
                print('features',n,'of',len(jobs),flush=True)
                write_json(DATA/'build_status.json',dict(stage='features',done=n,total=len(jobs)))
    shape=(len(arrays['symbols']),len(arrays['dates']),158)
    features=np.lib.format.open_memmap(DATA/'features.npy',mode='w+',dtype=np.float32,shape=shape)
    for i,s in enumerate(arrays['symbols']):
        features[i]=np.load(cache/f'{s}.npy')
    medians=[];scales=[]
    # Last training decision must realize its t+2 target within 2014.
    train=np.zeros(shape[:2],bool)
    ix=np.arange(20,len(arrays['dates'])-2)
    ix=ix[(arrays['dates'][ix]>=p['train'][0])&(arrays['dates'][ix+2]<=p['train'][1])]
    train[:,ix]=arrays['member'][:,ix]
    for j in range(158):
        raw=np.array(features[:,:,j])
        values=raw[train]
        med=float(np.nanmedian(values))
        scale=float((np.nanmedian(abs(values-med))+1e-12)*1.4826)
        assert np.isfinite(med) and scale>0
        features[:,:,j]=np.clip((raw-med)/scale,-3,3)
        medians.append(med);scales.append(scale)
    features.flush()
    np.savez(DATA/'normalization.npz',median=medians,scale=scales)
    names=Alpha158DL.get_feature_config()[1]
    manifest=dict(protocol_sha256=sha(ROOT/'protocol.json'),shape=shape,features=names,
        date_start=str(arrays['dates'][0]),date_end=str(arrays['dates'][-1]),
        source='CRSP CIZ daily, verified local full-universe partitions',
        source_sha256={str(SOURCE/f'{y}.parquet'):sha(SOURCE/f'{y}.parquet') for y in range(2007,2026)},
        output_sha256={p.name:sha(p) for p in sorted(DATA.glob('*.npy'))},
        builder_sha256=sha(__file__),normalization_last_realization=p['train'][1],
        vwap_substitution=p['vwap_substitution'])
    write_json(DATA/'manifest.json',manifest)
    write_json(DATA/'build_status.json',dict(stage='complete',shape=shape))
    print('US DATA COMPLETE',shape,flush=True)


if __name__=='__main__':
    build()
