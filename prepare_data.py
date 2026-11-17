"""Build auditable substitute universes from existing WRDS snapshots.

Source archives are read-only. Universe selection uses prices/capitalization at
2024-09-30 and history through that date, never test returns. GICS classifications
are the vendor snapshot, not a reconstructed point-in-time classification.
"""
from pathlib import Path
from datetime import date
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data'
WRDS = Path('/mnt/data/home/jkp/global-market-data/wrds')
CUT = pd.Timestamp('2024-09-30')
END = pd.Timestamp('2025-03-31')
START = pd.Timestamp('2012-01-01')
SECTORS = ['10','15','20','25','30','35','40','45','50','55','60']
SOURCES = {}


def read_table(path, columns=None, filt=None):
    path = Path(path)
    files = sorted(path.rglob('*.parquet')) if path.is_dir() else [path]
    for p in files:
        SOURCES[str(p)] = {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
    return ds.dataset([str(p) for p in files]).to_table(
        columns=columns, filter=filt).to_pandas()


def numeric(x, cols):
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors='coerce').astype(float)
    return x


def choose(meta, prices, n, per_sector, calendar):
    """Eligibility is determined entirely before the test period."""
    train = prices.reindex(calendar[calendar <= CUT])
    check = train.loc['2013-01-01':]
    eligible = check.notna().mean().ge(.98) & check.iloc[:5].notna().any()
    eligible &= check.ffill(limit=5).notna().all()
    meta = meta.loc[meta.id.isin(eligible[eligible].index)].copy()
    meta = meta.sort_values(['cap','id'], ascending=[False,True]).drop_duplicates('company')
    picked = []
    for s in SECTORS:
        picked.extend(meta.loc[meta.sector.eq(s)].head(per_sector).id.tolist())
    if len(picked) > n:
        # Keep at least three per sector before awarding remaining places by cap.
        core = []
        for s in SECTORS:
            core.extend(meta.loc[meta.sector.eq(s)].head(n//11).id.tolist())
        rest = meta.loc[meta.id.isin(picked) & ~meta.id.isin(core)].head(n-len(core)).id.tolist()
        picked = core + rest
    if len(picked) < n:
        picked += meta.loc[~meta.id.isin(picked)].head(n-len(picked)).id.tolist()
    if len(picked) != n:
        raise ValueError(f'Only {len(picked)} eligible stocks for requested {n}')
    return meta.set_index('id').loc[picked].reset_index()


def save_market(name, meta, prices, calendar, details):
    x = prices.reindex(index=calendar, columns=meta.id)
    missing = x.isna()
    # Suspended prices stay stale; never backfill using a future price.
    x = x.ffill()
    x = x.loc['2012-01-01':'2025-03-31']
    if x.loc['2013-01-01':].isna().any().any():
        raise ValueError(f'{name}: unresolved prices in experiment')
    r = np.log(x).diff()
    details.update({
        'stocks': len(meta), 'first_date':str(x.index.min().date()),
        'last_date':str(x.index.max().date()), 'sessions':len(x),
        'selection_date':str(CUT.date()), 'selection_uses_test_prices':False,
        'missing_prices_forward_filled_train':int(missing.loc[:CUT].sum().sum()),
        'missing_prices_forward_filled_test':int(missing.loc['2024-10-01':].sum().sum()),
        'sector_counts':meta.sector.value_counts().sort_index().to_dict(),
        'max_absolute_daily_log_return':float(r.abs().max().max()),
    })
    x.to_parquet(OUT/f'{name}_prices.parquet')
    meta.to_csv(OUT/f'{name}_universe.csv', index=False)
    (OUT/f'{name}_data_audit.json').write_text(json.dumps(details,indent=2))
    print(name, json.dumps(details), flush=True)


def us_data():
    base = Path('/home/jkp/data/crsp-daily/parquet')
    snap = read_table(base/'2024.parquet', ['permno','ticker','dlycap','dlycaldt'],
                      ds.field('dlycaldt') == CUT.date())
    snap = numeric(snap,['dlycap'])
    mem = read_table(WRDS/'crsp_a_indexes/dsp500list')
    mem = mem[(pd.to_datetime(mem.start)<=CUT)&(pd.to_datetime(mem.ending)>=CUT)]
    links = read_table('/home/jkp/data/compustat/ccm/primary_links.parquet')
    links = links[(pd.to_datetime(links.ccm_linkdt)<=CUT) &
                  (pd.to_datetime(links.ccm_linkenddt).isna() | (pd.to_datetime(links.ccm_linkenddt)>=CUT))]
    comp = read_table('/home/jkp/data/compustat/all-parquet/comp/company', ['gvkey','conm','gsector'])
    meta = snap[snap.permno.isin(mem.permno)].merge(links,on='permno').merge(comp,on='gvkey')
    meta = meta.rename(columns={'gsector':'sector','dlycap':'cap','gvkey':'company'})
    meta['id'] = meta.permno.astype(str)
    meta = meta[meta.sector.isin(SECTORS)].sort_values('cap',ascending=False).drop_duplicates('company')
    # Read candidates, plus GLD as the gold price proxy and SPY for an index proxy.
    ids = meta.permno.tolist()
    frames = []
    for y in range(2012,2026):
        filt = ds.field('permno').isin(ids) | ds.field('ticker').isin(['GLD','SPY'])
        x = read_table(base/f'{y}.parquet', ['permno','ticker','dlycaldt','dlyret','dlyprc'],filt)
        frames.append(numeric(x,['dlyret','dlyprc']))
    x = pd.concat(frames,ignore_index=True)
    x['date'] = pd.to_datetime(x.dlycaldt)
    x = x[x.date.between(START,END)].sort_values(['date','permno'])
    if x.duplicated(['date','permno']).any(): raise ValueError('duplicate CRSP rows')
    x['id'] = x.permno.astype(str)
    ret = x[x.permno.isin(ids)].pivot(index='date',columns='id',values='dlyret')
    calendar = ret.index
    # Total-return level; missing raw returns remain missing for eligibility.
    price = (1+ret.fillna(0)).cumprod().where(ret.notna())
    picked = choose(meta,price,55,5,calendar)
    save_market('US',picked,price,calendar,{
        'source':'CRSP CIZ dlyret, total returns including distributions',
        'universe':'S&P500 members at 2024-09-30; top five eligible companies per snapshot GICS sector',
        'deviations':['Selection date precedes test rather than paper 2025-03-31',
                      'GICS uses downloaded company snapshot', 'CRSP total-return levels substitute adjusted closes'],
    })
    proxy = x[x.ticker.isin(['GLD','SPY'])].pivot_table(index='date',columns='ticker',values='dlyprc',aggfunc='first').abs()
    proxy.to_parquet(OUT/'us_macro_proxies.parquet')


def global_data():
    base=WRDS/'comp_global_daily/g_secd'
    cols=['gvkey','iid','conm','curcdd','prccd','cshoc','gind','fic','tpci','exchg','qunit']
    snap=read_table(base/'2024',cols,ds.field('datadate')==CUT.date())
    snap=numeric(snap,['prccd','cshoc','qunit'])
    snap['cap']=snap.prccd/snap.qunit.replace(0,np.nan)*snap.cshoc
    snap['sector']=snap.gind.str[:2]
    snap['company']=snap.gvkey
    snap['id']=snap.gvkey+'_'+snap.iid
    # Keep ordinary shares quoted in the domestic currency; one issue/company.
    sec=read_table(WRDS/'comp_global_daily/g_security',['gvkey','iid','tic','isin','excntry'])
    snap=snap.merge(sec,on=['gvkey','iid'],how='left')
    comp=read_table(WRDS/'comp_global_daily/g_company',['gvkey','prirow'])
    snap=snap.merge(comp,on='gvkey',how='left')
    snap=snap[snap.tpci.eq('0')&snap.sector.isin(SECTORS)&snap.cap.gt(0)&snap.iid.eq(snap.prirow)]
    regions={
        'China':(['CHN'],'CNY',55,5),
        'Europe':(['AUT','BEL','DEU','ESP','FIN','FRA','GRC','IRL','ITA','LUX','NLD','PRT'],'EUR',37,4),
        'Korea':(['KOR'],'KRW',44,4),
    }
    candidates={}
    for name,(countries,currency,n,persector) in regions.items():
        m=snap[snap.fic.isin(countries)&snap.excntry.isin(countries)&snap.curcdd.eq(currency)].copy()
        m=m.sort_values(['cap','id'],ascending=[False,True]).drop_duplicates('company')
        # Enough reserve names for train-only full-history eligibility.
        m=m.groupby('sector',group_keys=False).head(35)
        candidates[name]=m
    gvkeys=sorted(set(pd.concat(candidates.values()).gvkey))
    hist=[]
    for y in range(2012,2026):
        f=ds.field('gvkey').isin(gvkeys)&(ds.field('datadate')<=END.date())
        x=read_table(base/str(y),['gvkey','iid','datadate','curcdd','prccd','ajexdi','trfd','qunit','cshtrd'],f)
        hist.append(numeric(x,['prccd','ajexdi','trfd','qunit','cshtrd']))
        print('global source year',y,'rows',len(x),flush=True)
    x=pd.concat(hist,ignore_index=True)
    x['id']=x.gvkey+'_'+x.iid
    x['date']=pd.to_datetime(x.datadate)
    x['price']=x.prccd/x.qunit.replace(0,np.nan)/x.ajexdi.replace(0,np.nan)*x.trfd
    for name,(_,currency,n,persector) in regions.items():
        m=candidates[name]
        z=x[x.id.isin(m.id)&x.curcdd.eq(currency)].copy()
        if z.duplicated(['date','id']).any(): raise ValueError('duplicate Compustat rows')
        price=z.pivot(index='date',columns='id',values='price').where(lambda q:q>0)
        vol=z.pivot(index='date',columns='id',values='cshtrd')
        # Exclude vendor holiday carry-forward rows using regional activity.
        active=(vol>0).sum(axis=1)/vol.notna().sum(axis=1).clip(lower=1)
        calendar=price.index[active.reindex(price.index).fillna(0)>.25]
        picked=choose(m,price,n,persector,calendar)
        save_market(name,picked,price,calendar,{
            'source':'WRDS Compustat Global g_secd',
            'price_formula':'prccd / qunit / ajexdi * trfd (total-return adjusted local-currency price)',
            'universe':'Regional primary ordinary shares, capitalization-ranked within snapshot GICS sectors',
            'deviations':['Index membership unavailable: regional large-cap proxy universe',
                          'Train-cutoff capitalization and history eligibility',
                          'Snapshot GICS classifications; region calendar requires >25% candidates with positive volume',
                          'Sector shortages filled with next-largest eligible companies'],
        })


def macros():
    v=read_table(WRDS/'cboe_all/cboe',['date','vix'])
    v['date']=pd.to_datetime(v.date);v=numeric(v,['vix']).set_index('date')
    p=pd.read_parquet(OUT/'us_macro_proxies.parquet')
    x=v.join(p,how='outer')
    for name in ['T10Y3M','DTWEXBGS']:
        path=OUT/f'{name}.csv'
        t=pd.read_csv(path,parse_dates=[0],na_values=['.']).set_index('observation_date')
        x=x.join(t,how='outer')
    x=x.sort_index().loc['2012-01-01':'2025-03-31']
    x.to_parquet(OUT/'global_macro.parquet')
    print('macro nonmissing range', {c:[str(x[c].first_valid_index()),str(x[c].last_valid_index())] for c in x},flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    us_data();global_data();macros()
    (OUT/'source_files.json').write_text(json.dumps(SOURCES,indent=2))
    outputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps({'sha256':outputs,
       'macro_urls':{s:f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd=2012-01-01&coed=2025-03-31' for s in ['T10Y3M','DTWEXBGS']}},indent=2))


if __name__=='__main__': main()
