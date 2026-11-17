"""Paper-date US universe reconstruction; never modifies the original run.

Selection uses 2025-03-31 membership/capitalization and full-sample availability,
as the paper describes. Vendor/GICS identity and eligibility ordering remain
explicit assumptions. No portfolio metrics are used in selection.
"""
from datetime import date
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
OUT = ROOT / 'data'
CRSP = Path('/home/jkp/data/crsp-daily/parquet')
INDEX = Path('/mnt/data/home/jkp/global-market-data/wrds/crsp_a_indexes')
CUT = pd.Timestamp('2025-03-31')
SOURCES = {}


def read(path, columns=None, filt=None):
    path = Path(path)
    files = sorted(path.rglob('*.parquet')) if path.is_dir() else [path]
    for file in files:
        SOURCES[str(file)] = {'bytes': file.stat().st_size,
                              'mtime_ns': file.stat().st_mtime_ns}
    return ds.dataset([str(f) for f in files]).to_table(
        columns=columns, filter=filt).to_pandas()


def main():
    snap = read(CRSP / '2025.parquet',
                ['permno', 'permco', 'ticker', 'dlycap', 'dlycaldt'],
                ds.field('dlycaldt') == date(2025, 3, 31))
    snap['dlycap'] = pd.to_numeric(snap.dlycap)
    members = read(INDEX / 'dsp500list_v2')
    members = members[(pd.to_datetime(members.mbrstartdt) <= CUT) &
                      (pd.to_datetime(members.mbrenddt) >= CUT)]
    links = read('/home/jkp/data/compustat/ccm/primary_links.parquet')
    links = links[(pd.to_datetime(links.ccm_linkdt) <= CUT) &
                  (links.ccm_linkenddt.isna() | (pd.to_datetime(links.ccm_linkenddt) >= CUT))]
    company = read('/home/jkp/data/compustat/all-parquet/comp/company',
                   ['gvkey', 'conm', 'gsector'])
    meta = snap[snap.permno.isin(members.permno)].merge(links, on='permno').merge(company, on='gvkey')
    sectors = [str(x) for x in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]]
    meta = meta[meta.gsector.isin(sectors)].copy()
    # Select one listed security per company, as in the original reconstruction.
    meta = meta.sort_values(['dlycap', 'permno'], ascending=[False, True]).drop_duplicates('gvkey')
    candidates = meta.groupby('gsector', group_keys=False).head(15).copy()
    frames = []
    for year in range(2012, 2026):
        filt = ds.field('permno').isin(candidates.permno.tolist()) & (ds.field('dlycaldt') <= CUT.date())
        frame = read(CRSP / f'{year}.parquet', ['permno', 'dlycaldt', 'dlyret', 'dlyprc'], filt)
        for col in ['dlyret', 'dlyprc']:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw['date'] = pd.to_datetime(raw.dlycaldt)
    if raw.duplicated(['date', 'permno']).any():
        raise ValueError('duplicate prices')
    rets = raw.pivot(index='date', columns='permno', values='dlyret').sort_index()
    close = raw.pivot(index='date', columns='permno', values='dlyprc').abs().reindex(rets.index)
    # A first listing-day close is usable even if its preceding daily return is absent.
    # Missing returns after inception must not silently become zero.
    first = close.notna() & close.notna().cumsum().eq(1)
    missing_returns = rets.isna() & close.notna() & ~first
    available = close.notna() & ~missing_returns
    full = available.loc['2013-01-01':'2025-03-31'].all()
    candidates['full_history_eligible'] = candidates.permno.map(full).fillna(False)
    candidates.to_csv(OUT / 'US_selection_candidates.csv', index=False)
    picked = candidates[candidates.full_history_eligible].groupby('gsector', group_keys=False).head(5)
    picked = picked.sort_values(['gsector', 'dlycap', 'permno'], ascending=[True, False, True])
    if len(picked) != 55 or not picked.groupby('gsector').size().eq(5).all():
        raise ValueError('cannot construct 55 assets with full history, five per sector')
    levels = (1 + rets.fillna(0)).cumprod().where(available).loc[:, picked.permno]
    # Forward fill only the warm-up panel; selected fit/test observations are all present.
    levels = levels.ffill().loc['2012-01-01':'2025-03-31']
    assert levels.loc['2013-01-01':].notna().all().all()
    levels.columns = levels.columns.astype(str)
    levels.to_parquet(OUT / 'US_prices.parquet')
    picked = picked.rename(columns={'permno': 'id', 'gsector': 'sector', 'dlycap': 'cap'})
    picked.to_csv(OUT / 'US_universe.csv', index=False)
    old = pd.read_csv(BASE / 'data/US_universe.csv')
    market = read(INDEX / 'dsp500_v2', ['caldt', 'spindx'])
    market['caldt'] = pd.to_datetime(market.caldt)
    market['spindx'] = pd.to_numeric(market.spindx, errors='coerce')
    market = market.set_index('caldt').sort_index().loc['2012-01-01':'2025-03-31']
    gm = pd.read_parquet(BASE / 'data/global_macro.parquet').join(market)
    assert gm.spindx.reindex(levels.index).notna().all()
    gm.to_parquet(OUT / 'global_macro.parquet')
    audit = {
        'selection_date': str(CUT.date()), 'selection_uses_test_end_information': True,
        'retrospective_paper_protocol': True, 'stocks': len(picked),
        'sector_counts': picked.sector.value_counts().sort_index().to_dict(),
        'added_vs_original': sorted(set(picked.ticker) - set(old.ticker)),
        'removed_vs_original': sorted(set(old.ticker) - set(picked.ticker)),
        'price_source': 'CRSP total-return levels; first observed close permits inception-day null return',
        'market_feature': 'CRSP dsp500_v2 spindx',
        'unresolved': ['Author IDs/vendor unavailable', 'GICS downloaded company snapshot, not verified March 2025 GICS',
                       'Top five full-history-eligible per sector interpretation; company capitalization represented by primary listed security',
                       'Other macro series/proxies retained from original reconstruction'],
    }
    (OUT / 'US_data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (OUT / 'source_files.json').write_text(json.dumps(SOURCES, indent=2) + '\n')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'}
    (OUT / 'manifest.json').write_text(json.dumps({'sha256': hashes}, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
