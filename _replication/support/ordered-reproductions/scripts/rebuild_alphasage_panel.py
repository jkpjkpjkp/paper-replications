"""Bounded, resumable Baostock CSI300 recovery; never overwrites earlier inputs.

This recovers adjusted prices, not the unavailable paper-vintage data. The
membership file is preserved exactly, including its known early coverage gap.
Both vendor responses are archived before a derived Qlib panel is produced.
"""
import argparse
import hashlib
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'csi/qlib/cn_data'
START, END, ANCHOR = '2009-01-01', '2025-03-31', '2024-12-31'
FIELDS = 'date,code,open,high,low,close,volume,amount,tradestatus,adjustflag'
PRICE = ['open', 'high', 'low', 'close']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def query(code, adjustment):
    response = bs.query_history_k_data_plus(code, FIELDS, start_date=START,
        end_date=END, frequency='d', adjustflag=adjustment)
    if response.error_code != '0':
        raise RuntimeError(f'{code}: vendor error {response.error_code}')
    rows = []
    while response.next():
        rows.append(response.get_row_data())
    if response.error_code != '0':
        raise RuntimeError(f'{code}: incomplete response {response.error_code}')
    data = pd.DataFrame(rows, columns=response.fields)
    if data.empty:
        raise ValueError(f'{code}: no observations')
    if data.date.duplicated().any() or not data.date.is_monotonic_increasing:
        raise ValueError(f'{code}: unordered or duplicate dates')
    for field in PRICE + ['volume', 'amount']:
        data[field] = pd.to_numeric(data[field], errors='coerce')
    if not data.code.eq(code).all() or not data.adjustflag.eq(adjustment).all():
        raise ValueError(f'{code}: wrong vendor series')
    return data


def adjust(raw, adjusted):
    if not raw[['date', 'code']].equals(adjusted[['date', 'code']]):
        raise ValueError('Raw and adjusted response keys differ')
    # The ratio removes adjustment events later than the frozen anchor.
    # Anchor-date scaling is a declared reconstruction choice, not paper provenance.
    factor = adjusted.close / raw.close
    usable = (raw.date <= ANCHOR) & np.isfinite(factor) & (factor > 0)
    if not usable.any():
        raise ValueError('No finite price before adjustment anchor')
    anchor_factor = float(factor.loc[usable].iloc[-1])
    relative = factor / anchor_factor
    frame = raw[['date', 'code', 'tradestatus']].copy()
    for field in PRICE:
        frame[field] = adjusted[field] / anchor_factor
    frame['factor'] = relative
    frame['volume'] = raw.volume / relative
    frame['vwap'] = raw.amount / frame.volume
    # Zero-volume/suspended days retain missing VWAP; never invent prices/labels.
    frame.loc[raw.volume.le(0), 'vwap'] = np.nan
    frame['vwap'] = frame.vwap.replace([np.inf, -np.inf], np.nan)
    good = raw[PRICE].notna().all(axis=1) & raw[PRICE].gt(0).all(axis=1)
    if not np.isfinite(frame.loc[good, PRICE + ['factor']]).all().all():
        raise ValueError('Invalid adjusted prices')
    if not np.allclose(frame.loc[good, 'close'], raw.loc[good, 'close'] * relative[good]):
        raise ValueError('Price adjustment identity failed')
    traded = good & raw.volume.gt(0) & raw.amount.gt(0)
    if not np.allclose((frame.vwap * frame.volume)[traded], raw.amount[traded]):
        raise ValueError('VWAP/volume identity failed')
    return frame, dict(anchor=ANCHOR, last_anchor_quote=str(raw.loc[usable, 'date'].iloc[-1]),
        rows=len(frame), traded_rows=int(traded.sum()), anchor_factor=anchor_factor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='pilot only; cannot create a full-panel receipt')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ['raw', 'adjusted', 'derived', 'receipts']:
        (out / name).mkdir(exist_ok=True)
    membership = pd.read_csv(BASE / 'instruments/csi300.txt', sep='\t', header=None,
        names=['stock', 'start', 'end'])
    relevant = membership.loc[(membership.start <= '2024-12-31') & (membership.end >= '2010-01-01')]
    stocks = sorted(relevant.stock.unique())
    if args.limit:
        stocks = stocks[:args.limit]
    spec = dict(vendor='Baostock', start=START, end=END, adjustment_anchor=ANCHOR,
        membership_sha256=sha(BASE / 'instruments/csi300.txt'), stock_count=len(stocks),
        pilot=bool(args.limit), fields=FIELDS, source_membership=str(BASE / 'instruments/csi300.txt'),
        scope='Reconstructed input panel only; not paper equivalence or a model result',
        limitations=['Current-vintage prices, not archived author snapshot',
            'Community historical constituent file; starts 2010-01-29, preserved without extrapolation',
            '2024-12-31 adjustment anchor is a declared reconstruction choice; not established by paper',
            'No model or CZ training is launched by this collector'])
    protocol = out / 'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise ValueError('Existing collection protocol differs')
    if not protocol.exists():
        save(protocol, spec)
    socket.setdefaulttimeout(30)
    started = time.monotonic()
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(f'Login failed: {login.error_code}')
    try:
        failures = {}
        for i, stock in enumerate(stocks):
            receipt = out / 'receipts' / (stock + '.json')
            if receipt.exists():
                for relative, expected in json.loads(receipt.read_text())['sha256'].items():
                    if sha(out / relative) != expected:
                        raise ValueError(f'{stock}: completed artifact changed')
            else:
                number, exchange = stock.split('.')
                code = exchange.lower() + '.' + number
                try:
                    raw, adjusted = query(code, '3'), query(code, '2')
                    frame, diagnostics = adjust(raw, adjusted)
                except Exception as error:
                    failures[stock] = repr(error)
                    save(out / 'failures.json', failures)
                    print(json.dumps(dict(stock=stock, state='failed_stock', error=repr(error))), flush=True)
                    save(out / 'status.json', dict(state='collecting', attempted=i + 1,
                        completed=i + 1 - len(failures), total=len(stocks), failures=len(failures),
                        seconds=time.monotonic() - started, last_stock=stock))
                    # Preserve missing stocks as errors. Never drop them from the universe.
                    continue
                files = []
                for subdir, data in [('raw', raw), ('adjusted', adjusted), ('derived', frame)]:
                    path = out / subdir / (stock + '.parquet')
                    data.to_parquet(path, index=False)
                    files.append(path)
                save(receipt, dict(stock=stock, downloaded_utc=datetime.now(timezone.utc).isoformat(),
                    diagnostics=diagnostics, sha256={str(p.relative_to(out)): sha(p) for p in files}))
            status = dict(state='collecting', attempted=i + 1, completed=i + 1 - len(failures),
                total=len(stocks), failures=len(failures),
                seconds=time.monotonic() - started, last_stock=stock)
            save(out / 'status.json', status)
            if (i + 1) % 20 == 0 or i == 0:
                print(json.dumps(status), flush=True)
        save(out / 'failures.json', failures)
        if failures:
            raise RuntimeError(f'{len(failures)} stocks missing; refusing to export an incomplete universe. Resume to retry.')
        # Export simple Qlib binary series onto a frozen observed trading calendar.
        calendar = pd.DatetimeIndex(pd.to_datetime((BASE / 'calendars/day.txt').read_text().splitlines()))
        older = pd.DatetimeIndex(pd.to_datetime((ROOT / 'data/crowd_data/calendars/day.txt').read_text().splitlines()))
        calendar = calendar.union(older).sort_values()
        calendar = calendar[(calendar >= START) & (calendar <= END)]
        target = out / 'qlib'
        (target / 'calendars').mkdir(parents=True, exist_ok=True)
        (target / 'instruments').mkdir(exist_ok=True)
        (target / 'calendars/day.txt').write_text('\n'.join(calendar.strftime('%Y-%m-%d')) + '\n')
        relevant.loc[relevant.stock.isin(stocks)].to_csv(target / 'instruments/csi300.txt',
            sep='\t', header=False, index=False)
        for stock in stocks:
            frame = pd.read_parquet(out / 'derived' / (stock + '.parquet'))
            frame.index = pd.to_datetime(frame.date)
            if not frame.index.isin(calendar).all():
                raise ValueError(f'{stock}: vendor date outside frozen calendar')
            frame = frame.reindex(calendar)
            directory = target / 'features' / stock.lower()
            directory.mkdir(parents=True, exist_ok=True)
            for field in PRICE + ['volume', 'vwap', 'factor']:
                np.r_[np.float32(0), frame[field].to_numpy(dtype=np.float32)].astype('<f4').tofile(directory / (field + '.day.bin'))
        save(out / 'complete.json', dict(state='pilot_complete' if args.limit else 'panel_reconstructed',
            stocks=len(stocks), calendar_days=len(calendar), seconds=time.monotonic() - started,
            protocol_sha256=sha(protocol), runner_sha256=sha(Path(__file__)),
            original_fidelity_passed=False, requires='Independent coverage, normalization and source-contract review before training'))
        save(out / 'status.json', json.loads((out / 'complete.json').read_text()))
        print((out / 'complete.json').read_text(), flush=True)
    except BaseException as error:
        save(out / 'status.json', dict(state='failed', error=repr(error), seconds=time.monotonic() - started))
        raise
    finally:
        bs.logout()


if __name__ == '__main__':
    main()
