"""Build a declared alternative NASDAQ/NYSE panel from the paper's RSR source.

This is reconstructed input, NOT the missing PDU preprocessing. Price features
are OHLC / maximum training close; volume / maximum training volume; trailing
MA5/10/20/30 divided by contemporaneous close minus one. No future-fit scaling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def graph_from_archive(market, n):
    union = np.zeros((n, n), dtype=bool)
    stats = {}
    with tarfile.open(ROOT / "reference-rsr/data/relation.tar.gz") as tar:
        for member in tar:
            if not member.name.endswith('.npy') or f'/{market}_' not in member.name:
                continue
            stream = tar.extractfile(member)
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            assert shape[:2] == (n, n) and not fortran
            count = shape[1] * shape[2]
            single = np.zeros((n, n), dtype=bool)
            # Stream one stock row at a time: NYSE tensor is 2.6 GB uncompressed.
            for i in range(n):
                row = np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype).reshape(shape[1:])
                single[i] = np.any(row != 0, axis=1)
            union |= single
            stats[Path(member.name).name] = {"shape": shape, "directed_edges_including_self": int(single.sum())}
    np.fill_diagonal(union, False)
    union |= union.T.copy()
    return np.asarray(np.nonzero(union), dtype=np.int64), stats


def prepare(market):
    source = ROOT / "reference-rsr/data"
    tickers = (source / f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv").read_text().split()
    raw_dates = (source / f"{market}_aver_line_dates.csv").read_text().splitlines()
    dates = pd.DatetimeIndex(sorted({pd.Timestamp(s.strip().strip('"')[:10]) for s in raw_dates}))
    dates = dates[(dates >= '2013-01-02') & (dates <= '2017-11-08')]
    n, t = len(tickers), len(dates)
    if t <= 1008:
        raise ValueError(f"Insufficient dates: {t}")
    features = np.zeros((n, t, 9), np.float32)
    price = np.ones((n, t), np.float32)
    opening = np.ones((n, t), np.float32)
    mask = np.zeros((n, t), bool)
    input_hashes, absent = {}, []
    for i, ticker in enumerate(tickers):
        path = source / f"google_finance/{market}_{ticker}_30Y.csv"
        if not path.exists():
            absent.append(ticker); continue
        input_hashes[ticker] = hashlib.sha256(path.read_bytes()).hexdigest()
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index.str[:10])
        df = df[~df.index.duplicated(keep='last')].sort_index()
        # Calendar-align before rolling so missing sessions do not become
        # apparently contiguous observations.
        # Use actual source trading dates, including pre-period warmup.
        full_dates = pd.DatetimeIndex(sorted({pd.Timestamp(s.strip().strip('"')[:10]) for s in raw_dates}))
        df = df.reindex(full_dates)
        close = df['Close']
        ma = pd.concat([close.rolling(w, min_periods=w).mean()/close - 1 for w in [5,10,20,30]],axis=1)
        frame = df.reindex(dates)
        train = frame.iloc[:756]
        scale, volscale = train.Close.max(), train.Volume.max()
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(volscale) or volscale <= 0:
            raise ValueError(f"Invalid training scaling for {ticker}")
        vals = np.column_stack([frame[['Open','High','Low','Close']].to_numpy()/scale,
                                frame.Volume.to_numpy()/volscale, ma.reindex(dates).to_numpy()])
        valid = np.isfinite(vals).all(axis=1) & (frame[['Open','High','Low','Close']] > 0).all(axis=1).to_numpy()
        features[i] = np.nan_to_num(vals, nan=0, posinf=0, neginf=0)
        price[i] = np.nan_to_num(frame.Close.to_numpy()/scale, nan=1)
        opening[i] = np.nan_to_num(frame.Open.to_numpy()/scale, nan=1)
        mask[i] = valid
    if absent:
        raise FileNotFoundError(f"Missing raw files ({len(absent)}): {absent[:10]}")
    edges, relation_stats = graph_from_archive(market, n)
    truth = np.zeros_like(price)
    truth[:, 1:] = price[:, 1:]/np.maximum(price[:, :-1], 1e-8) - 1
    overnight = np.zeros_like(price)
    overnight[:, 1:] = opening[:, 1:]/np.maximum(price[:, :-1], 1e-8) - 1
    dest = ROOT / "data"; dest.mkdir(exist_ok=True)
    outfile = dest / f"{market}.npz"
    np.savez_compressed(outfile, features=features, price=price, truth=truth, overnight=overnight,
                        mask=mask, edges=edges, dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str),
                        tickers=np.array(tickers))
    meta = {"status": "reconstructed_reference_data_not_author_PDU_bundle", "market": market,
        "source": "https://github.com/fulifeng/Temporal_Relational_Stock_Ranking",
        "source_commit": subprocess.check_output(['git','-C',str(ROOT/'reference-rsr'),'rev-parse','HEAD'],text=True).strip(),
        "shape": list(features.shape), "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
        "train_last": str(dates[755].date()), "validation_first": str(dates[756].date()),
        "validation_last": str(dates[1007].date()), "test_first": str(dates[1008].date()),
        "missing_stock_day_fraction": float(1-mask.mean()), "edges": edges.shape[1], "relations": relation_stats,
        "features": ['open','high','low','close','volume','ma5_relative','ma10_relative','ma20_relative','ma30_relative'],
        "scaling": "OHLC/max training close; volume/max training volume; MA(window)/current close - 1",
        "target": "next close / current close - 1; overnight target retained separately",
        "point_in_time_relations": False,
        "relation_archive_sha256": hashlib.sha256((source/'relation.tar.gz').read_bytes()).hexdigest(),
        "raw_sha256": input_hashes, "panel_sha256": hashlib.sha256(outfile.read_bytes()).hexdigest()}
    (dest / f"{market}.json").write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k!='raw_sha256'},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--market',choices=['NASDAQ','NYSE'],required=True)
    prepare(parser.parse_args().market)
