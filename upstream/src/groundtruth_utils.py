# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def parse_slices_jsonl(slices_file: str) -> List[Dict]:
    """
    Expect JSONL with fields including slice_id, test_start, test_end.
    Falls back to nested 'test': {'start','end'} if present.
    """
    out: List[Dict] = []
    with open(slices_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = str(rec.get("slice_id", "") or rec.get("id", "")).strip()
            if not sid:
                continue
            ts = rec.get("test_start") or (rec.get("test", {}) or {}).get("start")
            te = rec.get("test_end") or (rec.get("test", {}) or {}).get("end")
            if not ts or not te:
                raise ValueError(f"slice record missing test_start/test_end: {rec}")
            out.append(
                {
                    "slice_id": sid,
                    "test_start": ts,
                    "test_end": te,
                }
            )
    return out


# ---------- helpers ----------
def _tsns_scalar(d: date) -> pa.Scalar:
    """Make a timestamp[ns] scalar for Arrow filter."""
    return pa.scalar(pd.Timestamp(d), type=pa.timestamp("ns"))


def _to_date(x) -> date:
    return pd.Timestamp(x).date()


# ---------- labels: new BUY in test ----------
def build_newbuy_labels(
    tx_path: Path,
    test_start: date,
    test_end: date,
    user_whitelist: Optional[Set[str]] = None,
    chunk_rows: int = 200_000,
) -> Dict[str, Set[str]]:
    """
    Construct per-user set of asset_id with new BUY in test window.
    new BUY = first transaction in [test_start, test_end] for (user, asset) is BUY,
              and user had no positive holding before test_start.
    Notes:
    - Compare Arrow timestamp columns with timestamp[ns] scalars (not strings).
    - user_whitelist is applied on pandas after scan to avoid dtype mismatch.
    """
    ds_tx = ds.dataset(str(tx_path))

    # Holdings before test_start
    filt_pre = ds.field("date") < _tsns_scalar(test_start)
    sc_pre = ds.Scanner.from_dataset(
        ds_tx,
        filter=filt_pre,
        columns=["customer_id", "asset_id", "side", "quantity", "date"],
        batch_size=chunk_rows,
    )
    pre_pos = defaultdict(float)  # (user, asset) -> net qty
    for b in sc_pre.to_batches():
        df = b.to_pandas()
        if user_whitelist:
            df = df[df["customer_id"].astype(str).isin(user_whitelist)]
        for r in df.itertuples(index=False):
            uid = str(r.customer_id)
            aid = str(r.asset_id)
            q = float(r.quantity or 0.0)
            side = str(r.side).upper()
            if side == "BUY":
                pre_pos[(uid, aid)] += q
            elif side == "SELL":
                pre_pos[(uid, aid)] -= q

    # First txn in test window
    filt_test = (ds.field("date") >= _tsns_scalar(test_start)) & (ds.field("date") <= _tsns_scalar(test_end))
    sc_test = ds.Scanner.from_dataset(
        ds_tx,
        filter=filt_test,
        columns=["customer_id", "asset_id", "side", "date"],
        batch_size=chunk_rows,
    )

    first_tx: Dict[Tuple[str, str], Tuple[str, str]] = {}  # (uid, asset) -> (side, date_iso)
    for b in sc_test.to_batches():
        df = b.to_pandas()
        if user_whitelist:
            df = df[df["customer_id"].astype(str).isin(user_whitelist)]
        # stable sort
        df = df.sort_values(["customer_id", "asset_id", "date"], kind="mergesort")
        for r in df.itertuples(index=False):
            key = (str(r.customer_id), str(r.asset_id))
            if key in first_tx:
                continue
            first_tx[key] = (str(r.side).upper(), pd.Timestamp(r.date).strftime("%Y-%m-%d"))

    labels: Dict[str, Set[str]] = defaultdict(set)
    for (uid, aid), (side, _dt) in first_tx.items():
        if side == "BUY" and pre_pos.get((uid, aid), 0.0) <= 0.0:
            labels[uid].add(aid)
    return labels


# ---------- prices: start/end ----------
def load_prices_at_dates(
    prices_path: Path,
    asset_ids: Sequence[str],
    start: date,
    end: date,
    chunk_rows: int = 200_000,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Load close prices at start and end dates for asset universe.
    Returns: (start_close_map, end_close_map)
    Robust to timestamp[ns]/date32 by filtering a window and selecting exact dates in pandas.
    """
    ds_px = ds.dataset(str(prices_path))
    filt = (ds.field("date") >= _tsns_scalar(start)) & (ds.field("date") <= _tsns_scalar(end)) & (
        ds.field("asset_id").isin(list(asset_ids))
    )
    sc = ds.Scanner.from_dataset(
        ds_px,
        filter=filt,
        columns=["asset_id", "date", "close_eur"],
        batch_size=chunk_rows,
    )
    rows = []
    for b in sc.to_batches():
        rows.append(b.to_pandas()[["asset_id", "date", "close_eur"]])
    start_map: Dict[str, float] = {}
    end_map: Dict[str, float] = {}
    if not rows:
        return start_map, end_map
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    d_start = _to_date(start)
    d_end = _to_date(end)
    df_s = df[df["date"] == d_start]
    df_e = df[df["date"] == d_end]
    for r in df_s.itertuples(index=False):
        start_map[str(r.asset_id)] = float(r.close_eur or np.nan)
    for r in df_e.itertuples(index=False):
        end_map[str(r.asset_id)] = float(r.close_eur or np.nan)
    return start_map, end_map


# ---------- prices: full window ----------
def load_prices_window(
    prices_path: Path,
    asset_ids: Sequence[str],
    start: date,
    end: date,
    chunk_rows: int = 200_000,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load daily close prices in [start, end] for asset universe.
    Returns:
        dates: np.ndarray[datetime64[D]]
        asset_to_series: dict(asset_id -> np.ndarray of close prices aligned to dates)
    Missing dates for an asset are forward-filled (last observation carry-forward).
    """
    ds_px = ds.dataset(str(prices_path))
    filt = (ds.field("date") >= _tsns_scalar(start)) & (ds.field("date") <= _tsns_scalar(end)) & (
        ds.field("asset_id").isin(list(asset_ids))
    )
    sc = ds.Scanner.from_dataset(
        ds_px,
        filter=filt,
        columns=["asset_id", "date", "close_eur"],
        batch_size=chunk_rows,
    )
    # Collect into dataframe
    rows = []
    for b in sc.to_batches():
        rows.append(b.to_pandas()[["asset_id", "date", "close_eur"]])
    if not rows:
        return np.array([], dtype="datetime64[D]"), {}
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Pivot to wide (date x asset)
    pvt = df.pivot_table(index="date", columns="asset_id", values="close_eur", aggfunc="last").sort_index(kind="mergesort")
    # forward-fill per column
    pvt = pvt.ffill()
    dates = pd.to_datetime(pvt.index).to_numpy(dtype="datetime64[D]")
    asset_to_series = {str(c): pvt[c].to_numpy(dtype=float) for c in pvt.columns}
    return dates, asset_to_series
