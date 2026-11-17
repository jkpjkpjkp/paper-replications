#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from src.io_utils_fast import ensure_parents  # type: ignore
from src.metrics_core_fast import ndcg_at_k_binary, roi_from_prices
from src.groundtruth_utils import parse_slices_jsonl, build_newbuy_labels, load_prices_at_dates

ALGO_VERSION = "metrics_fast_v1.0.0"

@dataclass
class Args:
    runs_dir: str
    features_dir: str
    candidates_dir: str
    profiles_dir: str
    slices_file: str
    clean_dir: str
    out_dir: str
    methods: List[str]
    eval_mode: str
    cvar_mode: str
    alpha_cvar: float
    overall_weight: str
    fees_bps_buy: float
    fees_bps_sell: float
    slippage_bps: float
    device: str
    num_workers: int
    chunk_rows: int
    user_filter_file: Optional[str]
    progress_estimate_total: bool
    progress_every_users: int
    progress_time_sec: float
    seed: int
    log: str
    use_cov: bool
    mc_samples: int

def _parse_bool(x) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}

def _parse_methods(x: str) -> List[str]:
    xs = [t.strip() for t in (x or "").split(",")]
    return [t for t in xs if t]

def _setup_logging(log_path: str) -> None:
    ensure_parents(Path(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def _parse_args() -> Args:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=str, default="")
    ns_base, _ = base.parse_known_args()
    ap = argparse.ArgumentParser(
        description="Phase-9 Evaluation — fast, parallel, GPU-ready",
        parents=[base],
    )
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--features_dir", type=str, default="features")
    ap.add_argument("--candidates_dir", type=str, default="candidates_subset")
    ap.add_argument("--profiles_dir", type=str, default="profiles")
    ap.add_argument("--slices_file", type=str, default="splits/slices.jsonl")
    ap.add_argument("--clean_dir", type=str, default="clean")
    ap.add_argument("--out_dir", type=str, default="metrics")
    ap.add_argument("--methods", type=str, default="raw_blend,likelihood_only")
    ap.add_argument("--eval_mode", type=str, default="realized_ts", choices=["realized_ts", "modelled"])
    ap.add_argument("--cvar_mode", type=str, default="normal", choices=["normal", "mc"])
    ap.add_argument("--alpha_cvar", type=float, default=0.95)
    ap.add_argument("--overall_weight", type=str, default="users", choices=["users", "equal"])
    ap.add_argument("--fees_bps_buy", type=float, default=10.0)
    ap.add_argument("--fees_bps_sell", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=10.0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--num_workers", type=int, default=min(16, os.cpu_count() or 16))
    ap.add_argument("--chunk_rows", type=int, default=200_000)
    ap.add_argument("--user_filter_file", type=str, default="")
    ap.add_argument("--progress_estimate_total", type=_parse_bool, default=True)
    ap.add_argument("--progress_every_users", type=int, default=500)
    ap.add_argument("--progress_time_sec", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--log", type=str, default="logs/metrics.log")
    ap.add_argument("--use_cov", type=_parse_bool, default=True)
    ap.add_argument("--mc_samples", type=int, default=200000)
    ns = ap.parse_args()
    if ns_base.config and Path(ns_base.config).exists():
        with open(ns_base.config, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
        for k, v in conf.items():
            if hasattr(ns, k):
                setattr(ns, k, v)
    return Args(
        runs_dir=ns.runs_dir,
        features_dir=ns.features_dir,
        candidates_dir=ns.candidates_dir,
        profiles_dir=ns.profiles_dir,
        slices_file=ns.slices_file,
        clean_dir=ns.clean_dir,
        out_dir=ns.out_dir,
        methods=_parse_methods(ns.methods),
        eval_mode=ns.eval_mode,
        cvar_mode=ns.cvar_mode,
        alpha_cvar=float(ns.alpha_cvar),
        overall_weight=ns.overall_weight,
        fees_bps_buy=float(ns.fees_bps_buy),
        fees_bps_sell=float(ns.fees_bps_sell),
        slippage_bps=float(ns.slippage_bps),
        device=ns.device,
        num_workers=int(ns.num_workers),
        chunk_rows=int(ns.chunk_rows),
        user_filter_file=(ns.user_filter_file or None),
        progress_estimate_total=bool(ns.progress_estimate_total),
        progress_every_users=int(ns.progress_every_users),
        progress_time_sec=float(ns.progress_time_sec),
        seed=int(ns.seed),
        log=ns.log,
        use_cov=bool(ns.use_cov),
        mc_samples=int(ns.mc_samples),
    )

def _load_user_whitelist(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"user_filter_file not found: {p}")
    wl: Set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                wl.add(s)
    if not wl:
        raise RuntimeError(f"user_filter_file is empty: {p}")
    return wl

def _write_metrics_parquet(path: Path, rows: List[Dict]) -> None:
    if not rows:
        schema = pa.schema(
            [
                ("slice_id", pa.string()),
                ("user_id", pa.string()),
                ("method", pa.string()),
                ("nDCG@k", pa.float64()),
                ("ROI@k", pa.float64()),
                ("k_used", pa.int32()),
                ("sum_w", pa.float64()),
                ("n_items", pa.int32()),
                ("algo_version", pa.string()),
                ("seed", pa.int64()),
            ]
        )
        ensure_parents(path)
        with pq.ParquetWriter(path, schema=schema, compression="snappy", use_dictionary=True) as w:
            pass
        return
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    ensure_parents(path)
    pq.write_table(table, path, compression="snappy", use_dictionary=True)

def _aggregate_overall(metrics_dir: Path, slices: List[Dict], methods: List[str], overall_weight: str, out_csv: Path, out_parquet: Path) -> None:
    recs: List[Dict] = []
    for m in methods:
        s_means = []
        s_counts = []
        for sd in slices:
            sid = str(sd["slice_id"])
            p = metrics_dir / sid / f"{m}.parquet"
            if not p.exists():
                continue
            tbl = pq.read_table(p, columns=["user_id", "nDCG@k", "ROI@k"])
            if tbl.num_rows == 0:
                continue
            df = tbl.to_pandas()
            s_means.append((df["nDCG@k"].mean(), df["ROI@k"].mean()))
            s_counts.append(len(df))
        if not s_means:
            continue
        arr = np.asarray(s_means, dtype=float)
        if overall_weight == "users":
            w = np.asarray(s_counts, dtype=float)
            w = w / (w.sum() if w.sum() > 0 else 1.0)
            overall = (arr * w.reshape(-1, 1)).sum(axis=0)
        else:
            overall = arr.mean(axis=0)
        recs.append(
            {
                "method": m,
                "nDCG@k": overall[0],
                "ROI@k": overall[1],
                "slices_present": len(s_means),
                "weighting": overall_weight,
                "algo_version": ALGO_VERSION,
            }
        )
    if not recs:
        logging.getLogger("main").warning("No metrics to aggregate into overall.")
        return
    df = pd.DataFrame(recs)
    ensure_parents(out_csv)
    df.to_csv(out_csv, index=False)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_parquet, compression="snappy", use_dictionary=True)

def _process_one_slice(sd: Dict, args: Args, user_whitelist: Optional[Set[str]]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    slice_id = str(sd["slice_id"])
    test_start = pd.to_datetime(sd["test_start"]).date()
    test_end = pd.to_datetime(sd["test_end"]).date()
    logger = logging.getLogger(f"slice-{slice_id}")
    t0 = time.monotonic()
    stats: Dict[str, Dict[str, float]] = {}
    gt_t0 = time.monotonic()
    tx_path = Path(args.clean_dir) / "transactions.parquet"
    if not tx_path.exists():
        raise FileNotFoundError(f"missing transactions: {tx_path}")
    labels = build_newbuy_labels(
        tx_path=tx_path,
        test_start=test_start,
        test_end=test_end,
        user_whitelist=user_whitelist,
        chunk_rows=max(200_000, args.chunk_rows),
    )
    gt_sec = time.monotonic() - gt_t0
    logger.info(f"[{slice_id}] labels built: users={len(labels)} in {gt_sec:.1f}s")
    prices_path = Path(args.clean_dir) / "prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"missing prices: {prices_path}")
    for method in args.methods:
        run_path = Path(args.runs_dir) / slice_id / f"{method}.parquet"
        if not run_path.exists():
            logger.warning(f"[{slice_id}] runs not found for method={method}: {run_path}")
            continue
        mt0 = time.monotonic()
        n_users = 0
        n_rows = 0
        rows_out: List[Dict] = []
        need_cols = ["slice_id", "user_id", "item_id", "rank", "w", "r", "sigma2", "k"]
        sch = pq.read_schema(run_path)
        cols = [c for c in need_cols if c in sch.names]
        tbl = pq.read_table(run_path, columns=cols)
        if tbl.num_rows == 0:
            out_path = Path(args.out_dir) / slice_id / f"{method}.parquet"
            _write_metrics_parquet(out_path, [])
            stats[method] = {"users": 0, "rows": 0, "sec": 0.0}
            continue
        df_all = tbl.to_pandas()
        if user_whitelist is not None:
            df_all = df_all[df_all["user_id"].astype(str).isin(user_whitelist)]
        if len(df_all) == 0:
            out_path = Path(args.out_dir) / slice_id / f"{method}.parquet"
            _write_metrics_parquet(out_path, [])
            stats[method] = {"users": 0, "rows": 0, "sec": time.monotonic() - mt0}
            continue
        asset_union = sorted(set(df_all["item_id"].astype(str).tolist()))
        start_close, end_close = load_prices_at_dates(
            prices_path, asset_union, test_start, test_end, chunk_rows=max(200_000, args.chunk_rows)
        )
        last_log_t = time.monotonic()
        for uid, g in df_all.groupby("user_id", sort=False):
            uid = str(uid)
            n_users += 1
            k_used = int(len(g))
            n_rows += k_used
            items = g["item_id"].astype(str).to_numpy()
            w = g["w"].astype(float).to_numpy()
            lbl_set = labels.get(uid, set())
            ndcg = ndcg_at_k_binary(items, lbl_set, k_used)
            roi = roi_from_prices(items, w, start_close, end_close,
                                  fee_bps_buy=args.fees_bps_buy, fee_bps_sell=args.fees_bps_sell, slip_bps=args.slippage_bps)
            rows_out.append(
                {
                    "slice_id": slice_id,
                    "user_id": uid,
                    "method": method,
                    "nDCG@k": float(ndcg),
                    "ROI@k": float(roi),
                    "k_used": int(k_used),
                    "sum_w": float(w.sum()),
                    "n_items": int(k_used),
                    "algo_version": ALGO_VERSION,
                    "seed": int(args.seed),
                }
            )
            now = time.monotonic()
            if now - last_log_t >= args.progress_time_sec and (n_users % max(1, args.progress_every_users) == 0):
                logger.info(f"[{slice_id}][{method}] users={n_users}, rows={n_rows}")
                last_log_t = now
        out_path = Path(args.out_dir) / slice_id / f"{method}.parquet"
        _write_metrics_parquet(out_path, rows_out)
        stats[method] = {"users": n_users, "rows": n_rows, "sec": time.monotonic() - mt0}
        logger.info(f"[{slice_id}][{method}] done: users={n_users}, rows={n_rows}, out={out_path}")
    logger.info(f"[{slice_id}] slice done in {time.monotonic()-t0:.1f}s (labels={gt_sec:.1f}s)")
    return slice_id, stats

def main() -> None:
    args = _parse_args()
    _setup_logging(args.log)
    np.random.seed(args.seed)
    slices = parse_slices_jsonl(args.slices_file)
    if not slices:
        raise RuntimeError(f"No slices in {args.slices_file}")
    user_whitelist = _load_user_whitelist(args.user_filter_file)
    agg_stats: Dict[str, Dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = [ex.submit(_process_one_slice, sd, args, user_whitelist) for sd in slices]
        for fut in as_completed(futs):
            sid, st = fut.result()
            agg_stats[sid] = {f"{m}.sec": v.get("sec", 0.0) for m, v in st.items()}
            logging.getLogger("main").info(f"FINISHED slice={sid} | stats={st}")
    metrics_dir = Path(args.out_dir)
    out_csv = metrics_dir / "overall.csv"
    out_parquet = metrics_dir / "overall.parquet"
    _aggregate_overall(metrics_dir, slices, args.methods, args.overall_weight, out_csv, out_parquet)
    logging.getLogger("main").info(f"Overall written: {out_csv} , {out_parquet}")
    logging.getLogger("main").info("All done.")

if __name__ == "__main__":
    main()

