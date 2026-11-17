from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow.parquet as pq
import yaml

from src.io_utils_fast import (
    iter_users_from_all_parquet,
    load_market_stats,
    load_popularity,
    load_covariance_coo,
    ensure_parents,
    make_runs_writer,
    RunsWriter,
    count_users_in_all_parquet,
)
from src.rura_core_fast import (
    ruraConfig,
    ruraResultRow,
    run_rura_for_user,
    parse_method_flags,
)
from src.risk_utils import get_rho

ALGO_VERSION = "rura_fast_v1.2.1"


@dataclass
class Args:
    features_dir: str
    candidates_dir: str
    profiles_dir: str
    slices_file: str
    out_dir: str
    method: str
    lambda_div: float
    beta_pop: float
    use_cov: bool
    require_cov: bool
    L: int
    w_max: float
    eps_improve: float
    diag_eps: float
    device: str
    num_workers: int
    chunk_rows: int
    seed: int
    log: str
    config: Optional[str]
    progress_every_users: int
    progress_time_sec: float
    progress_print_user: bool
    progress_estimate_total: bool
    on_user_error: str
    user_filter_file: Optional[str]


def _parse_bool(x) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def _parse_args() -> Args:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=str, default="")
    ns_base, _ = base.parse_known_args()

    ap = argparse.ArgumentParser(description="rura Phase-7 high-throughput deterministic re-ranker", parents=[base])
    ap.add_argument("--features_dir", type=str, default="features")
    ap.add_argument("--candidates_dir", type=str, default="candidates")
    ap.add_argument("--profiles_dir", type=str, default="profiles")
    ap.add_argument("--slices_file", type=str, default="splits/slices.jsonl")
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--method", type=str, default="rura_la", help="rura_la|rura_uf|rura_la_no_sigma|rura_la_crra ...")
    ap.add_argument("--lambda_div", type=float, default=0.05)
    ap.add_argument("--beta_pop", type=float, default=0.10)
    ap.add_argument("--use_cov", type=_parse_bool, default=True)
    ap.add_argument("--require_cov", type=_parse_bool, default=True)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--w_max", type=float, default=0.4)
    ap.add_argument("--eps_improve", type=float, default=1e-9)
    ap.add_argument("--diag_eps", type=float, default=1e-10)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|gpu")
    ap.add_argument("--num_workers", type=int, default=min(16, os.cpu_count() or 16))
    ap.add_argument("--chunk_rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--log", type=str, default="logs/rura.log")
    ap.add_argument("--progress_every_users", type=int, default=1000, help="log progress every N users")
    ap.add_argument("--progress_time_sec", type=float, default=30.0, help="log progress if this many seconds have elapsed (whichever comes first)")
    ap.add_argument("--progress_print_user", type=_parse_bool, default=True, help="include the most recent user_id in logs")
    ap.add_argument("--progress_estimate_total", type=_parse_bool, default=False, help="pre-scan all candidates to estimate total users in advance (used for ETA calculation)")
    ap.add_argument("--on_user_error", type=str, default="skip", choices=["skip", "raise"])
    ap.add_argument("--user_filter_file", type=str, default="", help="process only user_ids listed in this file (one ID per line)")

    if ns_base.config and Path(ns_base.config).exists():
        with open(ns_base.config, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
        ap.set_defaults(**{k: v for k, v in conf.items() if k in {a.dest for a in ap._actions}})

    ns = ap.parse_args()

    return Args(
        features_dir=ns.features_dir,
        candidates_dir=ns.candidates_dir,
        profiles_dir=ns.profiles_dir,
        slices_file=ns.slices_file,
        out_dir=ns.out_dir,
        method=ns.method,
        lambda_div=ns.lambda_div,
        beta_pop=ns.beta_pop,
        use_cov=bool(ns.use_cov),
        require_cov=bool(ns.require_cov),
        L=ns.L,
        w_max=ns.w_max,
        eps_improve=ns.eps_improve,
        diag_eps=ns.diag_eps,
        device=ns.device,
        num_workers=ns.num_workers,
        chunk_rows=ns.chunk_rows,
        seed=ns.seed,
        log=ns.log,
        config=ns_base.config if ns_base.config else None,
        progress_every_users=ns.progress_every_users,
        progress_time_sec=ns.progress_time_sec,
        progress_print_user=bool(ns.progress_print_user),
        progress_estimate_total=bool(ns.progress_estimate_total),
        on_user_error=ns.on_user_error,
        user_filter_file=(ns.user_filter_file or None),
    )


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


def _read_slices(slices_file: str) -> List[str]:
    sids: List[str] = []
    with open(slices_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = str(rec.get("slice_id", "") or rec.get("id", "")).strip()
            if sid:
                sids.append(sid)
    if not sids:
        raise RuntimeError(f"No slice_ids found in {slices_file}")
    return sids


def _fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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


def _process_one_slice(
    slice_id: str,
    args: Args,
    bcfg: ruraConfig,
) -> Tuple[str, int, int, Dict[str, float]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    logger = logging.getLogger(f"slice-{slice_id}")
    logger.info(f"Start slice {slice_id} | device={bcfg.device} | method={args.method}")

    t_stage: Dict[str, float] = {}

    t0 = time.monotonic()
    fdir = Path(args.features_dir) / slice_id
    r_sigma_dict, sigma2_default = load_market_stats(fdir)
    s_dict = load_popularity(fdir)
    t_stage["features_load_sec"] = time.monotonic() - t0

    t0 = time.monotonic()
    cov_index = None
    want_cov = args.use_cov and parse_method_flags(args.method).use_cov
    if want_cov:
        cov_index = load_covariance_coo(fdir)
        if args.require_cov and (not cov_index):
            raise FileNotFoundError(f"[strict] covariance required but missing/empty: {fdir / 'covariance.parquet'}")
    t_stage["cov_load_sec"] = time.monotonic() - t0

    t0 = time.monotonic()
    prof_path = Path(args.profiles_dir) / slice_id / "user_params.parquet"
    if not prof_path.exists():
        raise FileNotFoundError(f"profiles not found: {prof_path}")
    prof_tbl = pq.read_table(prof_path, columns=["user_id", "risk_model", "gamma_or_eta", "B", "k"])
    prof_pd = prof_tbl.to_pandas()
    profiles: Dict[str, Tuple[str, float, float, int]] = {}
    for row in prof_pd.itertuples(index=False):
        profiles[str(row.user_id)] = (str(row.risk_model), float(row.gamma_or_eta), float(row.B), int(row.k))
    t_stage["profiles_load_sec"] = time.monotonic() - t0

    cand_path = Path(args.candidates_dir) / f"{slice_id}_all.parquet"
    if not cand_path.exists():
        raise FileNotFoundError(f"candidates-all not found: {cand_path}")

    total_users_est: Optional[int] = None
    if args.progress_estimate_total:
        t0 = time.monotonic()
        total_users_est = count_users_in_all_parquet(cand_path, batch_rows=max(200_000, args.chunk_rows))
        logger.info(f"[{slice_id}] estimated total users: {total_users_est} (prepass {time.monotonic()-t0:.1f}s)")

    out_dir = Path(args.out_dir) / slice_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method}.parquet"
    t0 = time.monotonic()
    writer: RunsWriter = make_runs_writer(out_path)
    writer_open_sec = time.monotonic() - t0

    n_users_yielded = 0
    n_users_processed = 0
    n_users_failed = 0
    n_rows_out = 0
    rows_seen = 0

    feature_lookup_sec = 0.0
    rura_core_sec = 0.0
    write_rows_sec = 0.0
    write_calls = 0

    sum_w_per_user: List[float] = []
    obj_per_user: List[float] = []
    k_sum = 0
    k_used_sum = 0
    cand_cnt_sum = 0

    start_t = time.monotonic()
    last_log_t = start_t
    last_logged_users = 0

    columns = ["user_id", "item_id", "p_i", "eligible", "in_holdings_flag"]
    user_whitelist = _USER_WHITELIST

    iter_t0 = time.monotonic()
    for uid, udata in iter_users_from_all_parquet(
        cand_path,
        columns=columns,
        batch_rows=args.chunk_rows,
        user_whitelist=user_whitelist,
    ):
        n_users_yielded += 1
        rows_seen += int(udata["item_id"].shape[0])
        cand_cnt_sum += int(udata["item_id"].shape[0])

        now = time.monotonic()
        need_log = False
        if (n_users_yielded - last_logged_users) >= args.progress_every_users:
            need_log = True
        if (now - last_log_t) >= args.progress_time_sec:
            need_log = True

        if need_log:
            elapsed = now - start_t
            rate_u = n_users_yielded / max(elapsed, 1e-9)
            rate_r = rows_seen / max(elapsed, 1e-9)
            msg = f"[{slice_id}] users={n_users_yielded}"
            if total_users_est:
                rem = max(0, total_users_est - n_users_yielded)
                eta_sec = rem / max(rate_u, 1e-9) if rate_u > 0 else float("inf")
                msg += f"/{total_users_est}  elapsed={_fmt_hms(elapsed)}  rate={rate_u:.1f} u/s, {rate_r:.0f} rows/s  ETA~{_fmt_hms(eta_sec)}"
            else:
                msg += f"  elapsed={_fmt_hms(elapsed)}  rate={rate_u:.1f} u/s, {rate_r:.0f} rows/s"
            if args.progress_print_user:
                msg += f"  last_user={uid}"
            logger.info(msg)
            last_log_t = now
            last_logged_users = n_users_yielded

        try:
            prof = profiles.get(uid)
            if prof is None:
                continue
            risk_model, gamma_or_eta, B, k = prof
            rho = get_rho(risk_model, gamma_or_eta)

            t1 = time.monotonic()
            mask = (udata["eligible"]) & (~udata["in_holdings_flag"])
            if not mask.any():
                continue
            item_ids = udata["item_id"][mask]
            p_i = udata["p_i"][mask]

            n = item_ids.shape[0]
            r = np.empty(n, dtype=np.float64)
            sigma2 = np.empty(n, dtype=np.float64)
            s = np.empty(n, dtype=np.float64)
            for idx, it in enumerate(item_ids):
                it_s = str(it)
                rs = r_sigma_dict.get(it_s)
                if rs is None:
                    r[idx] = 0.0
                    sigma2[idx] = sigma2_default
                else:
                    r[idx] = rs[0]
                    sigma2[idx] = rs[1]
                s[idx] = s_dict.get(it_s, 0.0)
            feature_lookup_sec += (time.monotonic() - t1)

            t2 = time.monotonic()
            rows: List[ruraResultRow] = run_rura_for_user(
                slice_id=slice_id,
                user_id=uid,
                item_ids=item_ids.astype(str),
                p_i=p_i,
                r=r,
                sigma2=sigma2,
                s=s,
                rho=rho,
                B=B,
                k=k,
                cfg=bcfg,
                cov_index=cov_index,
                algo_version=ALGO_VERSION,
                seed=args.seed,
            )
            rura_core_sec += (time.monotonic() - t2)

            n_users_processed += 1
            if rows:
                t3 = time.monotonic()
                writer.write_rows(rows)
                write_rows_sec += (time.monotonic() - t3)
                write_calls += 1

                n_rows_out += len(rows)
                k_sum += int(k)
                k_used_sum += len(rows)
                sum_w = float(sum(r.w for r in rows))
                sum_w_per_user.append(sum_w)
                obj_per_user.append(float(rows[0].obj_total))

        except Exception as e:
            n_users_failed += 1
            if args.on_user_error == "raise":
                raise
            logger.warning(f"[{slice_id}] user={uid} failed: {e}")

    iter_total_sec = time.monotonic() - iter_t0

    t0 = time.monotonic()
    writer.close()
    writer_close_sec = time.monotonic() - t0

    file_size_mb = 0.0
    try:
        file_size_mb = (out_path.stat().st_size or 0) / (1024.0 * 1024.0)
    except Exception:
        pass

    t0 = time.monotonic()

    def _pct50_95(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return (0.0, 0.0)
        arr = np.asarray(xs, dtype=float)
        return (float(np.percentile(arr, 50)), float(np.percentile(arr, 95)))

    sw50, sw95 = _pct50_95(sum_w_per_user)
    obj50, obj95 = _pct50_95(obj_per_user)
    avg_cand = (cand_cnt_sum / max(n_users_yielded, 1)) if n_users_yielded else 0.0
    k_fill = (k_used_sum / max(k_sum, 1)) if k_sum else 0.0

    summary_sec = time.monotonic() - t0

    total_elapsed = (time.monotonic() - start_t)

    logger.info(
        "Done slice %s: users(yielded/processed/failed)=%d/%d/%d%s, rows_out=%d, "
        "cand_avg=%.1f, k_fill=%.3f, sum_w p50/p95=%.3f/%.3f, obj p50/p95=%.6g/%.6g, "
        "time(iter_total=%s, feature=%.2fs, core=%.2fs, write_rows=%.2fs [%d calls], "
        "writer_open=%.2fs, writer_close=%.2fs, summary=%.2fs, total=%s), out=%s, size=%.1f MiB"
        % (
            slice_id, n_users_yielded, n_users_processed, n_users_failed,
            (f"/{total_users_est}" if total_users_est else ""),
            n_rows_out,
            avg_cand, k_fill, sw50, sw95, obj50, obj95,
            _fmt_hms(iter_total_sec),
            feature_lookup_sec, rura_core_sec, write_rows_sec, write_calls,
            writer_open_sec, writer_close_sec, summary_sec,
            _fmt_hms(total_elapsed),
            str(out_path), file_size_mb,
        )
    )

    t_stage["writer_open_sec"] = writer_open_sec
    t_stage["feature_lookup_sec"] = feature_lookup_sec
    t_stage["rura_core_sec"] = rura_core_sec
    t_stage["write_rows_sec"] = write_rows_sec
    t_stage["write_calls"] = float(write_calls)
    t_stage["writer_close_sec"] = writer_close_sec
    t_stage["summary_sec"] = summary_sec
    t_stage["iter_total_sec"] = iter_total_sec
    t_stage["scan_total_sec"] = iter_total_sec
    t_stage["total_sec"] = total_elapsed
    t_stage["file_size_mb"] = file_size_mb

    return slice_id, n_users_yielded, n_rows_out, t_stage


_USER_WHITELIST: Optional[Set[str]] = None


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log)
    logging.getLogger("main").info(
        f"CONFIG: device={args.device} method={args.method} num_workers={args.num_workers} "
        f"chunk_rows={args.chunk_rows} progress_every_users={args.progress_every_users} "
        f"progress_time_sec={args.progress_time_sec} estimate_total={args.progress_estimate_total} "
        f"user_filter_file={'yes' if args.user_filter_file else 'no'} on_user_error={args.on_user_error}"
    )

    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    _ = np.random.default_rng(args.seed)

    global _USER_WHITELIST
    _USER_WHITELIST = _load_user_whitelist(args.user_filter_file)

    sids = _read_slices(args.slices_file)

    bcfg = ruraConfig(
        method=args.method,
        lambda_div=args.lambda_div,
        beta_pop=args.beta_pop,
        L=args.L,
        w_max=args.w_max,
        eps_improve=args.eps_improve,
        diag_eps=args.diag_eps,
        device=args.device,
    )

    results: List[Tuple[str, int, int, Dict[str, float]]] = []
    stage_agg: Dict[str, float] = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = []
        for sid in sids:
            futs.append(ex.submit(_process_one_slice, sid, args, bcfg))
        for fut in as_completed(futs):
            sid, n_users, n_rows, t_stage = fut.result()
            results.append((sid, n_users, n_rows, t_stage))
            logging.info(
                f"FINISHED: {sid} users={n_users} rows={n_rows} "
                f"| writer_open={t_stage.get('writer_open_sec',0):.2f}s "
                f"close={t_stage.get('writer_close_sec',0):.2f}s "
                f"core={t_stage.get('rura_core_sec',0):.2f}s "
                f"write={t_stage.get('write_rows_sec',0):.2f}s/{int(t_stage.get('write_calls',0))} "
                f"iter_total={t_stage.get('iter_total_sec',0):.2f}s size={t_stage.get('file_size_mb',0):.1f}MiB"
            )
            for k, v in t_stage.items():
                if isinstance(v, (int, float)):
                    stage_agg[k] = stage_agg.get(k, 0.0) + float(v)

    if results:
        keys = [
            "features_load_sec", "cov_load_sec", "profiles_load_sec",
            "feature_lookup_sec", "rura_core_sec", "write_rows_sec",
            "writer_open_sec", "writer_close_sec", "summary_sec",
            "iter_total_sec", "total_sec",
        ]
        msg = " | ".join(f"{k}={stage_agg.get(k,0.0):.1f}s" for k in keys)
        logging.getLogger("main").info(f"All done. slices={len(results)} | {msg}")
    else:
        logging.getLogger("main").info("All done. slices=0")


if __name__ == "__main__":
    main()

