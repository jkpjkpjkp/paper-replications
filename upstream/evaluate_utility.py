#!/usr/bin/env python3

import argparse, os, sys, json, time, logging, glob, math
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def setup_logger(log_path: Optional[str]):
    logger = logging.getLogger("utility")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

MU_FALLBACK_NAMES = [
    "mu","ret_mu","exp_return","expected_return",
    "score","pred","pred_mu","r_hat","y_pred","pred_score",
    "return","ret","p","p_i"
]
VAR_CANDIDATE_NAMES = ["var","sigma2","variance"]

def _discover_mu_and_var(cols: List[str], mu_pref: List[str]) -> Tuple[Optional[str], Optional[str]]:
    for c in mu_pref + [x for x in MU_FALLBACK_NAMES if x not in mu_pref]:
        if c in cols:
            mu = c
            break
    else:
        mu = None
    var = next((c for c in VAR_CANDIDATE_NAMES if c in cols), None)
    return mu, var

def _load_profiles_rho(profiles_dir: str, slice_id: str, logger) -> pd.DataFrame:
    p = os.path.join(profiles_dir, slice_id, "user_params.parquet")
    tbl = pq.read_table(p, columns=["user_id","risk_model","gamma_or_eta"])
    df = tbl.to_pandas()
    df["rho"] = df["gamma_or_eta"].astype(float)
    bad = (df["rho"]<=0).sum()
    if bad>0:
        logger.warning(f"[slice={slice_id}] rho<=0 found: {bad}; clip -> 1e-12")
        df.loc[df["rho"]<=0,"rho"] = 1e-12
    return df[["user_id","rho"]]

def _load_user_categories(csv_path: str, rho_scale_map: Dict[str,float], logger) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"user_id","category"} <= set(df.columns):
        raise ValueError("user_category_csv must have columns: user_id, category")
    df["category"] = df["category"].astype(str).str.strip()
    df["rho_scale"] = df["category"].map(rho_scale_map).fillna(1.0)
    unk = (df["category"].map(rho_scale_map).isna()).sum()
    if unk>0: logger.info(f"[risk-map] {unk} users have unknown category; rho_scale=1.0")
    return df[["user_id","category","rho_scale"]]

def _load_mu_from_dir(dir_path: str, slice_id: str, mu_pref: List[str], logger):
    if not dir_path: return None, None, None, None
    sdir = os.path.join(dir_path, slice_id)
    for cand in [f"{slice_id}_all.parquet", f"{slice_id}.parquet"]:
        p = os.path.join(dir_path, cand)
        if os.path.exists(p):
            t = pq.read_table(p)
            cols = t.column_names
            mu_col, var_col = _discover_mu_and_var(cols, mu_pref)
            if mu_col is None: return None, None, None, None
            if {"user_id","item_id"} <= set(cols):
                sel = ["user_id","item_id",mu_col] + ([var_col] if var_col else [])
                df = t.select(sel).to_pandas().drop_duplicates(["user_id","item_id"])
                df = df.rename(columns={mu_col:"mu"})
                if var_col: df = df.rename(columns={var_col:"var"})
                else: df["var"]=np.nan
                return df, "ui", ("candidates" if "candidates" in dir_path else "features"), p
            elif {"item_id"} <= set(cols):
                sel = ["item_id",mu_col] + ([var_col] if var_col else [])
                df = t.select(sel).to_pandas().drop_duplicates(["item_id"])
                df = df.rename(columns={mu_col:"mu"})
                if var_col: df = df.rename(columns={var_col:"var"})
                else: df["var"]=np.nan
                return df, "i", ("candidates" if "candidates" in dir_path else "features"), p
            else:
                return None, None, None, None
    paths = sorted(glob.glob(os.path.join(sdir, "*.parquet")))
    for f in paths:
        try:
            t = pq.read_table(f)
            cols = t.column_names
            mu_col, var_col = _discover_mu_and_var(cols, mu_pref)
            if mu_col is None: continue
            if {"user_id","item_id"} <= set(cols):
                sel = ["user_id","item_id",mu_col] + ([var_col] if var_col else [])
                df = t.select(sel).to_pandas().drop_duplicates(["user_id","item_id"])
                df = df.rename(columns={mu_col:"mu"})
                if var_col: df = df.rename(columns={var_col:"var"})
                else: df["var"]=np.nan
                return df, "ui", ("candidates" if "candidates" in dir_path else "features"), f
            elif {"item_id"} <= set(cols):
                sel = ["item_id",mu_col] + ([var_col] if var_col else [])
                df = t.select(sel).to_pandas().drop_duplicates(["item_id"])
                df = df.rename(columns={mu_col:"mu"})
                if var_col: df = df.rename(columns={var_col:"var"})
                else: df["var"]=np.nan
                return df, "i", ("candidates" if "candidates" in dir_path else "features"), f
        except Exception as e:
            logger.warning(f"[slice={slice_id}] read fail {f}: {e}")
            continue
    return None, None, None, None

def _load_mu_slice_or_features(slice_id: str, candidates_dir: Optional[str], features_dir: Optional[str],
                               mu_pref: List[str], logger):
    df, keyt, src, path = _load_mu_from_dir(candidates_dir, slice_id, mu_pref, logger)
    if df is not None: return df, keyt, src, path
    df, keyt, src, path = _load_mu_from_dir(features_dir, slice_id, mu_pref, logger)
    if df is not None: return df, keyt, src, path
    return None, None, None, None

def _read_runs_method(runs_dir: str, slice_id: str, method: str):
    p = os.path.join(runs_dir, slice_id, f"{method}.parquet")
    if not os.path.exists(p): return None
    t = pq.read_table(p, columns=["user_id","item_id"])
    return t.to_pandas()

def _u_cara_vec(mu: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-rho * mu)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--profiles_dir", required=True)
    ap.add_argument("--slices_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--user_category_csv", required=True)
    ap.add_argument("--rho_scale_map", default="Conservative=1.20,Income=1.10,Balanced=1.00,Aggressive=0.85")
    ap.add_argument("--min_rho", type=float, default=1e-6)
    ap.add_argument("--features_dir", default="")
    ap.add_argument("--candidates_dir", default="")
    ap.add_argument("--mu_col_candidates", default="score_blend,score_tx,score_px,p_i,score,pred")
    ap.add_argument("--methods", default="rura_la")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--log", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--progress_every_users", type=int, default=1000)
    ap.add_argument("--progress_time_sec", type=int, default=20)
    args = ap.parse_args()

    logger = setup_logger(args.log)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    per_slice_dir = os.path.join(args.out_dir, "per_slice"); os.makedirs(per_slice_dir, exist_ok=True)

    rho_scale_map: Dict[str,float] = {}
    for kv in args.rho_scale_map.split(","):
        kv = kv.strip()
        if not kv: continue
        k, v = kv.split("=")
        rho_scale_map[k.strip()] = float(v)

    df_cat = _load_user_categories(args.user_category_csv, rho_scale_map, logger)
    mu_col_pref = [c.strip() for c in args.mu_col_candidates.split(",") if c.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    totals = []
    with open(args.slices_file,"r",encoding="utf-8") as f:
        for line in f:
            meta = json.loads(line); slice_id = meta["slice_id"]; t0=time.time()
            logger.info(f"=== [slice={slice_id}] start ===")
            df_rho = _load_profiles_rho(args.profiles_dir, slice_id, logger)
            df_rho = df_rho.merge(df_cat, on="user_id", how="left")
            df_rho["rho_scale"] = df_rho["rho_scale"].fillna(1.0)
            df_rho["rho_eff"] = np.maximum(args.min_rho, df_rho["rho"] * df_rho["rho_scale"])
            user_to_cat = df_rho.set_index("user_id")["category"].to_dict()
            slice_mu_df, slice_key, mu_src, mu_path = _load_mu_slice_or_features(
                slice_id, args.candidates_dir or None, args.features_dir or None, mu_col_pref, logger
            )
            if slice_mu_df is None:
                logger.warning(f"[slice={slice_id}] no mu source found; skip")
                continue
            logger.info(f"[slice={slice_id}] mu source={mu_src} file={mu_path} key={slice_key}  rows={len(slice_mu_df)}")
            try:
                desc = slice_mu_df["mu"].describe().to_dict()
                logger.info(f"[slice={slice_id}] mu describe: {desc}")
            except Exception:
                pass
            recs = []
            for m in methods:
                df_runs = _read_runs_method(args.runs_dir, slice_id, m)
                if df_runs is None or df_runs.empty:
                    logger.warning(f"[slice={slice_id}] runs missing or empty: {m}")
                    continue
                if slice_key == "ui":
                    df = df_runs.merge(slice_mu_df, on=["user_id","item_id"], how="left")
                else:
                    df = df_runs.merge(slice_mu_df[["item_id","mu"]], on="item_id", how="left")
                df = df.dropna(subset=["mu"])
                if df.empty:
                    logger.warning(f"[slice={slice_id}][{m}] no join rows with mu; skip")
                    continue
                df = df.merge(df_rho[["user_id","rho_eff","category"]], on="user_id", how="left")
                df = df.dropna(subset=["rho_eff"])
                if df.empty:
                    logger.warning(f"[slice={slice_id}][{m}] no rows with rho; skip")
                    continue
                u_row = 1.0 - np.exp(-df["rho_eff"].to_numpy(np.float64) * df["mu"].to_numpy(np.float64))
                df_user = pd.DataFrame({
                    "user_id": df["user_id"].to_numpy(),
                    "category": df["category"].to_numpy(),
                    "u": u_row
                })
                g = df_user.groupby(["user_id","category"], sort=False, observed=True)["u"].mean().reset_index()
                summ = g.groupby("category", observed=True)["u"].agg(["count","mean","median","std"]).reset_index()
                summ.insert(0, "method", m)
                recs.append(summ)
                logger.info(f"[slice={slice_id}][{m}] users={g.shape[0]}  rows_joined={df.shape[0]}  elapsed={time.time()-t0:.1f}s")
            if not recs:
                logger.warning(f"[slice={slice_id}] no method produced summary.")
                continue
            out_slice = pd.concat(recs, ignore_index=True)
            out_path = os.path.join(per_slice_dir, f"{slice_id}_summary_by_method.csv")
            out_slice.to_csv(out_path, index=False)
            logger.info(f"[slice={slice_id}] wrote {out_path} rows={len(out_slice)}  elapsed={time.time()-t0:.1f}s")
            totals.append(out_slice.assign(slice_id=slice_id))

    if totals:
        overall = pd.concat(totals, ignore_index=True)
        def wavg(g, col):
            w = g["count"].astype(float); x = g[col].astype(float)
            s = float(w.sum());  return float((w*x).sum()/s) if s>0 else np.nan
        rows = []
        for (m,c), g in overall.groupby(["method","category"], as_index=False):
            rows.append({
                "method": m, "category": c,
                "count_total": int(g["count"].sum()),
                "mean_w": wavg(g,"mean"),
                "median_w": wavg(g,"median"),
                "std_w": wavg(g,"std"),
                "n_slices": int(g.shape[0]),
            })
        out_overall = os.path.join(args.out_dir, "overall_by_method.csv")
        pd.DataFrame(rows).to_csv(out_overall, index=False)
        print(f"[overall] wrote {out_overall}")
    else:
        logger.warning("[overall] no slice summaries.")

if __name__ == "__main__":
    main()

