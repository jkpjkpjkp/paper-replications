from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def ensure_parents(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _detect_col_name(candidates, available):
    """Return the first matching column name from candidates; raise if none."""
    for c in candidates:
        if c in available:
            return c
    raise KeyError(f"None of {candidates} found in columns {sorted(available)}")


def load_market_stats(slice_dir: Path) -> Tuple[Dict[str, Tuple[float, float]], float]:
    """
    features/{slice_id}/market_stats.parquet :
        columns example:
            asset_id, asof_date, r, sigma2, hl_days, window_min_obs, ...
        or
            item_id, r, sigma2, ...
    Returns:
        (dict item_id -> (r, sigma2), sigma2_default)
    """
    p = slice_dir / "market_stats.parquet"
    if not p.exists():
        raise FileNotFoundError(f"market_stats not found: {p}")

    # Detect schema first (cheap) → pick correct id column without reading all columns
    pf = pq.ParquetFile(p)
    cols = {pf.schema_arrow.names[i] for i in range(len(pf.schema_arrow.names))}
    id_col = _detect_col_name(["item_id", "asset_id"], cols)
    # r/sigma2 must exist
    if "r" not in cols or "sigma2" not in cols:
        # Fallback: read table to double-check column aliases
        tbl_all = pq.read_table(p)
        names = set(tbl_all.column_names)
        if "r" not in names or "sigma2" not in names:
            raise KeyError(f"'r' or 'sigma2' missing in {p}, got {sorted(names)}")

    tbl = pq.read_table(p, columns=[id_col, "r", "sigma2"])
    pd = tbl.to_pandas()
    out: Dict[str, Tuple[float, float]] = {}
    for row in pd.itertuples(index=False):
        out[str(getattr(row, id_col))] = (float(getattr(row, "r")), float(getattr(row, "sigma2")))
    sigma2_default = max(1e-8, float(np.nanmedian(pd["sigma2"])) if len(pd) else 1e-6)
    return out, sigma2_default


def load_popularity(slice_dir: Path) -> Dict[str, float]:
    p = slice_dir / "popularity.parquet"
    if not p.exists():
        return {}
    pf = pq.ParquetFile(p)
    cols = set(pf.schema_arrow.names)
    id_col = _detect_col_name(["item_id", "asset_id"], cols)
    s_col = "s" if "s" in cols else (_detect_col_name(["popularity"], cols))
    tbl = pq.read_table(p, columns=[id_col, s_col])
    pd = tbl.to_pandas()
    return {str(getattr(row, id_col)): float(getattr(row, s_col)) for row in pd.itertuples(index=False)}


def load_covariance_coo(slice_dir: Path) -> Dict[Tuple[str, str], float]:
    """
    features/{slice_id}/covariance.parquet : COO (i<=j)
    Accepts columns like:
        asset_id_i, asset_id_j, cov
    or
        i, j, value
    Returns dict with key=(min(item_i,item_j), max(...)) -> cov
    """
    p = slice_dir / "covariance.parquet"
    if not p.exists():
        return {}
    tbl = pq.read_table(p)
    names = tbl.column_names
    # Normalize names
    i_col = "asset_id_i" if "asset_id_i" in names else ("i" if "i" in names else names[0])
    j_col = "asset_id_j" if "asset_id_j" in names else ("j" if "j" in names else names[1 if len(names) > 1 else 0])
    v_col = "cov" if "cov" in names else ("value" if "value" in names else names[-1])
    pd = tbl.select([i_col, j_col, v_col]).to_pandas()
    d: Dict[Tuple[str, str], float] = {}
    for row in pd.itertuples(index=False):
        a = str(getattr(row, i_col))
        b = str(getattr(row, j_col))
        key = (a, b) if a <= b else (b, a)
        d[key] = float(getattr(row, v_col))
    return d


def iter_users_from_all_parquet(
    parquet_path: Path,
    columns: List[str],
    batch_rows: int = 200_000,
    user_whitelist: Optional[Set[str]] = None,
):
    """
    Robust per-user streaming iterator.
    Yields:
      (user_id, {"item_id": np.ndarray[str], "p_i": np.ndarray[float64],
                 "eligible": np.ndarray[bool], "in_holdings_flag": np.ndarray[bool]})
    """
    dataset = ds.dataset(str(parquet_path))


    need = ["user_id", "item_id", "p_i", "eligible", "in_holdings_flag"]
    for c in need:
        if c not in columns:
            columns.append(c)
    proj_cols = list(dict.fromkeys(need))
    scanner = ds.Scanner.from_dataset(dataset, columns=proj_cols, batch_size=batch_rows)

    cur_uid: Optional[str] = None
    buf = {c: [] for c in proj_cols if c != "user_id"}

    def _emit(uid: str):
        if uid is None:
            return
        if (user_whitelist is not None) and (uid not in user_whitelist):
            # drop silently
            return
        import numpy as _np
        def _boolify(seq):
            return [_ if isinstance(_, (bool, np.bool_)) else bool(_) for _ in seq]
        out = {
            "item_id": _np.asarray(buf["item_id"], dtype=object).astype(str),
            "p_i": _np.asarray(buf["p_i"], dtype=_np.float64),
            "eligible": _np.asarray(_boolify(buf["eligible"]), dtype=bool),
            "in_holdings_flag": _np.asarray(_boolify(buf["in_holdings_flag"]), dtype=bool),
        }
        if out["item_id"].size:
            yield (uid, out)

    for batch in scanner.to_batches():
        uids = batch.column("user_id").to_pylist()
        cols = {c: batch.column(c).to_pylist() for c in buf.keys()}

        for i, uid in enumerate(uids):
            uid = str(uid)
            if (cur_uid is not None) and (uid != cur_uid):
                for yielded in _emit(cur_uid) or ():
                    yield yielded
                for c in buf.keys():
                    buf[c].clear()

            cur_uid = uid
            for c in buf.keys():
                buf[c].append(cols[c][i])

    if cur_uid is not None:
        for yielded in _emit(cur_uid) or ():
            yield yielded


# ----- Parquet Writer for runs -----

RUNS_SCHEMA = pa.schema(
    [
        pa.field("slice_id", pa.string()),
        pa.field("user_id", pa.string()),
        pa.field("item_id", pa.string()),
        pa.field("rank", pa.int32()),
        pa.field("w", pa.float64()),
        pa.field("p_i", pa.float64()),
        pa.field("r", pa.float64()),
        pa.field("sigma2", pa.float64()),
        pa.field("s", pa.float64()),
        pa.field("cov_used", pa.bool_()),
        pa.field("risk_model", pa.string()),
        pa.field("gamma_or_eta", pa.float64()),
        pa.field("B", pa.float64()),
        pa.field("k", pa.int32()),
        pa.field("term_utility", pa.float64()),
        pa.field("term_diversity", pa.float64()),
        pa.field("term_pop_penalty", pa.float64()),
        pa.field("term_cov", pa.float64()),
        pa.field("obj_total", pa.float64()),
        pa.field("method", pa.string()),
        pa.field("algo_version", pa.string()),
        pa.field("seed", pa.int32()),
    ]
)


@dataclass
class RunsWriter:
    path: Path
    writer: pq.ParquetWriter

    def write_rows(self, rows: List) -> None:
        cols = {name: [] for name in RUNS_SCHEMA.names}
        for r in rows:
            cols["slice_id"].append(r.slice_id)
            cols["user_id"].append(r.user_id)
            cols["item_id"].append(r.item_id)
            cols["rank"].append(r.rank)
            cols["w"].append(r.w)
            cols["p_i"].append(r.p_i)
            cols["r"].append(r.r)
            cols["sigma2"].append(r.sigma2)
            cols["s"].append(r.s)
            cols["cov_used"].append(r.cov_used)
            cols["risk_model"].append(r.risk_model)
            cols["gamma_or_eta"].append(r.gamma_or_eta)
            cols["B"].append(r.B)
            cols["k"].append(r.k)
            cols["term_utility"].append(r.term_utility)
            cols["term_diversity"].append(r.term_diversity)
            cols["term_pop_penalty"].append(r.term_pop_penalty)
            cols["term_cov"].append(r.term_cov)
            cols["obj_total"].append(r.obj_total)
            cols["method"].append(r.method)
            cols["algo_version"].append(r.algo_version)
            cols["seed"].append(r.seed)
        tbl = pa.table(cols, schema=RUNS_SCHEMA)
        self.writer.write_table(tbl)

    def close(self) -> None:
        self.writer.close()


def make_runs_writer(path: Path) -> RunsWriter:
    ensure_parents(path)
    if path.exists():
        path.unlink()
    writer = pq.ParquetWriter(str(path), RUNS_SCHEMA, compression="snappy", use_dictionary=True)
    return RunsWriter(path=path, writer=writer)


def count_users_in_all_parquet(
    parquet_path: Path,
    batch_rows: int = 1_000_000,
) -> int:
    dataset = ds.dataset(str(parquet_path))
    scanner = ds.Scanner.from_dataset(dataset, columns=["user_id"], batch_size=batch_rows)
    last_uid: Optional[str] = None
    n_users = 0
    for batch in scanner.to_batches():
        uids = batch.column("user_id").to_pylist()
        if not uids:
            continue
        if last_uid is None:
            last_uid = uids[0]
            n_users += 1
        for uid in uids:
            if uid != last_uid:
                n_users += 1
                last_uid = uid
    return n_users

