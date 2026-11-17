#!/usr/bin/env python3
"""Exhaustive factor × entry × exit event regressions, development sample only.

This is an event-weighted analytical screening step. Its return and Sharpe
proxies do NOT describe an implementable calendar-time, dollar-neutral book.
The parent study must exact-simulate candidate and baseline holdings, overlap,
cash days, fees and daily neutralization before making performance claims.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 to avoid oversubscription.
Input contract is opportunities_nNN.npz: rank[K,F], returns[K,H], duration[K,H],
entry_idx[K] (first earned-return session), forecast_idx[K], exit_idx[K,H],
event_id[K], permno[K]. Ranks are signed, finite month-end percentiles or NaN.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
DATA = HERE / "horizon_data"
OUT = HERE / "horizon_results"
METADATA = HERE / "data" / "metadata.json"
CALENDAR = HERE.parent / "results" / "trading_calendar.csv"


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _divide(num, den, minimum=0):
    """Return NaN for undefined ratios instead of silently imputing zero."""
    num, den = np.broadcast_arrays(num, den)
    return np.divide(
        num, den, out=np.full(num.shape, np.nan, dtype=np.float64), where=den > minimum
    )


def regress_matrix(ranks, outcomes, labels_valid, clusters):
    """OLS with intercept for every factor/horizon; forecast-date clustered SE.

    Returns arrays shaped (factor, horizon). Missing ranks and labels are
    omitted pairwise. The finite-sample cluster correction is G/(G-1) times
    (n-1)/(n-2); p-values use Student t with G-1 degrees of freedom. This is
    one-way date clustering, not a causal estimate or two-way firm clustering.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    labels_valid = np.asarray(labels_valid, dtype=bool) & np.isfinite(outcomes)
    clusters = np.asarray(clusters)
    assert ranks.ndim == outcomes.ndim == 2
    assert len(ranks) == len(outcomes) == len(clusters)
    assert labels_valid.shape == outcomes.shape
    order = np.argsort(clusters, kind="stable")
    ranks, outcomes, labels_valid = ranks[order], outcomes[order], labels_valid[order]
    clusters = clusters[order]
    starts = np.r_[0, np.flatnonzero(clusters[1:] != clusters[:-1]) + 1]
    if len(ranks) == 0:
        shape = (ranks.shape[1], outcomes.shape[1])
        return {k: np.full(shape, np.nan) for k in
                ["n", "alpha", "beta", "cluster_se", "cluster_t", "cluster_p", "clusters"]}

    available = np.isfinite(ranks)
    a = available.astype(np.float64)
    x = np.where(available, ranks - .5, 0.)
    v = labels_valid.astype(np.float64)
    y = np.where(labels_valid, outcomes, 0.)
    count = a.T @ v
    sum_x = x.T @ v
    sum_y = a.T @ y
    sum_xx = (x * x).T @ v
    sum_xy = x.T @ y
    xbar = _divide(sum_x, count)
    ybar = _divide(sum_y, count)
    centered_xx = sum_xx - _divide(sum_x * sum_x, count)
    centered_xy = sum_xy - _divide(sum_x * sum_y, count)
    # Count-scaled tolerance excludes effectively constant ranks and roundoff.
    identified = (count > 2) & (centered_xx > np.maximum(1e-14, count * 1e-12))
    beta = np.divide(centered_xy, centered_xx,
                     out=np.full(count.shape, np.nan), where=identified)
    alpha = ybar - beta * xbar
    se = np.full(count.shape, np.nan)
    groups = np.zeros(count.shape, dtype=np.int32)

    for h in range(outcomes.shape[1]):
        valid_h = labels_valid[:, h]
        # The centered regressor score gives the slope row of (X'X)^-1 U_g.
        residual_score = -x * np.nan_to_num(beta[:, h])[None, :]
        residual_score += y[:, h, None] - np.nan_to_num(alpha[:, h])[None, :]
        residual_score *= x - np.nan_to_num(xbar[:, h])[None, :]
        residual_score *= available
        residual_score[~valid_h] = 0.
        group_score = np.add.reduceat(residual_score, starts, axis=0)
        group_count = np.add.reduceat(available & valid_h[:, None], starts, axis=0)
        g = (group_count > 0).sum(axis=0)
        groups[:, h] = g
        finite_sample = _divide(g, g - 1) * _divide(count[:, h] - 1, count[:, h] - 2)
        variance = _divide((group_score * group_score).sum(axis=0),
                           centered_xx[:, h] * centered_xx[:, h]) * finite_sample
        se[:, h] = np.where(identified[:, h] & (g > 1), np.sqrt(variance), np.nan)
    t_value = _divide(beta, se)
    # Exact constant outcomes can yield zero SE and zero beta. They are not a
    # significant finding; nonzero/zero cases remain undefined rather than inf.
    p_value = 2 * student_t.sf(np.abs(t_value), np.maximum(groups - 1, 1))
    return dict(n=count, alpha=alpha, beta=beta, cluster_se=se,
                cluster_t=t_value, cluster_p=p_value, clusters=groups)


def tail_screen(ranks, outcomes, durations, labels_valid, cost=.0005, borrow=.01):
    """Event-weighted thirds and net single-spread approximations, not NAV P&L."""
    ranks = np.asarray(ranks, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    valid = (np.asarray(labels_valid, bool) & np.isfinite(outcomes)
             & np.isfinite(durations) & (durations > 0))
    v = valid.astype(np.float64)
    y = np.where(valid, outcomes, 0.)
    d = np.where(valid, durations, 0.)
    long = (np.isfinite(ranks) & (ranks > 2/3)).astype(np.float64)
    short = (np.isfinite(ranks) & (ranks < 1/3)).astype(np.float64)
    nl, ns = long.T @ v, short.T @ v
    lm, sm = _divide(long.T @ y, nl), _divide(short.T @ y, ns)
    ld, sd = _divide(long.T @ d, nl), _divide(short.T @ d, ns)
    lvar = _divide(long.T @ (y*y) - nl * lm*lm, nl - 1)
    svar = _divide(short.T @ (y*y) - ns * sm*sm, ns - 1)
    lvar, svar = np.maximum(lvar, 0), np.maximum(svar, 0)
    duration = _divide(nl*ld + ns*sd, nl + ns)
    gross = lm-sm
    # One dollar on each side: entry dollars 2, exit marks 2+Lmean+Smean.
    # This ignores reinvestment, NAV scaling, neutralization and overlap.
    fees = cost * (4 + lm + sm)
    borrow_cost = borrow * sd / 252
    net = gross-fees-borrow_cost
    annual = net * _divide(252., duration)
    spread_std = np.sqrt(lvar + svar)
    sharpe = _divide(net, spread_std) * np.sqrt(_divide(252., duration))
    return dict(long_trades=nl, short_trades=ns, long_mean=lm, short_mean=sm,
                long_duration=ld, short_duration=sd, mean_duration=duration,
                gross_per_trade=gross, fees_per_trade=fees,
                borrow_per_trade=borrow_cost, net_per_trade=net,
                annual_mean_proxy=annual, sharpe_proxy=sharpe)


def screen(data_dir=DATA, out_dir=OUT, entries=range(1,31), min_tail=500):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(METADATA.read_text())
    factors = meta["factors"]
    old = set(meta["published_by2004"])
    cal = pd.read_csv(CALENDAR, parse_dates=["date"])
    train = cal[cal.date.between("1985-01-01", "2004-12-31")]
    lo, hi = int(train.idx.min()), int(train.idx.max())
    parts, audit = [], []
    for n_before in entries:
        source = data_dir/f"opportunities_n{n_before:02d}.npz"
        log(f"OLS and event screening: N={n_before}, {source.name}")
        with np.load(source) as z:
            forecast = z["forecast_idx"]
            keep = ((forecast >= lo+30) & (forecast <= hi-40)
                    & (z["entry_idx"] >= lo) & (z["entry_idx"] <= hi))
            ranks = z["rank"][keep]
            y = z["returns"][keep]
            duration = z["duration"][keep]
            exits = z["exit_idx"][keep]
            clusters = forecast[keep]
            entry = z["entry_idx"][keep]
            event_id = z["event_id"][keep]
            permno = z["permno"][keep]
        assert ranks.shape[1] == len(factors)
        assert y.shape == duration.shape == exits.shape
        assert len(set(zip(event_id.tolist(), permno.tolist()))) == len(event_id), "Duplicate stock/state opportunity"
        valid = ((exits >= entry[:, None]) & (exits <= hi)
                 & (duration == exits-entry[:,None]+1) & (duration > 0)
                 & np.isfinite(y))
        assert not np.any(y[valid] < -1-1e-10), "Stock total return below -100%"
        regress = regress_matrix(ranks, y, valid, clusters)
        tails = tail_screen(ranks, y, duration, valid)
        nf, nh = ranks.shape[1], y.shape[1]
        base = dict(factor=np.repeat(factors, nh), buy_before=n_before,
                    sell_after=np.tile(np.arange(1,nh+1), nf))
        for name, values in {**regress, **tails}.items():
            base[name] = values.ravel()
        part = pd.DataFrame(base)
        part["published_by2004"] = part.factor.isin(old)
        part["eligible"] = ((part.long_trades >= min_tail)
                            & (part.short_trades >= min_tail)
                            & np.isfinite(part.sharpe_proxy))
        parts.append(part)
        audit.append(dict(buy_before=n_before, common_purged_opportunities=int(len(y)),
                          forecast_clusters=int(len(np.unique(clusters))),
                          valid_labels=int(valid.sum()),
                          invalid_labels=int((~valid).sum()),
                          minimum_exit_idx=int(exits.min()) if len(exits) else None,
                          maximum_exit_idx=int(exits.max()) if len(exits) else None))
    result = pd.concat(parts, ignore_index=True)
    baseline = result[(result.buy_before==1)&(result.sell_after==1)].set_index("factor")
    assert len(baseline)==len(factors), "Need all factor -1/+1 baselines"
    for column in ["net_per_trade", "annual_mean_proxy", "sharpe_proxy", "mean_duration"]:
        result["baseline_"+column] = result.factor.map(baseline[column])
    result["delta_annual_proxy"] = result.annual_mean_proxy-result.baseline_annual_mean_proxy
    result["delta_net_per_trade"] = result.net_per_trade-result.baseline_net_per_trade
    result.to_csv(out_dir/"regression_screen.csv", index=False)
    receipt = dict(
        training_start="1985-01-01", training_end="2004-12-31",
        earliest_forecast_idx=lo+30, latest_forecast_idx=hi-40,
        factor_count=len(factors), specification_count=len(result),
        grid="209 factors × N=1..30 × M=1..30 when all entries supplied",
        response="Per-stock cumulative holding total return, before portfolio costs",
        regressor="Signed previous-month factor percentile minus 0.5, with intercept",
        covariance="One-way forecast-date clustered CR1, Student t(G-1)",
        screen="Event-weighted approximate spreads; no calendar portfolio claim",
        cost_per_dollar=0.0005, short_borrow_per_annum=0.01,
        minimum_trades_per_tail=min_tail, later_returns_used=False,
        source_audit=audit,
    )
    (out_dir/"regression_screen_audit.json").write_text(json.dumps(receipt, indent=2))
    log(f"Wrote {len(result):,} exhaustive development specifications")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--min-tail", type=int, default=500)
    args = parser.parse_args()
    screen(args.data_dir, args.out_dir, min_tail=args.min_tail)
