# FILE: src/rura_core_fast.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp  # optional
    from cupyx.scipy import sparse as cupy_sparse  # noqa: F401
    _CUPY_OK = True
except Exception:
    _CUPY_OK = False


@dataclass
class ruraConfig:
    method: str = "rura_la"        # rura_la | rura_uf | rura_la_no_sigma | rura_la_crra ...
    lambda_div: float = 0.05
    beta_pop: float = 0.10
    L: int = 128                   # budget splits
    w_max: float = 0.4
    eps_improve: float = 1e-9
    diag_eps: float = 1e-10
    device: str = "auto"           # auto|cpu|gpu


@dataclass
class ruraResultRow:
    slice_id: str
    user_id: str
    item_id: str
    rank: int
    w: float
    p_i: float
    r: float
    sigma2: float
    s: float
    cov_used: bool
    risk_model: str
    gamma_or_eta: float
    B: float
    k: int
    term_utility: float
    term_diversity: float
    term_pop_penalty: float
    term_cov: float
    obj_total: float
    method: str
    algo_version: str
    seed: int


@dataclass
class MethodFlags:
    la: bool
    use_cov: bool
    use_crra: bool


def parse_method_flags(method: str) -> MethodFlags:
    m = method.lower()
    la = "uf" not in m
    use_cov = "no_sigma" not in m and "nosigma" not in m and "no-sigma" not in m
    use_crra = "crra" in m
    return MethodFlags(la=la, use_cov=use_cov, use_crra=use_crra)


def _device_pick(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "gpu":
        return "gpu" if _CUPY_OK and _gpu_available() else "cpu"
    # auto
    return "gpu" if _CUPY_OK and _gpu_available() else "cpu"


def _gpu_available() -> bool:
    try:
        import cupy as cp  # noqa
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _stable_sort_w_itemid(item_ids: np.ndarray, w: np.ndarray) -> np.ndarray:
    # Sort by w desc, then item_id asc
    # argsort is ascending; use negative weight for descending
    idx_item = np.argsort(item_ids, kind="mergesort")
    idx_w = np.argsort(-w, kind="mergesort")
    # Stable two-key: first by item_id asc, then by w desc (mergesort stable)
    # But we want w desc primary; so first sort by item_id, then stable sort by -w
    order = idx_item
    order = order[np.argsort(-w[order], kind="mergesort")]
    return order


def _entropy_term(w: np.ndarray, lam: float, eps: float) -> float:
    # -lambda * sum_i w_i log max(w_i, eps)
    if lam == 0.0:
        return 0.0
    w_safe = np.maximum(w, eps)
    return -lam * float(np.sum(w * np.log(w_safe)))


def _pop_penalty_term(s: np.ndarray, w: np.ndarray, beta: float) -> float:
    # -beta * sum_i s_i w_i
    if beta == 0.0:
        return 0.0
    return -beta * float(np.dot(s, w))


def _utility_term_LAUF(p: np.ndarray, r: np.ndarray, sigma2: np.ndarray, w: np.ndarray, rho: float, B: float, la: bool) -> float:
    # Sum_i [ p_i * (B*r_i*w_i - 0.5*rho*B^2*sigma2_i*w_i^2) ] ; if UF, p_i=1
    scale = p if la else np.ones_like(p)
    term = float(np.sum(scale * (B * r * w - 0.5 * rho * (B**2) * sigma2 * (w**2))))
    return term


def _cov_term(w: np.ndarray, S_idx: List[int], cov_index: Optional[Dict[Tuple[str, str], float]], item_ids: np.ndarray, rho: float, B: float) -> float:
    # -0.5 * rho * B^2 * sum_{i != j, in S} w_i w_j Sigma_ij  (symmetric; count once)
    if cov_index is None or len(S_idx) <= 1:
        return 0.0
    coeff = -0.5 * rho * (B**2)
    total = 0.0
    for a_pos, i in enumerate(S_idx):
        for j in S_idx[a_pos + 1 :]:
            key = (item_ids[i], item_ids[j]) if item_ids[i] <= item_ids[j] else (item_ids[j], item_ids[i])
            cij = cov_index.get(key)
            if cij is None:
                continue
            total += (w[i] * w[j]) * cij
    return coeff * total


def _cov_sum_with_S(i: int, S_idx: List[int], w: np.ndarray, cov_index: Optional[Dict[Tuple[str, str], float]], item_ids: np.ndarray) -> float:
    if cov_index is None or not S_idx:
        return 0.0
    s = 0.0
    iid = item_ids[i]
    for j in S_idx:
        if j == i:
            continue
        key = (iid, item_ids[j]) if iid <= item_ids[j] else (item_ids[j], iid)
        cij = cov_index.get(key)
        if cij is None:
            continue
        s += w[j] * cij
    return s


def _marginal_gain(
    i: int,
    w: np.ndarray,
    delta: float,
    p: np.ndarray,
    r: np.ndarray,
    sigma2: np.ndarray,
    s_pop: np.ndarray,
    rho: float,
    B: float,
    lam: float,
    beta: float,
    la: bool,
    use_cov: bool,
    S_idx: List[int],
    cov_index: Optional[Dict[Tuple[str, str], float]],
    item_ids: np.ndarray,
    eps: float,
    allow_new_item: bool,
) -> float:
    # If new item is not allowed, and w_i == 0 -> block
    if not allow_new_item and w[i] == 0.0:
        return -np.inf

    scale = p[i] if la else 1.0
    # Utility increment (exact difference for quadratic in w_i)
    # ΔU = scale * [B*r_i*δ - 0.5*rho*B^2*sigma2_i*((w_i+δ)^2 - w_i^2)]
    d_util = scale * (B * r[i] * delta - 0.5 * rho * (B**2) * sigma2[i] * ((w[i] + delta) ** 2 - w[i] ** 2))
    # Entropy (diversity) exact difference
    prev = max(w[i], eps)
    now = max(w[i] + delta, eps)
    d_div = -lam * ((w[i] + delta) * math.log(now) - w[i] * math.log(prev))
    # Popularity penalty (linear)
    d_pop = -beta * s_pop[i] * delta
    # Covariance linearized incremental term
    d_cov = 0.0
    if use_cov and (S_idx or w[i] > 0.0):
        cross = _cov_sum_with_S(i, S_idx, w, cov_index, item_ids)
        # Note: if i not in S yet, S_idx cross already covers all j ∈ S
        d_cov = - (rho * (B**2)) * delta * cross

    return d_util + d_div + d_pop + d_cov


def run_rura_for_user(
    slice_id: str,
    user_id: str,
    item_ids: np.ndarray,  # shape [n], str
    p_i: np.ndarray,       # [n]
    r: np.ndarray,         # [n]
    sigma2: np.ndarray,    # [n]
    s: np.ndarray,         # [n]
    rho: float,
    B: float,
    k: int,
    cfg: ruraConfig,
    cov_index: Optional[Dict[Tuple[str, str], float]],
    algo_version: str,
    seed: int,
) -> List[ruraResultRow]:
    n = item_ids.shape[0]
    if n == 0 or k <= 0:
        return []

    flags = parse_method_flags(cfg.method)
    use_cov = flags.use_cov and (cov_index is not None)

    # Device selection
    dev = _device_pick(cfg.device)

    # Determinism
    rng = np.random.default_rng(seed)

    # Budget token
    delta = 1.0 / float(cfg.L)

    # Initialize
    w = np.zeros(n, dtype=np.float64)
    active_mask = np.zeros(n, dtype=bool)
    S_idx: List[int] = []
    tokens_used = 0

    # Pre-alloc marginal gains
    # We will use lazy recomputation: store last seen w_i to detect staleness
    last_w_seen = np.zeros(n, dtype=np.float64)
    gains = np.full(n, -np.inf, dtype=np.float64)

    # Initial gains (no covariance cross yet)
    allow_new_item = True
    for i in range(n):
        gains[i] = _marginal_gain(
            i, w, delta, p_i, r, sigma2, s, rho, B,
            cfg.lambda_div, cfg.beta_pop, flags.la, use_cov, S_idx, cov_index, item_ids, cfg.diag_eps,
            allow_new_item=True,
        )
        last_w_seen[i] = w[i]

    # Greedy allocation
    while tokens_used < cfg.L:
        # If S already at capacity, block new items
        allow_new_item = len(S_idx) < k

        # Pick best index
        best_i = int(np.argmax(gains))
        best_gain = gains[best_i]

        if best_gain <= cfg.eps_improve:
            break  # no improving move

        # Recompute lazy if stale
        if not np.isclose(last_w_seen[best_i], w[best_i]):
            gains[best_i] = _marginal_gain(
                best_i, w, delta, p_i, r, sigma2, s, rho, B,
                cfg.lambda_div, cfg.beta_pop, flags.la, use_cov, S_idx, cov_index, item_ids, cfg.diag_eps,
                allow_new_item=allow_new_item,
            )
            last_w_seen[best_i] = w[best_i]
            # Re-select
            best_i = int(np.argmax(gains))
            best_gain = gains[best_i]
            if best_gain <= cfg.eps_improve:
                break

        # Check capacity & w_max
        if w[best_i] == 0.0 and not allow_new_item:
            # This is a new item but S is full → invalidate this gain and continue
            gains[best_i] = -np.inf
            continue

        # Determine actual step size (respect w_max and remaining budget)
        step = min(delta, cfg.w_max - w[best_i], 1.0 - np.sum(w))
        if step <= 0.0 + 1e-18:
            # No room for this item; invalidate and continue
            gains[best_i] = -np.inf
            continue

        # Commit
        w[best_i] += step
        tokens_used += int(round(step / delta))  # 1 in most cases

        if not active_mask[best_i]:
            active_mask[best_i] = True
            S_idx.append(best_i)

        # Update gains lazily: only mark stale for indices that depend on w[best_i]
        last_w_seen[best_i] = w[best_i]
        # Recompute this item's own gain for next step
        gains[best_i] = _marginal_gain(
            best_i, w, delta, p_i, r, sigma2, s, rho, B,
            cfg.lambda_div, cfg.beta_pop, flags.la, use_cov, S_idx, cov_index, item_ids, cfg.diag_eps,
            allow_new_item=True,  # already in S
        )
        # For all others, their gain may have changed via covariance with best_i; mark stale
        if use_cov:
            for j in range(n):
                if j == best_i:
                    continue
                last_w_seen[j] = -1.0  # force recompute on demand

        # If budget saturated
        if np.sum(w) >= 1.0 - 1e-15:
            break

    # Optional 1↔1 swap (lightweight)
    # Try replace the smallest-weight active item with the best outside item of same total weight
    def _objective_total(wv: np.ndarray) -> Tuple[float, float, float, float, float]:
        term_u = _utility_term_LAUF(p_i, r, sigma2, wv, rho, B, flags.la)
        term_d = _entropy_term(wv, cfg.lambda_div, cfg.diag_eps)
        term_p = _pop_penalty_term(s, wv, cfg.beta_pop)
        term_c = _cov_term(wv, [i for i, a in enumerate(wv) if a > 0.0], cov_index if use_cov else None, item_ids, rho, B)
        total = term_u + term_d + term_p + term_c
        return total, term_u, term_d, term_p, term_c

    # One round
    if len(S_idx) == k and k > 0:
        # identify smallest w
        j_star = min(S_idx, key=lambda j: w[j])
        wj = w[j_star]
        if wj > 0.0:
            # find best outsider
            outsiders = [i for i in range(n) if not active_mask[i]]
            best_total = None
            best_i = None
            base_total, _, _, _, _ = _objective_total(w)
            for i in outsiders:
                w_try = w.copy()
                # remove j_star
                w_try[j_star] = 0.0
                # add to i
                w_try[i] = min(cfg.w_max, wj)
                # keep sum <= 1.0
                # If spare due to cap, discard remaining for swap test
                tot, *_ = _objective_total(w_try)
                if (best_total is None) or (tot > best_total):
                    best_total = tot
                    best_i = i
            if best_total is not None and best_total - base_total > cfg.eps_improve:
                # Commit swap
                active_mask[j_star] = False
                S_idx.remove(j_star)
                w[j_star] = 0.0
                w[best_i] = min(cfg.w_max, wj)
                active_mask[best_i] = True
                S_idx.append(best_i)

    # Final stable ordering
    if np.sum(w) > 1.0 + 1e-9:
        # Normalize slightly to respect sum<=1 (should not happen with the guards)
        w *= (1.0 / np.sum(w))

    # Prepare outputs: take only nonzeros, order by w desc -> item_id asc, keep top-k
    nz = np.where(w > 0.0)[0]
    if nz.size == 0:
        return []
    order = _stable_sort_w_itemid(item_ids[nz], w[nz])
    sel = nz[order][:k]

    # Objective terms for the final portfolio (over selected only)
    w_sel = np.zeros_like(w)
    w_sel[sel] = w[sel]
    total, term_u, term_d, term_p, term_c = _objective_total(w_sel)

    # Build rows
    rows: List[ruraResultRow] = []
    # Rank assignment by sorted order
    rank_order = np.argsort(-w[sel], kind="mergesort")
    for rank_pos, idx in enumerate(sel[rank_order], start=1):
        rows.append(
            ruraResultRow(
                slice_id=slice_id,
                user_id=user_id,
                item_id=str(item_ids[idx]),
                rank=rank_pos,
                w=float(w[idx]),
                p_i=float(p_i[idx]),
                r=float(r[idx]),
                sigma2=float(sigma2[idx]),
                s=float(s[idx]),
                cov_used=bool(use_cov),
                risk_model="CRRA" if flags.use_crra else "CARA",
                gamma_or_eta=float(rho),
                B=float(B),
                k=int(k),
                term_utility=float(term_u),
                term_diversity=float(term_d),
                term_pop_penalty=float(term_p),
                term_cov=float(term_c),
                obj_total=float(total),
                method=str(cfg.method),
                algo_version=str(algo_version),
                seed=int(seed),
            )
        )
    return rows

