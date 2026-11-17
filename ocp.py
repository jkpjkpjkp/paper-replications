"""Auditable online conformal predictors for Liu et al. (ICML 2026).

`UniversalPortfolio` follows Algorithm 1 literally. `run` also exposes the
authors' experimental implementation (zero initialization, no radius clipping)
under the separate name UP. All predictions precede the current observation.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit


@dataclass
class UniversalPortfolio:
    alpha: float = 0.05
    t: int = 1
    misses: int = 0
    log_wealth: float = 0.0

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

    def predict(self):
        lam = (self.misses + 0.5) / self.t
        beta = (lam - self.alpha) / (self.alpha * (1 - self.alpha))
        return 0.0 if beta <= 0 else math.exp(self.log_wealth + math.log(beta))

    def update(self, score):
        if not np.isfinite(score) or score < 0:
            raise ValueError("score must be finite and nonnegative")
        lam = (self.misses + 0.5) / self.t
        if score > self.predict():
            self.misses += 1
            self.log_wealth += math.log(lam / self.alpha)
        else:
            self.log_wealth += math.log((1 - lam) / (1 - self.alpha))
        self.t += 1


@njit(cache=True)
def _basic(scores, alpha, method, lr):
    n = len(scores)
    radii = np.empty(n)
    wealth, beta, radius, grad_sq = 1.0, 0.0, 0.0, 0.0
    covered_count = 0
    log_wealth = 0.0
    for i in range(n):
        t = i + 1
        lam = (i - covered_count + 0.5) / t
        if method == 3:  # Literal Algorithm 1.
            fraction = (lam - alpha) / (alpha * (1 - alpha))
            radius = 0.0 if fraction <= 0 else np.exp(log_wealth + np.log(fraction))
        radii[i] = radius
        covered = scores[i] <= radius
        g = alpha if covered else -(1.0 - alpha)
        covered_count += int(covered)
        if method == 0:  # Experimental UP (no implicit alpha correction).
            wealth = wealth - g * radius
            x = (t - covered_count + 0.5) / (t + 1)
            beta = -1 / (1 - alpha) + x * (1 / alpha + 1 / (1 - alpha))
            radius = beta * wealth
        elif method == 1:  # KT.
            wealth = wealth - g * radius
            beta = (t / (t + 1)) * beta - (1 / (t + 1)) * g
            radius = beta * wealth
        elif method == 2:  # Official SF-OGD (no epsilon).
            grad_sq += g * g
            radius -= lr * (g / np.sqrt(grad_sq))
        else:
            log_wealth += np.log((1 - lam) / (1 - alpha) if covered else lam / alpha)
    return radii


@njit(cache=True)
def _quantile_sorted(history, n, p):
    pos = (n - 1) * p
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    a, b = history[lo], history[hi]
    if frac >= 0.5:
        return b - (b - a) * (1 - frac)
    return a + (b - a) * frac


@njit(cache=True)
def _cdf_sorted(history, n, score):
    # Match core.functional.compute_empirical_cdf, including ties and its
    # scipy.bisect tolerance. Do not substitute an empirical rank CDF.
    if n == 0 or score >= history[n - 1]:
        return 1.0
    if score <= history[0]:
        return 0.0
    left, delta = 0.0, 1.0
    mid = 0.5
    for _ in range(100):
        delta *= 0.5
        mid = left + delta
        fmid = _quantile_sorted(history, n, mid) - score
        if fmid < 0:
            left = mid
        if fmid == 0 or delta < 2e-12 + 8.881784197001252e-16 * abs(mid):
            return mid
    return mid


@njit(cache=True)
def _dtaci(scores, alpha):
    n = len(scores)
    radii = np.empty(n)
    history = np.empty(n)
    levels = np.full(8, alpha)
    weights = np.full(8, 1 / 8)
    lrs = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128])
    radius = 0.0
    for i in range(n):
        radii[i] = radius
        score = scores[i]
        beta = 1 - _cdf_sorted(history, i, score)
        for j in range(8):
            u = beta - levels[j]
            loss = max(-(1 - alpha) * u, alpha * u)
            weights[j] *= np.exp(-np.e * loss)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        weights = (1 - 0.001) * weights + 0.001 / 8
        # Preserve the official ordering: aggregate old expert levels, then
        # update experts, then take the new-history quantile at the old level.
        current_alpha = np.dot(weights, levels)
        for j in range(8):
            levels[j] += lrs[j] * (alpha if levels[j] <= beta else -(1 - alpha))
        k = i
        while k > 0 and history[k - 1] > score:
            history[k] = history[k - 1]
            k -= 1
        history[k] = score
        if current_alpha < 0:
            radius = np.inf
        elif current_alpha > 1:
            radius = 0.0
        else:
            radius = _quantile_sorted(history, i + 1, 1 - current_alpha)
    return radii


@njit(cache=True)
def _controller(scores, alpha, lr, window, ki, csat):
    """Angelopoulos et al. reference P/PI, ahead=1, scorecast=False.

    Integrator excludes the just-observed error, exactly as the reference.
    The proportional step scales by the preceding window's range, not max.
    """
    radii = np.empty(len(scores))
    radius, q, error_sum = 0.0, 0.0, 0.0
    for t in range(len(scores)):
        radii[t] = radius
        step = lr
        if t > 0:
            previous = scores[max(0, t - window):t]
            step *= np.max(previous) - np.min(previous)
        error = 1.0 if scores[t] > radius else 0.0
        integrator = 0.0
        if ki != 0:
            angle = error_sum * np.log(t + 1) / (csat * (t + 1))
            if angle >= np.pi / 2:
                integrator = np.inf
            elif angle <= -np.pi / 2:
                integrator = -np.inf
            else:
                integrator = ki * np.tan(angle)
        q += step * (error - alpha)
        radius = q + integrator
        error_sum += error - alpha
    return radii


def controller(scores, alpha=0.05, lr=0.1, window=100, ki=0.0, csat=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("scores must be a finite, nonnegative vector")
    if not 0 < alpha < 1 or window < 1 or csat <= 0 or ki < 0 or lr < 0:
        raise ValueError("invalid controller settings")
    return _controller(scores, float(alpha), float(lr), int(window), float(ki), float(csat))


def run(scores, method="UP", alpha=0.05, lr=1.0):
    """Return radii for an already sliced, post-burn-in score sequence."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("scores must be a finite, nonnegative vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if method == "DtACI":
        return _dtaci(scores, alpha)
    codes = {"UP": 0, "KT": 1, "SFOGD": 2, "UP-paper": 3}
    if method not in codes:
        raise ValueError(f"Unknown method: {method}")
    return _basic(scores, alpha, codes[method], float(lr))


def metrics(scores, radii):
    scores, radii = np.asarray(scores), np.asarray(radii)
    if len(scores) != len(radii) or not len(scores) or np.isnan(radii).any():
        raise ValueError("Invalid score/radius arrays")
    covered = scores <= radii
    longest = streak = 0
    for hit in covered:
        streak = 0 if hit else streak + 1
        longest = max(longest, streak)
    # A negative radius represents the empty set, whose length is zero.
    widths = 2 * np.maximum(radii, 0)
    # Explicit order-statistic interpolation avoids inf - inf -> NaN.
    ordered = np.sort(widths)
    qs = []
    for p in (0.5, 0.75, 0.9, 0.95):
        pos = (len(ordered) - 1) * p
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        if pos == lo:
            q = ordered[lo]
        elif np.isinf(ordered[hi]):
            q = np.inf
        else:
            q = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
        qs.append(float(q))
    return dict(coverage=float(np.mean(covered)), longest_error_streak=longest,
                mean_width=float(np.mean(widths)), median_width=qs[0],
                q75_width=qs[1], q90_width=qs[2], q95_width=qs[3],
                infinite_fraction=float(np.mean(np.isinf(widths))),
                negative_radius_fraction=float(np.mean(radii < 0)), n=len(scores))
