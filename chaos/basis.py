"""All stochastic integrals use increments, never cumulative Brownian values."""
from itertools import combinations_with_replacement
from math import factorial
import numpy as np


def cosine_basis(n: int, modes: int) -> np.ndarray:
    """Rows of the orthogonal DCT-II matrix (discrete L2 normalization)."""
    if not 1 <= modes <= n:
        raise ValueError("modes must be in [1, n]")
    x = (np.arange(n) + 0.5) / n
    b = np.cos(np.pi * np.arange(modes)[:, None] * x) * np.sqrt(2 / n)
    b[0] /= np.sqrt(2)
    return b


def discrete_haar(n: int) -> np.ndarray:
    """Orthonormal Haar tree, including unequal splits for non-powers of two."""
    rows = [np.ones(n) / np.sqrt(n)]
    queue = [(0, n)]
    while queue:
        left, right = queue.pop(0)
        if right - left <= 1:
            continue
        mid = (left + right) // 2
        a, b = mid - left, right - mid
        row = np.zeros(n)
        row[left:mid] = np.sqrt(b / (a * (a + b)))
        row[mid:right] = -np.sqrt(a / (b * (a + b)))
        rows.append(row)
        queue.extend([(left, mid), (mid, right)])
    return np.stack(rows)


def haar_values(n: int, count: int = 64) -> np.ndarray:
    """Continuous Haar functions on [0,1], sampled at interval midpoints."""
    if count < 1:
        raise ValueError("count must be positive")
    x = (np.arange(n) + 0.5) / n
    rows = [np.ones(n)]
    level = 0
    while len(rows) < count:
        for k in range(2**level):
            a, m, b = k / 2**level, (k + 0.5) / 2**level, (k + 1) / 2**level
            row = np.sqrt(2**level) * (((x >= a) & (x < m)).astype(float)
                                          - ((x >= m) & (x < b)).astype(float))
            rows.append(row)
            if len(rows) == count:
                break
        level += 1
    return np.stack(rows)


def coordinates(dw: np.ndarray, dt: float, basis: np.ndarray) -> np.ndarray:
    """dw: (..., time), output (..., modes), covariance I for standard noise."""
    return (dw / np.sqrt(dt)) @ basis.T


def multi_indices(dim: int, order: int) -> np.ndarray:
    """Complete total-degree set, including all mixed products and constant."""
    rows = [np.zeros(dim, dtype=np.int16)]
    for degree in range(1, order + 1):
        for combination in combinations_with_replacement(range(dim), degree):
            rows.append(np.bincount(combination, minlength=dim).astype(np.int16))
    return np.stack(rows)


def wick_features(z: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Normalized probabilists' Hermites, prod He_alpha(z)/sqrt(alpha!)."""
    z = np.asarray(z)
    degree = int(indices.max())
    h = [np.ones_like(z), z]
    for k in range(1, degree):
        h.append(z * h[-1] - k * h[-2])
    features = []
    for alpha in indices:
        value = np.ones(z.shape[:-1], dtype=z.dtype)
        for j in np.flatnonzero(alpha):
            k = int(alpha[j])
            value = value * h[k][..., j] / np.sqrt(factorial(k))
        features.append(value)
    return np.stack(features, axis=-1).astype(np.float32)
