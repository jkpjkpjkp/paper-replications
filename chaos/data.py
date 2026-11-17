"""Synthetic paired-noise datasets; explicit local choices are in configs/*.json."""
import numpy as np
from scipy.fft import rfft, irfft


def ou(dw, dt, theta=2.0, mu=1.0, sigma=0.3, x0=0.0):
    x = np.empty((len(dw), dw.shape[1] + 1), dtype=np.float64)
    x[:, 0] = x0
    for k in range(dw.shape[1]):
        x[:, k + 1] = x[:, k] + theta * (mu - x[:, k]) * dt + sigma * dw[:, k]
    return x


def ou_coefficients(basis, dt, theta=2.0, mu=1.0, sigma=0.3, x0=0.0):
    """Exact EM chaos coefficients, enabling separate projection/network errors."""
    n = basis.shape[1]
    coeff = np.zeros((n + 1, 1 + len(basis)))
    coeff[0, 0] = x0
    for k in range(n):
        coeff[k + 1] = (1 - theta * dt) * coeff[k]
        coeff[k + 1, 0] += theta * mu * dt
        coeff[k + 1, 1:] += sigma * np.sqrt(dt) * basis[:, k]
    return coeff


def heston(dw_independent, dt, mu=0.05, kappa=2.0, theta=0.04,
           zeta=0.2, rho=-0.7, s0=1.0, v0=0.04):
    """Full-truncation Euler-Maruyama; output (price, nonnegative variance)."""
    n, steps, _ = dw_independent.shape
    x = np.empty((n, steps + 1, 2), dtype=np.float64)
    x[:, 0] = [s0, v0]
    for k in range(steps):
        s, v = x[:, k, 0], np.maximum(x[:, k, 1], 0)
        dws = dw_independent[:, k, 0]
        dwv = rho * dws + np.sqrt(1 - rho**2) * dw_independent[:, k, 1]
        x[:, k + 1, 0] = s + mu * s * dt + np.sqrt(v) * s * dws
        x[:, k + 1, 1] = np.maximum(v + kappa * (theta - v) * dt + zeta * np.sqrt(v) * dwv, 0)
    return x


def phi4_solve(u0, dw, dt, nonlinear=True):
    """Periodic finite-difference Laplacian, implicit diffusion, explicit drift.

    This avoids the unstable explicit heat step at dt=1e-3, dx=1/128.
    The inverse diffusion operator also acts on the noise, matching the cited
    torchspde general_solver.ipynb / SPDE.Parabolic implementation.
    """
    n, steps, nx = dw.shape
    lap = -4 * nx**2 * np.sin(np.pi * np.arange(nx // 2 + 1) / nx)**2
    denominator = 1 - dt * lap
    u = np.empty((n, steps + 1, nx), dtype=np.float32)
    u[:, 0] = u0
    for k in range(steps):
        old = u[:, k].astype(np.float64)
        drift = 3 * old - (old**3 if nonlinear else 0)
        u[:, k + 1] = irfft(rfft(old + dt * drift + dw[:, k], axis=-1) / denominator,
                            n=nx, axis=-1)
    return u


def phi4_dataset(n, seed, random_initial=False, nx=128, steps=50, dt=0.001, sigma=0.1,
                 initial_amplitude=0.3, initial_decay=2.0):
    rng = np.random.default_rng(seed)
    grid = np.arange(nx) / nx
    # Paper says torus but specifies sine(j*pi*x). Keep this noise prescription
    # and document that it is not translation-invariant periodic Fourier noise.
    spatial = np.sqrt(2) * np.sin(np.pi * np.arange(nx + 1)[:, None] * grid)
    independent = rng.normal(size=(n, steps, nx + 1)) * np.sqrt(dt)
    dw = (sigma * (independent @ spatial)).astype(np.float32)
    qstd = (sigma * np.sqrt(np.sum(spatial**2, axis=0))).astype(np.float32)
    u0 = np.zeros((n, nx), np.float32)
    if random_initial:
        frequencies = np.arange(1, 9)
        weights = rng.normal(size=(n, 8)) / frequencies**initial_decay
        waves = np.sin(2 * np.pi * frequencies[:, None] * grid)
        u0 = (initial_amplitude * weights @ waves).astype(np.float32)
    return {"dw": dw, "u0": u0, "qstd": qstd, "u": phi4_solve(u0, dw, dt)}
