import numpy as np
from cvxpy_toy_kkt import cvxpy_toy_parallel_kkt, task_loss
import torch
from torch.distributions.normal import Normal

from typing import Optional, Tuple, Union, Sequence
import numpy as np

def gen_toy_data(m: int,
            x_dim: int,
            a: float = 5.0,
            b: float = 0.9,
            p: float = 0.7,
            noise_std: float = 0.0,
            seed: int | None = None,
            x_dependent_prob: bool = False,
            logit_scale: float = 1.0):
    assert a > 0, "require a > 0"
    assert 0.0 < b < 1.0, "require 0 < b < 1 to stay in log domain"
    assert 0.0 < p < 1.0, "require 0 < p < 1"

    rng = np.random.default_rng(seed)


    X = rng.standard_normal((m, x_dim)).astype(np.float32)


    if x_dependent_prob:
        theta = rng.standard_normal((x_dim, 1)).astype(np.float64) * float(logit_scale)
        bias  = np.log(p / (1 - p))
        logits = (X.astype(np.float64) @ theta).reshape(-1) + bias
        p_i = 1.0 / (1.0 + np.exp(-logits))

        eps = 1e-3
        p_i = np.clip(p_i, eps, 1 - eps)
    else:
        p_i = np.full(m, p, dtype=np.float64)


    comp = rng.binomial(1, p_i, size=m).astype(bool)

    Y = np.empty((m, 2), dtype=np.float64)
    Y[~comp, :] = np.array([a, -b], dtype=np.float64)
    Y[comp, :] = np.array([-b, a], dtype=np.float64)

    if noise_std > 0:
        Y += rng.normal(scale=noise_std, size=Y.shape)

        Y = np.maximum(Y, -1.0 + 1e-3)

    info = {
        "a": float(a),
        "b": float(b),
        "p": float(p),
        "x_dependent_prob": bool(x_dependent_prob),
        "logit_scale": float(logit_scale),
        "noise_std": float(noise_std),
        "y_dim": 2,
        "p_i_min": float(p_i.min()), "p_i_max": float(p_i.max()), "p_i_mean": float(p_i.mean()),
    }
    return X.astype(np.float32), Y.astype(np.float32), info

def gen_toy_data_simple(
    m: int,
    x_dim: int,
    a: float = 3.0,
    b: float = -0.5,
    p: float = 0.5,
    c: float = 0.15,
    noise_std: float = 0.0,
    seed: int | None = None,
    x_dependent_prob: bool = False,
    logit_scale: float = 1.0,
    min_margin: float = 0,
):

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, x_dim)).astype(np.float32)

    clip_min = -0.99
    assert a > clip_min and b > clip_min and c > clip_min, "a,b,c must > -1"
    assert b < c < a, "need b < c < a"

    if x_dependent_prob:
        theta = rng.standard_normal(x_dim) * logit_scale
        logits = X @ theta
        p_x = 1.0 / (1.0 + np.exp(-logits))


        p_star = (c - b) / (a - b)

    else:
        p_x = np.full(m, float(p))

    choose_a = rng.random(m) < p_x


    y1 = np.where(choose_a, a, b).astype(np.float64)

    y2 = np.full(m, c, dtype=np.float64)

    if noise_std > 0:
        y1 += rng.normal(0.0, noise_std, size=m)
        y2 += rng.normal(0.0, noise_std, size=m)

    y1 = np.maximum(y1, clip_min)
    y2 = np.maximum(y2, clip_min)

    Y = np.stack([y1, y2], axis=1).astype(np.float32)
    data_info = {
        "a": float(a), "b": float(b), "c": float(c),
        "p_const": float(p), "x_dependent_prob": bool(x_dependent_prob),
        "mu1_const": float(p*a + (1-p)*b) if not x_dependent_prob else None,
    }
    return X.astype(np.float32), Y, data_info

def z_star_empirical(Y, C=1.0, grid=5001):
    z = torch.linspace(0., C, grid, device=Y.device, dtype=Y.dtype)
    F = torch.exp(-Y[:, None] * z[None, :]).mean(0)
    return float(z[torch.argmin(F)])


def generate_data(
    n: int,
    x_dim: int,
    C: float = 4.0,
    p_pos: float = 0.9,
    a_pos: float = 1.0,
    b_neg: float = 3.0,
    sigma: float = 0.15,
    y_scale: float = 0.5,
    seed: Optional[int] = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    if seed is not None:
        torch.manual_seed(seed)

    X = torch.randn(n, x_dim, device=device, dtype=dtype)

    bern = torch.bernoulli(torch.full((n,), p_pos, device=device, dtype=dtype)).bool()
    eps  = sigma * torch.randn(n, device=device, dtype=dtype)
    y_raw = torch.where(bern, a_pos + eps, -b_neg + eps)

    Y = y_scale * y_raw

    info = {
        "C": float(C),
        "p_pos": float(p_pos),
        "a_pos": float(a_pos),
        "b_neg": float(b_neg),
        "sigma": float(sigma),
        "y_scale": float(y_scale),
        "y_shift": 0.0,
    }
    return X, Y, info


def generate_data_multi_dim(
    n: int,
    x_dim: int,
    y_dim: int = 1,
    C: float = 4.0,
    p_pos: Union[float, Sequence[float]] = 0.9,
    a_pos: Union[float, Sequence[float]] = 1.0,
    b_neg: Union[float, Sequence[float]] = 3.0,
    sigma: Union[float, Sequence[float]] = 0.15,
    y_scale: Union[float, Sequence[float]] = 0.5,
    y_shift: Union[float, Sequence[float]] = 0.0,
    seed: Optional[int] = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    if seed is not None:
        torch.manual_seed(seed)

    def _to_1d_param(val, name):
        t = torch.as_tensor(val, device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.repeat(y_dim)
        elif t.numel() != y_dim:
            raise ValueError(f"{name} must be scalar or length {y_dim}, got shape {tuple(t.shape)}")
        return t

    p_pos_t  = _to_1d_param(p_pos,  "p_pos")
    a_pos_t  = _to_1d_param(a_pos,  "a_pos")
    b_neg_t  = _to_1d_param(b_neg,  "b_neg")
    sigma_t  = _to_1d_param(sigma,  "sigma")
    y_scale_t= _to_1d_param(y_scale,"y_scale")
    y_shift_t= _to_1d_param(y_shift,"y_shift")

    X = torch.randn(n, x_dim, device=device, dtype=dtype)

    probs = p_pos_t.view(1, y_dim).expand(n, y_dim)
    bern  = torch.bernoulli(probs).bool()
    eps   = torch.randn(n, y_dim, device=device, dtype=dtype) * sigma_t.view(1, y_dim)

    means_pos = a_pos_t.view(1, y_dim)
    means_neg = (-b_neg_t).view(1, y_dim)
    y_raw = torch.where(bern, means_pos + eps, means_neg + eps)

    Y = y_scale_t.view(1, y_dim) * y_raw + y_shift_t.view(1, y_dim)

    info = {
        "C": float(C),
        "p_pos": p_pos_t.detach().cpu().tolist(),
        "a_pos": a_pos_t.detach().cpu().tolist(),
        "b_neg": b_neg_t.detach().cpu().tolist(),
        "sigma": sigma_t.detach().cpu().tolist(),
        "y_scale": y_scale_t.detach().cpu().tolist(),
        "y_shift": y_shift_t.detach().cpu().tolist(),
    }
    return X, Y, info


def sample_y_from_env(X_test, y_dim, mc_samples, params, seed=0, simple=False):
    B = X_test.shape[0]
    if simple:
        _, Y_flat, _ = gen_toy_data_simple(x_dim=X_test.shape[1], m=B*mc_samples, seed=seed)
    else:
        if y_dim == 1:
            _, Y_flat, _ = generate_data(x_dim=X_test.shape[1], n=B * mc_samples, seed=seed, C=params["C"])
        else:
            _, Y_flat, _ = generate_data_multi_dim(x_dim=X_test.shape[1], y_dim=y_dim, n=B * mc_samples, seed=seed, C=params["C"])

    Y_flat = np.asarray(Y_flat)
    if Y_flat.ndim == 1:
        Y_flat = Y_flat[:, None]

    n = Y_flat.shape[1]
    Y_samples = Y_flat.reshape(B, mc_samples, n)
    return Y_samples

def comp_true_obj(
    X_test, Y_test, params, device="cpu",
    mc_env: int = 10, seed: int = 0, simple=False
):
    y_dim = Y_test.shape[1]
    batch_size = Y_test.shape[0]
    n = params["n"]

    layer_regret = cvxpy_toy_parallel_kkt(params, mc_samples=batch_size)
    if Y_test.ndim == 1:
        Y_test = Y_test.unsqueeze(0)
    else:
        Y_test = Y_test.unsqueeze(0)
    z_star_tuple, _ = layer_regret(Y_test)
    z_stars_regret = z_star_tuple[0]

    f_evals_regret = task_loss(
        z_stars_regret.to(device=device, dtype=torch.float32), Y_test, params,
    )

    g0 = -(Y_test).mean()
    g1 = -(Y_test * torch.exp(-Y_test)).mean()
    print("F'(0)=", g0.item(), "   F'(1)=", g1.item())


    layer_env = cvxpy_toy_parallel_kkt(params, mc_samples=mc_env*batch_size)
    y_samples = sample_y_from_env(
        X_test, y_dim, mc_env, params, seed=seed, simple=simple
    )
    y_samples = y_samples.reshape(1, -1, y_dim)
    z_star_tuple, _ = layer_env(y_samples)
    z_stars_env = z_star_tuple[0]
    z_stars_env = z_stars_env.to(device=device, dtype=torch.float32)

    f_evals_env = task_loss(
        z_stars_env.to(device=device, dtype=torch.float32), Y_test, params,
    )

    return f_evals_regret, f_evals_env, z_stars_regret, z_stars_env
