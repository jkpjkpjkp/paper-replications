import cvxpy as cp
import numpy as np
import torch
from autograd import jacobian
import autograd.numpy as auto_np
from cvxpylayer_local import utils
from cvxpylayer_local.cvxpylayer import CvxpyLayer
import wandb
import os
from joblib import Parallel, delayed, parallel_backend


def dg_dz(y_preds: torch.Tensor, z_star: torch.Tensor, params):
    alpha = params["alpha"]

    if y_preds.dim() == 1:
        s = (y_preds * z_star).sum()
        return alpha * s * y_preds - y_preds

    if y_preds.dim() == 2:
        s = (y_preds * z_star).sum(dim=-1, keepdim=True)
        return alpha * y_preds * s - y_preds

    if y_preds.dim() == 3:
        if z_star.dim() == 1:
            zs = z_star.view(1, 1, -1)
        elif z_star.dim() == 2 and z_star.shape[0] == y_preds.shape[0]:
            zs = z_star.unsqueeze(1)
        else:
            raise ValueError(f"Bad z_star shape {tuple(z_star.shape)} for y_preds {tuple(y_preds.shape)}")

        s = (y_preds * zs).sum(dim=-1, keepdim=True)
        return alpha * y_preds * s - y_preds

    raise ValueError(f"Unexpected y_preds shape: {tuple(y_preds.shape)}")

def d2g_dzdy(y_preds, z_star, params):
    """
    y_preds: (B, M, n) or (B, n)  (interpreted as (B,1,n))
    z_star : (B, n)
    returns: (B, n, n) for ∂^2 g / (∂z ∂y)
             where g(z;Y) = (1/M) sum_j [ 0.5*alpha*(y_j^T z)^2 - (y_j^T z) ]
             so  ∂^2 g / (∂z ∂y) = alpha*( mean_y z^T + mean_s * I ) - I
             with mean_y = (1/M)∑ y_j, mean_s = (1/M)∑ (y_j^T z)
    """
    alpha = params["alpha"]

    if y_preds.dim() == 2:
        y_preds = y_preds.unsqueeze(1)
    z_unsq = z_star.unsqueeze(1)


    s = (y_preds * z_unsq).sum(dim=-1)
    mean_s = s.mean(dim=1)


    mean_y = y_preds.mean(dim=1)

    Bsz, n = z_star.shape
    I = torch.eye(n, device=z_star.device, dtype=z_star.dtype).unsqueeze(0).expand(Bsz, n, n)


    yzT = torch.einsum('bi,bj->bij', mean_y, z_star)


    H = alpha * (yzT + mean_s.view(-1, 1, 1) * I) - I
    return H


#             # TODO


os.environ["PYDEVD_DISABLE_SUBPROCESS_TRACE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def cvxpy_portfolio_parallel_kkt(params, mc_samples: int = 1, n_jobs: int = 32):
    n = params["n"]
    alpha = params["alpha"]
    z_max = 1.0

    def _worker_init():
        _worker_init.Z = cp.Variable(n)

        _worker_init.Y = cp.Parameter((mc_samples, n))

        t_vec = _worker_init.Y @ _worker_init.Z
        obj = (0.5 * alpha * cp.sum_squares(t_vec) - cp.sum(t_vec)) / mc_samples

        cons = [
            cp.sum(_worker_init.Z) == 1,
            -_worker_init.Z <= 0,
            _worker_init.Z <= z_max
        ]
        _worker_init.CON = cons
        _worker_init.PROB = cp.Problem(cp.Minimize(obj), cons)

    def _solve_one(Y_i):
        if not hasattr(_worker_init, "PROB"):
            _worker_init()
        W = _worker_init

        W.Y.value = Y_i
        try:
            W.PROB.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            W.PROB.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-1)


        try:
            z_star = W.Z.value.astype(np.float64)
        except:
            W.PROB.solve(solver=cp.SCS, verbose=False, eps=1e-1)
            z_star = W.Z.value.astype(np.float64)

        lam_eq    = np.atleast_1d(W.PROB.constraints[0].dual_value).astype(np.float64)
        lam_lower = np.asarray(W.PROB.constraints[1].dual_value, dtype=np.float64).ravel()
        lam_upper = np.asarray(W.PROB.constraints[2].dual_value, dtype=np.float64).ravel()

        H = alpha * (Y_i.T @ Y_i) / mc_samples
        H = H.astype(np.float64)


        G = np.vstack([-np.eye(n), np.eye(n)])

        h_val = np.concatenate([-z_star, z_star - z_max]).astype(np.float64)

        lam = np.concatenate([lam_lower, lam_upper]).astype(np.float64)
        DG  = np.diag(lam) @ G
        Dh  = np.diag(h_val)

        Q = np.ones((1, n), dtype=np.float64)

        top = np.concatenate([H, G.T, Q.T], axis=1)
        mid = np.concatenate([DG, Dh, np.zeros((2 * n, 1), dtype=np.float64)], axis=1)
        bottom = np.concatenate([Q,  np.zeros((1,2*n), dtype=np.float64),np.zeros((1,1), dtype=np.float64)], axis=1)
        KKT = np.concatenate([top, mid, bottom], axis=0)

        eps = 1e-6
        KKT += eps * np.eye(3 * n + 1)
        return z_star.astype(np.float32), KKT.astype(np.float32)

    def layer_fn(y_preds: torch.Tensor):
        device = y_preds.device
        y_np = y_preds.detach().cpu().numpy()

        if y_np.ndim == 2:
            y_np = y_np[:, None, :]
        assert y_np.ndim == 3 and y_np.shape[1] == mc_samples and y_np.shape[2] == n,\
            f"Expected y_preds shape (B,{mc_samples},{n}) or (B,{n}); got {y_preds.shape}"

        with Parallel(n_jobs=n_jobs, backend="loky", initializer=_worker_init) as pool:
            results = pool(delayed(_solve_one)(y_np[i]) for i in range(len(y_np)))

        z_list, KKT_list = map(np.array, zip(*results))
        z_t = torch.from_numpy(z_list).to(device)

        info = {"KKT_matrices": np.stack(KKT_list, axis=0)}
        return (z_t,), info

    return layer_fn
