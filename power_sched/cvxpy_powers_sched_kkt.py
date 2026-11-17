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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")


def dg_dz(y_pred, z_star, params):
    gs, ge = params["gamma_under"], params["gamma_over"]
    under = - gs * ((y_pred - z_star) > 0).float()
    over = ge * ((z_star - y_pred) > 0).float()
    quad_term = z_star - y_pred
    return under + over + quad_term


def jacobian_h_auto(z, c_ramp):
    n = len(z)
    J = np.zeros((2 * (n - 1), n), np.float64)
    for i in range(n - 1):
        J[i, i + 1], J[i, i] = 1.0, -1.0
        J[n - 1 + i, i], J[n - 1 + i, i + 1] = 1.0, -1.0
    return J

def constraint_fn(z, c_ramp):
    n = len(z)
    g = np.zeros(2 * (n - 1), np.float64)
    for i in range(n - 1):
        g[i] = z[i + 1] - z[i] - c_ramp
        g[n - 1 + i] = z[i] - z[i + 1] - c_ramp
    return g

def cvxpy_ps_slack(params, mc_samples: int):
    n = params["n"]
    c_ramp = params["c_ramp"]
    gs, ge = params["gamma_under"], params["gamma_over"]

    z_var = cp.Variable(n)
    y_param = cp.Parameter([mc_samples, n])

    diff = y_param - z_var

    under = cp.pos(diff)
    over = cp.pos(-diff)
    quad = 0.5 * cp.sum_squares(diff)
    obj = (gs * cp.sum(under) + ge * cp.sum(over) + quad) / float(mc_samples)

    constraints = [
        z_var[1:] - z_var[:-1] <= c_ramp,
        z_var[:-1] - z_var[1:] <= c_ramp
    ]
    problem = cp.Problem(cp.Minimize(obj), constraints)

    def layer_fn(y_preds):
        device = y_preds.device
        batch_size = y_preds.shape[0]

        if y_preds.ndim == 3:
            y_np = y_preds.cpu().detach().numpy()
        else:
            batch_size = 1
            y_np = y_preds.cpu().detach().numpy()

        z_batch_np = np.zeros((y_np.shape[0], n), dtype=np.float32)
        KKT_list = []
        eye_n = np.eye(n, dtype=np.float32)

        total_time = 0
        for i in range(batch_size):
            y_param.value = y_np[i]


            try:
                problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            except cp.SolverError:
                problem.solve(solver=cp.OSQP, warm_start=True, verbose=True, eps_abs=1e-1)


            total_time += problem.solver_stats.solve_time

            z_star = z_var.value.copy().astype(np.float64)

            lam_u = problem.constraints[0].dual_value
            lam_o = problem.constraints[1].dual_value
            lam_u[np.abs(lam_u) < 1e-10] = 0
            lam_o[np.abs(lam_o) < 1e-10] = 0
            lam = np.concatenate([lam_u, lam_o], axis=0).astype(np.float64)

            dh = jacobian_h_auto(z_star, c_ramp).astype(np.float64)
            Dlam = np.diag(lam)
            H = eye_n
            hdiag = np.diag(constraint_fn(z_star, c_ramp).astype(np.float64))

            top = np.concatenate([H, dh.T], axis=1)
            bottom = np.concatenate([Dlam.dot(dh), hdiag], axis=1)
            _KKT = np.concatenate([top, bottom], axis=0)


            if np.abs(np.linalg.eigvals(_KKT)).min() < 0.1:
                _KKT = _KKT + np.eye(top.shape[1]) * 1e-5
            z_star = z_star.astype(np.float32)
            z_batch_np[i] = z_star

            KKT_list.append(_KKT)


        KKT_np = np.stack(KKT_list, axis=0)

        z_t  = torch.from_numpy(z_batch_np).to(device)

        info = {"KKT_matrices": KKT_np}


        return (z_t,), info

    return layer_fn


os.environ["PYDEVD_DISABLE_SUBPROCESS_TRACE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def cvxpy_ps_parallel_kkt(params, mc_samples):
    try:
        def layer_fn(y_preds):
            n_jobs = 64
            y_cpu = y_preds.detach().cpu().numpy().astype('float64')
            with Parallel(
                n_jobs=n_jobs,
                backend="loky",
                prefer="processes",
                initializer=_worker_init,
                initargs=(params, mc_samples),
                batch_size=1
            ) as pool:
                results = pool(
                    delayed(_solver_one_sample)(y_cpu[i]) for i in range(len(y_cpu))
                )
            z_np, KKT_np = map(np.array, zip(*results))
            return (torch.from_numpy(z_np).to(y_preds.device),),\
                {"KKT_matrices": np.stack(KKT_np, 0)}
        return layer_fn
    except (RuntimeError, torch.OutOfMemoryError) as e:
        raise torch.OutOfMemoryError(str(e))

def _worker_init(params, mc_samples):
    global CON_ID0, CON_ID1, C_RAMP, PROB, Y_VAR, Z_VAR
    n, C_RAMP = params["n"], params["c_ramp"]
    gs, ge = params["gamma_under"], params["gamma_over"]

    Z_VAR  = cp.Variable(n)
    Y_VAR  = cp.Parameter((mc_samples, n))
    diff = Y_VAR - Z_VAR
    obj  = (gs*cp.sum(cp.pos(diff)) +
            ge*cp.sum(cp.pos(-diff)) +
            0.5*cp.sum_squares(diff)) / float(mc_samples)
    CON_ID0 =  Z_VAR[1:] - Z_VAR[:-1] <= C_RAMP
    CON_ID1 =  Z_VAR[:-1] - Z_VAR[1:] <= C_RAMP
    PROB = cp.Problem(cp.Minimize(obj), [CON_ID0, CON_ID1])

    Y_VAR.value = np.zeros([mc_samples, n])
    PROB.solve(solver="OSQP", warm_start=False, verbose=False,
               eps_abs=1e-5, eps_rel=1e-5, polish=False)

def _solver_one_sample(y_pred):
    Y_VAR.value = y_pred
    try:
        PROB.solve(solver="OSQP", warm_start=True, verbose=False)
    except cp.SolverError:
        PROB.solve(solver="OSQP", warm_start=True, verbose=True, eps_abs=1e-1)

    z_star = Z_VAR.value.astype(np.float32)

    lam_u  = CON_ID0.dual_value
    lam_o  = CON_ID1.dual_value
    lam_u[np.abs(lam_u) < 1e-10] = 0
    lam_o[np.abs(lam_o) < 1e-10] = 0
    lam = np.concatenate([lam_u, lam_o], axis=0).astype(np.float64)
    dh = jacobian_h_auto(z_star, C_RAMP).astype(np.float64)
    Dlam = np.diag(lam)
    H = np.eye(z_star.shape[0])
    hdiag  = np.diag(constraint_fn(z_star, C_RAMP).astype(np.float64))

    top    = np.concatenate([H, dh.T], 1)
    bottom = np.concatenate([Dlam.dot(dh), hdiag], 1)
    KKT    = np.concatenate([top, bottom], 0)


    return z_star, KKT
