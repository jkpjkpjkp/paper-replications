"""Audited portfolio solver and diffusion sampling; see REPRODUCTION.txt for changes."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stock_portfolio'))
from model import MLP, GaussianMLP, PortfolioPolicyNet
from diffusion_opt import Diffsion


def solve_portfolio(samples, alpha=1e-6, ridge=1e-6, kkt=True):
    """Solve original convex QP, certify easy vertex cases, return regularized KKT.

    Vertex shortcut is exact when its first-order optimality certificate holds.
    ridge regularizes the backward system only, following the release.
    """
    y = samples.detach().double()
    b, m, n = y.shape
    mu = y.mean(1)
    hess = alpha * y.transpose(1, 2).bmm(y) / m
    idx = mu.argmax(1)
    z = F.one_hot(idx, n).double()
    grad = hess.bmm(z.unsqueeze(-1)).squeeze(-1) - mu
    gaps = grad - grad.gather(1, idx[:, None])
    certified = (gaps >= -1e-12).all(1)
    lower = gaps.clamp_min(0)
    upper = torch.zeros_like(z)
    for i in (~certified).nonzero().flatten().tolist():
        import cvxpy as cp
        w = cp.Variable(n)
        yn = y[i].cpu().numpy()
        problem = cp.Problem(cp.Minimize(.5*alpha*cp.sum_squares(yn@w)/m - mu[i].cpu().numpy()@w),
                             [cp.sum(w) == 1, w >= 0, w <= 1])
        problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
        if problem.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(f'QP failed: {problem.status}')
        z[i] = torch.as_tensor(w.value, device=y.device)
        lower[i] = torch.as_tensor(problem.constraints[1].dual_value, device=y.device)
        upper[i] = torch.as_tensor(problem.constraints[2].dual_value, device=y.device)
    if (z.sum(1)-1).abs().max() > 1e-6 or z.min() < -1e-6 or z.max() > 1+1e-6:
        raise RuntimeError('Infeasible portfolio')
    if not kkt:
        return z.to(samples.dtype), None
    g = torch.cat([-torch.eye(n, device=y.device, dtype=y.dtype), torch.eye(n, device=y.device, dtype=y.dtype)])
    lam = torch.cat([lower, upper], 1)
    slack = torch.cat([-z, z-1], 1)
    mat = y.new_zeros(b, 3*n+1, 3*n+1)
    mat[:, :n, :n] = hess
    mat[:, :n, n:3*n] = g.T
    mat[:, n:3*n, :n] = lam.unsqueeze(-1)*g
    mat[:, n:3*n, n:3*n] = torch.diag_embed(slack)
    mat[:, :n, -1] = 1
    mat[:, -1, :n] = 1
    mat += ridge * torch.eye(3*n+1, device=y.device, dtype=y.dtype)
    return z.to(samples.dtype), mat


def adjoint(mat, dz):
    rhs = F.pad(dz.double(), (0, mat.shape[-1]-dz.shape[-1]))
    return torch.linalg.solve(mat.transpose(-1, -2), rhs.unsqueeze(-1)).squeeze(-1)[:, :dz.shape[-1]].to(dz.dtype)


def sample_vjp(y, z, v, alpha):
    """-v^T d/dy [alpha*y*(y^T z)-y], including cross and -I terms."""
    s = (y*z[:, None]).sum(-1, keepdim=True)
    vy = (y*v[:, None]).sum(-1, keepdim=True)
    return -(alpha*vy*z[:, None] + (alpha*s-1)*v[:, None]) / y.shape[1]


class PortfolioLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, samples, alpha, ridge):
        z, mat = solve_portfolio(samples, alpha, ridge)
        ctx.save_for_backward(samples, z, mat)
        ctx.alpha = alpha
        return z

    @staticmethod
    def backward(ctx, dz):
        samples, z, mat = ctx.saved_tensors
        return sample_vjp(samples, z, adjoint(mat, dz), ctx.alpha), None, None


def score_weights(y, z, mat, labels, alpha):
    # Derivative of mean true decision loss, then KKT adjoint.
    actual = (labels*z).sum(-1, keepdim=True)
    dz = (alpha*actual-1)*labels / len(labels)
    v = adjoint(mat, dz)
    dg = alpha*y*(y*z[:, None]).sum(-1, keepdim=True)-y
    return -(dg*v[:, None]).sum(-1).detach()/y.shape[1]


def ddim(diffusion, x, samples, steps=1000, grad=False, checkpoint=False):
    """DDIM eta=0. 1000 steps matches release; fewer steps are labelled approximation."""
    b = len(x)
    context = x.repeat_interleave(samples, 0)
    y = torch.randn(b*samples, diffusion.y_dim, device=x.device)
    schedule = np.linspace(0, diffusion.timesteps-1, steps, dtype=int)[::-1].copy()
    with torch.set_grad_enabled(grad):
        for j, t in enumerate(schedule):
            times = torch.full((len(y),), int(t), device=x.device, dtype=torch.long)
            if checkpoint and grad:
                from torch.utils.checkpoint import checkpoint as ckpt
                eps = ckpt(diffusion.model_net, y, times, context, use_reentrant=False)
            else:
                eps = diffusion.model_net(y, times, context)
            at = diffusion.bar_alpha[int(t)]
            ap = diffusion.bar_alpha[int(schedule[j+1])] if j+1 < len(schedule) else y.new_tensor(1.)
            y0 = (y-(1-at).sqrt()*eps)/at.sqrt()
            y = ap.sqrt()*y0+(1-ap).sqrt()*eps
    return y.reshape(b, samples, -1)


def gaussian_samples(model, x, samples):
    mu, logvar = model(x)
    y = mu[:, None] + torch.exp(.5*logvar[:, None])*torch.randn(len(x), samples, mu.shape[-1], device=x.device)
    return y, mu, logvar


def diffusion_score_loss(diffusion, y, x, weights, timesteps, time_start=None):
    """Uniform-time Monte Carlo of the released negative denoising score surrogate."""
    b, m, d = y.shape
    clean = y.detach().reshape(b*m, d).repeat_interleave(timesteps, 0)
    context = x.repeat_interleave(m*timesteps, 0)
    if time_start is not None:
        t = torch.arange(time_start, time_start+timesteps, device=x.device).repeat(b*m)
    elif timesteps == diffusion.timesteps:
        t = torch.arange(timesteps, device=x.device).repeat(b*m)
    else:
        t = torch.randint(diffusion.timesteps, (len(clean),), device=x.device)
    noise = torch.randn_like(clean)
    noisy = diffusion.q_sample(clean, t, noise)
    prediction = diffusion.model_net(noisy, t, context)
    score = -(prediction-noise).square().sum(-1).reshape(b, m, timesteps).mean(-1)
    return (weights*score).sum()
