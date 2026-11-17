"""Independent checks for solver, implicit derivatives, data timing, and sampling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
import cvxpy as cp
from core import solve_portfolio, PortfolioLayer, Diffsion, ddim, sample_vjp, score_weights
from run import load_data


def test_qp_matches_independent_solver():
    torch.manual_seed(12)
    for alpha in [1e-6, 10., 1000.]:
        y = torch.randn(3, 10, 4, dtype=torch.double)*.1
        z, _ = solve_portfolio(y, alpha)
        for yy, zz in zip(y.numpy(), z.numpy()):
            w = cp.Variable(4)
            loss = .5*alpha*cp.sum_squares(yy@w)/10-cp.sum(yy@w)/10
            p = cp.Problem(cp.Minimize(loss), [w>=0, w<=1, cp.sum(w)==1])
            p.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
            actual = .5*alpha*np.mean((yy@zz)**2)-np.mean(yy@zz)
            assert abs(actual-p.value) < 2e-8


def test_implicit_gradient_finite_difference():
    torch.manual_seed(71)
    y = (.02*torch.randn(2, 10, 3, dtype=torch.double)).requires_grad_()
    assert torch.autograd.gradcheck(lambda a: PortfolioLayer.apply(a, 1000., 0.),
                                   (y,), eps=1e-5, atol=1e-3, rtol=1e-2)


def test_cross_derivative_includes_all_terms():
    torch.manual_seed(9)
    y = torch.randn(2, 5, 3, dtype=torch.double, requires_grad=True)
    z, v = torch.randn(2,3,dtype=torch.double), torch.randn(2,3,dtype=torch.double)
    alpha = .4
    g = (alpha*y*(y*z[:,None]).sum(-1,keepdim=True)-y).mean(1)
    expected = torch.autograd.grad(-(v*g).sum(), y)[0]
    torch.testing.assert_close(sample_vjp(y,z,v,alpha),expected)


def test_score_adjoint_against_distribution_perturbation():
    # Change scenario probabilities, re-solve the optimization independently,
    # and compare outer-loss finite differences to the score-function adjoint.
    scenarios=np.concatenate([.2*np.eye(3),-.2*np.eye(3)])+np.array([.01,.02,.03])
    labels=np.array([.1,-.2,.3]);alpha=10.
    samples=torch.tensor(scenarios[None],dtype=torch.double)
    z,mat=solve_portfolio(samples,alpha,0.)
    assert z.min()>.05
    w=score_weights(samples,z,mat,torch.tensor(labels[None]),alpha)[0].numpy()
    predicted=w-w.mean()
    def objective(logits):
        probs=np.exp(logits);probs/=probs.sum()
        decision=cp.Variable(3)
        p=cp.Problem(cp.Minimize(.5*alpha*cp.sum(cp.multiply(probs,cp.square(scenarios@decision)))-probs@(scenarios@decision)),
                     [decision>=0,decision<=1,cp.sum(decision)==1])
        p.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_feas=1e-12,tol_gap_rel=1e-12)
        r=labels@decision.value
        return .5*alpha*r*r-r
    numeric=[]
    for j in range(6):
        delta=np.zeros(6);delta[j]=1e-4
        numeric.append((objective(delta)-objective(-delta))/2e-4)
    np.testing.assert_allclose(predicted,numeric,atol=2e-5,rtol=2e-3)


def test_full_ddim_matches_release():
    torch.set_num_threads(2)
    torch.manual_seed(2)
    d = Diffsion(4, 3, timesteps=1000)
    x = torch.randn(2,4)
    torch.manual_seed(3)
    ours = ddim(d,x,1,1000).squeeze(1)
    # The release draws unused noise at each eta=0 step. That affects RNG state,
    # but its deterministic trajectory after the initial draw is identical.
    torch.manual_seed(3)
    original = d.sample_elbo(x,test_mode=True)
    torch.testing.assert_close(ours,original,rtol=5e-5,atol=5e-5)


def test_split_scaler_and_portfolio_alignment():
    x,y,s,meta = load_data(0,50,'cpu')
    assert all(len(a)>0 for a in s)
    assert max(meta['target_dates'][s[0]]) < min(meta['dates'][s[1]])
    assert max(meta['target_dates'][s[1]]) < min(meta['dates'][s[2]])
    assert (meta['target_dates']>meta['dates']).all()
    assert np.max(np.abs(x[s[0]].numpy().reshape(-1,28).mean(0))) < 1e-4
    assert len(np.unique(meta['permnos'])) == 50
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
