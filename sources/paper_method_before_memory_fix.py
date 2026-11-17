"""Literal Algorithm 1 / Eq. (5), isolated from the released-code replay.

The printed right-hand side of Eq. (5) uses ||L y||, t>=0. We preserve it,
including its inconsistency with Sigma=L L.T. This is not a corrected sparse
optimizer. Eq. (12) soft gates do not impose an exact nonzero holding count.
Unpublished experimental choices are recorded by paper_method_study.py.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/src"))
from model import dense_nn_mlp
from selector import soft_topk_1d


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def vector_loss(prediction, target, weights=None, alpha=.5):
    """Algorithm 1: same y_i in both terms; sum over assets, mean over batch."""
    squared = (prediction-target).square().sum()
    if weights is None:
        return squared
    return alpha * (-target @ weights) + (1-alpha) * squared


def covariance(window):
    """Appendix B.1: 10% identity shrinkage, jitter only if Cholesky needs it."""
    cov = torch.cov(window.T)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype)
    cov = .9*cov + .1*torch.trace(cov)/len(cov)*eye
    cov = (cov+cov.T)/2
    L, info = torch.linalg.cholesky_ex(cov)
    if int(info) != 0:
        # Magnitude not reported; explicit reconstruction choice, in raw units.
        cov = cov + max(1e-12, 1e-12-float(torch.linalg.eigvalsh(cov).min()))*eye
        L = torch.linalg.cholesky(cov)
    return cov, L


class Panel:
    def __init__(self, path, lookback=100, train_ratio=.8):
        self.path = Path(path)
        self.frame = pd.read_csv(path, index_col=0)
        values = self.frame.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not self.frame.index.is_unique or not self.frame.index.is_monotonic_increasing:
            raise ValueError("Nonfinite, duplicate, or unordered panel")
        self.raw = torch.tensor(values, dtype=torch.float64)
        self.n = self.raw.shape[1]
        self.lookback = lookback
        self.cut = int(len(self.raw)*train_ratio)
        self.train = list(range(lookback, self.cut))
        # Test targets after the chronological split; windows can use earlier
        # training history. No target crosses backward into the training set.
        self.test = list(range(self.cut, len(self.raw)))
        self.train_mean = self.raw[:self.cut].mean(0)
        self._covariances = {}

    def sample(self, index):
        window = self.raw[index-self.lookback:index]
        if index not in self._covariances:
            self._covariances[index] = covariance(window)
        cov, L = self._covariances[index]
        return window.flatten().float(), self.raw[index], cov, L


class PrintedDecision:
    def __init__(self, n, k, beta=5., offset=1e-8, solver_eps=1e-6, precondition=True):
        self.n, self.k = n, k
        self.beta, self.offset, self.solver_eps = beta, offset, solver_eps
        self.precondition = precondition
        self.layers = {}

    def layer(self, n):
        if n not in self.layers:
            if self.precondition:
                # q=F@y, a=F^{-T}1, b=F^{-T}mu. Eliminating t gives
                # max b.q s.t. ||q||<=1 and a.q>=0. Positive normalization
                # of a and b changes neither the feasible set nor argmax.
                q=cp.Variable(n)
                b,a=cp.Parameter(n),cp.Parameter(n)
                problem=cp.Problem(cp.Maximize(b@q),[cp.norm(q,2)<=1,a@q>=0])
                self.layers[n]=CvxpyLayer(problem,parameters=[b,a],variables=[q])
            else:
                y, t = cp.Variable(n), cp.Variable()
                mu, factor = cp.Parameter(n), cp.Parameter((n,n))
                problem = cp.Problem(cp.Maximize(mu@y),
                                     [cp.sum(y)==t, cp.norm(factor@y,2)<=1, t>=0])
                self.layers[n] = CvxpyLayer(problem, parameters=[mu,factor], variables=[y,t])
        return self.layers[n]

    def solve(self, mean, L, covariance_risk=False):
        factor = L.T.contiguous() if covariance_risk else L
        if self.precondition:
            a=torch.linalg.solve(factor.T,torch.ones_like(mean))
            b=torch.linalg.solve(factor.T,mean)
            q,=self.layer(len(mean))(b/torch.linalg.vector_norm(b).clamp_min(1e-30),
                                    a/torch.linalg.vector_norm(a),
                                    solver_args={"eps":self.solver_eps,"max_iters":10000})
            y=torch.linalg.solve(factor,q)
            t=y.sum()
        else:
            y, t = self.layer(len(mean))(mean, factor, solver_args={"eps":self.solver_eps,"max_iters":10000})
        w = y/(t+self.offset)
        if not torch.isfinite(w).all():
            raise FloatingPointError("Nonfinite weights from the published t>=0 formulation")
        with torch.no_grad():
            diagnostics = {"t":float(t), "near_zero_t":bool(abs(float(t))<1e-6),
                           "budget_error":float(abs(w.sum()-1)),
                           "conic_equality_error":float(abs(y.sum()-t)),
                           "conic_risk_excess":max(0.,float(torch.linalg.vector_norm(factor@y))-1),
                           "gross_exposure":float(w.abs().sum()),
                           "negative_t_violation":max(0.,-float(t))}
        return w, diagnostics

    def soft(self, mean, L):
        initial, d0 = self.solve(mean, L)
        scores = (L.T@initial).abs()
        gate = soft_topk_1d(scores, self.k, alpha=self.beta, iters=32)
        w, d1 = self.solve(mean*gate, L)
        info = {**d1, "initial_t":d0["t"], "initial_near_zero_t":d0["near_zero_t"],
                "gate_mass":float(gate.detach().sum()), "gate_mass_error":float(abs(gate.detach().sum()-self.k))}
        return w, info

    def hard(self, mean, cov, L):
        # Conventional OSCAR is defined in Sec. 3 by Eq. (1), whose actual
        # covariance variance is ||L.T y||^2, followed by restricted support.
        initial, _ = self.solve(mean, L, covariance_risk=True)
        support = torch.topk((L.T@initial).abs(), self.k).indices
        subcov = cov[support][:,support]
        subw, info = self.solve(mean[support], torch.linalg.cholesky(subcov), covariance_risk=True)
        w = torch.zeros_like(mean)
        w[support] = subw
        return w, info

    def release(self):
        for layer in self.layers.values():
            layer.info.clear()
        gc.collect(0)


def batches(indices, batch_size, rng):
    order = list(indices)
    rng.shuffle(order)
    return [order[start:start+batch_size] for start in range(0,len(order),batch_size)]


def train_epoch(model, panel, decision, optimizer, method, rng, output, epoch, update, history, diagnostic):
    model.train()
    for batch in batches(panel.train,64,rng):
        optimizer.zero_grad()
        # Forward all forecasts, then reverse-order sample backpropagation to
        # bound conic derivative memory. Mathematically the batch-average loss.
        predictions = [model(panel.sample(i)[0]).double() for i in batch]
        loss_values = []
        for j in reversed(range(len(batch))):
            i = batch[j]
            _, target, cov, L = panel.sample(i)
            if method == "dfl":
                weights, info = decision.soft(predictions[j], L)
                diagnostic["training_decisions"] += 1
                diagnostic["training_near_zero_t"] += int(info["near_zero_t"])
                diagnostic["training_budget_error_max"] = max(diagnostic["training_budget_error_max"],info["budget_error"])
                diagnostic["training_gate_mass_error_max"] = max(diagnostic["training_gate_mass_error_max"],info["gate_mass_error"])
            else:
                weights = None
            loss = vector_loss(predictions[j],target,weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss on {panel.frame.index[i]}")
            loss_values.append(float(loss.detach()))
            (loss/len(batch)).backward()
            predictions[j] = None
            del loss
            decision.release()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient; no clipping or skipped samples permitted")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite parameter after the published gradient update")
        update += 1
        history.append({"epoch":epoch,"update":update,"batch_size":len(batch),"batch_mean_loss":float(np.mean(loss_values))})
        save_json(output/"progress.json",{"epoch":epoch,"update":update,"last_batch_loss":history[-1]["batch_mean_loss"],**diagnostic})
        print(json.dumps(history[-1]),flush=True)
    return update


def evaluate(model, panel, decision, method, output):
    rows=[]
    if model is not None:
        model.eval()
    for i in panel.test:
        x,target,cov,L=panel.sample(i)
        with torch.no_grad():
            mean = panel.train_mean if method=="historic" else model(x).double()
            w,info = decision.soft(mean,L) if method=="dfl" else decision.hard(mean,cov,L)
            realized=float(target@w)
            if not np.isfinite(realized):
                raise FloatingPointError("Nonfinite test return; no dropped dates")
            rows.append({"date":str(panel.frame.index[i]),"return":realized,
                         "holdings_1e-8":int((w.abs()>1e-8).sum()),**info})
        decision.release()
    trace=pd.DataFrame(rows)
    trace.to_csv(output/"test_trace.csv",index=False)
    sr=float(trace["return"].mean()/(trace["return"].std(ddof=1)+1e-12))
    valid_budget=bool((trace.budget_error<=1e-3).all())
    return {"test_days":len(trace),"daily_sharpe_diagnostic":sr,
            "budget_valid_all_test_days":valid_budget,
            "test_budget_violations":int((trace.budget_error>1e-3).sum()),
            "test_budget_error_max":float(trace.budget_error.max()),
            "test_near_zero_t_count":int(trace.near_zero_t.sum()),
            "test_holdings_min":int(trace["holdings_1e-8"].min()),
            "test_holdings_max":int(trace["holdings_1e-8"].max()),
            "test_gross_exposure_max":float(trace.gross_exposure.max()),
            "test_gate_mass_error_max":float(trace.gate_mass_error.max()) if method=="dfl" else None,
            "valid_daily_sharpe":sr if valid_budget else None}


def run(path, output, method, seed, rho, epochs, optimizer_name="sgd"):
    output=Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to mix/overwrite experiment: {output}")
    output.mkdir(parents=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    panel=Panel(path)
    k=round(panel.n*rho/100)
    decision=PrintedDecision(panel.n,k)
    model=None if method=="historic" else dense_nn_mlp(panel.n*100,panel.n,hidden_dims=[512,256],activation="tanh",dropout=0.)
    config={"method":method,"seed":seed,"rho_percent":rho,"k":k,"n_assets":panel.n,"epochs":epochs,
            "optimizer":optimizer_name,"train_windows":len(panel.train),"test_windows":len(panel.test),
            "data_path":str(path),"data_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256":hashlib.sha256((ROOT/"reproduction/paper_method_protocol.json").read_bytes()).hexdigest(),
            "exact_published_experiment_recovered":False}
    save_json(output/"config.json",config)
    history=[];update=0;stage="initializing";epoch=0;start=time.monotonic()
    diagnostic={"training_decisions":0,"training_near_zero_t":0,"training_budget_error_max":0.,"training_gate_mass_error_max":0.}
    try:
        if model is not None:
            optimizer=(torch.optim.SGD if optimizer_name=="sgd" else torch.optim.Adam)(model.parameters(),lr=.001)
            rng=random.Random(seed)
            for epoch in range(1,epochs+1):
                stage="training"
                update=train_epoch(model,panel,decision,optimizer,method,rng,output,epoch,update,history,diagnostic)
                torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"rng":rng.getstate(),"torch_rng":torch.get_rng_state(),"epoch":epoch,"update":update},output/"checkpoint.pt")
                pd.DataFrame(history).to_csv(output/"history.csv",index=False)
        stage="evaluating"
        metrics=evaluate(model,panel,decision,method,output)
        result={**config,"status":"completed" if metrics["budget_valid_all_test_days"] else "completed_invalid_budget",**metrics,**diagnostic}
    except Exception as error:
        result={**config,"status":"failed","failure_stage":stage,"epoch_at_failure":epoch,
                "error_type":type(error).__name__,"error":str(error),**diagnostic}
        if model is not None:
            torch.save(model.state_dict(),output/"failed_model.pt")
        pd.DataFrame(history).to_csv(output/"history.csv",index=False)
    result.update(completed_updates=len(history),elapsed_seconds=time.monotonic()-start)
    save_json(output/"result.json",result)
    print(json.dumps(result),flush=True)
    return result


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--method",choices=["dfl","pfl","historic"],required=True)
    p.add_argument("--seed",type=int,default=16)
    p.add_argument("--rho",type=int,choices=[10,15,20],default=10)
    p.add_argument("--epochs",type=int,required=True,help="Explicit reconstruction budget; not specified by the paper")
    p.add_argument("--optimizer",choices=["sgd","adam"],default="sgd")
    args=p.parse_args()
    result=run(args.data,args.output,args.method,args.seed,args.rho,args.epochs,args.optimizer)
    raise SystemExit(2 if result["status"]=="failed" else 0)
