"""Diagnostic reading of paper Algorithm 2; not the fixed-anchor PSGA solver.

The pseudocode omits initialization for inner w/d. Here both start at outer w,
mu resets to zero each outer iteration, and line 6 uses the changing inner w.
These assumptions are explicit; results cannot establish author-code fidelity.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stco.market import load_dataset, predict, transaction_fraction
from stco.solver import project_simplex, _topk


@njit(cache=True)
def literal(q, h, outer_max=100, inner_max=100000):
    w = np.ones(len(q))/len(q)
    total_inner = 0
    largest_primal = 0.
    for k in range(outer_max):
        anchor = w.copy()
        d = anchor.copy()
        mu = np.zeros(len(q))
        v = _topk(anchor, 10)
        for j in range(inner_max):
            old = w.copy()
            w = project_simplex((old/.0025+.618*d+q+8e-6*v-mu)/(1/.0025+.618))
            phi = w+mu/.618-h
            d = h+np.sign(phi)*np.maximum(np.abs(phi)-.025/.618, 0)
            mu += .618*(w-d)
            if np.linalg.norm(w-old)/np.linalg.norm(w) <= 1e-8:
                break
        else:
            return w, k+1, total_inner+j+1, False, largest_primal
        total_inner += j+1
        largest_primal = max(largest_primal, np.linalg.norm(w-d))
        if np.linalg.norm(w-anchor)/np.linalg.norm(w) <= 1e-8:
            return w, k+1, total_inner, True, largest_primal
    return w, outer_max, total_inner, False, largest_primal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--days", type=int, default=1043)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x = load_dataset("MSCI", args.data)[:args.days]
    h = np.zeros(x.shape[1]); capital = 1.
    rows = []; weights = []
    start = time.time()
    for t in range(len(x)):
        w = np.ones(x.shape[1])/x.shape[1]
        outer, inner, ok, primal = 0, 0, True, 0.
        if t:
            q = predict(x,t-1,1,4)
            w, outer, inner, ok, primal = literal(q,h)
        if not ok:
            (args.output/"incomplete.json").write_text(json.dumps({
                "day_zero_based":t,"outer_iterations":outer,"inner_iterations":inner,
                "elapsed_seconds":time.time()-start,"diagnostic_outer_cap":100,
                "diagnostic_inner_cap":100000,"paper_cap":100000000,
                "scope":"Preflight; no completed representative financial result"},indent=2)+"\n")
            print("INCOMPLETE",t,outer,inner,time.time()-start,flush=True)
            return
        gross = w@x[t]
        theta = transaction_fraction(h,w,.0025)
        capital *= gross*theta
        h = w*x[t]/gross
        rows.append({"day":t,"wealth":float(capital),"outer_iterations":outer,
                     "inner_iterations":inner,"max_inner_primal_residual":float(primal)})
        weights.append(w)
        if t%50 == 0 or t == len(x)-1:
            print(t,capital,"seconds",time.time()-start,flush=True)
    np.savez_compressed(args.output/"weights.npz",weights=np.array(weights))
    (args.output/"results.json").write_text(json.dumps({"scope":"Diagnostic moving-anchor Algorithm 2 reading, MSCI STCO1 gamma .0025",
        "paper_wealth":2.65,"local_wealth":float(capital),"days":len(x),
        "complete_dataset":len(x)==1043,"elapsed_seconds":time.time()-start,"rows":rows},indent=2)+"\n")


if __name__ == "__main__":
    main()
