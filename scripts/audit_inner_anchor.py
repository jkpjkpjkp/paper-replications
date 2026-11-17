"""Compare both printed anchor readings on four archived MSCI subproblems."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stco.market import load_dataset, predict
from stco.solver import prox_exact, prox_admm, _topk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    x = load_dataset("MSCI", args.archive/"upstream/STCO-code")
    saved = np.load(args.archive/"results/main/runs/msci_STCO2_g0.0025_w4.npz")
    rows = []
    for t in [1, 100, 500, 1042]:
        w = saved["weights"][t-1]
        h = w*x[t-1]/(w@x[t-1])
        anchor = np.ones(24)/24
        q = predict(x, t-1, 2, 4)
        v = _topk(anchor, 10)
        exact = prox_exact(anchor+.0025*(q+8e-6*v), h, .0025*.025)
        for literal in [False, True]:
            result, info = prox_admm(anchor, q, h, v, lam=.025, tol=1e-8,
                                    stopping="relative", max_iter=100000,
                                    literal_algorithm2=literal)
            rows.append({"day_zero_based": t, "moving_anchor": literal,
                         "max_weight_difference_from_exact_prox":
                         float(abs(result-exact).max()), "info": info})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    main()
