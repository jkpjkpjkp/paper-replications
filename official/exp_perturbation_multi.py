"""
Multi-trial perturbation stability: average Spearman ρ across 3 train seeds.
Strengthens single-trial result in App G / EXP-7.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_perturbation import run as run_single


def aggregate_trials(dataset_name, gpu, trials, num_eval_nodes, out_json):
    per_trial = []
    for t in range(trials):
        tmp_path = f"/tmp/perturb_{dataset_name}_{t}.json"
        run_single(dataset_name, gpu,
                   sigma_list=[0.01, 0.05, 0.1],
                   num_eval_nodes=num_eval_nodes,
                   trial=t, out_json=tmp_path)
        with open(tmp_path) as f:
            per_trial.append(json.load(f))

    aggregated = {
        "dataset": dataset_name,
        "n_trials": trials,
        "per_trial": per_trial,
    }
    # Averaged stats
    sigmas = [str(s) for s in [0.01, 0.05, 0.1]]
    aggregated["averaged"] = {}
    for s in sigmas:
        rho_means = [d["per_sigma"][s]["rho_mean"] for d in per_trial
                     if d["per_sigma"][s]["rho_mean"] is not None]
        rho_medians = [d["per_sigma"][s]["rho_median"] for d in per_trial
                       if d["per_sigma"][s]["rho_median"] is not None]
        top_match = [d["per_sigma"][s]["top_component_match_rate"]
                     for d in per_trial
                     if d["per_sigma"][s]["top_component_match_rate"] is not None]
        aggregated["averaged"][s] = {
            "rho_mean_avg": float(np.mean(rho_means)) if rho_means else None,
            "rho_mean_std": float(np.std(rho_means)) if rho_means else None,
            "rho_median_avg": float(np.mean(rho_medians)) if rho_medians else None,
            "top_match_avg": float(np.mean(top_match)) if top_match else None,
            "top_match_std": float(np.std(top_match)) if top_match else None,
        }

    aurocs = [d["model_AUROC"] for d in per_trial]
    aggregated["model_AUROC_mean"] = float(np.mean(aurocs))
    aggregated["model_AUROC_std"] = float(np.std(aurocs))

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n>>> Multi-trial saved {out_json}")
    for s in sigmas:
        a = aggregated["averaged"][s]
        if a["rho_mean_avg"] is None: continue
        print(f"  sigma={s}: rho_mean={a['rho_mean_avg']:.3f}±{a['rho_mean_std']:.3f} "
              f"top_match={a['top_match_avg']*100:.0f}%±{a['top_match_std']*100:.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--num_eval_nodes", type=int, default=200)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    aggregate_trials(args.dataset, args.gpu, args.trials,
                     args.num_eval_nodes, args.out_json)
