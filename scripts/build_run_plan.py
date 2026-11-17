"""Enumerate required fits, not runnable jobs. No dummy results are emitted."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
models = ["Set-Sequence", "LongConv", "S4", "H3", "Transformer", "Hyena"]
jobs = []
for objective, end_year in [("gross_sharpe", 2021), ("net_sharpe", 2016)]:
    for model in models:
        for year in range(2002, end_year + 1):
            for seed_slot in range(5):
                jobs.append(dict(task="equity", model=model, objective=objective,
                    train_start=f"{year-8}-01-01", train_end=f"{year-1}-12-31",
                    test_start=f"{year}-01-01", test_end=f"{year}-12-31",
                    seed_slot=seed_slot, actual_seed="",
                    status="blocked_exact_panel_and_training_details"))
with (ROOT / "configs/equity_required_fits.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(jobs[0]))
    writer.writeheader()
    writer.writerows(jobs)
summary = dict(
    gross_equity_fits=600, net_equity_fits=450,
    total_equity_fits=len(jobs),
    estimate_at_paper_reported_minutes_per_fit=30,
    implied_a6000_gpu_hours=len(jobs)*.5,
    caveat="Planning extrapolation from paper timings, not a local benchmark. Excludes domain baselines and mortgages. Assumes 5 seeds for net experiments as for gross; Table 9 does not explicitly restate seed count.",
    table5_set_sequence="Can evaluate 2002-2016 subset of gross fits; verify whether authors used identical fits.",
    additional_required=[
        "Table 5 domain baselines: CNN-Transformer K=5,8,10,15; Fourier-FNN K=10,15; OU-Threshold K=10,15",
        "Table 9 domain baselines: CNN-Transformer/IPCA K=0,1,3,5,10,15",
        "Table 6: six sequence/set models plus 5-layer NN and logistic regression",
        "Figures 3 and 13: ten evaluation subsets, 25 sequences x 2500 loans per subset",
        "Figure 14: no-refit, uniform-refit, 24-month-half-life refit",
        "Figures 15-16: Set-Sequence, gated Set-Sequence, NN, logistic yearly refits",
        "Figures 8-12 and 17: aligned returns, transition summaries and learned representations"
    ])
(ROOT / "configs/run_plan_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
