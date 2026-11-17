# FraudGNAM: Inherently Interpretable Spectral GNN for Graph Fraud Detection

Official code for the UAI 2026 paper
**"FraudGNAM: Inherently Interpretable Spectral GNN for Graph Fraud Detection"**
(Suan Lee, Jinho Kim).

FraudGNAM (**Fraud** **G**raph **N**eural **A**dditive **M**odel) is the first
*component-level inherently interpretable* spectral GNN for fraud detection. It
decomposes each fraud logit into a transparent scalar sum of auditable
components — per-feature contributions, pairwise interactions, multi-scale
spectral band responses, and explicit feature–spectrum interactions:

```
s(v) = β₀ + Σ_c α_c · ψ_c(v)
```

The decomposition *is* the model (not a post-hoc approximation), so component
contributions are exact at inference (LOCO MAE < 10⁻⁶). A deterministic,
theory-guided adaptive rule sets the spectral order, high-pass filter count, and
contrastive regularization from measurable graph statistics, eliminating
per-dataset hyperparameter search.

This repository is built on top of
[GADBench](https://github.com/squareRoot3/GADBench) (NeurIPS 2023) and reuses its
dataset loaders and evaluation protocol.

## Installation

```bash
# 1. PyTorch + DGL (CUDA 11.7 example — match to your CUDA version)
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c dglteam/label/cu117 dgl

# 2. Remaining Python dependencies
pip install -r requirements.txt
```

## Data

Datasets follow the GADBench format and are **not** included here. Download the
preprocessed bundle from the GADBench
[Google Drive link](https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing)
and unzip into `datasets/`:

```
datasets/
├── reddit/        # included upstream (no download needed)
├── amazon/  yelp/  weibo/  tfinance/  tolokers/  questions/  tsocial/
└── hetero/
```

`tfinance`/`tsocial` are large; `DGraph-Fin` and `Elliptic` require manual
download due to licensing (see the GADBench README). The paper uses seven
datasets: `yelp, amazon, weibo, reddit, tfinance, tolokers, questions`.

## Quick start

```bash
# Train + evaluate FraudGNAM vs the strongest baselines (10 trials)
python benchmark.py --trials 10 --models FraudGNAM-SECGFD-GAGA-PMP \
                    --datasets 2,3 --gpu 0
```

Datasets are indexed `0=reddit 1=weibo 2=amazon 3=yelp 4=tfinance 5=elliptic
6=tolokers 7=questions 8=dgraphfin 9=tsocial`. Models are registered in
`utils.py` (`model_detector_dict`).

## Reproducing the paper

Each result in the paper maps to a script below. Most experiment scripts accept
`--gpu`, `--datasets` (dataset *names*, e.g. `yelp,amazon`), and `--trials`.
Pre-computed per-trial outputs (≈410 trials) are provided in `results/`; the
scripts also read from and write to that directory.

| Paper item | Script |
|---|---|
| Table 1 — main results vs black-box baselines | `benchmark.py` |
| §5 / App — refined adaptive rule (R1/R2/R3 cutoffs) | `exp_refined_rule.py` |
| Table R1/R2 — matched-seed paired tests vs SEC-GFD | `exp_real_paired_stats.py`, `exp_secgfd_paired.py` |
| Table R3 — interpretable baseline (GNAN) | `exp_gnan.py`, `exp_gnan_paired.py` |
| Table R4 — threshold sensitivity (14-cell grid) | `run_threshold_sensitivity.py`, `exp_sensitivity_grid.py` |
| App — leave-one-dataset-out configuration | `exp_refined_lodo.py` |
| Table R5 — NCE component gradient flow (Gini) | `exp_nce_flow.py` |
| Table R6 — per-variant memory profile | `run_memory_profile.py`, `exp_memory.py` |
| Ablation (no_highpass / no_nce / order2 / …) | `run_ablation.py` |
| Faithfulness (LOCO exactness, top-k sufficiency) | `evaluate_faithfulness.py` |
| Perturbation / adversarial stability | `exp_perturbation.py`, `exp_perturbation_multi.py` |
| Collapsed-MLP capacity control | `exp_collapsed_mlp.py` |
| Filter-response figures | `exp_filter_response.py`, `plot_filter_response.py` |
| Explanation comparison (vs GNAN) | `exp_explanation_compare.py` |
| Per-feature shape functions (Yelp) | `exp_yelp_shape_functions.py` |
| Label-noise robustness | `exp_label_noise.py` |
| Component interpretability extraction | `extract_interpretability.py` |
| Post-hoc baseline (GNNExplainer) | `evaluate_gnnexplainer.py` |

Aggregate/format collected runs with `aggregate_results.py` and
`final_rebuttal_report.py`.

Example:

```bash
python run_ablation.py --ablation all --datasets all --gpu 0
python evaluate_faithfulness.py --datasets amazon,yelp --gpu 0 --trial 0
python run_threshold_sensitivity.py --sweep h_order --datasets all --gpu 0 --trials 10
```

## Repository structure

```
models/
├── gnn.py                 # FraudGNAM (class at L1393) + all baseline GNNs
├── fraudgnam_detector.py  # training / evaluation wrapper + NCE loss
├── secgfd_detector.py  gnan_detector.py  gaga_detector.py  pmp_detector.py
├── detector.py            # BaseDetector
└── attention.py
utils.py                   # Dataset loader + model registry
benchmark.py               # main benchmark entrypoint
exp_*.py / run_*.py / evaluate_*.py   # paper experiment scripts
results/                   # pre-computed per-trial outputs (≈410 trials)
                           # (figures are written to rebuttal_figures/ at run time)
```

## Citation

```bibtex
@inproceedings{lee2026fraudgnam,
  title     = {FraudGNAM: Inherently Interpretable Spectral GNN for Graph Fraud Detection},
  author    = {Lee, Suan and Kim, Jinho},
  booktitle = {Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2026}
}
```

## License

Released under the MIT License (see [LICENSE](LICENSE)). Dataset loaders and the
evaluation harness derive from [GADBench](https://github.com/squareRoot3/GADBench),
also MIT-licensed.
