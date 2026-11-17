"""
Threshold sensitivity analysis for FraudGNAM paper.
Sweeps homophily thresholds to show robustness of configuration rules.

Two sweeps:
1. h_order ∈ {0.80, 0.85, 0.90}: boundary for order=4 vs order=2/3
2. h_nce ∈ {0.90, 0.95, 1.00}: boundary for NCE/highpass activation

Usage:
    python run_threshold_sensitivity.py --sweep h_order --gpu 0
    python run_threshold_sensitivity.py --sweep h_nce --gpu 1
    python run_threshold_sensitivity.py --sweep all --gpu 0
"""
import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import pandas
import warnings
warnings.filterwarnings("ignore")

from utils import Dataset, save_results
from models.fraudgnam_detector import FraudGNAMDetector

seed_list = list(range(3407, 10000, 10))

ALL_DATASETS = ['yelp', 'amazon', 'weibo', 'reddit', 'tfinance', 'tolokers', 'questions']


def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class ThresholdFraudGNAMDetector(FraudGNAMDetector):
    """FraudGNAM with overridable homophily thresholds for sensitivity analysis."""

    _threshold_overrides = {}  # {'h_order': float, 'h_nce': float}

    def __init__(self, train_config, model_config, data):
        # Store overrides before parent init
        self._overrides = dict(ThresholdFraudGNAMDetector._threshold_overrides)

        # Monkey-patch the threshold checks in parent __init__
        # We need to intercept the homophily-based decisions
        super().__init__(train_config, model_config, data)

        overrides = self._overrides
        if not overrides:
            return

        need_rebuild = False

        # Override h_order: changes spectral order decision
        if 'h_order' in overrides:
            h_order = overrides['h_order']
            homophily = self.homophily
            graph = self.source_graph
            num_nodes = graph.num_nodes()
            num_edges = graph.num_edges()
            avg_degree = num_edges / num_nodes
            sparse_graph = avg_degree < 30

            if homophily < h_order:
                new_order = 4
            elif sparse_graph:
                new_order = 3
            else:
                new_order = 2

            if model_config.get('order') != new_order:
                model_config['order'] = new_order
                need_rebuild = True

        # Override h_nce: changes NCE and highpass activation threshold
        if 'h_nce' in overrides:
            h_nce = overrides['h_nce']
            homophily = self.homophily

            labels_np = self.labels.cpu().numpy()
            fraud_ratio = labels_np.mean()
            train_fraud_count = labels_np[self.train_mask.cpu().numpy()].sum()

            # Highpass: use h_nce as the upper threshold (h < h_nce enables HP)
            # Original: h < 0.85 -> 2, h < 0.95 -> 1, else -> 0
            # Shifted: h < (h_nce - 0.10) -> 2, h < h_nce -> 1, else -> 0
            h_hp_low = h_nce - 0.10  # e.g., 0.90 -> 0.80, 0.95 -> 0.85, 1.00 -> 0.90
            if homophily < h_hp_low:
                new_num_hp = 2
            elif homophily < h_nce:
                new_num_hp = 1
            else:
                new_num_hp = 0

            if model_config.get('num_highpass') != new_num_hp:
                model_config['num_highpass'] = new_num_hp
                need_rebuild = True

            # NCE: use h_nce threshold
            if train_fraud_count >= 500 and homophily < h_nce:
                new_use_nce = True
                new_nce_lambda = 0.05 if fraud_ratio < 0.05 else 0.1
            else:
                new_use_nce = False
                new_nce_lambda = 0.0

            if model_config.get('use_nce') != new_use_nce:
                model_config['use_nce'] = new_use_nce
                need_rebuild = True
            self.nce_lambda = new_nce_lambda

            # Adjust warmup/grad_clip based on new NCE/HP state
            if new_use_nce and new_num_hp > 0:
                self.warmup_epochs = 10
                self.grad_clip = 1.5
            elif new_use_nce:
                self.grad_clip = 1.5
                self.warmup_epochs = 0
            else:
                self.grad_clip = 2.0
                self.warmup_epochs = 0

        if need_rebuild:
            print(f"  [THRESHOLD] Overrides: {overrides}, rebuilding model")
            self.model = type(self.model)(**model_config).to(train_config['device'])
        else:
            print(f"  [THRESHOLD] Overrides: {overrides}, no config change for this dataset")


# Sweep configurations
SWEEP_CONFIGS = {
    'h_order': [
        {'name': 'h_order=0.80', 'h_order': 0.80},
        {'name': 'h_order=0.85', 'h_order': 0.85},  # default
        {'name': 'h_order=0.90', 'h_order': 0.90},
    ],
    'h_nce': [
        {'name': 'h_nce=0.90', 'h_nce': 0.90},
        {'name': 'h_nce=0.95', 'h_nce': 0.95},  # default
        {'name': 'h_nce=1.00', 'h_nce': 1.00},
    ],
}


def run_sweep(sweep_name, dataset_names, gpu, trials=10):
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    configs = SWEEP_CONFIGS[sweep_name]

    all_results = []

    for config in configs:
        config_name = config['name']
        overrides = {k: v for k, v in config.items() if k != 'name'}
        ThresholdFraudGNAMDetector._threshold_overrides = overrides

        for dataset_name in dataset_names:
            auc_list, pre_list, rec_list = [], [], []
            time_cost = 0

            for t in range(trials):
                torch.cuda.empty_cache()
                print(f"\n=== {config_name}, {dataset_name}, Trial {t} ===")

                data = Dataset(dataset_name)
                data.split(0, t)
                seed = seed_list[t]
                set_seed(seed)

                train_config = {
                    'device': device,
                    'epochs': 200,
                    'patience': 50,
                    'metric': 'AUPRC',
                    'inductive': False,
                    'seed': seed,
                }
                model_config = {'model': 'FraudGNAM', 'lr': 0.01, 'drop_rate': 0}

                detector = ThresholdFraudGNAMDetector(train_config, model_config, data)
                st = time.time()
                try:
                    test_score = detector.train()
                    auc_list.append(test_score['AUROC'])
                    pre_list.append(test_score['AUPRC'])
                    rec_list.append(test_score['RecK'])
                except Exception as e:
                    print(f"ERROR in trial {t}: {e}")
                    import traceback; traceback.print_exc()
                    auc_list.append(0.5)
                    pre_list.append(0.0)
                    rec_list.append(0.0)
                ed = time.time()
                time_cost += ed - st
                del detector, data

            result = {
                'config': config_name,
                'dataset': dataset_name,
                'AUROC_mean': np.mean(auc_list),
                'AUROC_std': np.std(auc_list),
                'AUPRC_mean': np.mean(pre_list),
                'AUPRC_std': np.std(pre_list),
                'RecK_mean': np.mean(rec_list),
                'RecK_std': np.std(rec_list),
                'time_per_trial': time_cost / trials,
            }
            all_results.append(result)

            print(f"\n--- {config_name} / {dataset_name} ---")
            print(f"  AUROC: {np.mean(auc_list):.4f} +/- {np.std(auc_list):.4f}")
            print(f"  AUPRC: {np.mean(pre_list):.4f} +/- {np.std(pre_list):.4f}")
            print(f"  RecK:  {np.mean(rec_list):.4f} +/- {np.std(rec_list):.4f}")

    # Save as DataFrame
    df = pandas.DataFrame(all_results)
    file_id = save_results(df, None)
    print(f"\nResults saved to results/{file_id}.xlsx")

    # Also print pivot table for easy reading
    print(f"\n{'='*60}")
    print(f"AUROC Summary for sweep: {sweep_name}")
    print(f"{'='*60}")
    pivot = df.pivot(index='config', columns='dataset', values='AUROC_mean')
    print(pivot.to_string(float_format='%.4f'))

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, default='h_order',
                       help='Sweep type: h_order, h_nce, or all')
    parser.add_argument('--datasets', type=str, default='all',
                       help='Comma-separated dataset names or "all"')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()

    if args.datasets == 'all':
        dataset_names = ALL_DATASETS
    else:
        dataset_names = [d.strip() for d in args.datasets.split(',')]

    if args.sweep == 'all':
        sweeps = list(SWEEP_CONFIGS.keys())
    else:
        sweeps = [args.sweep]

    for sweep in sweeps:
        print(f"\n{'='*60}")
        print(f"Running sweep: {sweep}")
        print(f"Datasets: {dataset_names}")
        print(f"Trials: {args.trials}")
        print(f"{'='*60}")
        run_sweep(sweep, dataset_names, args.gpu, args.trials)
