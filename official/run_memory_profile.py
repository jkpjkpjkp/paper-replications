"""
Peak GPU Memory Profiling for GADBench models.
Measures peak GPU memory allocation during a single training trial.
"""

import argparse
import time
import os
import warnings
import random
import numpy as np
import torch
import pandas as pd

warnings.filterwarnings("ignore")

from utils import Dataset, model_detector_dict, save_results


def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def profile_memory(model_name, dataset_name, device='cuda'):
    """Run a single trial and measure peak GPU memory."""
    set_seed(3407)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    data = Dataset(dataset_name)
    data.split(semi_supervised=0, trial_id=0)

    train_config = {
        'device': device,
        'epochs': 200,
        'patience': 50,
        'metric': 'AUPRC',
        'inductive': False,
        'seed': 3407,
    }
    model_config = {'model': model_name, 'lr': 0.01, 'drop_rate': 0}

    try:
        detector = model_detector_dict[model_name](train_config, model_config, data)
        test_score = detector.train()
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  {model_name} on {dataset_name}: Peak={peak_mem_mb:.0f} MB, "
              f"AUROC={test_score['AUROC']:.4f}")
    except Exception as e:
        peak_mem_mb = float('nan')
        print(f"  {model_name} on {dataset_name}: FAILED ({e})")
    finally:
        del detector, data
        torch.cuda.empty_cache()

    return peak_mem_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--models', type=str, default='GCN,BWGNN,SECGFD,FraudGNAM,GAGA,PMP')
    parser.add_argument('--datasets', type=str, default='yelp,amazon,weibo,reddit,tfinance,tolokers,questions')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    models = args.models.split(',')
    datasets = args.datasets.split(',')

    results = {}
    for model_name in models:
        results[model_name] = {}
        for dataset_name in datasets:
            print(f"\nProfiling {model_name} on {dataset_name}...")
            peak_mb = profile_memory(model_name, dataset_name, device)
            results[model_name][dataset_name] = peak_mb

    # Print table
    print("\n\n=== Peak GPU Memory (MB) ===")
    header = f"{'Method':<12} " + " ".join(f"{d:>10}" for d in datasets)
    print(header)
    print("-" * len(header))
    for model_name in models:
        row = f"{model_name:<12} "
        row += " ".join(f"{results[model_name].get(d, float('nan')):>10.0f}" for d in datasets)
        print(row)

    # Save to file
    df = pd.DataFrame(results).T
    df.to_csv('results/memory_profile.csv')
    print(f"\nSaved to results/memory_profile.csv")


if __name__ == '__main__':
    main()
