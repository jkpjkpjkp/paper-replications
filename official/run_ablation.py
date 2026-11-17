"""
Ablation study for FraudGNAM v2 paper.
Runs 6 ablation variants on 4 datasets (Yelp, Amazon, Weibo, Reddit) with 10 trials.

Usage:
    python run_ablation.py --ablation no_highpass --datasets yelp --gpu 0
    python run_ablation.py --ablation no_nce --datasets yelp,amazon,weibo,reddit --gpu 1
    python run_ablation.py --ablation all --datasets all --gpu 0
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

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class AblationFraudGNAMDetector(FraudGNAMDetector):
    """FraudGNAM with post-init overrides for ablation."""

    _ablation_overrides = {}  # Class-level dict for overrides

    def __init__(self, train_config, model_config, data):
        # Let parent do full adaptive config
        super().__init__(train_config, model_config, data)

        overrides = AblationFraudGNAMDetector._ablation_overrides
        if not overrides:
            return

        # Apply overrides - need to rebuild model if model_config changed
        need_rebuild = False
        for key, val in overrides.items():
            if key in ['nce_lambda']:
                self.nce_lambda = val
            elif key in ['focal_gamma']:
                self.focal_gamma = val
            elif key in ['grad_clip']:
                self.grad_clip = val
            elif key in ['warmup_epochs']:
                self.warmup_epochs = val
            else:
                # Model architecture change
                if model_config.get(key) != val:
                    model_config[key] = val
                    need_rebuild = True

        if need_rebuild:
            print(f"  [ABLATION] Overrides: {overrides}")
            # When NCE disabled, also disable related training params
            if overrides.get('use_nce') == False:
                self.nce_lambda = 0.0
                self.grad_clip = 2.0
                self.warmup_epochs = 0
            self.model = type(self.model)(**model_config).to(train_config['device'])


ABLATION_CONFIGS = {
    'full': {},  # No overrides = full model
    'no_highpass': {'num_highpass': 0},
    'no_nce': {'use_nce': False},
    'no_efi': {'use_efi': False},
    'no_pairwise': {'use_pairwise': False, 'num_pairwise': 0},
    'order2': {'order': 2},
    'order4': {'order': 4},
}

DATASET_MAP = {
    'yelp': 'yelp',
    'amazon': 'amazon',
    'weibo': 'weibo',
    'reddit': 'reddit',
}

def run_ablation(ablation_name, dataset_names, gpu, trials=10):
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

    overrides = ABLATION_CONFIGS[ablation_name]
    AblationFraudGNAMDetector._ablation_overrides = overrides

    columns = ['name']
    for ds in dataset_names:
        for metric in ['AUROC mean', 'AUROC std', 'AUPRC mean', 'AUPRC std',
                       'RecK mean', 'RecK std', 'Time']:
            columns.append(f'{ds}-{metric}')

    results = pandas.DataFrame(columns=columns)
    model_result = {'name': f'FraudGNAM_{ablation_name}'}

    for dataset_name in dataset_names:
        auc_list, pre_list, rec_list = [], [], []
        time_cost = 0

        for t in range(trials):
            torch.cuda.empty_cache()
            print(f"\n=== Dataset {dataset_name}, Ablation {ablation_name}, Trial {t} ===")

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

            detector = AblationFraudGNAMDetector(train_config, model_config, data)
            st = time.time()
            try:
                test_score = detector.train()
                auc_list.append(test_score['AUROC'])
                pre_list.append(test_score['AUPRC'])
                rec_list.append(test_score['RecK'])
            except Exception as e:
                print(f"ERROR in trial {t}: {e}")
                auc_list.append(0.5)
                pre_list.append(0.0)
                rec_list.append(0.0)
            ed = time.time()
            time_cost += ed - st
            del detector, data

        model_result[f'{dataset_name}-AUROC mean'] = np.mean(auc_list)
        model_result[f'{dataset_name}-AUROC std'] = np.std(auc_list)
        model_result[f'{dataset_name}-AUPRC mean'] = np.mean(pre_list)
        model_result[f'{dataset_name}-AUPRC std'] = np.std(pre_list)
        model_result[f'{dataset_name}-RecK mean'] = np.mean(rec_list)
        model_result[f'{dataset_name}-RecK std'] = np.std(rec_list)
        model_result[f'{dataset_name}-Time'] = time_cost / trials

        print(f"\n--- {dataset_name} {ablation_name} ---")
        print(f"  AUROC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
        print(f"  AUPRC: {np.mean(pre_list):.4f} ± {np.std(pre_list):.4f}")
        print(f"  RecK:  {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}")

    model_result_df = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result_df])
    file_id = save_results(results, None)
    print(f"\nResults saved to results/{file_id}.xlsx")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default='full',
                       help='Ablation variant: full, no_highpass, no_nce, no_efi, no_pairwise, order2, order4, or "all"')
    parser.add_argument('--datasets', type=str, default='yelp',
                       help='Comma-separated dataset names or "all"')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()

    if args.datasets == 'all':
        dataset_names = list(DATASET_MAP.values())
    else:
        dataset_names = [DATASET_MAP.get(d, d) for d in args.datasets.split(',')]

    if args.ablation == 'all':
        ablations = list(ABLATION_CONFIGS.keys())
    else:
        ablations = [args.ablation]

    for abl in ablations:
        print(f"\n{'='*60}")
        print(f"Running ablation: {abl}")
        print(f"Datasets: {dataset_names}")
        print(f"{'='*60}")
        run_ablation(abl, dataset_names, args.gpu, args.trials)
