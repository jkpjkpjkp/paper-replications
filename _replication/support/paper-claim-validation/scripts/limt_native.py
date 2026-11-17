#!/usr/bin/env python3
"""Replay original-market LiMT with the exact hash-recorded model source."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'limt-reproduction'
OUT = ROOT/'paper-claim-validation/results'
sys.path[:0] = [str(PROJECT), str(PROJECT/'scripts')]
from limt.data import Panel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    historical = ROOT/'paper-claim-validation/evidence/limt_original_model.py'
    trainer = ROOT/'paper-claim-validation/evidence/limt_initial_train.py'
    module = runpy.run_path(str(historical))
    module['LiMT'].__init__.__globals__['ROOT'] = PROJECT
    # Exact source identity: current joint model prefix is byte-identical to
    # training time. The later appended class cannot change these definitions.
    current_prefix = (PROJECT/'limt/model.py').read_text().split('class SingleTaskLiMT')[0].rstrip()+'\n'
    assert current_prefix.encode() == historical.read_bytes()
    old_train = trainer.read_text()
    corrected = old_train.replace('torch.set_rng_state(ck["rng"])', 'torch.set_rng_state(ck["rng"].cpu())')
    corrected = corrected.replace('torch.cuda.set_rng_state(ck["cuda_rng"],device)',
                                  'torch.cuda.set_rng_state(ck["cuda_rng"].cpu(),device)')
    assert corrected == (PROJECT/'scripts/train.py').read_text()
    rows, checks = [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for universe in ['csi300','csi500']:
        panel = Panel(PROJECT/'data'/universe)
        for seed in range(3):
            folder = PROJECT/f'results/main/{universe}/seed{seed}'
            provenance = json.loads((folder/'provenance.json').read_text())
            assert provenance['model_code_sha256'] in {sha(historical),sha(PROJECT/'limt/model.py')}
            assert provenance['trainer_sha256'] in {sha(trainer),sha(PROJECT/'scripts/train.py')}
            assert provenance['best_sha256'] == sha(folder/'best.pt')
            assert provenance['data_manifest_sha256'] == sha(panel.folder/'manifest.json')
            predictions = pd.read_parquet(folder/'test_predictions.parquet')
            saved = json.loads((folder/'metrics.json').read_text())['test']['return']
            assert not predictions.duplicated(['date','symbol']).any()
            groups = dict(tuple(predictions.groupby('date')))
            days = panel.split('test')
            assert sorted(groups) == list(panel.dates[days])
            ic, rankic = [], []
            for t in days:
                f = groups[panel.dates[t]]
                ids, _, _, target = panel.day(t)
                np.testing.assert_array_equal(f.symbol.to_numpy(), panel.symbols[ids])
                for k, name in enumerate(['return','volume','volatility']):
                    np.testing.assert_allclose(f[f'label_{name}'],target[:,k],rtol=0,atol=0,equal_nan=True)
                ic.append(f.pred_return.corr(f.label_return))
                rankic.append(f.pred_return.corr(f.label_return,method='spearman'))
            observed = dict(IC=np.nanmean(ic),ICIR=np.nanmean(ic)/np.nanstd(ic,ddof=1),
                            RankIC=np.nanmean(rankic),RankICIR=np.nanmean(rankic)/np.nanstd(rankic,ddof=1))
            for name,value in observed.items():
                np.testing.assert_allclose(value,saved[name],rtol=1e-7,atol=1e-9)
            cfg = json.loads((folder/'config.json').read_text())
            model = module['LiMT'](experts=cfg['experts'],dropout=cfg['dropout'],variant=cfg['variant']).to(device)
            model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True))
            model.eval()
            replayed=[]
            for t in days[[0,len(days)//2,-1]]:
                ids, x, _, _ = panel.day(t)
                with torch.no_grad():
                    pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
                f = groups[panel.dates[t]]
                np.testing.assert_allclose(pred,f[['pred_return','pred_volume','pred_volatility']],rtol=1e-5,atol=2e-6)
                replayed.append(str(panel.dates[t]))
            rows.append(dict(universe=universe,seed=seed,**observed))
            checks.append(dict(universe=universe,seed=seed,forecasts=len(predictions),days=len(days),
                               checkpoint_replay_dates=replayed,device=device,
                               recorded_model_sha256=provenance['model_code_sha256'],
                               recorded_trainer_sha256=provenance['trainer_sha256']))
    pd.DataFrame(rows).to_csv(OUT/'limt_native_metrics.csv',index=False)
    result=dict(passed=True,checks=checks,exact_historical_model_source_sha256=sha(historical),
                exact_historical_trainer_source_sha256=sha(trainer),
                current_model_change='Only appended independent SingleTaskLiMT class',
                current_trainer_change='Only move restored RNG-state tensors to CPU',
                published_numbers_replicated=False,
                limitation='Alternative 2010-start data; paper trains from 2008. APO engine reconstructed.')
    (OUT/'limt_native_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
