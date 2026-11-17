"""Fixed-seed rolling tests and frozen-checkpoint extension; no test-based selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hermes.evaluation import predict_checkpoint,evaluated_artifacts
from hermes.metrics import PORTFOLIO_PROTOCOL


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    manifest=json.loads((ROOT/'data/recent/manifest.json').read_text())
    result_root=ROOT/'results/recent';result_root.mkdir(exist_ok=True)
    frozen=result_root/'frozen-no_skip-seed0';frozen.mkdir(exist_ok=True)
    if not (frozen/'metrics.json').exists():
        data=np.load(ROOT/'data/recent/frozen.npz')
        start=manifest['frozen_test_start'];end=len(data['dates'])
        prediction,saved=predict_checkpoint(ROOT/'results/sp500-no_skip-seed0/best.pt',data,start,end,args.device)
        metrics,arrays=evaluated_artifacts(prediction,data['returns'][start:end],data['valid'][start:end],
                                         data['dates'][start:end],data['tickers'],5,data['pnl_returns'][start:end])
        np.savez_compressed(frozen/'test_predictions.npz',**arrays)
        result={'protocol':'frozen original checkpoint; CRSP vendor extension','seed':0,
                'source_checkpoint':'results/sp500-no_skip-seed0/best.pt','source_best_epoch':saved['epoch'],
                'test_dates':[str(data['dates'][start]),str(data['dates'][-1])],
                'evaluation_protocol':PORTFOLIO_PROTOCOL,'test':metrics}
        (frozen/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        print('frozen',json.dumps(metrics),flush=True)
    for fold in manifest['folds']:
        for seed in [0,1,2]:
            label=f'rolling-{fold["year"]}-no_skip-seed{seed}';dest=result_root/label
            if (dest/'metrics.json').exists():
                print('completed, skip:',label,flush=True);continue
            cmd=[sys.executable,str(ROOT/'scripts/train.py'),'--data',fold['data'],'--output',str(dest),
                 '--seed',str(seed),'--ablation','no_skip','--train-end',str(fold['train_end']),
                 '--val-end',str(fold['val_end']),'--device',args.device]
            (result_root/f'{label}.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            print('running:',label,flush=True)
            with (result_root/f'{label}.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    main()
