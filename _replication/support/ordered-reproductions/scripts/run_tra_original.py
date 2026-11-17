"""Run the original ALSTM initialization and TRA experiment with the released data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import qlib
from qlib.utils import init_instance_by_config

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'sources/TRA-qlib'
CODE=REPO/'examples/benchmarks/TRA'
OUT=ROOT/'results/tra-original'


def save(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    tmp=OUT/(name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False))
    tmp.replace(OUT/name)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists() and not args.preflight:
        raise RuntimeError('Existing complete experiment; refusing overwrite')
    torch.set_num_threads(2)
    assert torch.cuda.is_available(),'This full TRA experiment requires the prepared CUDA runtime'
    torch.backends.cudnn.benchmark=False
    # No shape, batch, training budget, loss, or precision changes from the released YAMLs.
    started=time.monotonic()
    provenance={'source_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
                'seed':1000,'torch':torch.__version__,'qlib':qlib.__version__,
                'runner_sha256':sha(__file__),
                'data_sha256':{p.name:sha(p) for p in (ROOT/'data/TRA/kdd21').glob('*.pkl')},
                'scope':'One ALSTM+TRA experiment, including the required initialization phase.',
                'limitations':['Released paper reports qlib==0.7.1. This compatibility runtime uses the pinned author Qlib fork and PyTorch 2.4.1 CUDA.',
                               'Paper table is a multi-seed aggregate; seed1000 is one complete experiment, not that aggregate.']}
    assert set(provenance['data_sha256'])=={'feature.pkl','label.pkl','ret.pkl'}
    save('protocol.json',provenance)
    os.chdir(CODE)
    qlib.init(provider_uri=str(ROOT/'data/crowd_data'),region='cn',kernels=2,
              joblib_backend='threading',expression_cache=None,dataset_cache=None)
    phases=[]
    for stage,filename in [('initialization','config_alstm_tra_init.yaml'),('tra','config_alstm_tra.yaml')]:
        cfg=yaml.safe_load((CODE/'configs'/filename).read_text())
        task=cfg['task']
        data_config=task['dataset']['kwargs']['handler']['kwargs']['data_loader']['kwargs']['config']
        data_config.update(feature=str(ROOT/'data/TRA/kdd21/feature.pkl'),label=str(ROOT/'data/TRA/kdd21/label.pkl'))
        for k in ['model','dataset']:
            task[k]['module_path']=str(CODE/task[k]['module_path'])
            task[k].pop('trusted',None) # Absent in the older Qlib config schema; source modules are pinned above.
        task['model']['kwargs']['logdir']=str(OUT/stage)
        if stage=='tra':task['model']['kwargs']['model_init_state']=str(OUT/'initialization/model.bin')
        save(stage+'-config.json',cfg)
        if args.preflight and stage=='tra':break
        save('status.json',dict(stage=stage,phase='loading',wall_seconds=time.monotonic()-started))
        dataset=init_instance_by_config(task['dataset'])
        model=init_instance_by_config(task['model'])
        if args.preflight:
            batch=next(iter(dataset.prepare('train')))
            save('data-preflight.json',dict(rows=len(dataset._index),features=dataset._data.shape[1],
                 batch_shapes={k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')},
                 source_files=provenance['data_sha256']))
            save('status.json',dict(stage='data_preflight_complete'))
            return
        original_epoch=model.train_epoch
        epoch=[0]
        def logged_epoch(data_set):
            save('status.json',dict(stage=stage,phase='training',epoch=epoch[0],wall_seconds=time.monotonic()-started))
            result=original_epoch(data_set)
            epoch[0]+=1
            return result
        model.train_epoch=logged_epoch
        model.fit(dataset)
        phases.append(dict(stage=stage,epochs=epoch[0]))
        del model,dataset
        torch.cuda.empty_cache()
    info=json.loads((OUT/'tra/info.json').read_text())
    pred=pd.read_pickle(OUT/'tra/pred.pkl')
    assert not pred.index.duplicated().any()
    def daily_metrics(g):
        a=g.score.rank(method='average',pct=True)
        b=g.label.rank(method='average',pct=True)
        return pd.Series(dict(MSE=float(((a-b)**2).mean()),MAE=float(abs(a-b).mean()),IC=float(a.corr(b))))
    daily=pred.groupby(level='datetime').apply(daily_metrics)
    daily.to_csv(OUT/'independent-daily-metrics.csv')
    independent={n:float(daily[n].mean()) for n in ['MSE','MAE','IC']}
    independent['ICIR']=float(daily.IC.mean()/daily.IC.std(ddof=1))
    for name,value in independent.items():
        assert np.isclose(value,info['metric'][name],rtol=1e-6,atol=1e-8),(name,value,info['metric'][name])
    target=dict(MSE=.157,MAE=.318,IC=.059,ICIR=.460)
    save('summary.json',dict(status='experiment_complete_pending_fidelity_review',
                            phases=phases,source_report=info,wall_seconds=time.monotonic()-started,
                            independently_verified=independent,
                            comparison={n:dict(paper=v,run=independent[n],difference=independent[n]-v) for n,v in target.items()},
                            metric_definition='Per-day percentile-rank errors and Spearman IC, matching original TRA evaluate()',
                            test_dates=[str(daily.index.min()),str(daily.index.max())],trading_days=len(daily),
                            full_cz_authorized_by_gate=False))
    save('status.json',dict(stage='experiment_complete_pending_fidelity_review'))


if __name__=='__main__':
    try:main()
    except Exception as e:
        save('status.json',dict(stage='failed',error=repr(e)))
        raise
