"""CZ202 monthly adapter for the pinned TRA trainer and calendar-safe dataset."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from qlib.utils import init_instance_by_config

ROOT=Path('/home/jkp/data')
HERE=Path(__file__).resolve().parent
BASE=ROOT/'ordered-reproductions'
OUT=HERE/'results/TRA'
CODE=BASE/'sources/TRA-qlib/examples/benchmarks/TRA'

def save(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/name; t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,default=str));t.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((BASE/'data/cz-monthly/manifest.json').read_text())
    names=manifest['features'];assert len(names)==202
    panel=pd.read_parquet(BASE/'data/cz-monthly/panel.parquet')
    assert panel.lagged_market_cap.ge(panel.nyse20).all()
    dates=pd.to_datetime(panel.return_month.astype(str),format='%Y%m')+pd.offsets.MonthEnd(0)
    idx=pd.MultiIndex.from_arrays([dates,panel.permno.astype(str)],names=['datetime','instrument'])
    features=pd.DataFrame(panel[names].to_numpy(dtype=np.float32),index=idx).sort_index()
    y=panel.groupby('return_month').target_return.transform(lambda x:(x-x.mean())/(x.std(ddof=0)+1e-12))
    assert y[panel.return_month.ge(200901)].notna().all()
    # Pre-training context labels never enter fitting, routing memory, or evaluation.
    labels=pd.DataFrame({'label':y.fillna(0).to_numpy(dtype=np.float32)},index=idx).sort_index()
    assert np.isfinite(features).all().all() and np.isfinite(labels).all().all()
    features.to_pickle(OUT/'features.pkl');labels.to_pickle(OUT/'labels.pkl')
    save('protocol.json',dict(scope='User-authorized CZ adaptation; original-paper numerical gate superseded for this run',
        features=names,train=[200901,201912],valid=[202001,202112],test=[202201,202512],
        sequence_months=12,horizon_months=1,label='Next-month return cross-sectional z-score',
        timing='Row return_month m contains CZ/mcap from m-1; only losses for months <m enter routing',
        model='Released ALSTM+TRA, hidden256 x2, initialization3 / router10, seed1000, released 500-epoch cap and patience20',
        source=str(CODE/'src/model.py'),dataset=str(BASE/'sources/tra_dataset_calendar_corrected.py'),
        universe='All eligible US common stocks; lagged NYSE20',cost_bps=10,
        adaptation='202 inputs; 12 monthly histories instead of 60 daily; next-month instead of 20-day labels',
        source_sha256=hashlib.sha256((CODE/'src/model.py').read_bytes()).hexdigest()))
    for phase,filename in [('initialization','config_alstm_tra_init.yaml'),('tra','config_alstm_tra.yaml')]:
        if (OUT/phase/'info.json').exists() and not args.preflight:continue
        task=yaml.safe_load((CODE/'configs'/filename).read_text())['task']
        for key in ['model','dataset']:task[key].pop('trusted',None)
        task['model']['module_path']=str(CODE/'src/model.py')
        task['dataset']['module_path']=str(BASE/'sources/tra_dataset_calendar_corrected.py')
        dk=task['dataset']['kwargs'];mk=task['model']['kwargs']
        dk['handler']['kwargs']['data_loader']['kwargs']['config'].update(feature=str(OUT/'features.pkl'),label=str(OUT/'labels.pkl'))
        dk.update(seq_len=12,horizon=1,pin_memory=False,segments=dict(train=['2009-01-01','2019-12-31'],valid=['2020-01-01','2021-12-31'],test=['2022-01-01','2025-12-31']))
        mk['model_config']['input_size']=202;mk['logdir']=str(OUT/phase)
        if phase=='tra':mk['model_init_state']=str(OUT/'initialization/model.bin')
        dataset=init_instance_by_config(task['dataset'])
        model=init_instance_by_config(task['model'])
        if args.preflight:
            ds=dataset.prepare('valid');ds.eval();batch=next(iter(ds))
            # Input and history-memory construction must not depend on query labels.
            a=batch['data'].clone();ix=batch['index'].cpu().numpy()
            saved=ds._label.iloc[ix].copy();ds._label.iloc[ix]=999
            b=next(iter(ds))['data'];ds._label.iloc[ix]=saved
            assert torch.equal(a,b)
            assert torch.count_nonzero(a[:,-1,-3:])==0
            assert a.shape[1:]==(12,205)
            assert all(dataset.restore_index(ix).get_level_values(1).year==2020)
            save('preflight.json',dict(passed=True,shape=list(a.shape),future_label_perturbation_invariant=True,query_loss_memory_zero=True,features=202))
            return
        original=model.train_epoch;epoch=[0]
        def train(ds):
            save('status.json',dict(stage=phase,epoch=epoch[0],pid=os.getpid(),updated=time.time()))
            result=original(ds);epoch[0]+=1;return result
        model.train_epoch=train
        model.fit(dataset)
        del model,dataset
        torch.cuda.empty_cache()
    pred=pd.read_pickle(OUT/'tra/pred.pkl').reset_index()
    pred['return_month']=pred.datetime.dt.year*100+pred.datetime.dt.month
    pred['permno']=pred.instrument.astype(int)
    pred[['return_month','permno','score']].to_csv(OUT/'scores.csv',index=False)
    import subprocess
    if not (OUT/'evaluation/report.json').exists():
        subprocess.run([sys.executable,str(BASE/'scripts/evaluate_cz.py'),'--scores',str(OUT/'scores.csv'),'--output',str(OUT/'evaluation')],check=True)
    save('status.json',dict(stage='complete',updated=time.time()))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('status.json',dict(stage='failed',error=repr(exc)));raise
