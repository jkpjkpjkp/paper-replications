"""Run the author's training loop on original data with an explicit config.

Usage: author_run.py --data-root PATH [--config resolved-config.json]
Without --config this is a download/hash preflight and executes no pickle.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--config',type=Path);args=parser.parse_args()
    files=json.loads((HERE/'author-files.json').read_text())['files'];receipt=[];missing=[]
    for rec in files:
        # Accept either the share-root directory or its enclosing AAAI folder.
        relative=Path(*Path(rec['path']).parts[-2:]);p=args.data_root/relative
        if not p.exists():missing.append(str(p));continue
        if p.stat().st_size!=rec['size']:raise ValueError(f'Incomplete/wrong-size author file: {p}')
        receipt.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)))
    result=dict(status='missing_original_files' if missing else 'download_sizes_and_hashes_checked',files=receipt,missing=missing)
    (HERE/'intake.json').write_text(json.dumps(result,indent=2)+'\n')
    if missing:
        print(json.dumps(result,indent=2));raise SystemExit(2)
    if not args.config:print(json.dumps(result,indent=2));return
    cfg=json.loads(args.config.read_text())
    for key in ['universe','seed','model_class','model_kwargs','training_kwargs','selection_provenance','data_semantics']:
        if key not in cfg:raise ValueError(f'Unresolved configuration field: {key}; no silent defaults')
    import pickle,numpy as np,pandas as pd,torch
    # Only the retrieved author datasets are loaded, after the size/hash intake.
    folder=args.data_root/f'R_Vol2_Vola2_{cfg["universe"]}_Alpha158'
    loaders={}
    for split in ['train','valid','test']:
        with (folder/f'dl_{split}.pkl').open('rb') as f:loaders[split]=pickle.load(f)
        if not hasattr(loaders[split],'get_index'):raise ValueError('Inspect author object schema before adapting it')
        ix=loaders[split].get_index();dates=pd.to_datetime(ix.get_level_values('datetime'))
        expected={'train':('2008-01-01','2014-12-31'),'valid':('2015-01-01','2016-12-31'),'test':('2017-01-01','2020-08-01')}[split]
        if dates.min()<pd.Timestamp(expected[0]) or dates.max()>pd.Timestamp(expected[1]):raise ValueError(f'Unexpected {split} dates')
        if split=='train' and dates.min().year!=2008:raise ValueError('Original 2008-start training coverage not present')
    original=module('base_model',ROOT/'upstream/base_model.py')
    tree=ast.parse((ROOT/'upstream/limt.py').read_text())
    # Remove only the unresolved unused Embed import. All selected class bodies
    # and the original trainer/loss/selection functions remain author code.
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='Embed')]
    scope={'__name__':'author_limt'}
    exec(compile(tree,str(ROOT/'upstream/limt.py'),'exec'),scope)
    np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    out=HERE/'author-results'/cfg['universe']/f'seed{cfg["seed"]}'
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    trainargs=dict(cfg['training_kwargs'],seed=cfg['seed'],save_path=str(out)+'/',save_prefix='model',log_save_path=str(out)+'/')
    runner=original.SequenceModel(**trainargs)
    runner.model=scope[cfg['model_class']](**cfg['model_kwargs'])
    runner.init_model()
    runner.fit(loaders['train'],loaders['valid'],None)
    for split in ['valid','test']:
        predictions,metrics=runner.predict(loaders[split],None,runner.mode,cfg['training_kwargs']['loss_type'])
        predictions.to_pickle(out/f'{split}_predictions.pkl')
        (out/f'{split}_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(data=receipt,config_sha256=sha(args.config),
        author_model_sha256=sha(ROOT/'upstream/limt.py'),trainer_sha256=sha(ROOT/'upstream/base_model.py'),
        runner_sha256=sha(Path(__file__)),torch=torch.__version__,
        note='Original author train/loss/checkpoint rules; no test evaluation during training. Exact published-config identity depends on supplied selection provenance.'),indent=2)+'\n')


if __name__=='__main__':main()
