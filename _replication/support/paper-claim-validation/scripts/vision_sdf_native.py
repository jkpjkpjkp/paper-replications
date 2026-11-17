#!/usr/bin/env python3
"""Run the released 200-seed SDF procedure with isolated resumable outputs.

Numerical function bodies are reused verbatim. A case-only CSV adapter maps
released ViT/CNN columns to the lowercase names expected by those scripts.
The literal released MSE broadcasting and penalty-selection rules are retained
and documented; their agreement with published results is an empirical test.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'paper-claim-validation'
OUT=BASE/'results/vision_sdf'
SOURCE=ROOT/'vision-to-value-evaluation/upstream/sdf_analysis'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,value):
    temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['data','factor_port']:
        p=OUT/name
        if not p.exists():p.symlink_to(SOURCE/name,target_is_directory=True)
    names=[SOURCE/'double_selection_lasso.py',ROOT/'vision-to-value-evaluation/sources/sdf_loading.py',
           SOURCE/'data/beta_k.csv',*sorted((SOURCE/'data/processed_kelly').glob('*.csv')),
           *sorted((SOURCE/'factor_port').glob('*.csv'))]
    manifest=dict(models=['vit_Cap_VW','cnn_Cap_VW'],seeds=list(range(200)),folds=10,
        lambda_grid='exp(linspace(0,-35,100))',source_sha256={str(p):sha(p) for p in names},
        adapter='Only model choice and released factor CSV column capitalization; no numerical function-body changes',
        known_literal_issues=['First LASSO prediction (assets,) subtracts mean-return (assets,1), broadcasting pairwise MSE.',
                              'One-SE rule uses CSV string-column indexing and max eligible index as released.',
                              'Published results are compared to literal code outputs, not forced to match.'])
    path=BASE/'evidence/vision_sdf_config.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Input/source changed during SDF suite'
    else:save(path,manifest)


def execute_source(path,model,stop_at_ols=False,published_penalties=False):
    tree=ast.parse(path.read_text())
    if published_penalties:
        def assigns(node,name):return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
        start=next(i for i,n in enumerate(tree.body) if assigns(n,'best_lambda1_candidate'))
        stop=next(i for i,n in enumerate(tree.body) if assigns(n,'NumLambda'))
        selected=ast.parse(f'first_lambda_idx=8\nsecond_lambda_idx={8 if model.startswith("vit") else 6}\n').body
        tree.body=tree.body[:start]+selected+tree.body[stop:]
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model' for t in node.targets):
            node.value=ast.Constant(model)
        nodes.append(node)
        if stop_at_ols and isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ols_model' for t in node.targets):break
    tree.body=nodes
    ast.fix_missing_locations(tree)
    namespace=dict(__name__='native_sdf_reproduction',__file__=str(path))
    original_read=pd.read_csv
    def read_case_adapter(path,*a,**kw):
        result=original_read(path,*a,**kw)
        if Path(path).parent.name=='factor_port':result=result.rename(columns={'ViT':'vit','CNN':'cnn'})
        return result
    pd.read_csv=read_case_adapter
    try:exec(compile(tree,str(path),'exec'),namespace)
    finally:pd.read_csv=original_read
    return namespace


def complete_seed(model,seed):
    folder=OUT/'result'/model
    for stage in [1,2]:
        p=folder/f'lasso{stage}_mse_seed{seed}.csv'
        if not p.exists():return False
        try:frame=pd.read_csv(p)
        except (OSError,pd.errors.EmptyDataError,pd.errors.ParserError):return False
        if frame.shape!=(10,100) or not np.isfinite(frame.to_numpy()).all():return False
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed-limit',type=int,default=200)
    p.add_argument('--validated-gram',action='store_true')
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    assert 1<=args.seed_limit<=200
    assert args.workers in [1,2]
    prepare();os.chdir(OUT)
    lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.validated_gram:
        proof=BASE/'results/vision_sdf_backend_equivalence.json'
        assert proof.exists() and json.loads(proof.read_text())['passed'],'Gram backend requires completed equivalence check'
        save(BASE/'evidence/vision_sdf_optimization.json',dict(
            optimization='Scikit-learn Lasso(precompute=True); same statistical objective and convergence settings',
            equivalence_proof=str(proof),equivalence_proof_sha256=sha(proof),
            checks=json.loads(proof.read_text()),runner_sha256=sha(__file__),
            seed_order='1..199, then 0; reuse literal seed-zero results when available',
            full_paper_reproduced=False))
    state_path=OUT/('FULL_STATUS.json' if args.seed_limit==200 else 'STATUS.json')
    records=[]
    for model in ['vit_Cap_VW','cnn_Cap_VW']:
        namespace=execute_source(SOURCE/'double_selection_lasso.py',model)
        original_lasso=namespace['Lasso']
        progress={'fits':0,'seed':None,'start':None}
        class AuditedLasso(original_lasso):
            def fit(self,*a,**kw):
                if args.validated_gram:self.precompute=True
                result=super().fit(*a,**kw)
                progress['fits']+=1
                if progress['fits']%100==0:
                    value=dict(status='cross_validation_fit_progress',model=model,seed=progress['seed'],
                        completed_fits=progress['fits'],fits_per_seed=2000,
                        solver_backend='validated_Gram' if args.validated_gram else 'literal',
                        elapsed_seconds=time.monotonic()-progress['start'],
                        last_solver_iterations=int(np.max(self.n_iter_)),updated_utc=datetime.now(timezone.utc).isoformat())
                    save(OUT/'CURRENT_FIT.json',value);print(json.dumps(value),flush=True)
                return result
        namespace['Lasso']=AuditedLasso
        save(state_path,dict(status='running',model=model,required_seeds_per_model=200,
            model_completed_seeds=sum(complete_seed(model,s) for s in range(200)),workers=args.workers,
            solver_backend='validated_Gram' if args.validated_gram else 'literal',
            runner_sha256=sha(__file__),updated_utc=datetime.now(timezone.utc).isoformat()))
        seeds=list(range(1,200))+[0] if args.validated_gram and args.seed_limit==200 else list(range(args.seed_limit))
        if args.workers>1:
            assert args.seed_limit==200
            from joblib import Parallel,delayed
            def worker(seed):
                start=time.monotonic();progress.update(fits=0,seed=seed,start=start)
                original_csv=pd.DataFrame.to_csv
                def atomic_csv(frame,path,*a,**kw):
                    path=Path(path);temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
                    result=original_csv(frame,temporary,*a,**kw);temporary.replace(path);return result
                pd.DataFrame.to_csv=atomic_csv
                try:namespace['run_one_seed'](seed)
                finally:pd.DataFrame.to_csv=original_csv
                assert complete_seed(model,seed)
                return dict(status='running',model=model,last_seed=seed,seed_elapsed_seconds=time.monotonic()-start,
                    required_seeds_per_model=200,solver_backend='validated_Gram' if args.validated_gram else 'literal',
                    workers=args.workers,updated_utc=datetime.now(timezone.utc).isoformat())
            pending=[seed for seed in range(1,200) if not complete_seed(model,seed)]
            completed=Parallel(n_jobs=args.workers,return_as='generator_unordered',batch_size=1,pre_dispatch=args.workers)(
                delayed(worker)(seed) for seed in pending)
            for result in completed:
                result['model_completed_seeds']=sum(complete_seed(model,s) for s in range(200))
                save(state_path,result);print(json.dumps(result),flush=True)
            seeds=[0]
        for seed in seeds:
            if complete_seed(model,seed):continue
            if seed==0 and args.seed_limit==200:
                while subprocess.run(['systemctl','--user','is-active','paper-claim-vision-sdf.service'],capture_output=True,text=True).stdout.strip()=='active':
                    if complete_seed(model,seed):break
                    save(state_path,dict(status='waiting_for_literal_seed_zero',model=model,
                        updated_utc=datetime.now(timezone.utc).isoformat()))
                    time.sleep(20)
            if complete_seed(model,seed):continue
            start=time.monotonic()
            progress.update(fits=0,seed=seed,start=start)
            namespace['run_one_seed'](seed)
            assert complete_seed(model,seed)
            status=dict(status='running',model=model,last_seed=seed,
                model_completed_seeds=sum(complete_seed(model,s) for s in range(200)),
                required_seeds_per_model=200,seed_elapsed_seconds=time.monotonic()-start,
                updated_utc=datetime.now(timezone.utc).isoformat())
            status['solver_backend']='validated_Gram' if args.validated_gram else 'literal'
            save(state_path,status);print(json.dumps(status),flush=True)
        if args.seed_limit<200:continue
        namespace=execute_source(ROOT/'vision-to-value-evaluation/sources/sdf_loading.py',model,stop_at_ols=True)
        fit=namespace['ols_model'];i=namespace['var_names'].index(model.split('_')[0])
        first=namespace['first_lambda_idx'];second=namespace['second_lambda_idx']
        expected_log1=-2.828;expected_log2=-2.828 if model.startswith('vit') else -2.121
        expected_coef=.052 if model.startswith('vit') else .054
        expected_t=2.948 if model.startswith('vit') else 4.458
        row=dict(model=model,seeds=200,first_lambda_idx=first,second_lambda_idx=second,
            first_log_lambda=float(np.log(namespace['lambdas1'][first])),
            second_log_lambda=float(np.log(namespace['lambdas2'][second])),
            sdf_loading=float(np.asarray(fit.params)[i]),sdf_loading_t=float(np.asarray(fit.tvalues)[i]),
            paper_first_log_lambda=expected_log1,paper_second_log_lambda=expected_log2,
            paper_loading=expected_coef,paper_loading_t=expected_t,
            selected_controls=namespace['feat_names'],n_test_portfolios=int(fit.nobs),
            code_level_reproduction_complete=True,full_paper_reproduced=False)
        row['paper_table3_matches']={key:abs(row[key]-row['paper_'+other])<.000500001 for key,other in
            [('first_log_lambda','first_log_lambda'),('second_log_lambda','second_log_lambda'),
             ('sdf_loading','loading'),('sdf_loading_t','loading_t')]}
        records.append(row);save(OUT/(model+'_table3.json'),row)
        (OUT/(model+'_OLS.txt')).write_text(fit.summary().as_text()+'\n')
    status=dict(status='complete' if args.seed_limit==200 else 'limited_preflight_complete',
                completed_seeds={m:sum(complete_seed(m,s) for s in range(200)) for m in ['vit_Cap_VW','cnn_Cap_VW']},
                native_SDF_code_reexecuted=True,full_paper_reproduced=False,table3=records,
                updated_utc=datetime.now(timezone.utc).isoformat())
    status['solver_backend']='validated_Gram' if args.validated_gram else 'literal'
    save(state_path,status);print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
