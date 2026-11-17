#!/usr/bin/env python3
"""Independently reconcile saved forecasts/targets and compare every variant.

No model training, no strategy construction and no inferred portfolio returns.
Run under the shared guard; only one prediction file is loaded at a time.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
from collections import defaultdict

import numpy as np

ROOT=Path(__file__).resolve().parent


def atomic(path,value):
 temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path,rows):
 with path.open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def paper_targets():
 text=(ROOT/'sources/paper.txt').read_text();start=text.index('Table 3: TFB benchmark results')
 rows=[]
 for line in text[start:].splitlines():
  match=re.match(r'^\s*(?:(NASDAQ|NYSE)\s+)?(24|36|48|60)\s+(.+)$',line)
  if not match:continue
  values=[float(v) for v in re.findall(r'\d*\.\d+|\d+',match[3])]
  if len(values)!=22:continue
  rows.append((int(match[2]),values))
  if len(rows)==8:break
 if [h for h,_ in rows]!=[24,36,48,60]*2:raise RuntimeError('Could not verify paper Table3 row structure')
 methods=['RISE','GTR','DUET','RAFT','iTransformer','PatchTST','DLinear','TimeMixer','Persistence','RW_drift','ARIMA111']
 targets={}
 for i,(H,values) in enumerate(rows):
  dataset='NASDAQ' if i<4 else 'NYSE'
  for j,method in enumerate(methods):targets[dataset,H,method]={'paper_mse':values[j*2],'paper_mae':values[j*2+1]}
 atomic(ROOT/'results/paper-table3-targets.json',{'source':str(ROOT/'sources/paper-viewer-export.pdf'),'source_text':str(ROOT/'sources/paper.txt'),
        'source_pdf_sha256':digest(ROOT/'sources/paper-viewer-export.pdf'),'entries':[{'dataset':d,'horizon':h,'method':m,**v} for (d,h,m),v in targets.items()]})
 return targets


def main():
 summaries=[json.loads((ROOT/'results'/name/'summary.json').read_text()) for name in ['tfb_rise','tfb_neural_baselines','tfb_remaining']]
 classical=json.loads((ROOT/'results/tfb_classical/summary.json').read_text())
 raw_cache={};checks=[];rows=[]
 for layout in ['tfb_native_blocks','date_complete']:
  for dataset in ['NASDAQ','NYSE']:
   p=ROOT/'data'/f'{dataset}.{layout}.csv'
   a=np.loadtxt(p,delimiter=',',skiprows=1);raw=a[:,1:]
   n=len(raw);train_end=int(n*.7);test_start=n-int(n*.2)
   mean=raw[:train_end].mean(0);scale=raw[:train_end].std(0)
   raw_cache[dataset,layout]={'z':(raw-mean)/scale,'n':n,'train_end':train_end,'test_start':test_start,'sha256':digest(p),'mean':mean,'scale':scale}
 for summary in summaries:
  for run in summary['all_runs']:
   method=run['method'] if run['method']!='RISE_equation_reimplementation' else 'RISE'
   d=raw_cache[run['dataset'],run['layout']];H=run['horizon']
   expected_origins=np.arange(d['test_start'],d['n']-H+1)
   with np.load(run['predictions_path'],allow_pickle=False) as arrays:
    p=arrays['test_prediction'];target=arrays['test_target'];origins=arrays['test_origins']
    if not np.array_equal(origins,expected_origins):raise RuntimeError('Missing/extra/nonchronological saved neural forecast origins')
    if not np.allclose(arrays['mean'],d['mean'],rtol=0,atol=1e-12):raise RuntimeError('Saved mean is not training-only source mean')
    if not np.allclose(arrays['scale'],d['scale'],rtol=0,atol=1e-12):raise RuntimeError('Saved scale is not training-only source scale')
    truth=np.stack([d['z'][origin:origin+H] for origin in origins]).astype(np.float32)
    if p.shape!=truth.shape or not np.array_equal(target,truth):raise RuntimeError('Saved neural target differs from independent source reconstruction')
    e=p.astype(np.float64)-target.astype(np.float64);mse=float(np.mean(e*e));mae=float(np.mean(np.abs(e)))
    if abs(mse-run['metrics']['mse'])>1e-10 or abs(mae-run['metrics']['mae'])>1e-10:raise RuntimeError('Saved neural metrics do not reconcile to forecast arrays')
    val_origins=arrays['validation_origins']
    if np.any(val_origins<d['train_end']) or np.any(val_origins+H>d['test_start']):raise RuntimeError('Validation labels cross partition')
   rows.append({'method':method,'dataset':run['dataset'],'layout':run['layout'],'horizon':H,'seed':run['seed'],'mse':mse,'mae':mae,
                'origins':len(origins),'prediction_path':run['predictions_path'],'scope':'partial_methodological_forecast_only'})
   checks.append({'method':method,'dataset':run['dataset'],'layout':run['layout'],'horizon':H,'seed':run['seed'],
                  'source_hash':d['sha256'],'origins_reconciled':True,'targets_exactly_reconstructed':True,'metrics_reconciled':True})
 for run in classical['results']:
  d=raw_cache[run['dataset'],run['layout']]
  with np.load(run['predictions_path'],allow_pickle=False) as arrays:
   for metric in run['metrics']:
    H=metric['horizon'];method=metric['method'];origins=arrays[f'origins_H{H}'];p=arrays[f'{method}_H{H}'];target=arrays[f'target_H{H}']
    if not np.array_equal(origins,np.arange(d['test_start'],d['n']-H+1)):raise RuntimeError('Classical origin mismatch')
    truth=np.stack([d['z'][origin:origin+H] for origin in origins])
    if not np.allclose(target,truth,rtol=1e-12,atol=1e-12):raise RuntimeError('Classical target source mismatch')
    e=p-target;mse=float(np.mean(e*e));mae=float(np.mean(np.abs(e)))
    if abs(mse-metric['mse'])>1e-10 or abs(mae-metric['mae'])>1e-10:raise RuntimeError('Classical metric reconciliation failed')
    rows.append({'method':method,'dataset':run['dataset'],'layout':run['layout'],'horizon':H,'seed':'deterministic','mse':mse,'mae':mae,
                 'origins':len(origins),'prediction_path':run['predictions_path'],'scope':'partial_methodological_forecast_only'})
    checks.append({'method':method,'dataset':run['dataset'],'layout':run['layout'],'horizon':H,'seed':'deterministic',
                   'source_hash':d['sha256'],'origins_reconciled':True,'targets_exactly_reconstructed':'within1e-12 across pinned NumPy versions','metrics_reconciled':True})
 if len(rows)!=688:raise RuntimeError(f'Expected688 original-grid/layout seed scores, found{len(rows)}')
 write_csv(ROOT/'results/all-forecast-variants.csv',rows)
 targets=paper_targets();grouped=defaultdict(list)
 for row in rows:grouped[row['layout'],row['dataset'],row['horizon'],row['method']].append(row)
 comparison=[]
 for (layout,dataset,H,method),values in grouped.items():
  reference=targets[dataset,H,method];mse=float(np.mean([v['mse'] for v in values]));mae=float(np.mean([v['mae'] for v in values]))
  comparison.append({'layout':layout,'dataset':dataset,'horizon':H,'method':method,'seeds':len(values),'mse_mean':mse,'mae_mean':mae,
                    **reference,'mse_delta':mse-reference['paper_mse'],'mae_delta':mae-reference['paper_mae'],
                    'claim':'methodological comparison; original preprocessing/tuning unresolved'})
 if len(comparison)!=176:raise RuntimeError('Expected176 data-layout/dataset/horizon/method aggregates')
 write_csv(ROOT/'results/paper-table3-comparison.csv',comparison)
 overall=[]
 for layout in ['tfb_native_blocks','date_complete']:
  for method in ['RISE','GTR','DUET','RAFT','iTransformer','PatchTST','DLinear','TimeMixer','Persistence','RW_drift','ARIMA111']:
   a=[x for x in comparison if x['layout']==layout and x['method']==method]
   wins=0
   for row in a:
    peers=[x['mse_mean'] for x in comparison if x['layout']==layout and x['dataset']==row['dataset'] and x['horizon']==row['horizon']]
    wins+=row['mse_mean']<=min(peers)+1e-12
   overall.append({'layout':layout,'method':method,'mse':float(np.mean([x['mse_mean'] for x in a])),
                   'mae':float(np.mean([x['mae_mean'] for x in a])),'mse_first_place_count':int(wins),'configurations':8})
 write_csv(ROOT/'results/overall-method-comparison.csv',overall)
 result={'status':'passed_for_partial_methodological_scope','neural_fits':640,'classical_metric_settings':48,'all_variant_seed_scores':688,
         'dataset_layout_horizon_method_aggregates':176,'all_forecast_targets_metrics_and_partitions_reconciled':True,
         'checks':checks,'overall_metrics':overall,'trading_variants_evaluated':0,'market_neutral_nocost':'blocked',
         'return_inference':'None; ordinal OHLCV arrays are not genuine daily portfolio returns. Seeds are not additional independent days.',
         'original_paper_reproduction_complete':False,'limitations':['Two declared preprocessing interpretations; exact original TFB grid/source code unavailable.','Eight curated Yahoo datasets and source-dependent ablations/sensitivity remain unexecuted.','ARIMA229/4495 fits nonconverged, retained with warnings.','No calendar-dated executable asset/cashflow evidence for market-neutral evaluation.']}
 atomic(ROOT/'results/verification-and-comparison.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['checks','overall_metrics']}),flush=True)


if __name__=='__main__':main()
