#!/usr/bin/env python3
"""Audit Table 3 baseline protocol, preserving the exploratory search.

This is a post-hoc protocol diagnostic, not confirmation of the original setup.
Reads archived data/predictions; writes only to a distinct output directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HORIZONS = [24, 36, 48, 60]
# Transcribed from original PDF page 8, Table 3 (visually verified 2026-11-17).
TARGET = {
    'NASDAQ': {
        'Persistence': [[.684,.582],[1.180,.776],[1.720,.938],[2.082,1.029]],
        'RW_train_drift': [[.666,.575],[1.138,.762],[1.653,.918],[1.999,1.006]],
        'RISE': [[.553,.531],[.809,.655],[1.077,.750],[1.259,.811]],
    },
    'NYSE': {
        'Persistence': [[.213,.276],[.377,.374],[.572,.468],[.824,.591]],
        'RW_train_drift': [[.193,.260],[.330,.343],[.485,.420],[.677,.522]],
        'RISE': [[.183,.267],[.309,.342],[.446,.421],[.617,.511]],
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(path, layout):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    channels = list(dict.fromkeys(r['cols'] for r in rows))
    grouped = {c: [r for r in rows if r['cols'] == c] for c in channels}
    if layout == 'tfb_native_blocks':
        assert len(set(map(len, grouped.values()))) == 1
        return np.array([[float(grouped[c][i]['data']) for c in channels]
                         for i in range(len(grouped[channels[0]]))])
    pairs = {(float(r['date']), r['cols']): float(r['data']) for r in rows}
    assert len(pairs) == len(rows)
    common = sorted(set.intersection(*[{float(r['date']) for r in grouped[c]}
                                      for c in channels]))
    return np.array([[pairs[t,c] for c in channels] for t in common])


def errors(prediction, target):
    e = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(e**2)), float(np.mean(np.abs(e)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fresh-run', type=Path,
                    help='Verify a completed L36/date-complete/within_split suite')
    args = ap.parse_args()
    assert args.archive.resolve() != args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, selected, rise, inputs = [], [], [], []
    for dataset in ['NASDAQ', 'NYSE']:
        original = args.archive / 'data' / f'{dataset}.csv'
        inputs.append({'path':str(original), 'sha256':sha(original)})
        for layout in ['tfb_native_blocks', 'date_complete']:
            raw = reconstruct(original, layout)
            retained = np.loadtxt(args.archive/'data'/f'{dataset}.{layout}.csv',
                                  delimiter=',',skiprows=1)[:,1:]
            assert np.array_equal(raw, retained), 'Raw-source reconstruction mismatch'
            n = len(raw)
            tr, test = int(n*.7), n-int(n*.2)
            mean, sd = raw[:tr].mean(0), raw[:tr].std(0,ddof=0)
            z = (raw-mean)/sd
            drift = (z[tr-1]-z[0])/(tr-1)
            for offset in [0,36,48,60,96]:
                for hi, h in enumerate(HORIZONS):
                    origins = np.arange(test+offset,n-h+1)
                    targets = np.stack([z[t:t+h] for t in origins])
                    baseline = np.repeat(z[origins-1,None,:],h,axis=1)
                    models = {'Persistence':baseline,
                              'RW_train_drift':baseline + np.arange(1,h+1)[None,:,None]*drift}
                    for lookback in [36,96]:
                        window_drift = (z[origins-1]-z[origins-lookback])/(lookback-1)
                        models[f'RW_window_drift_L{lookback}'] = baseline + np.arange(1,h+1)[None,:,None]*window_drift[:,None,:]
                    for method,pred in models.items():
                        mse,mae = errors(pred,targets)
                        paper = TARGET[dataset]['Persistence' if method=='Persistence' else 'RW_train_drift'][hi]
                        row = {'dataset':dataset,'layout':layout,'offset':offset,'horizon':h,
                               'method':method,'origins':len(origins),'mse':mse,'mae':mae,
                               'paper_mse':paper[0],'paper_mae':paper[1],
                               'mse_error':mse-paper[0],'mae_error':mae-paper[1],
                               'mse_matches_rounding':abs(mse-paper[0])<=.0005,
                               'mae_matches_rounding':abs(mae-paper[1])<=.0005}
                        rows.append(row)
                        if layout=='date_complete' and offset==36 and method in ['Persistence','RW_train_drift']:
                            # Independent raw-space formula checks numerical calculation.
                            independent=[]
                            rd=(raw[tr-1]-raw[0])/(tr-1)
                            for t in origins:
                                forecast=np.tile(raw[t-1],(h,1))
                                if method=='RW_train_drift':
                                    forecast=forecast+np.arange(1,h+1)[:,None]*rd
                                independent.append((forecast-raw[t:t+h])/sd)
                            independent=np.stack(independent)
                            assert np.isclose(mse,np.mean(independent**2),atol=1e-12,rtol=0)
                            assert np.isclose(mae,np.mean(np.abs(independent)),atol=1e-12,rtol=0)
                            row['first_forecast_origin']=int(origins[0])
                            row['last_forecast_origin']=int(origins[-1])
                            row['independent_raw_scale_check']=True
                            selected.append(row)
            if layout!='date_complete':
                continue
            for hi,h in enumerate(HORIZONS):
                for seed in range(5):
                    p=args.archive/'results/tfb_rise'/layout/dataset/f'H{h}'/f'seed{seed}'/'predictions.npz'
                    with np.load(p,allow_pickle=False) as a:
                        keep=a['test_origins']>=test+36
                        origins=a['test_origins'][keep]
                        assert np.array_equal(origins,np.arange(test+36,n-h+1))
                        y=np.stack([z[t:t+h] for t in origins]).astype(np.float32)
                        assert np.array_equal(a['test_target'][keep],y)
                        mse,mae=errors(a['test_prediction'][keep],y)
                    rise.append({'dataset':dataset,'horizon':h,'seed':seed,'mse':mse,'mae':mae,
                                 'paper_mse':TARGET[dataset]['RISE'][hi][0],
                                 'paper_mae':TARGET[dataset]['RISE'][hi][1],
                                 'prediction_path':str(p),'sha256':sha(p)})
    seed_totals=[]
    for seed in range(5):
        subset=[r for r in rise if r['seed']==seed]
        seed_totals.append([np.mean([r['mse'] for r in subset]),np.mean([r['mae'] for r in subset])])
    output={
        'assessment_date':'2026-11-17', 'scope':'post-hoc deterministic baseline protocol reconstruction; archived RISE re-evaluation',
        'grade_scope':'Numerical baseline match is exploratory; coordinator assigns the project grade. No RISE main-method match.',
        'protocol_not_author_verified':True,
        'exploration':'Persistence first checked for layouts native/date-complete and offsets 0,36,48,60,96. Offset36/date-complete matched. RW per-window L36/L96 did not; fixed training-split drift independently matched. All alternatives retained.',
        'inference':'The full pair of baseline curves is consistent with date-complete source data, training-only scaling, test origins beginning 36 observations after split, and training-fitted drift. It is not proof of the unreleased author pipeline.',
        'paper_support':'PDF pp6/8 specify 7:1:2, training-only z-score, M setting; p9 Table6 includes L36; exact split-border/window and drift-fitting conventions are omitted.',
        'rounding_tolerance':.0005,
        'baseline_matched_cells':sum(r['mse_matches_rounding']+r['mae_matches_rounding'] for r in selected),
        'baseline_total_cells':2*len(selected),
        'selected_baseline_rows':selected,'all_diagnostic_rows':rows,
        'archived_rise_reevaluation':rise,
        'archived_rise_overall_mean':np.mean(seed_totals,axis=0).tolist(),
        'archived_rise_overall_seed_sd':np.std(seed_totals,axis=0,ddof=1).tolist(),
        'paper_rise_overall':[.657,.537],
        'overall_summary_caveat':'Persistence unrounded overall MSE/MAE 0.956427506756/0.629450213564 differs from printed 0.957/0.630 by 0.000572493244/0.000549786436. Averaging the rounded dataset summaries produces the printed overall values. The 32-cell claim covers dataset/horizon rows, not these two overall summaries.',
        'precision_caveat':'NASDAQ H48 RW MSE=1.653499997256722 is only 2.74e-9 inside the 0.0005 rounding bound around 1.653. Different floating-point evaluation could change the rounded last digit.',
        'inputs':inputs,
        'paper_sha256':sha(args.archive/'sources/paper-viewer-export.pdf'),
        'audit_source_sha256':sha(Path(__file__)),
        'fresh_training':bool(args.fresh_run),
        'baseline_and_archived_audit_fresh_training':False,
        'limitations':['Protocol choices were explored after inspecting target values, so this is exploratory rather than confirmatory.',
                       'No author-released RISE implementation or exact TFB grid found in the current search.',
                       'Archived RISE model remains L96 and may have different training/validation windows from the author; re-evaluation does not fix its training.',
                       'Eight curated Yahoo datasets and full ablations/sensitivities remain unavailable.'],
    }
    if args.fresh_run:
        summary=json.loads((args.fresh_run/'summary.json').read_text())
        fresh=[]
        for run in summary['all_runs']:
            dataset,h=run['dataset'],run['horizon']
            assert run['layout']=='date_complete' and run['lookback']==36
            assert run['split_context']=='within_split'
            raw=reconstruct(args.archive/'data'/f'{dataset}.csv','date_complete')
            n=len(raw);tr=int(n*.7);test=n-int(n*.2)
            mean,sd=raw[:tr].mean(0),raw[:tr].std(0)
            z=(raw-mean)/sd
            path=Path(run['predictions_path'])
            with np.load(path,allow_pickle=False) as arrays:
                assert np.allclose(arrays['mean'],mean,rtol=0,atol=1e-12)
                assert np.allclose(arrays['scale'],sd,rtol=0,atol=1e-12)
                for split,start,end in [('test',test,n),('validation',tr,test)]:
                    origins=arrays[f'{split}_origins']
                    assert np.array_equal(origins,np.arange(start+36,end-h+1))
                    targets=np.stack([z[t:t+h] for t in origins]).astype(np.float32)
                    assert np.array_equal(arrays[f'{split}_target'],targets)
                    pred=arrays[f'{split}_prediction']
                    assert np.isfinite(pred).all()
                    mse,mae=errors(pred,targets)
                    saved=run['metrics' if split=='test' else 'validation_metrics']
                    assert abs(mse-saved['mse'])<1e-12 and abs(mae-saved['mae'])<1e-12
                mse,mae=errors(arrays['test_prediction'],arrays['test_target'])
            paper=TARGET[dataset]['RISE'][HORIZONS.index(h)]
            fresh.append({'dataset':dataset,'horizon':h,'seed':run['seed'],
                          'mse':mse,'mae':mae,'paper_mse':paper[0],'paper_mae':paper[1],
                          'mse_error':mse-paper[0],'mae_error':mae-paper[1],
                          'source_targets_partitions_metrics_verified':True,
                          'predictions_sha256':sha(path)})
        seeds=sorted(set(r['seed'] for r in fresh))
        assert len(fresh)==8*len(seeds)
        assert len({(r['dataset'],r['horizon'],r['seed']) for r in fresh})==len(fresh)
        output['fresh_run']={'rows':fresh,'seeds':seeds,
            'mse_mean':float(np.mean([r['mse'] for r in fresh])),
            'mae_mean':float(np.mean([r['mae'] for r in fresh])),
            'seed_uncertainty':'not measured' if len(seeds)==1 else 'see per-seed results',
            'scope':'fresh training of locked exploratory reconstruction; not author-exact protocol'}
    (args.output/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['baseline_matched_cells','baseline_total_cells','archived_rise_overall_mean','archived_rise_overall_seed_sd','grade_scope']}))


if __name__=='__main__':
    main()
