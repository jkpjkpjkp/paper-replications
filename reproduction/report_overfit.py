"""Render the completed, post-hoc mean-baseline audit from its saved evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import subprocess
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/overfit_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    runs=pd.read_csv(OUT/'prediction_runs.csv')
    comp=pd.read_csv(OUT/'prediction_comparison.csv')
    curves=pd.read_csv(OUT/'learning_curves.csv')
    portfolios=pd.read_csv(OUT/'portfolio_comparison.csv')
    literal=pd.read_csv(OUT/'literal_prediction_runs.csv')
    evidence=json.loads((OUT/'prediction_verification.json').read_text())
    curve_checks=json.loads((OUT/'learning_curve_verification.json').read_text())
    models=runs[runs.method.isin(['PFL','DFL'])]
    assert len(models)==80 and len(comp)==16 and len(portfolios)==36
    assert curves.groupby('seed').size().to_dict()==dict.fromkeys([16,37,47,214,1947],10)
    assert len(curve_checks['checks'])==5 and all(x['final_model_max_abs_error']==0 for x in curve_checks['checks'])
    for receipt in evidence['receipts']:
        path=ROOT/(receipt.get('model') or receipt['data'])
        assert sha(path)==receipt['sha256'],f'Artifact changed: {path}'
    source_check=subprocess.run(['git','-C',str(ROOT/'upstream'),'diff','--exit-code','--','src'],capture_output=True,text=True)
    assert source_check.returncode==0,'Official source was modified'
    baseline=runs[runs.method=='train_mean'].set_index('market')
    checks={}
    for unit in ['normalized','raw']:
        checks[f'{unit}_train_better']=int(sum(r[f'train_{unit}_mse']<baseline.loc[r['market'],f'train_{unit}_mse'] for r in models.to_dict('records')))
        checks[f'{unit}_test_worse']=int(sum(r[f'test_{unit}_mse']>baseline.loc[r['market'],f'test_{unit}_mse'] for r in models.to_dict('records')))
        assert checks[f'{unit}_train_better']==80 and checks[f'{unit}_test_worse']==80
    summary=[]
    for market,g in comp.groupby('market',sort=False):
        pfl=g[g.method=='PFL'].iloc[0];dfl=g[g.method=='DFL']
        summary.append({'market':market,'PFL_train_MSE_over_mean':pfl.train_normalized_ratio,
                        'PFL_test_MSE_over_mean':pfl.test_normalized_ratio,
                        'DFL_train_ratio_min':dfl.train_normalized_ratio.min(),'DFL_train_ratio_max':dfl.train_normalized_ratio.max(),
                        'DFL_test_ratio_min':dfl.test_normalized_ratio.min(),'DFL_test_ratio_max':dfl.test_normalized_ratio.max(),
                        'PFL_raw_test_ratio':pfl.test_raw_ratio,
                        'DFL_raw_test_ratio_min':dfl.test_raw_ratio.min(),'DFL_raw_test_ratio_max':dfl.test_raw_ratio.max()})
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'summary.csv',index=False)
    curve=curves.groupby('updates')[['train_mse','test_mse']].mean()
    curve['train_ratio_to_fixed_mean']=curve.train_mse/baseline.loc['euro50','train_normalized_mse']
    curve['test_ratio_to_fixed_mean']=curve.test_mse/baseline.loc['euro50','test_normalized_mse']
    curve.to_csv(OUT/'learning_curve_average.csv')
    lb=literal[literal.method=='train_mean'].set_index('market')
    ld=literal[literal.method.isin(['pfl','dfl'])].copy()
    for split in ['train','test']:
        ld[f'{split}_ratio']=[getattr(r,f'{split}_raw_mse')/lb.loc[r.market,f'{split}_raw_mse'] for r in ld.itertuples()]
        ld[f'{split}_initial_ratio']=[getattr(r,f'{split}_initial_raw_mse')/lb.loc[r.market,f'{split}_raw_mse'] for r in ld.itertuples()]
    # The three PFL support levels share one identical predictor per seed.
    lp=ld[(ld.market=='euro50')&(ld.method=='pfl')&(ld.rho==10)]
    literal_receipts=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted((ROOT/'results/paper_method/study').glob('*/checkpoint.pt'))]
    verification={'created_utc':datetime.now(timezone.utc).isoformat(),'models_checked':80,
                  'prediction_counts':checks,'prediction_saved_mse_max_abs_error':max(x['saved_test_mse_abs_error'] for x in evidence['saved_mse_checks']),
                  'full_learning_curve_final_models_bitwise_equal':5,'mean_baseline_portfolio_traces':24,
                  'portfolio_comparison_cells':36,'original_source_unchanged':True,
                  'bootstrap_ci20_and_ci60_all_normalized_mse_ratios_above_one':bool((comp.test_normalized_ratio_ci20_low>1).all() and (comp.test_normalized_ratio_ci60_low>1).all()),
                  'literal_checkpoint_receipts':literal_receipts,
                  'audit_script_sha256':sha(ROOT/'reproduction/audit_overfit.py'),'report_script_sha256':sha(__file__),
                  'exact_original_paper_model_evidence':False}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    sections=[]
    def paragraph(title,text,table=None):
        sections.append((title,text,table))
    paragraph('Finding',
        'The saved released-code replay shows substantial predictive overfitting. All 80 final predictors '
        '(20 PFL and 60 DFL at alpha=0.5) beat a fixed per-asset training mean on training MSE and lose '
        'to that same mean on test MSE, in both standardized and raw-return units. Five complete Euro PFL '
        'learning curves show training error falling while test error rises. This supports a claim about '
        'these reproduced predictors. It does not establish what happened in the authors\' unavailable original experiment.')
    paragraph('Matched comparison',
        'The baseline is the vector of per-asset arithmetic means over every row before the frozen 80% chronological split. '
        'It is held fixed for every test date. This is different from the release\'s Historic predictor, which uses a '
        'changing 100-day mean. Both predictors are scored on exactly the saved release test dates, after its 100-day '
        'post-split warmup. Models remain in eval mode. All five seeds and all three DFL support settings are included; '
        'none was selected by test performance. Ratios below 1 favor the model; ratios above 1 favor the mean.',summary)
    paragraph('Units and robustness',
        'Standardized MSE is the source training metric, averaged over dates and assets. Raw-unit errors inverse-transform '
        'the forecasts and compare them with actual simple returns. Inverse-transforming the constant baseline recovers '
        'exactly the raw training mean. All 80 models also lose to the training-target-only mean, zero raw-return forecast, '
        'and the rolling 100-day mean on test prediction MSE. The source full-panel scaler leakage is retained to inspect '
        'the existing models; this is not a new leakage-free training experiment. It does not invalidate the matched '
        'forecast-error calculation, but constrains broader generalization claims.')
    paragraph('Uncertainty',
        'For each market and DFL support, errors are averaged over its five seeds and all assets on each date. '
        'The same whole-date blocks are resampled for model and baseline. In 2,000 circular moving-block resamples '
        'with 20-day and 60-day blocks, every standardized test-MSE ratio has its unadjusted 95% interval above 1. '
        'Assets, seeds and support levels are not treated as independent financial samples. These are post-hoc '
        'diagnostic intervals, not a pre-registered multiple-testing claim.',
        comp[['market','method','rho','test_normalized_ratio','test_normalized_ratio_ci20_low','test_normalized_ratio_ci20_high',
              'test_normalized_ratio_ci60_low','test_normalized_ratio_ci60_high']])
    paragraph('Training trajectory, not just a train/test volatility difference',
        'The unmodified author train_model was rerun for all five Euro PFL seeds. Read-only hooks measured full train '
        'and test MSE at fixed update counts, with model mode restored and Torch RNG checked unchanged. Every final '
        'parameter tensor exactly equals its original saved counterpart (maximum difference 0). Thus the diagnostic '
        'did not change the training path. Mean training MSE falls from 1.0502 to 0.2531, while test MSE rises from '
        '0.8932 to 1.2867; the fixed mean test MSE is 0.8757. Updates here are the source\'s single sampled batch, '
        'not full passes through the data. No checkpoint was promoted using this test curve.',curve.reset_index())
    paragraph('Capacity and training controls',
        'The replay fits 2.55–10.73 million parameters from 1,853–1,956 overlapping training windows, uses the '
        'released 300 sampled-batch Adam updates, and has no validation-based early stopping. The source does '
        'use dropout 0.1 and weight decay 1e-4. These facts make memorization plausible; this audit does not '
        'causally isolate capacity, update count, regularization or market regime. The learning curves directly '
        'establish deterioration on the held-out Euro prediction task.')
    paragraph('Portfolio returns are a different claim',
        'Higher forecast MSE does not imply lower portfolio Sharpe, especially for a decision-focused loss. '
        'To check this separately, the fixed training mean was fed into the unchanged soft OSCAR and hard OSCAR '
        'paths, using the same rolling covariance, support parameter and test dates. The learned model has higher '
        'test daily Sharpe in all 36 five-seed comparisons (including the soft PFL sensitivity); every individual '
        'seed is also higher in those comparisons. However, these paths preserve the release\'s full-panel demeaning '
        'and use standardized forecasts/covariances for raw-return weights. The fixed-mean strategy therefore '
        'inherits the known preprocessing/unit defects and its poor Sharpe is not evidence of an economically '
        'sound learned edge. No cost deduction, market hedge or corrected-unit claim is made. DFL soft weights '
        'remain dense. A prediction-error finding must not be relabeled as proof that every learned investment '
        'strategy underperforms the mean strategy.',
        portfolios[['market','rho','method','selector','fixed_train_mean_daily_sr','model_daily_sr_mean','sr_delta',
                    'sr_delta_ci20_low','sr_delta_ci20_high']])
    paragraph('Separate literal-paper reconstruction: worse than overfitting alone',
        f'The separate raw-return, plain-SGD, five-full-epoch Euro reconstruction is not the released-code run. '
        f'Its five unique PFL predictors have average training MSE {lp.train_ratio.mean():.2f} times the training-mean baseline '
        f'and test MSE {lp.test_ratio.mean():.2f} times the baseline. Their initial ratios were '
        f'{lp.train_initial_ratio.mean():.2f} and {lp.test_initial_ratio.mean():.2f}, respectively: optimization improves '
        'the predictor but has not even beaten a constant in-sample. That cannot be diagnosed simply as memorization '
        'with low training error. Some literal DFL runs have enormous errors already on the training data, together '
        'with the separately recorded decision-layer numerical/budget failures. Those require optimization and '
        'numerical diagnosis. The raw-return/SGD/epoch choices are explicit reconstructions of missing settings, '
        'not recovered original author settings. Three duplicate PFL support evaluations per seed are not three '
        'independent predictors.',ld[['market','method','rho','seed','status','train_ratio','test_ratio','train_initial_ratio','test_initial_ratio']])
    paragraph('Implications',
        'Treat the released networks as failing the basic held-out prediction baseline. For any subsequent corrected '
        'experiment, include the fixed mean and zero forecast, fit scaling on training only, use chronological '
        'validation for regularization/early stopping, and keep a fresh untouched final test segment. The present '
        'test set has already been inspected and should not be used to choose a winning checkpoint or tuning setting. '
        'The independent original-setting replication queue and all of its frozen artifacts remain unchanged.')
    paragraph('Verification and reproduction',
        f'80 source saved-test-MSE values independently recomputed; maximum absolute discrepancy '
        f'{verification["prediction_saved_mse_max_abs_error"]:.3g}. Five complete training reruns match saved weights '
        'bit-for-bit. 24 constant-mean portfolio traces use the author evaluator and match existing model dates. '
        'Source Git diff is empty; all inspected model/data SHA-256 receipts remain valid. Full results and hashes '
        'are in the CSV/JSON files next to this report. Commands: .venv/bin/python reproduction/audit_overfit.py '
        '{predictions|curves|portfolios|literal}; .venv/bin/python reproduction/report_overfit.py.')
    txt=['Frozen-predictor audit against the training mean',verification['created_utc'],'']
    fragments=[]
    for title,text,table in sections:
        txt.extend([title,'',text,''])
        fragments.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>')
        if table is not None:
            txt.extend([table.to_string(index=False,float_format=lambda x:f'{x:.6g}'),''])
            fragments.append('<div class="scroll">'+table.to_html(index=False,float_format=lambda x:f'{x:.4g}',border=0)+'</div>')
        fragments.append('</section>')
    (OUT/'report.txt').write_text('\n'.join(txt))
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Training-mean and overfitting audit</title>'
    page+='<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#192635;background:#fafbfd}h1{font-size:30px}h2{font-size:22px;margin-top:34px}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:8px 12px;border-bottom:1px solid #dce3eb;text-align:right}th{background:#e9eef4}.scroll{overflow-x:auto}p{max-width:1000px}a{color:#145ba1}</style>'
    page+='<h1>Frozen predictors versus the training mean</h1><p>'+html.escape(verification['created_utc'])+'</p>'+''.join(fragments)
    page+='<p><a href="summary.csv">Summary CSV</a> · <a href="prediction_runs.csv">All prediction runs</a> · <a href="verification.json">Verification</a> · <a href="report.txt">Plain text</a></p></html>'
    (OUT/'report.html').write_text(page)
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()
