"""Evaluate all seeds without treating overlapping Top-K settings as replicates."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from metrics import ranking,portfolio,inference,holm
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def main():
    cfg=json.loads((ROOT/'protocol.json').read_text());daily=[];pred={}
    reference=None
    for variant in ['return_only','multitask']:
        zz=[]
        for seed in cfg['seeds']:
            path=OUT/f'{variant}_seed{seed}/predictions.npz'
            a=np.load(path);zz.append(a['pred'])
            if reference is None:reference={k:a[k] for k in ['y','dates','symbols']}
            else:
                for k in reference:assert np.array_equal(a[k],reference[k],equal_nan=True) if k=='y' else np.array_equal(a[k],reference[k])
            pred[variant,f'seed{seed}']=a['pred']
        pred[variant,'ensemble']=np.mean(zz,axis=0)
    y=reference['y'][...,0]
    for (variant,member),p in pred.items():
        for i,date in enumerate(reference['dates']):
            # Ranking is scored on observed return labels, without conditioning on future risk-label availability.
            valid=np.isfinite(y[i]);assert valid.sum()>=10
            for k in [3,5,10]:
                prec,mrr,r=ranking(p[i,valid,0],y[i,valid],k)
                daily.append({'variant':variant,'member':member,'date':date,'k':k,'precision':prec,'mrr_eq20':mrr,'portfolio_return':r,'stocks_scored':int(valid.sum())})
    daily=pd.DataFrame(daily);daily.to_csv(OUT/'daily_metrics.csv',index=False)
    summary=daily.groupby(['variant','member','k'])[['precision','mrr_eq20']].mean().reset_index()
    summary.to_csv(OUT/'ranking_summary.csv',index=False)
    ens=daily[daily.member=='ensemble'].groupby(['variant','date'])[['precision','mrr_eq20']].mean()
    tests=[];sens=[]
    for metric in ['precision','mrr_eq20']:
        a=ens.loc['multitask',metric];b=ens.loc['return_only',metric]
        assert a.index.equals(b.index)
        d=a.to_numpy()-b.to_numpy()
        tests.append({'metric':metric,'return_only':float(b.mean()),'multitask':float(a.mean()),**inference(d)})
        for block in [5,20]:sens.append({'metric':metric,**inference(d,block=block)})
    for key in ['bootstrap_p_two_sided','hac_p_two_sided']:
        adj=holm([r[key] for r in tests])
        for r,p in zip(tests,adj):r[key+'_holm']=float(p)
    pd.DataFrame(tests).to_csv(OUT/'primary_tests.csv',index=False)
    pd.DataFrame(sens).to_csv(OUT/'block_sensitivity.csv',index=False)
    # Per-seed summaries retain algorithmic variability; seeds are not multiplied by days in tests.
    per_seed=[]
    for member in [f'seed{s}' for s in cfg['seeds']]:
        for metric in ['precision','mrr_eq20']:
            q=daily[daily.member==member].groupby('variant')[metric].mean()
            per_seed.append({'member':member,'metric':metric,'return_only':q['return_only'],'multitask':q['multitask'],'delta':q['multitask']-q['return_only']})
    pd.DataFrame(per_seed).to_csv(OUT/'per_seed.csv',index=False)
    pp=[]
    for (variant,member,k),g in daily.groupby(['variant','member','k']):
        pp.append({'variant':variant,'member':member,'k':int(k),**portfolio(g.sort_values('date').portfolio_return.to_numpy())})
    pd.DataFrame(pp).to_csv(OUT/'portfolio_summary.csv',index=False)
    # Verify prediction provenance against frozen inputs and all completed fits.
    lock=json.loads((OUT/'run_lock.json').read_text())
    assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256']
    assert hashlib.sha256((ROOT/'data/price_only.npz').read_bytes()).hexdigest()==lock['data_sha256']
    for name,h in lock['source_sha256'].items():assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==h
    checks=json.loads((OUT/'checks.json').read_text());assert checks['passed']
    verification=json.loads((OUT/'verification.json').read_text());assert verification['passed']
    audit=json.loads((OUT/'paper_audit.json').read_text())
    out=['HGA-MT REPLICATION AUDIT AND REDUCED EXPERIMENT','',
      'Verdict: the published results have NOT been fully replicated here. No independent full',
      'replication or relevant public author code was found in the saved searches. Absence from',
      'these searches is not proof that no replication exists anywhere.','',
      'EXACT REPLICATION BLOCKERS',
      'The paper explicitly says the authors lack permission to share data. Original RESSET',
      '39-feature inputs, Hundsun news/MSC checkpoint and 336 Qichacha typed edges are missing.',
      'No author daily predictions, seeds or fully specified walk-forward schedule were found.',
      'Tables 5-8, five baselines, other ablations, sensitivity sweeps and Top1 are not fully replicated.','',
      'PUBLISHED DEFINITION / TABLE INCONSISTENCIES',
      f'Eq20 maximum Top10 MRR = H10/10 = {audit["eq20_top10_upper_bound"]:.9f}.',
      'Table7 reports 0.4587 for Ours and 0.4485 for w/o MT; all six Top10 MRR entries exceed',
      'that maximum under distinct positions in the stated ordered list. The table therefore',
      'requires an alternative metric definition or corrected values. We do not guess which.',
      'All seven Avg. Sharpe entries in Table8 repeat Avg. IRR, contradicting the Top-K rows.',
      'Recomputed averages: Ours 2.033433, w/o MT 1.870367, w/o Risk 1.926100.',
      'Eq14 omits squared deviations although prose defines a standard deviation. Eq23 does',
      'not explain reconciling daily returns and annual risk-free rate. These need clarification.','',
      'COMPLETED REDUCED DIAGNOSTIC (NOT TABLE7 OR TABLE8 REPLICATION)',
      'Real local Tushare/Qlib prices, the same 86 AppendixB stock identifiers, seven-day',
      'lookback, test targets 2020-09-22 through 2021-06-02 (166 dates), three matched seeds.',
      '24 price/volume features; no news, fundamentals or graphs; independently implemented',
      'BiLSTM with 200-dimensional representation and two 64/32 task heads, MAE losses,',
      'alpha=.3, Adam .001, dropout .2, 100 epochs maximum and patience10. Same initialization,',
      'shuffles, dropout seeds, samples and validation-return-MAE checkpoint rule for both fits.',
      'Single fixed split; not the unspecified paper walk-forward. No target standardization.',
      'Risk labels follow the prose: population std of ten close prices at origin t-4 through',
      't+5. Train/validation samples whose risk labels cross the split cutoff are purged.',
      'Inputs use training-only median imputation and min-max fitting. The local calendar has',
      '580 train dates in the stated date range versus the paper\'s581. Full provenance is saved.',
      'Static published universe and vendor data vintage may retain survivorship/revision bias.',
      'This historical follow-up is exploratory; no newly untouched holdout is claimed.','',
      'PRIMARY RESULTS: THREE-SEED ENSEMBLE, MEAN ACROSS TOP3/5/10']
    for r in tests:
        out.append(f'{r["metric"]}: return-only={r["return_only"]:.6f}, multitask={r["multitask"]:.6f}, delta={r["delta"]:+.6f}; 95% block CI [{r["ci_low"]:+.6f}, {r["ci_high"]:+.6f}], two-sided bootstrap p={r["bootstrap_p_two_sided"]:.4f}, Holm p={r["bootstrap_p_two_sided_holm"]:.4f}.')
    out+=['','At a 5% familywise threshold, neither primary improvement is statistically established',
      'after Holm correction (both bootstrap adjusted p=.0592). HAC20 adjusted p=.1062.',
      'Unadjusted confidence intervals above are pointwise, not simultaneous intervals.',
      'Per-seed precision improves for two seeds and declines for one; MRR improves for all three.',
      'These conclusions concern this reduced diagnostic only, not the original model.',
      '', 'Inference pairs daily metrics on the same dates, averages Top-K within each date,',
      'and uses 10-day circular block resampling (5000 draws), with blocks5/20 and HAC20',
      'sensitivities. Seeds are averaged before ranking for primary tests; per-seed results',
      'are separate. Stock-day observations and overlapping Top-Ks are not independent replicates.',
      'Only two primary metrics share the Holm correction; portfolio summaries are exploratory.','',
      'PORTFOLIO LIMITS',
      'The return-only rankings are used for both training variants. IRR follows Eq21 (SUM of',
      'daily equal-weight returns), not compounding. We also report compounded wealth and MDD,',
      'and an explicitly annualized Sharpe using252 sessions and2.5% annual risk-free rate.',
      'The paper\'s precise Sharpe implementation cannot be recovered. Costs are zero; execution',
      'at origin close is assumed despite close-derived signals, so this is not an execution-',
      'realistic trading backtest. Daily ranking scores exclude unobserved future returns,',
      'which is an evaluation sample restriction, not an investable missing-return policy.',
      'No claim that return supervision improves volatility forecasting: no volatility-only',
      'baseline is tested. No result from this diagnostic validates or refutes the full model.','',
      'REUSE',
      'Existing local CSI data and financial-graph-reproduction Python environment reused.',
      'Paired HAC/circular bootstrap adapted from monthly-coforecast/scripts/evaluate.py.',
      'No relevant HGA-MT author code was found; model reconstruction is independent.','',
      'RUN',
      'From the prepared HGA-MT workspace use .venv/bin/python:',
      'hga-mt-reproduction/scripts/prepare.py; scripts/audit_paper.py; scripts/checks.py;',
      'scripts/run.py; scripts/verify.py; scripts/evaluate.py (prefix each script with hga-mt-reproduction/).',
      'Run.py skips completed fits and refuses changed input/code hashes. Preserve protocol',
      'and lock files; changed designs require a separate archived result directory.']
    (ROOT/'REPORT.txt').write_text('\n'.join(out)+'\n')
    status={'exact_replication':False,'exact_replication_status':'blocked_missing_original_data_code_and_metric_clarification',
            'reduced_diagnostic':'complete','completed_fits':6,'test_days':166,'verification_passed':True,
            'primary_results':tests,'published_table_inconsistencies':{'mrr_bound_violations':audit['table7_mrr_bound_violations'],'table8_average_sharpe_errors':audit['table8_average_sharpe_errors']}}
    (ROOT/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(pd.DataFrame(tests).to_string(index=False))

if __name__=='__main__':main()
