"""Consolidate actual completed experiments without selecting by performance."""
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path('/home/jkp/data');HERE=Path(__file__).resolve().parent
rows=[];checks=[];statuses=[]
def check_outputs(directory,verification):
    v=json.loads(verification.read_text())
    assert v.get('passed',v.get('status')=='passed')
    n=0
    for name,digest in v.get('output_sha256',{}).items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name;n+=1
    checks.append(dict(directory=str(directory),saved_verification_passed=True,output_hashes_rechecked=n))
def monthly(name,path,reused):
    if not (path/'evaluation/report.json').exists():
        status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {'stage':'not_started'}
        statuses.append(dict(paper=name,reused=False,**status));return
    report=json.loads((path/'evaluation/report.json').read_text())
    verified=(path/'evaluation/verification.json').exists() and json.loads((path/'evaluation/verification.json').read_text()).get('passed',False)
    for side,v in report['results'].items():
        rows.append(dict(paper=name,cost_bps=0 if side=='gross' else 10,annual_mean_return=v['mean_annual_return'],annual_alpha=v['annual_alpha'],hac_t=v['hac_t'],
            p=v['p_value'],ci_low=v['annual_alpha_ci95'][0],ci_high=v['annual_alpha_ci95'][1],frequency='monthly',
            test='2022-2025',universe='all eligible US common stocks',reused=reused,verified=verified,source=str(path/'evaluation/report.json')))
    statuses.append(dict(paper=name,stage='complete' if verified else 'evaluation_pending_verification',reused=reused))
monthly('DoubleAdapt',ROOT/'ordered-reproductions/results/doubleadapt-cz',True)
for name,rel in [('LiMT','limt-us-reproduction/experiments/standard_characteristics'),('PRISM-VQ','prism-vq-reproduction/experiments/cz_vw')]:
    directory=ROOT/rel/'analysis';check_outputs(directory,directory/'verification.json')
    data=pd.read_csv(directory/'alpha.csv');performance=pd.read_csv(directory/'performance.csv')
    if name=='LiMT':
        data=data.loc[data.weighting.eq('value')&data.window.eq('full-2017-2025')&data.factor_model.eq('FF5+UMD')]
        performance=performance.loc[performance.weighting.eq('value')&performance.window.eq('full-2017-2025')]
        if len(data)!=2:raise ValueError('Unexpected LiMT factor model label: inspect alpha.csv')
    assert len(data)==2
    for _,r in data.iterrows():
        mean_return=float(performance.loc[performance.cost_bp.eq(r.cost_bp),'annual_mean'].iloc[0])
        rows.append(dict(paper=name,cost_bps=r.cost_bp,annual_mean_return=mean_return,annual_alpha=r.annual_alpha,hac_t=r.get('hac_t',r.get('t')),p=r.p,
            ci_low=r.ci_low,ci_high=r.ci_high,frequency='daily',test='2017-2025' if name=='LiMT' else '2022-2024',
            universe='historical US large500' if name=='LiMT' else 'historical S&P500',reused=True,verified=True,source=str(directory/'alpha.csv')))
    statuses.append(dict(paper=name,stage='complete',reused=True))
for name in ['TRA','AlphaSAGE','AlphaPROBE']:monthly(name,HERE/'results'/name,False)
df=pd.DataFrame(rows);df['p_bonferroni_12']=(df.p*12).clip(upper=1);df.to_csv(HERE/'results_summary.csv',index=False)
(HERE/'status.json').write_text(json.dumps(dict(papers=statuses,reuse_checks=checks),indent=2))
lines=['CZ202 protocol.txt — six papers','',
 'Scope authorized 2026-11-16: AlphaSAGE, DoubleAdapt, TRA, PRISM-VQ, LiMT, AlphaPROBE. HERMES excluded.',
 'Original numerical-fidelity gates do not block these explicitly requested CZ adaptations.',
 'All primary portfolios: lagged NYSE20 exclusion; 202 CZ inputs; value-weighted D10-D1; +1 long/-1 short; 10bp per traded dollar.',
 'FF5+UMD with intercept; no second RF subtraction for zero-investment spread. Bartlett HAC: monthly6 / daily20, 95% CI and two-sided p.',
 'Existing daily large-cap experiments and new monthly all-US experiments have different samples/dates. Do not rank them as a controlled head-to-head comparison.',
 'Training for new runs: 2009-2019; validation2020-2021; test2022-2025; source verified monthly panel reused. Seed and budgets fixed before test.',
 'AlphaSAGE: released GFN/GNN, 10000 episodes, pool50; 202 CZ expression terminals. AlphaPROBE: released graph and paper-aligned retrieval, 20 rounds, pool50, 29 fixed CZ seed characteristics; DeepSeek Flash + local Qwen3 4B Q8.',
 'TRA: released two-stage ALSTM+TRA, 3/10 routing states, hidden256x2; 12 monthly histories and next-month label; calendar-safe dataset.',
 'Symbolic models: frozen mined expressions; expanding historical RankIC filtering and OLS fitted only through previous realized month.',
 '', 'Paper | status | reused']
lines += [f"{r['paper']} | {r['stage']} | {r['reused']}" for r in statuses]
lines += ['', 'Completed results: annual alpha and CI in percent; nominal p; family-adjusted p in CSV',
 'Paper | cost bp | annual arithmetic return % | annual alpha % | HAC t | 95% CI % | p | test | frequency']
lines += [f"{r.paper} | {r.cost_bps:g} | {r.annual_mean_return*100:.4f} | {r.annual_alpha*100:.4f} | {r.hac_t:.4f} | [{r.ci_low*100:.4f}, {r.ci_high*100:.4f}] | {r.p:.6g} | {r.test} | {r.frequency}" for r in df.itertuples()]
lines += ['', 'Sources are in results_summary.csv. Reused results were not retrained. New complete status requires evaluation and independent verification.',
 'No historical data vintage equivalence is claimed. Borrow, financing and market impact are outside the retained trading-cost model.',
 'Refresh: ordered-reproductions/.venv-tra/bin/python protocol-six-20261116/summarize.py']
(HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print(df[['paper','cost_bps','annual_alpha','hac_t','p']].to_string(index=False))
