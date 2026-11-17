"""Summarize completed CZ202 runs without mixing them with daily CSI results."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/cz202'
completion = json.loads((OUT/'completion.json').read_text())
expected = [(v,int(s)) for v in completion['variants'].split(',') for s in completion['seeds'].split(',')]
runs = [json.loads((OUT/f'{v}-seed{s}'/'metrics.json').read_text()) for v,s in expected]
assert all(r['checks']['passed'] and r['test_months']==48 for r in runs)
summary = {}
lines = ['CZ202 monthly US adaptation — completed 2026-11-17',
         '================================================',
         'Original intraday paper replication grade: 0 (no comparable private-panel experiment).',
         'This report describes fresh monthly training, not an original Table 3 match.',
         'Training 2009-2019; validation 2020-2021; test Jan 2022-Dec 2025 (48 months).',
         '202 characteristics; 16 calendar months; full cross sections; fixed seeds.',
         'Value-weighted D10-D1 (+1/-1), lagged NYSE20 size screen, 10bp/traded dollar.',
         'Source gaps and imputation: ../../DATA.txt. Frozen settings: protocol.json.',
         '', 'Per-run test results:',
         'Variant / seed | IC | Rank IC | Gross Sharpe | Net Sharpe | chosen epoch']
for r in runs:
    lines.append(f"{r['variant']} / {r['seed']} | {r['ic']:.6f} | {r['rank_ic']:.6f} | {r['gross_sharpe']:.6f} | {r['net_sharpe']:.6f} | {r['best_epoch']}")
for variant in sorted({r['variant'] for r in runs}):
    group = [r for r in runs if r['variant']==variant]
    summary[variant] = {key:dict(mean=float(np.mean([r[key] for r in group])),
        seed_sd=float(np.std([r[key] for r in group],ddof=1)) if len(group)>1 else None)
        for key in ['ic','rank_ic','gross_sharpe','net_sharpe']}
    lines += ['',variant+' equal-seed mean +/- seed SD (SD is not a confidence interval):']
    lines += [f"  {key}: {x['mean']:.6f} +/- {x['seed_sd']:.6f}" if x['seed_sd'] is not None else f"  {key}: {x['mean']:.6f} (one seed)" for key,x in summary[variant].items()]
lines += ['', 'FF5+UMD annualized alpha (%), HAC t, 95% CI (%), nominal two-sided p:',
          'Bartlett six monthly lags, corrected covariance, Student t with T-7 df.',
          'These are time-series uncertainty estimates; three seeds share the same test period.']
for r in runs:
    for kind in ['gross','net']:
        a=r['regression'][kind];lo,hi=a['annual_alpha_ci95']
        lines.append(f"{r['variant']} / {r['seed']} {kind}: {100*a['annual_alpha']:.4f}%; t={a['hac_t']:.4f}; CI [{100*lo:.4f}%, {100*hi:.4f}%]; p={a['p_value']:.6f}")
lines += ['', 'Validation and limitations:',
          f'All {len(runs)} runs completed; saved checkpoint samples replay exactly; decile-return',
          'and fee-accounting identities pass. Source hash, complete CRSP universe, lagged',
          'capitalization/NYSE20, calendar timing and realized returns independently checked.',
          'No missing return labels; substantial missing features/history are neutral-filled.',
          'Current-vintage inputs are not an as-published vintage. Costs omit borrow fees,',
          'financing, heterogeneous spreads, market impact and taxes. No test tuning.',
          'The shared evaluator JSON also includes a legacy Bonferroni-12 value for its',
          'original six-paper study; it is not this experiment\'s multiplicity correction.',
          'Paper-versus-local absolute error and tolerance: N/A (different data/protocol).',
          'Original 27 daily CSI results remain separate; original private intraday inputs',
          'and specification gaps are unresolved. See ../../STATUS.txt and ../../DATA.txt.']
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
(OUT/'summary.json').write_text(json.dumps(dict(completed_runs=len(runs),summary=summary),indent=2)+'\n')
print('\n'.join(lines))
