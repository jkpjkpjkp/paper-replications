from pathlib import Path
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
f = pd.read_csv(OUT/'summary.csv')
main = f[~f.diagnostic].copy()


def find(project, case):
    return f[(f.project == project) & (f.case == case)].iloc[0]


def describe(row):
    return (f'{row.annual_mean:.2%} arithmetic annual mean; {row.cagr:.2%} CAGR; '
            f'Sharpe {row.sharpe:.2f}; HAC 95% CI [{row.ci_low:.2%}, {row.ci_high:.2%}]; '
            f'p={row.p:.5g}, Holm across audit={row.p_holm_audit:.5g}; '
            f'{row.n} {row.frequency} observations, {row.start} through {row.end}.')


a = find('LiMT', 'csi300-3seed')
b = find('LiMT', 'csi500-2seed')
h = find('HERMES', 'rolling-2023-2025-3seed')
old = find('HERMES', 'sp500-no_skip-seed0')
alpha = find('AlphaPROBE', 'csi1000')
arb = find('ArbNN', 'arbnn-q0.1')
prophet = find('Prophets Profit', 'brier-dollar_beta_industry_size')

text = f'''ZERO-COST EQUAL-DOLLAR LONG/SHORT AUDIT — 7 November 2026

Yes: LiMT shows statistically significant positive historical gains before all
implementation costs, in both universes with completed saved forecasts. HERMES
has positive nominal evidence that does not survive this audit's multiple-test
correction. No significant positive gross gain was found for AlphaPROBE,
Sparse Tangent's new neutral adaptation, ArbNN, or the Prophets Profit equity
adaptation. This is an audit of fixed local experiments, not an assertion that
every possible implementation of these papers succeeds or fails.

Principal results

LiMT CSI300, equal sleeves across all three completed seeds:
{describe(a)}
Bootstrap20 annual-mean CI [{a.bootstrap_ci_low:.2%}, {a.bootstrap_ci_high:.2%}].
Maximum drawdown {a.max_drawdown:.2%}. Mean netted gross exposure
{a.mean_gross_exposure:.2%} of NAV, approximately {a.mean_gross_exposure/2:.2%}
each side; the remainder is uninvested. This result is not scaled to 100% gross.

LiMT CSI500, equal sleeves across both completed seeds:
{describe(b)}
Bootstrap20 annual-mean CI [{b.bootstrap_ci_low:.2%}, {b.bootstrap_ci_high:.2%}].
Maximum drawdown {b.max_drawdown:.2%}. Mean netted gross exposure
{b.mean_gross_exposure:.2%}, approximately {b.mean_gross_exposure/2:.2%} each side.
The third seed has no test predictions; its saved status stopped at epoch43.
Both LiMT aggregate gains also have positive HAC(5), HAC(60) and 60-session
bootstrap lower bounds. Every one of the five available individual seeds is
positive with nominal HAC p<0.000001. Seeds are not independent market samples.

HERMES recent rolling no-skip, equal sleeves across all three seeds:
{describe(h)}
Mean netted gross exposure {h.mean_gross_exposure:.2%}; equal dollar books.
Historical no-skip seed0, chosen after the old test results were inspected:
{describe(old)}
Both are nominally significant; neither passes Holm across the 18 HERMES
aggregate rows or all57 audit rows. The original full-model three-seed result
is only +0.11% annual mean, Sharpe0.01. The old selected ablation is not the
headline HERMES model, and the later tests reuse the originally fixed universe.

AlphaPROBE, all three original universes:
'''
for u in ['csi300','csi500','csi1000']:
    text += f'{u}: {describe(find("AlphaPROBE", u))}\n'
text += f'''
Even CSI1000 has a confidence interval spanning zero. Equal dollars hold at
the open; maximum intraday net drift ranges from 1.7% to 2.7% of NAV across
the universes. Executability restrictions
and stale positions can move gross exposure slightly above or below100%.

Sparse Tangent:
A genuine equal-dollar adaptation was required: the old5bp analysis had
100% net stock exposure. We subtract a contemporaneously eligible equal-weight
stock hedge from each saved portfolio, net the stock positions, then normalize
to +50%/-50% daily. All four markets, three cardinalities per market, DFL/PFL
and five seeds were evaluated. None of the12 DFL or12 PFL seed-average settings
has a significant positive mean, even before correction. DFL annual means
range from -5.27% to +1.97%. This overlay changes the original sparsity and
rebalancing rule. It assumes ideal daily closing fills and available shorts;
the Korean short book is hypothetical despite the original conservative
long-only implementation. It is not a literal paper reproduction.

ArbNN, primary top/bottom10% ensemble:
{describe(arb)}
The20% basket also loses before costs (-0.41% annual mean). The earlier report
used +100%/-100%; dividing gross returns by two gives the +50%/-50% convention
here. This changes annual mean/compound return but leaves mean-test p and
Sharpe invariant. Monthly source returns include dividends.

Prophets Profit equity adaptation, stronger neutrality:
{describe(prophet)}
Dollar-only Brier is -1.18% annual mean, Sharpe-0.18, p0.430. All eight saved
forecast-sizing/neutrality variants lose before all costs. A cost_bps=0 row in
the old file still charges borrowing; this audit uses gross_return instead.
The original event-contract ROI results are not equal-long/short stock returns.

song26l US minute experiment:
No completed test forecasts or return ledger exists. The saved coordinator
status says building_available_tensors; no reproduction/results/status.json
exists. No neutral return or significance can be inferred from downloads or
engineering checks. No duplicate training/download job was launched.

Separate 437-paper programme:
The saved canonical reviews contain435 pending cost-extension tests and2
not_applicable entries. They are not435 failed or nonprofitable tests. This
audit does not claim to have run those absent experiments; canonical ownership
and review files are preserved. This distinction is recorded in
programme-coverage.json.

Statistical interpretation and scope

The audit contains57 aggregate comparisons and134 individual-seed diagnostics,
191 result rows in total. Days/months are the sampling units; initializations
on the same dates are never counted as extra observations. Annual mean is
252x daily mean or12x monthly mean; CAGR and Sharpe are separate columns.
The zero benchmark is the self-financing overlay with no cash/collateral yield.
All commissions, borrowing, financing, impact and slippage are set to zero.
This is a zero-implementation-cost experiment; capital and market risk remain.

Mean inference uses two-sided Bartlett HAC,20 daily or6 monthly lags, a
small-sample correction and t reference distribution. Block-bootstrap intervals
use5000 circular resamples,20-session/12-month blocks, with60-session daily
sensitivity. Independent Bartlett arithmetic agrees with statsmodels for every
reported lag. Holm corrections cover the enumerated aggregate tests within
each project and across this audit. They do not undo earlier unrecorded model
choices, test inspection, or the wider search over hundreds of papers.

All experiments are retrospective. LiMT's significance is strong conditional on
this dataset/strategy, but does not certify a future edge, absence of leakage,
or actual fills. Its test period is2017-2020, not recent market performance.
Original author preprocessing/private execution code is unavailable. The local
vendor dataset starts2010 rather than the paper's2008; adjusted-close dividend
and historical-constituent conventions remain vendor-dependent. Execution uses
signal t, t+1 close fills and t+2 returns, with future realized execution-day
volume limiting fills. That is an idealized fill model using same-session
volume, not an independently observed auction execution record. Suspended
prices are carried forward. Actual stock-borrow availability is assumed.

AlphaPROBE additionally uses a contemporary LLM and an already inspected test
sample; historical universe/delisting and adjustment coverage remain unresolved.
HERMES uses a fixed historical474-security universe; old public data have
documented stale-delisting issues. Recent CRSP tests improve payoff data but
do not replace the universe with a dynamic point-in-time one. ArbNN/Prophets
use long historical CRSP samples but retain source-vintage/missing-settlement
and ideal monthly execution limitations. Dollar neutrality throughout this
report is the user's definition; it does not assert zero market beta.

Verification and artifacts

- summary.csv: all estimates, CIs, raw/adjusted p-values and lag sensitivities.
- return-series.csv: the actual calendar-time gross portfolio streams.
- *_zero_ledger.csv / *_zero_weights.npz: LiMT replay ledgers/weights and
  Sparse Tangent neutral overlay weights.
- verification.json:304 hashed source inputs, software versions, numerical
  invariants and incomplete-run notes. Final source stability is checked in
  final-checks.json. LiMT's four relevant existing neutral-engine tests passed.
- PROTOCOL.txt: fixed evaluation rules; session-scope.json: user-request trace.
- effect_intervals.png: annual means and95% HAC intervals for key comparisons.
- report.html: formatted report with all57 aggregate rows.

Rerun from /home/jkp/data:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 no-cost-neutral-audit/.venv/bin/python no-cost-neutral-audit/analyze.py
  no-cost-neutral-audit/.venv/bin/python no-cost-neutral-audit/make_report.py
Environment: requirements.lock.txt. No original experiment outputs were edited.

Methods and workflow sources

statsmodels official robust covariance documentation, checked7 November2026:
https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html
The statsmodels skill informed robust inference and multiplicity reporting.
Its requested software/workflow attribution is included here:
Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026).
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
https://doi.org/10.48550/arXiv.2609.00065
Current arXiv record v2, author list/year checked7 November2026; no journal
reference was listed. This is workflow attribution, not evidence of profitability.
'''
# Improve spacing in the prose without changing identifiers or numeric values.
for before, after in {
    'epoch43':'epoch 43', 'all57':'all 57', 'the 18 HERMES':'the 18 HERMES',
    'Sharpe0.01':'Sharpe 0.01', 'historical474':'historical 474',
    'old5bp':'old 5 bp', 'the12':'the 12', 'or12':'or 12',
    'below100%':'below 100%', 'above100%':'above 100%',
    'top/bottom10%':'top/bottom 10%', 'The20%':'The 20%',
    'Sharpe-0.18':'Sharpe -0.18', 'p0.430':'p 0.430',
    'contain435':'contain 435', 'and2':'and 2', 'not435':'not 435',
    'contains57':'contains 57', 'and134':'and 134',
    'or12x':'or 12 x', 'HAC,20':'HAC, 20', 'or6':'or 6',
    'use5000':'use 5,000', 'resamples,20':'resamples, 20',
    'with60':'with 60', 'for 100%':'for 100%',
    'period is2017':'period is 2017', 'starts2010':'starts 2010',
    "paper's2008":"paper's 2008", 'with60':'with 60',
    'documentation, checked7 November2026':'documentation, checked 7 November 2026',
    'checked7 November2026':'checked 7 November 2026',
    'and95%':'and 95%', 'all57':'all 57',
}.items():
    text = text.replace(before, after)
(OUT/'REPORT.txt').write_text(text)

keys = [('LiMT','csi300-3seed','LiMT CSI300 (3 seeds)'),
        ('LiMT','csi500-2seed','LiMT CSI500 (2 seeds)'),
        ('HERMES','rolling-2023-2025-3seed','HERMES rolling 2023–2025'),
        ('HERMES','sp500-no_skip-seed0','HERMES selected 2021–2022 ablation'),
        ('AlphaPROBE','csi300','AlphaPROBE CSI300'),
        ('AlphaPROBE','csi500','AlphaPROBE CSI500'),
        ('AlphaPROBE','csi1000','AlphaPROBE CSI1000'),
        ('ArbNN','arbnn-q0.1','ArbNN top/bottom 10%'),
        ('ArbNN','arbnn-q0.2','ArbNN top/bottom 20%'),
        ('Prophets Profit','brier-dollar','Prophets Profit: dollar neutral'),
        ('Prophets Profit','brier-dollar_beta_industry_size','Prophets Profit: stronger neutrality')]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, ax = plt.subplots(figsize=(11.2,7.2))
fig.subplots_adjust(left=.37, right=.98, bottom=.18, top=.9)
for i, (project, case, label) in enumerate(keys):
    row = find(project, case)
    color = '#167b62' if row.positive_audit_adjusted else ('#be7c18' if row.positive_nominal else '#526779')
    ax.errorbar(row.annual_mean*100, i,
                xerr=np.array([[row.annual_mean-row.ci_low], [row.ci_high-row.annual_mean]])*100,
                fmt='o', color=color, capsize=3, markersize=5)
ax.set_yticks(range(len(keys)), [k[2] for k in keys])
ax.invert_yaxis()
ax.axvline(0, color='#333333', lw=.8)
ax.set_xlabel('Annual arithmetic mean (%) and individual 95% HAC confidence interval')
ax.set_title('Zero-cost equal-dollar long/short returns', loc='left', weight='bold')
ax.grid(axis='x', alpha=.18)
fig.text(.01,.035, 'Green: passes audit Holm correction. Gold: nominal only. All tests retrospective; portfolio exposures differ.', fontsize=8)
fig.savefig(OUT/'effect_intervals.png', dpi=180, bbox_inches='tight')
plt.close(fig)

display = main[['project','case','start','end','annual_mean','cagr','sharpe',
                'ci_low','ci_high','p','p_holm_project','p_holm_audit','bootstrap_ci_low','bootstrap_ci_high']].copy()
for c in ['annual_mean','cagr','ci_low','ci_high','bootstrap_ci_low','bootstrap_ci_high']:
    display[c] = display[c].map(lambda x:f'{x:.2%}')
for c in ['p','p_holm_project','p_holm_audit']:
    display[c] = display[c].map(lambda x:f'{x:.5g}')
display.sharpe = display.sharpe.map(lambda x:f'{x:.2f}')
page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zero-cost neutral returns audit</title><style>
body{font:16px/1.55 system-ui;margin:32px auto;max-width:1300px;padding:0 24px;color:#1c2933}
pre{white-space:pre-wrap;font:14px/1.65 system-ui;max-width:1050px}
table{border-collapse:collapse;font-size:12px}td,th{border-bottom:1px solid #ddd;padding:7px;text-align:right}
td:nth-child(-n+2){text-align:left}img{max-width:100%;height:auto}.table{overflow-x:auto}
a{color:#116a69}h1{font-size:29px}</style><h1>Zero-cost equal-dollar long/short audit</h1>
<p><strong>LiMT passes the enumerated significance checks. HERMES is nominal only; the other tested methods do not show significant positive gains.</strong></p>
<p><a href="summary.csv">All 191 rows</a> · <a href="return-series.csv">Return streams</a> ·
<a href="verification.json">Numerical/source verification</a> · <a href="PROTOCOL.txt">Protocol</a></p>
<img src="effect_intervals.png" alt="Annual mean gross returns with HAC confidence intervals">
<h2>All 57 aggregate comparisons</h2><p>Annual mean and interval columns use arithmetic annualization. P-values are two-sided. Seeds on the same dates are not independent samples.</p>
<div class="table">'''+display.to_html(index=False, escape=True)+'''</div><h2>Evidence, limitations and reproduction</h2><pre>'''+html.escape(text)+'''</pre></html>'''
(OUT/'report.html').write_text(page)
print('Created REPORT.txt, report.html, effect_intervals.png')
