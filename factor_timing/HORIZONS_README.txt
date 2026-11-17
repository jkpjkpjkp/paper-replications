THREE-PARAMETER FACTOR / ENTRY / EXIT STUDY

Primary result folder: horizon_results/
Realized-date descriptive diagnostic: horizon_oracle_results/

Baseline requested by user: same factor, buy at announcement forecast minus
one trading-session close, sell one session after actual release (subject to
the data's availability delay). For each N,M candidate the direct baseline
uses the same factor and its own -1/+1 eligible event opportunities.

Grid: all209 downloaded stock-level predictors, N=1..30, M=1..30.
There are188100 event-return regressions. Regression screens are event-weighted
approximations, not calendar portfolio performance. All final strategy returns
are exact daily portfolios from a frozen training shortlist.

From /home/jkp/data/earnings-timing:
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/horizons.py prepare
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/horizon_regress.py
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/horizons.py train
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/horizons.py evaluate
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/horizon_report.py
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/horizon_validate.py

Optional separate full actual-date diagnostic:
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/horizon_oracle.py all

Entry convention: forecast F, buy close C=F-N, first earned return C+1.
Rank snapshot: previous completed month relative to first return day. This
permits that just-completed month at a month-end entry close. Rank/sign freezes
at entry. Daily same-leg rescaling is required to maintain dollar neutrality.
Exit close=min(F+40,max(next actual release+M,next data-known close)); unknown
next release uses F+40 timeout. Actual future dates never select primary entry.
Holding continues through analyst coverage/regularity changes. Missing listing
rows force exit at last mark. Exactly scheduled new entries only; no delayed
starts or reentries of abandoned spells.

Portfolio: +100% long/-100% short of after-fee NAV when both legs can form.
At least one stock each side in primary analysis; report concentration.
When either side empties, flatten the other and abandon those spells. A min5
per side sensitivity is reported. Every dollar bought or sold costs5bp; short
borrow is1% annual; every study endpoint includes terminal liquidation. Gross
and10bp sensitivities are provided for primary/baseline. Bankruptcy, if any,
stops that scenario and excludes a training strategy from selection.

Training1985-2004; labels purged to common forecast range beginning30sessions
after training start and ending40sessions before training end. Freeze choices
before2005+ evaluation. Validation2005-2014, holdout2015-2025. Related holdouts
were previously inspected, so retrospective search/publication hindsight remains.

The actual-date oracle aligns entries using the realized future release date.
It describes an event pattern but cannot establish an implementable strategy
without a point-in-time historical announcement schedule. It is never used
for tradable parameter selection or described as live out-of-sample evidence.

Accounting and label validation: horizon_validate.py uses a separate absolute-
dollar portfolio ledger with scalar bisection and raw daily return products.
