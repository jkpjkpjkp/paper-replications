DECILE FACTOR COMBINATIONS AND THE EARNINGS ANNOUNCEMENT CYCLE

This is a separate, retrospective extension of Linnainmaa and Zhang (2018),
The Earnings Announcement Return Cycle (../13699_paper_YrbKK6Zn.pdf).
See design.json for the finite grid, timing conventions, selection and costs.
The source data and earlier studies are not modified.

QUESTION AND COMPARISON
Does conditioning an original factor's extreme long/short portfolios on the
announcement cycle increase gross returns, and does that increase survive
costs? Every combination is compared with the SAME original-factor portfolio,
characteristic half/aggregation, EW/VW weighting and activity threshold with
the timing gate removed. All market dates, including cash days, count.
Positive own return and positive incremental return are separate requirements.

GRID
Six existing factors: MS, CompEquIss, InvestPPEInv, Cash, RIO_Disp and RDIPO.
Monthly source signals are signed using SignalDoc.Sign and ranked within the
source month/universe, with average ranks for ties; signals enter only in the
next calendar month, with no forward fill.

D10: factor long rank>0.90, short rank<=0.10. Cross with:
 - no characteristic split; or size, ADV, previous63-session total volatility;
 - low half, high half, or balanced equal halves for each characteristic;
 - EW or lagged-market-cap VW;
 - no timing, all4 high/low timing-decile leg pairs, and all9 paper phase pairs.
This is280 configurations per factor.

S30: factor long rank>0.70, short rank<=0.30, crossed independently with the
same characteristic halves and EW/VW choices, with no timing or9 phase pairs.
This is180 configurations per factor. The 2x3 grid means characteristic halves
crossed with factor low30%, middle40%, high30%; the middle is not held. It is
not a reconstruction of NYSE-breakpoint Fama-French portfolios.

Total2760 factor configurations plus2 timing-only comparators. This searches
all members of the stated grid, not all conceivable factor or timing rules.
MS, RIO_Disp and RDIPO are discrete. Ties are NEVER split with security IDs
or random noise. Tail sizes can differ from10%/30%, and a missing factor tail
makes its strategy inactive. A binary RDIPO split cannot be called a genuine
equal-count decile. Inactivity is a substantive result, not repaired by ties.

TIMING AND WEIGHTS
The exact paper phases are P1 age6..20, P2 age21..35, P3 age36..50 trading days
since latest known announcement (Table3, printedp37). Paper baseline buysP1
and shortsP3, equal-weight and daily rebalanced.

The continuous score cos(2*pi*(age-13)/65) is a HAND-CHOSEN extension: peak at
phase1 midpoint13, trough45.5, fixed65-session period. It is not a formula
from the paper or a fit to this grid's returns. Daily average-tie ranks across
all eligible stocks define high>0.90 and low<=0.10. For pair H_L, original
factor longs need high timing, original factor shorts need low timing. The cosine repeats near age78 while the panel allows age<=80; that repeat
does not assert that another announcement occurred. Every
corner and every phase pair is tested without reversing original factor sign.

Characteristic half ranks are independent of factor and timing, daily across
eligible stocks: low<=0.5, high>0.5, average ties. All/low/high portfolios target
+1 long and-1 short per1 NAV. Balanced assigns+0.5 long,-0.5 short to EACH
half. Every required leg/cell needs>=5 names; otherwise the whole book is cash.
EW/VW normalizes inside each funded cell. VW is uncapped; a name-count minimum
does not guarantee low concentration. Cached factor-rank cutoffs share the
float32 rank representation to preserve exact endpoint inclusion. Repeat selected rules with minimum20
as a concentration sensitivity. Portfolios rebalance daily.

GROSS FIRST, THEN COSTS
Training is1985-2004. Per factor and D10/S30 family, retain best positive gross
uplift and best positive HAC20 t of paired uplift, deduplicated. Candidates
must have positive own mean, positive incremental mean,>=50% active dates and
matched baseline>=80% active dates. No passing strategy means no selection.
The primary rule maximizes training paired HAC20 t across all eligible models.
Choices freeze before selected later-period evaluation. Later-period all-grid
rankings are descriptive ex-post searches, not independently chosen winners.
Correct full-grid inference for the full tested comparison family, not just
the handful of winners. Corrections do not cover the preceding related studies.

Selected models and matched baselines run with separate exact NAV/holdings
accounts at0,1,5,10,25bp per dollar traded. Nonzero-cost scenarios also charge
1%/year short borrow; zero costs have no borrow. Turnover counts all bought/
sold dollars, not half turnover. Initial opening and terminal liquidation are
charged at each simulation endpoint. Daily fees reduce investable capital.
Dividends remain cash until the next rebalance. Observed missing returns earn
zero; missing held securities close at last observed marks with charged costs.
Annual arithmetic mean, CAGR and Sharpe are distinct statistics. Cost scenarios
omit impact, stock-specific borrow availability, financing, collateral rebates
and cash interest; they are not evidence of live implementability.

PAPER DEVIATIONS AND EVIDENCE LIMITS
The paper uses>=5 analysts in the prior3 months, at least4 releases in400 days,
latest gap70..110 calendar days, and quarter-fixed characteristic quintiles.
The reused regular65 panel instead uses the>=5/180-day active analyst proxy,
last3 gaps each50..80 market sessions with median60..70, and age<=80. Events
are sourced from IBES rather than Compustat rdq. sigma63 is total volatility,
not one-year Carhart residual volatility; ADV is not Amihud illiquidity.
This study's deciles, 2x3, daily characteristic sorts and cost models are
extensions, not exact replication of the original paper.

No realized future announcement is used. known_idx<idx and event_idx<idx,
lagged prices/market cap/liquidity, and prior-month factors define decisions.
Historical signal revisions, exact-CUSIP retrospective links and imperfect
vendor vintages remain inherited limitations.

All six factors were published AFTER2004: MS2005,CompEquIss2006,
InvestPPEInv2008,Cash2012,RIO_Disp2005,RDIPO2006. The paper is dated2018, and
related validation/holdout returns were already examined. Therefore the
1985-2004 selection and2005-2025 checks are retrospective chronological
research, not a pristine holdout or prospective test. Returns and paired
uplift are not causal alpha. Post2018 results are useful regime evidence,
not a new untouched holdout.

READ AND REPRODUCE
Run from /home/jkp/data/earnings-timing. Set OPENBLAS_NUM_THREADS=1 and
OMP_NUM_THREADS=1 in the shell environment, then run in this order:

 .venv/bin/python decile_grid/screen.py --start 1985 --end 2004
 .venv/bin/python decile_grid/screen.py --start 2005 --end 2025 --no-freeze
 .venv/bin/python decile_grid/full_grid_report.py
 .venv/bin/python decile_grid/evaluate.py --start 1985 --end 2004
 .venv/bin/python decile_grid/evaluate.py --start 2005 --end 2025
 .venv/bin/python decile_grid/evaluate.py --start 2005 --end 2025 --min-names 20
 .venv/bin/python decile_grid/validate.py
 .venv/bin/python decile_grid/report.py

Optional descriptive after-cost evaluation of per-factor/family highest paired gross
uplift and highest paired gross HAC20 t selected USING2015-2025 itself:

 .venv/bin/python decile_grid/evaluate.py --start 2015 --end 2025 --expost-holdout
 .venv/bin/python decile_grid/report.py

These ex-post selectors require>=50% candidate and>=80% baseline active days
in2015-2025 but do not require positive own return, uplift or training eligibility.
Outputs use the expost_exact_2015_2025_min5 prefix and leave frozen.json
unchanged. Their cost settings are the same five scenarios without retuning.
They test whether ex-post gross point maxima survive those costs; selecting
them on the evaluation years does not create independently validated winners.

validate.py independently checks all2762 model weights on two actual dates,
at minimum5 and20, against a scalar reference; source-month ranks and raw
CRSPret/retx; rank ties and sparse balanced cells; HAC against statsmodels;
and an absolute-dollar accounting simulation with held-share drift, dividends,
missing data, all five cost scenarios and terminal liquidation. It also
reports full-period maximum single-security weight of selected VW models.
