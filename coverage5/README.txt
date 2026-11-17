FIVE-ANALYST EARNINGS-DATE-ONLY TRADING RESEARCH

LATEST BROAD LONG/SHORT SEARCH: exploration/results/report.html.
See exploration/README.txt for the complete reproduction workflow. It screens
7,650 date-window pairs, exactly evaluates shortlisted pairs and smooth
quarterly scores, tests slower adjustments and annual rolling/expanding refits.
All new search portfolios are zero-net-target long/short books.

CURRENT USER-REQUESTED FOCUS: regular approximately 65-trading-session releases.
LATEST: equal-dollar SPY hedge check in results_regular65/spy_hedge_report.html.
Reproduce with audit_long_only.py then hedge_spy.py using the Python below.
The hedge holds $1 short SPY against $1 long announcement basket at each daily
rebalance; both legs are zero when fewer than 20 long candidates qualify.
Results include raw-return ledger validation, cost/borrow scenarios, and an
audit of portfolio exposure versus individual-stock holding periods.

Open results_regular65/report.html and results_regular65/summary.txt.
Reproduce with analyze_regular65.py and validate_regular65.py using the same
Python executable below. This profile requires all last three known release
gaps in 50..80 trading sessions, median 60..70, and current age <=80 sessions.
Its primary forecast is latest release + median of those trading-session gaps.
Candidate nearest-clock bins run only -40..+35; stale +81..130 is excluded.
All five-analyst, execution and cost definitions below otherwise apply.

The original broad-universe run below is retained as a preliminary analysis.
Open results/report.html for the findings and exact trading rules.
Read results/summary.txt for the primary chronological evaluation.

Reproduce from this directory:
  /home/jkp/data/earnings-timing/.venv/bin/python analyze.py
  /home/jkp/data/earnings-timing/.venv/bin/python validate.py

Use --reuse-panel to reuse the derived coverage states and stock-day panels.
Use --reuse-returns only when regenerating statistical tables/reports from
unchanged daily returns. The frozen design rejects changed definitions.
Dependencies are in ../requirements.txt.

The existing ../results study does NOT apply the analyst-coverage restriction.
This separate analysis reconstructs it from completed det_epsus files, using
distinct analyst identifiers, conservative announcement/activation availability,
and trailing 180 calendar days of annual EPS updates. A 90-day version is a
sensitivity. Forecast values, future actuals and reviews never enter signals.
The count is an active-coverage proxy, not vendor NUMEST or a reconstruction
of all analyst withdrawals. All qualifying US ordinary common stocks are used;
there is no primary price/size/liquidity threshold beyond valid lagged marks.

The signal is a signed nearest known/forecast quarterly earnings clock.
Future dates are seasonal forecasts from prior earnings (or valid median gaps),
not eventual actual dates. Quarterly earnings are the announcement type in scope.
Windows are selected on 1985-2004, then frozen for 2005-2014 and 2015-2025.
This remains retrospective exploratory evidence given prior related studies.

All primary long/short rules target 200% gross exposure on $1 NAV. The separate
long-only and covered-universe benchmark target 100% long. Cash earns zero.
Costs charge every dollar bought and sold, with separate exact NAV/holdings
per scenario. Gross uses no fees/borrow; 1,5,10 bps use 1% annual short borrow;
25 bps uses 2%. Stock-specific borrow availability, impact and financing are
not observed. Annual mean is arithmetic; CAGR is compounded.

Raw inputs and the prior studies are unchanged. Checksums and receipts are
verified. The original event linkage and conservative event information states
are reused explicitly; linkage/vintage limitations remain in the report.

Outputs: design.json, input_audit.json, coverage_states.parquet, panel/YYYY.parquet,
selected_windows.json, universe_by_year.csv, universe_audit.json, daily_bins.csv,
bin_statistics.csv, daily_returns.csv, performance.csv, annual_returns.csv,
inference_robustness.csv, breakeven_costs.csv, timing_results.png, report.html,
summary.txt and validation_checks.json. See validate.py for independent raw
analyst-count, historical-clock, cost ledger and inference checks.
