LATEST USER RULE: minimum10 factors; open min10/results/report.html.
The original empty-pool results below are retained as historical analysis.

ROLLING FIVE-YEAR SIGNIFICANT SHARPE / PAPER EARC ADDITION

Open results/report.html for the Chinese report and results/summary_chinese.csv
for the compact result table. Primary: every previous calendar year separately
significant (positive mean, two-sided IID p<0.05), complete12months per year,
publication year before pool year. All original data/results remain unchanged.

The2026 primary pool is empty; no2026 return data exist locally.
EARC is independently reconstructed from Compustat RDQ, historical CCM links,
CRSP daily returns, and retained90-day analyst coverage states.
Primary costs:5bp per dollar bought/sold, separate holdings/NAV, no borrow.
Zoo control returns remain GROSS because positions/turnover are unavailable.
This is NOT an all-components-net implementable factor-fund backtest.

Reproduce from /home/jkp/data/earnings-timing:
  .venv/bin/python rolling_sharpe5/audit/extend_selection_audit.py
  .venv/bin/python rolling_sharpe5/build_paper_factor.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python rolling_sharpe5/analyze_increment.py
  .venv/bin/python rolling_sharpe5/audit/validate_increment_analysis.py
  .venv/bin/python rolling_sharpe5/report.py

Sensitivity reconstruction: build_paper_factor.py --event-source ibes.
Archived initial IBES/min20 results have explicit _min20 filenames.
Main uses both legs nonempty, as paper has no minimum20 constraint.
Pooled60 tests are alternative interpretation, with2023 missing option returns;
only continuous1990-2022 comparison is reported rather than filling missing data.

Software-method sources, including the used statistical skill reference, are
listed in report.html. See audit/ for independent checks and paper definitions.
