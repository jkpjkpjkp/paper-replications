EARC COMPANY AND CALENDAR-TIME DIAGNOSTICS

Open results/report.html or results/company_time.png.
Run from /home/jkp/data/earnings-timing:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python company_time/analyze.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python company_time/validate.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python company_time/normality.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python company_time/outliers.py

This diagnostic reuses the final primary Compustat EARC announcement rules,
retained analyst states, and raw CRSP daily returns; see
../rolling_sharpe5/returns_spec.txt. Existing inputs/results are read-only.
No large stock-day cache is retained. Results cover 1985-01-02..2025-12-31.

The phase score is +1 in the long age6..20 window and -1 in short age36..50.
Company correlation is corr(phase, daily stock total return) over eligible
stock-days grouped by CRSP PERMCO; time correlation is the daily cross-sectional
correlation across selected securities. Neutral phases are excluded. Stable
company IDs group changing tickers; share classes remain separate stock-day
observations, matching the primary. Company summary statistics require at
least120 stock-days in each phase and equal-weight companies; date summaries
equal-weight dates. A company-ID Pearson correlation would have no meaning.

spread_raw = mean daily long-phase stock return - mean daily short-phase stock
return, as a decimal. Multiply by10000 for bp/selected day. This company phase
spread is not a company's all-calendar daily trading-strategy return. Companies
with different eligible histories/weights have means that do not aggregate to
the daily EARC portfolio. annual_gross_contribution instead uses actual daily
portfolio weights and sums across all companies to252*mean(gross EARC return).

Market-adjusted returns subtract the retained local value-weighted CRSP market
each day. At a fixed date this subtracts one common constant, leaving the IC
and long-short spread unchanged; it can change the company-time-series IC.

Complete-cycle diagnostics compare the15 observed long-session returns with
the15 observed short-session returns for the same security/announcement. Mean
daily differences and15-session compounded differences are saved separately.
These windows occur at different dates. They are historical phase contrasts,
not a simultaneous long-short portfolio. Complete share classes at one company
event are averaged before company persistence and fixed-effect calculations.
Persistence compares company mean cycle spreads in1985-2004 vs2005-2025, with
at least8 complete cycles per side. Bootstrap intervals resample companies
2000times, seed20261109, conditional on observed periods. They do not account
for all common time shocks. This is a descriptive overlap sample, not a
tradable company-selection backtest. Named-company tests requested afterward
are in results/outliers/report.html; they include all-company Holm correction,
earlier/later estimates, change tests, and single-cycle influence diagnostics.

Categorical variance output reports descriptive one- and two-way fixed-effect
R-squared for company and announcement calendar quarter on complete cycles.
Adjusted R-squared uses category-design rank, including disconnected-component
adjustment. In-sample group associations are not a decomposition of expected
profits or evidence that company means can be predicted. Daily-series mean
inference uses statsmodels OLS with an intercept and Newey-West HAC21lags.

Output files:
  compact_summary.csv / distribution_summary.csv: distributions, overall and
    pre2005/post2005/recent2015 periods
  company_distribution.csv: every company and period, phase ICs, return spreads,
    history filters, complete-cycle means, actual gross portfolio contributions
  daily_cross_section.csv: daily phase IC, spread, counts and exact5bp net return
  company_cycles.csv: complete historical company-cycle contrasts
  company_split_*.csv: same-company mean contrasts before/after2005
  annual_time_distribution.csv / monthly_time_distribution.csv: time paths
  categorical_variance.csv: categorical associations
  mean_inference.csv: HAC mean estimates and confidence intervals
  normality.csv: bias-corrected skewness, excess/Pearson kurtosis and Jarque-Bera
    descriptive statistics for each distribution; normality_validation.json
    verifies the coefficients against independent central-moment formulas.
    Normal references are skewness0, excess kurtosis0, Pearson kurtosis3.
    JB p-values assume IID sampling, not dependence-adjusted inference.
  audit.json: definitions, counts, source hashes, reconciliation, correlations
  outliers/report.html: named tail examples and >=40-cycle examples, quarterly
    HAC significance, chronological halves/change tests, and influence checks.
    Holm family:4865companies x2outcomes x4tests=38920. Results are exploratory.

Validation: all10330 daily gross returns and long/short counts agree with the
saved primary; maximum absolute return error2.61e-16. Gross contributions
reconcile by date and in every reported period. All44missing selected returns
are0, matching the primary. Same inherited historical-vintage/linkage limits.

Methods/software reference (required by the used statistical-modeling skill):
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
Verified at https://arxiv.org/abs/2609.00065 on2026-11-09: latest version2,
revised2026-09-02, five listed authors above; no journal reference listed.
This is a workflow citation, not evidence for the finance results.
