ANNOUNCEMENT-CYCLE FACTOR OVERLAYS AND DIAGNOSTIC PLOTS

Start with findings.html. Plots are standalone PNG/PDF under plots/.
Primary methods/results: results/report.html.
Target-horizon sensitivity: results_h5/report.html.
Cost-veto: results_cost_guard/performance.csv and design.json.

Only factor_overlay/ was added or edited; earlier studies/raw inputs unchanged.
All six requested factors, six fixed regression families, matched baselines.
2005–2025 annual walk-forward fits on preceding10years; no current/future
labels in features. Earlier related holdouts had already been inspected.

Reproduce from /home/jkp/data/earnings-timing, with OPENBLAS_NUM_THREADS=1
and OMP_NUM_THREADS=1 in the environment:
 .venv/bin/python factor_overlay/data.py
 .venv/bin/python factor_overlay/run.py all
 .venv/bin/python factor_overlay/horizon_labels.py
 .venv/bin/python factor_overlay/run.py all --horizon 5
 .venv/bin/python factor_overlay/report.py
 .venv/bin/python factor_overlay/report.py --horizon 5
 .venv/bin/python factor_overlay/diagnose_costs.py
 .venv/bin/python factor_overlay/diagnose_forecasts.py
 .venv/bin/python factor_overlay/diagnose_correlations.py
 .venv/bin/python factor_overlay/diagnose_weights.py
 .venv/bin/python factor_overlay/cost_guard.py

Additional scripts in this directory implement weekly timing correlations,
pre-2018 slices, full-stock ground-truth correlations, calibration bins and
plots. Consult their headers/CLI before running. requirements.txt records
versions; NumPy implementations supply regression models without sklearn.
Run plot_diagnostics.py and build_overview.py after their input CSVs exist.

Independent validations:
 validate_models.py: independent SciPy/statsmodels model reproduction.
 portfolio.py: independent convex optimizer and absolute-dollar ledger.
 validate_run.py: raw2005 return and position replay.
 validate_horizon_labels.py: independent five-session return labels.
 Individual diagnostics write validation/audit files in diagnostics/.

Important interpretation:
 - Daily/annual arithmetic means, CAGR and variance are different quantities.
 - 5bp means each dollar bought/sold, plus1% annual short borrow.
 - RDIPO requires a separate concentrated min5-short sensitivity.
 - Local factor constructions are not exact original-paper portfolios.
 - Original timing score = +1 forecast pre1..5, -1 pre21..40, else0.
 - Pairwise signal matrices use different finite overlapping universes and
   may not be positive-semidefinite; they are not optimizer covariance inputs.
 - Direct5day labels purge calendar-year crossing observations.
 - Five-day, cost-veto, diagnostics and2018cutoff follow primary results;
   they are explicitly post-result robustness/mechanism investigations.
 - Holm72/108 correction covers current target horizons/cost-veto candidates
   per displayed period/statistic, not all previous research or cutoff choices.
 - A local fee veto guarantees lower trade cost from identical current
   holdings, not lower fees over a separately evolved entire baseline path.
