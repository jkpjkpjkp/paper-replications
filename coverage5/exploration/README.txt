DATE-ONLY DOLLAR-NEUTRAL STRATEGY EXPLORATION

Open results/report.html. It reports all frozen families, low-cost selections,
and eight annual-refit variants, with no long-only strategy in this search.

From this directory, run the following with
  /home/jkp/data/earnings-timing/.venv/bin/python

  explore.py prepare
  explore.py screen
  explore.py train
  explore.py evaluate
  validate.py
  adaptive.py
  report.py

Universe and information chronology inherit the validated regular-65,
at-least-five-active-analyst panel from ../results_regular65/panel.
Prices/returns do not select directions or act as alpha predictors; they are
used to fit historical date-return relationships and to account for shares,
cash, trading costs and daily equal-dollar neutrality.

Search: 7,650 ordered window pairs screened on 1985-2004 only, then exact
simulation of shortlisted pairs, smooth learned/fixed quarterly curves,
and slower adjustments. 10-bp and 1-bp objectives choose separate winners
using development only. 2005-2014 and 2015-2025 are chronological comparisons,
not untouched prospective evidence. Adaptive curves use only strictly earlier
years, but were specified after the fixed-rule later-period results were seen.

Every book is $1 long and $1 short per $1 after-cost NAV when active. Gross
exposure is 200%, and target net is zero at every daily rebalance. Price
drift between rebalances is corrected the next day. Both sides need >=20
stocks. Fees charge all dollars bought/sold; short borrow assumptions and
missing financing, impact and actual stock-borrow availability are explicit.

Key outputs:
  design.json, adaptive_design.json: frozen definitions and provenance
  input_audit.json: source-panel checksums and inherited raw-audit references
  window_screen.csv, screen_audit.json: the complete screening search
  stage1_models.json, stage2_models.json: exact model and adjustment definitions
  frozen_choices.json: training-only decisions before later evaluation
  performance.csv, adaptive_performance.csv: every evaluated scenario/period
  daily_returns.csv, annual_returns.csv, adaptive_daily_returns.csv
  inference_robustness.csv, validation_checks.json, adaptive_validation.json
  family_comparison.csv, low_cost_comparison.csv, adaptive_comparison.csv
  selected_rules.json, summary.txt, report.html, exploration_results.png

Compressed NPZ files retain daily simulation arrays and screening leg caches.
The two training simulations are reused only when their saved definitions
exactly match. The exact full-period simulation reproduces its training prefix.
