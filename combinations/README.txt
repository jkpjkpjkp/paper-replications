EARNINGS TIMING OVERLAYS ON EXISTING STRATEGIES

Read results/report.html for findings. The primary question is incremental
net return versus the same baseline, not a new stand-alone earnings book.

Reproduce from this directory:
  ../.venv/bin/python combine.py
  ../.venv/bin/python validate.py

Use --reuse-features only to reuse the existing lagged-signal feature cache.
The default recomputes features. Dependencies are in ../requirements.txt.

Frozen design: results/design.json, written before computing strategy returns.
All six overlay/baseline combinations are reported. No holdout tuning.
Prior standalone timing holdout results were already known, so this is a
retrospective exploratory extension rather than prospective validation.

Baselines:
  reversal: buy the bottom quintile / short the top quintile of prior-five-
    session cumulative returns, equal weight, daily rebalance.
  momentum: buy the top quintile / short the bottom quintile of prior 12-2
    returns (t-252 through t-22), equal weight, monthly rebalance. Require
    at least 200 valid returns in the 231-session window.

Overlays (all confined to current baseline candidate positions):
  cycle_tilt: +6..20 sessions since known earnings is favorable to long
    positions; +36..50 is unfavorable. Multiply weights by 1 +/- 0.5*clock
    and renormalize each leg. Neutral clocks include firms without links.
  cycle_veto: omit late-phase longs and early-phase shorts, renormalize.
  earnings_blackout: omit both sides for 0..5 sessions since known earnings
    and 1..5 sessions before the prior-year/52-week predicted announcement.

Overlay target changes use the baseline's rebalance dates only. Monthly
momentum carries shares between rebalances. A name leaving the lagged
eligible universe is closed at the last mark and incurs exit turnover,
even on a non-rebalance date. This convention is shared by all variants.

Costs are based on final combined net orders. Baseline and overlay orders
can offset; costs are not charged on two independently traded portfolios.
Cost scenarios: gross (no fees or borrow), and 1, 5, 10, 25 one-way bps
plus 1% annual short borrow. The primary scenario is 10 bps.

At a scheduled rebalance solve x = 1 - c*sum(abs(x*w - v)), where w is the
target portfolio and v is pre-trade marked holdings divided by current NAV.
After-cost dollars invested are x*w. Daily borrow applies to actual current
short exposure. Each scenario maintains its own cost-specific holdings/NAV.
Initial target gross exposure is 200%; momentum exposure drifts between
monthly rebalances. Cash earns zero. Total returns determine P&L, while
ex-dividend returns determine share-value drift. Distributions remain cash.

Output files:
  design.json: fixed rules, costs, periods, and inference protocol
  input_audit.json: checksums of all raw price and parent panel inputs
  features/YYYY.parquet: lagged signals, source returns, and known clocks
  daily_returns.csv: all strategies/scenarios, costs, borrow, turnover, NAV returns
  performance.csv: full strategy results across subperiods/cost scenarios
  paired_incremental_results.csv: baseline versus overlay return differences,
    marginal turnover/costs, confidence intervals and adjusted p-values
  development_choices.json: overlay selected per baseline using 1985-2004
  inference_robustness.csv: daily HAC 5/20/60 and monthly HAC3 sensitivity
  annual_returns.csv: compounded calendar-year returns and trading costs
  validation_checks.json: raw-window, execution, chronology, cost and paired tests
  incremental_edge.png: holdout NAVs and incremental-return confidence intervals

All annual_mean and annual_increment estimates are daily arithmetic mean
times 252, rather than compounded annual growth. Increments are percentage
points. annual_before_fee_increment - annual_trading_cost_increment -
annual_borrow_increment = annual_increment for each matched comparison.
Before-fee increments use each cost scenario's own holdings, so they need
not equal the independently simulated all-gross scenario's increment.

Monthly compound differences are evaluated separately, not obtained by
compounding daily return differences. Holm adjustment covers six baseline/
overlay pairs within each period/scenario. 2020-2025 overlaps the holdout.

Limitations inherited from the original study include retrospective exact-
CUSIP linkage, missing historical availability versions, no official factor
series, no intraday execution quotes and no stock-specific borrow/impact.
Current and future realized announcement dates never enter a trading signal.
Missing daily returns are carried as zero, and absent names are closed at
their last mark. No terminal liquidation, financing or collateral rebate
is included. These implementations are transparent variants of established
strategy families, not exact replications of a published factor portfolio.

The report includes primary strategy references and the statistical workflow
software reference used in the original analysis.
