#!/usr/bin/env python3
"""Execute the unmodified released table script in an isolated output folder.

This reproduces table calculations from released monthly portfolio artifacts;
it cannot validate absent images, model training, or raw-price sorting inputs.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper-claim-validation/results/vision'
SOURCE = ROOT / 'vision-to-value-evaluation/upstream/sdf_analysis'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['data', 'sorted_portfolio']:
        p = OUT / name
        if not p.exists():
            p.symlink_to(SOURCE / name, target_is_directory=True)
    (OUT / 'factor_port').mkdir(exist_ok=True)
    script = ROOT / 'vision-to-value-evaluation/sources/portfolio_performance.py'
    os.chdir(OUT)
    import matplotlib
    matplotlib.use('Agg')
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    log = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        warnings.simplefilter('ignore')
        result = runpy.run_path(str(script), run_name='native_table_replay')
    (OUT / 'author_stdout.txt').write_text(log.getvalue())
    metrics = result['risk_metric'].astype(float)
    metrics.to_csv(OUT / 'replayed_table2.csv')
    result['summary_stat'].to_csv(OUT / 'replayed_table2_monthly.csv')
    result['total_return_table'].to_csv(OUT / 'replayed_monthly_returns.csv')
    # Independent, unit-consistent diagnostics, kept separate from the literal
    # reproduction. The two factor portfolios are zero-investment spreads.
    returns=result['total_return_table']
    rf=result['r_f'].reindex(returns.index)
    corrected=[]
    for column in returns.columns:
        series=returns[column]
        excess=series if column in ['ViT','CNN'] else series-rf
        corrected.append(dict(portfolio=column,annual_mean=series.mean()*12,
                              annual_volatility=series.std(ddof=1)*np.sqrt(12),
                              sharpe_units_consistent=excess.mean()/series.std(ddof=1)*np.sqrt(12),
                              risk_free_subtracted=column not in ['ViT','CNN']))
    pd.DataFrame(corrected).to_csv(OUT/'unit_consistent_diagnostics.csv',index=False)
    # Scalar witness for the original sorting code's common fee cancellation.
    fee=.001; high=.05; low=-.02
    legacy=lambda r:(1-fee)*(1+r)*(1-fee)-1
    witness=dict(one_way_fee=fee,high_leg_return=high,low_leg_return=low,
                 gross_spread=high-low,released_fee_spread=legacy(high)-legacy(low),
                 two_full_round_trip_leg_costs_approx=high-low-4*fee,
                 released_formula='(1-fee)^2 * (r_high-r_low)',
                 consequence='Constant round-trip leg fees cancel on differencing; this is not a long-short trade-cost ledger')
    np.testing.assert_allclose(witness['released_fee_spread'],(1-fee)**2*(high-low),atol=1e-15)
    (OUT/'fee_cancellation_witness.json').write_text(json.dumps(witness,indent=2)+'\n')
    # Exact displayed cells from SSRN PDF Table 2, columns in paper order.
    expected = {
        'Annualized return': [2.646,7.579,5.426,9.388,6.249,1.396,6.640,6.659,9.448,6.637,6.700,5.575,7.809],
        'Annualized volatility': [5.127,17.482,18.842,20.558,22.396,6.269,17.419,18.965,20.618,22.238,18.208,15.971,21.493],
        'Sharpe ratio': [.513,.433,.287,.456,.278,.220,.380,.350,.457,.298,.367,.348,.363],
        'Maximum drawdown': [23.113,50.871,60.874,43.503,55.263,33.640,49.729,59.009,47.501,53.534,53.818,53.992,55.574],
    }
    comparisons = []
    for metric, targets in expected.items():
        for column, target in zip(metrics.columns, targets):
            observed = float(metrics.loc[metric, column])
            comparisons.append(dict(metric=metric,portfolio=column,paper=target,replayed=observed,
                                    absolute_error=abs(observed-target),
                                    agrees_at_printed_precision=abs(observed-target)<.000500001))
    pd.DataFrame(comparisons).to_csv(OUT / 'paper_comparison.csv', index=False)
    manifest = []
    for p in [script, SOURCE/'data/processed_kelly/ff3.csv',
              *sorted((SOURCE/'sorted_portfolio').glob('*.csv'))]:
        manifest.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    record = dict(kind='released_portfolio_artifact_table_reproduction',
                  unmodified_author_script_executed=True,
                  compared_cells=len(comparisons),
                  matching_cells=sum(x['agrees_at_printed_precision'] for x in comparisons),
                  full_image_training_reproduced=False,
                  native_sorting_from_raw_prices_reproduced=False,
                  source_manifest=manifest,
                  caveat='Matching released monthly returns validates table arithmetic only. '
                         'The original script positionally relabels dates and applies RF in mixed units; '
                         'these literal conventions are retained solely to reproduce the printed table.')
    (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='source_manifest'},indent=2))


if __name__ == '__main__':
    main()
