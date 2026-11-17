"""Exploratory significance checks of the already-inspected no-skip result.

Days are the sampling units. All p-values are two-sided and unadjusted for
multiple comparisons or selection of this variant. HAC uses Bartlett weights,
an intercept-only n/(n-1) correction, and asymptotic normal reference values.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hermes.metrics import PORTFOLIO_PROTOCOL


def describe(values):
    x = np.asarray(values,dtype=float)
    n = len(x)
    centered = x - x.mean()
    iid = stats.ttest_1samp(x,0)
    result = {'days':n,'mean':float(x.mean()),'std_sample':float(x.std(ddof=1)),
              'iid_t':float(iid.statistic),'iid_p_two_sided':float(iid.pvalue),'hac':{}}
    for lag in [5,10,20]:
        long_run_variance = centered @ centered / n
        for k in range(1,lag+1):
            long_run_variance += 2*(1-k/(lag+1))*(centered[k:] @ centered[:-k])/n
        standard_error = np.sqrt(long_run_variance/(n-1))
        t = x.mean()/standard_error
        result['hac'][str(lag)] = {
            't':float(t),'p_two_sided_normal':float(2*stats.norm.sf(abs(t))),
            'ci95_mean':[float(x.mean()-stats.norm.ppf(.975)*standard_error),
                         float(x.mean()+stats.norm.ppf(.975)*standard_error)]}
    return result


def main():
    no_skip = np.load(ROOT / 'results/sp500-no_skip-seed0/test_predictions.npz')
    full = np.load(ROOT / 'results/sp500-full-seed0/test_predictions.npz')
    np.testing.assert_array_equal(no_skip['dates'],full['dates'])
    np.testing.assert_array_equal(no_skip['target'],full['target'])
    portfolio = no_skip['portfolio_return'].astype(float)
    net = no_skip['net_portfolio_return'].astype(float)
    weights=no_skip['portfolio_weights']
    np.testing.assert_allclose(weights.sum(1),0,atol=1e-12)
    np.testing.assert_allclose(np.abs(weights).sum(1),1,atol=1e-12)
    result = {
        'status':'exploratory; variant was noticed after examining the test results',
        'p_value_convention':'two-sided, unadjusted; no correction for multiple variants or model selection',
        'portfolio_protocol':PORTFOLIO_PROTOCOL,
        'benchmark':'zero return for the dollar-neutral strategy; paired full-model returns use identical exposures and cost convention',
        'tests':{
            'mean_daily_IC_above_zero':describe(no_skip['daily_ic']),
            'mean_gross_portfolio_return_above_zero':describe(portfolio),
            'mean_net_portfolio_return_above_zero':describe(net),
            'paired_IC_difference_vs_full_seed0':describe(no_skip['daily_ic']-full['daily_ic']),
            'paired_return_difference_vs_full_seed0':describe(portfolio-full['portfolio_return']),
            'paired_net_return_difference_vs_full_seed0':describe(net-full['net_portfolio_return']),
        },
        'portfolio_cumulative_return':float(np.prod(1+portfolio)-1),
        'net_cumulative_return':float(np.prod(1+net)-1),
        'mean_buy_turnover':float(no_skip['buy_turnover'].mean()),
        'mean_fee_bp':float(no_skip['transaction_fees'].mean()*10000),
        'gross_annualized_sharpe':float(portfolio.mean()/portfolio.std()*np.sqrt(252)),
        'net_annualized_sharpe':float(net.mean()/net.std()*np.sqrt(252)),
        'yearly':[{ 'year':str(year), 'days':int(np.char.startswith(no_skip['dates'],year).sum()),
                    'gross_return':float(np.prod(1+portfolio[np.char.startswith(no_skip['dates'],year)])-1),
                    'net_return':float(np.prod(1+net[np.char.startswith(no_skip['dates'],year)])-1)}
                  for year in sorted(set(str(x)[:4] for x in no_skip['dates']))],
        'validation':json.loads((ROOT/'results/sp500-no_skip-seed0/metrics.json').read_text())['validation'],
        'limitations':['single training seed','352 test days','fixed historical universe and sector assignments',
                       '5bp buy-only costs; other borrow/financing/execution charges not modeled','post-hoc selection among several reported variants',
                       'stationarity and finite-sample limitations of asymptotic HAC inference'],
        'workflow_tooling_reference':'https://doi.org/10.48550/arXiv.2609.00065 (metadata previously verified; also in README.txt)',
    }
    dest = ROOT/'results/no_skip_significance.json'
    dest.write_text(json.dumps(result,indent=2)+'\n')
    for name, values in result['tests'].items():
        print(name,'mean=',values['mean'],'HAC(5) t=',values['hac']['5']['t'],
              'p=',values['hac']['5']['p_two_sided_normal'])
    print('Saved',dest)


if __name__ == '__main__':
    main()
