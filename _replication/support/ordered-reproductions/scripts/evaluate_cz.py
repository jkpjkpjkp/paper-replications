"""Evaluate complete, out-of-sample monthly scores under protocol.txt.

Input CSV: permno, return_month (YYYYMM), score. No future-return universe filtering.
Costs use a conventional gross-portfolio turnover ledger, not a financing simulator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ['Mkt-RF','SMB','HML','RMW','CMA','UMD']


def portfolios(panel, scores, cost_bps=10):
    keys = ['return_month','permno']
    if scores.duplicated(keys).any() or panel.duplicated(keys).any():
        raise ValueError('Duplicate stock-month')
    data = panel.merge(scores[keys+['score']], on=keys, how='outer', validate='one_to_one', indicator=True)
    if not data._merge.eq('both').all() or not np.isfinite(data.score).all():
        raise ValueError('Scores must cover exactly the eligible universe, including stocks with missing future returns')
    if not data.lagged_market_cap.ge(data.nyse20).all():
        raise ValueError('NYSE20 screen violation')
    old = pd.Series(dtype=float)
    returns, holdings = [], []
    for month, group in data.groupby('return_month', sort=True):
        g = group.sort_values(['score','permno']).copy()
        if len(g)<10: raise ValueError('Fewer than ten eligible stocks')
        # Deterministic, nearly equal stock counts; market cap weights within each decile.
        g['decile'] = np.arange(len(g))*10//len(g)+1
        g['weight'] = g.lagged_market_cap/g.groupby('decile').lagged_market_cap.transform('sum')
        if not np.isfinite(g.target_return).all():
            raise ValueError(f'Missing/nonfinite realized return in {month}; resolve data, do not drop ex post')
        deciles = (g.weight*g.target_return).groupby(g.decile).sum()
        g['ls_weight'] = g.weight*np.where(g.decile.eq(10),1,np.where(g.decile.eq(1),-1,0))
        current = g.set_index('permno').ls_weight
        gross = float(deciles[10]-deciles[1])
        if gross<=-1: raise ValueError('Gross strategy insolvency')
        turnover = float(current.subtract(old,fill_value=0).abs().sum())
        selected = g.ls_weight.ne(0)
        if not np.isfinite(g.loc[selected,'price_return']).all():
            raise ValueError(f'Missing selected-stock price return in {month}')
        # Price-only appreciation leaves dividends in cash. Delisted stock has zero stock holdings.
        end = g.set_index('permno').loc[current.ne(0)].copy()
        end['dollar_position'] = end.ls_weight*(1+end.price_return)
        end.loc[end.delisting_month.eq('Y'),'dollar_position'] = 0
        old = end.dollar_position/(1+gross)
        row = dict(return_month=int(month),stocks=len(g),gross=gross,turnover=turnover,
                   terminal_turnover=0.0,net=gross-cost_bps*1e-4*turnover)
        row.update({f'D{i}':float(deciles[i]) for i in range(1,11)})
        returns.append(row)
        holdings.append(g[keys+['decile','score','lagged_market_cap','weight','ls_weight']])
    # Close remaining shares at final month end; quoted in beginning-of-last-month NAV units.
    terminal = float(end.dollar_position.abs().sum())
    returns[-1]['terminal_turnover'] = terminal
    returns[-1]['net'] -= cost_bps*1e-4*terminal
    return pd.DataFrame(returns), pd.concat(holdings,ignore_index=True)


def regress(returns, factors, maxlags=6):
    joined = returns.merge(factors,on='return_month',how='left',validate='one_to_one')
    if not np.isfinite(joined[FACTORS+['gross','net']]).all().all(): raise ValueError('Missing/nonfinite returns or factors')
    if len(joined)<=len(FACTORS)+maxlags+1: raise ValueError('Insufficient months')
    x = sm.add_constant(joined[FACTORS],has_constant='add')
    if np.linalg.matrix_rank(x)!=len(FACTORS)+1: raise ValueError('Rank-deficient factor design')
    output = {}
    for name in ['gross','net']:
        # Zero-investment long-short return: do NOT subtract RF again.
        fit = sm.OLS(joined[name],x,missing='raise').fit(cov_type='HAC',
                    cov_kwds={'maxlags':maxlags,'use_correction':True},use_t=True)
        ci = fit.conf_int(alpha=.05).loc['const']*12
        output[name] = dict(months=len(joined),annual_alpha=float(fit.params['const']*12),
            hac_t=float(fit.tvalues['const']),p_value=float(fit.pvalues['const']),
            p_bonferroni_12_planned_tests=min(1.0,float(fit.pvalues['const'])*12),
            annual_alpha_ci95=[float(v) for v in ci],r_squared=float(fit.rsquared),
            coefficients={k:float(v) for k,v in fit.params.items()},
            mean_annual_return=float(joined[name].mean()*12),
            annual_volatility=float(joined[name].std(ddof=1)*np.sqrt(12)))
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scores',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cost-bps',type=float,default=10)
    ap.add_argument('--hac-lags',type=int,default=6)
    args=ap.parse_args()
    if args.cost_bps<0 or args.hac_lags<0: raise ValueError('Negative cost or lag count')
    if args.output.exists(): raise ValueError('Refusing to overwrite an existing evaluation')
    scores=pd.read_csv(args.scores)
    panel=pd.read_parquet(ROOT/'data/cz-monthly/panel.parquet',columns=[
        'permno','return_month','lagged_market_cap','nyse20','target_return','price_return','delisting_month'])
    panel=panel.loc[panel.return_month.between(202201,202512)]
    assert len(panel.return_month.unique())==48
    returns, holdings=portfolios(panel,scores,args.cost_bps)
    factors=pd.read_csv(ROOT/'data/factors-monthly/ff5-umd.csv')
    result=regress(returns,factors,args.hac_lags)
    report=dict(results=result,protocol=dict(universe='All eligible US common stocks, lagged NYSE20 screen',
        test=[202201,202512],weighting='Market cap within each decile; long D10 +1, short D1 -1; gross exposure 2',
        cost_bps_per_dollar_traded=args.cost_bps,
        turnover='Sum absolute target-minus-price-drifted weights; gross NAV denominator; initial and terminal trades included',
        cost_accounting='Net return = gross return minus turnover charge; costs do not feed back into gross target weights',
        omitted_costs='Borrow fees, financing, spread heterogeneity, market impact and taxes',
        hac=dict(kernel='Bartlett',maxlags=args.hac_lags,finite_sample_correction=True,reference_distribution='Student t, T-7 df'),
        annualization='Monthly intercept and its confidence limits multiplied by 12',
        inference='Nominal two-sided 95% intervals and p-values; additional Bonferroni p for six planned papers times gross/net = 12 tests'),
        scores_sha256=hashlib.sha256(args.scores.read_bytes()).hexdigest())
    args.output.mkdir(parents=True)
    returns.to_csv(args.output/'monthly-returns.csv',index=False)
    holdings.to_parquet(args.output/'holdings.parquet',index=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
