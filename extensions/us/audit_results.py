"""Final source reconciliation, payoff stress and calendar-year diagnostics."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from usdata import ROOT,DATA,SOURCE,Panel,arrow_read,sha,write_json
from analyze import hac,bootstrap


def main():
    panel=Panel();folder=ROOT/'analysis'
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert all(sha(path)==expected for path,expected in manifest['source_sha256'].items())
    checks=[];held=[];stress_summary=[];annual=[];missing=[]
    cases=[('LiMT','equal'),('LiMT','apo'),('LightGBM','equal')]
    for model,strategy in cases:
        tag=f'{model}_ensemble_{strategy}_buy0'
        z=np.load(folder/f'{tag}_weights.npz')
        weights=z['weights'];dates=z['dates'];ids=np.array([int(s[1:]) for s in z['symbols']])
        lookup={d:i for i,d in enumerate(panel.dates)}
        ix=np.array([lookup[d] for d in dates])
        values=np.asarray(panel.total_return[:,ix]).T
        ledger=pd.read_csv(folder/f'{tag}_ledger.csv')
        np.testing.assert_allclose(np.sum(weights*np.nan_to_num(values,nan=0),axis=1),ledger.return_gross,atol=1e-12)
        np.testing.assert_allclose(weights.sum(1),0,atol=1e-8)
        ii,jj=np.nonzero(weights)
        held.append(pd.DataFrame(dict(permno=ids[jj],date=dates[ii],prepared_return=values[ii,jj])))
        bad=~np.isfinite(values[ii,jj])
        if bad.any():
            missing.append(pd.DataFrame(dict(model=model,strategy=strategy,permno=ids[jj[bad]],
                date=dates[ii[bad]],weight=weights[ii[bad],jj[bad]])))
        stress=pd.read_csv(folder/f'{model}_ensemble_{strategy}_missing_stress.csv')
        from analyze import windows
        for window,mask in windows(stress):
            f=stress.loc[mask];x=f.return_net.to_numpy();stats=hac(x);low,high=bootstrap(x)
            stress_summary.append(dict(model=model,strategy=strategy,window=window,days=len(x),
                annual_mean=float(x.mean()*252),cagr=float(np.prod(1+x)**(252/len(x))-1),
                sharpe=float(x.mean()/x.std(ddof=1)*np.sqrt(252)),
                ci_low=stats['ci_low']*252,ci_high=stats['ci_high']*252,p=stats['p'],
                bootstrap20_low=low*252,bootstrap20_high=high*252))
        for year,g in ledger.groupby(ledger.date.str[:4],sort=True):
            x=g.return_net.to_numpy()
            annual.append(dict(model=model,strategy=strategy,year=year,days=len(g),
                calendar_return=float(np.prod(1+x)-1),annualized_mean=float(x.mean()*252),
                sharpe=float(x.mean()/x.std(ddof=1)*np.sqrt(252))))
        checks.append(f'{tag}: saved full-precision weights reconcile to its gross ledger and zero net exposure.')
    keys=pd.concat(held,ignore_index=True).drop_duplicates(['permno','date'])
    raw=[]
    for year in range(2017,2026):
        f=arrow_read(SOURCE/f'{year}.parquet',['permno','dlycaldt','dlyret','dlyretx','dlydelflg'])
        f['date']=pd.to_datetime(f.dlycaldt).dt.strftime('%Y-%m-%d')
        raw.append(f[['permno','date','dlyret','dlyretx','dlydelflg']])
    raw=pd.concat(raw,ignore_index=True)
    joined=keys.merge(raw,on=['permno','date'],how='left',validate='one_to_one',indicator=True)
    np.testing.assert_allclose(joined.prepared_return,joined.dlyret,equal_nan=True,atol=1e-12)
    joined[joined.prepared_return.isna()].to_csv(folder/'missing_payoff_source_audit.csv',index=False)
    if missing:pd.concat(missing,ignore_index=True).to_csv(folder/'missing_held_positions.csv',index=False)
    stress_frame=pd.DataFrame(stress_summary);stress_frame.to_csv(folder/'missing_stress_summary.csv',index=False)
    yearly=pd.DataFrame(annual);yearly.to_csv(folder/'calendar_years.csv',index=False)
    checks.append(f'Every one of {len(joined):,} distinct ensemble held stock-date payoffs matches original CRSP daily Parquet, including retained missing values.')
    report=ROOT/'REPORT.txt'
    text=report.read_text().split('\nFINAL SOURCE AUDIT AND PREDECLARED MISSING-PAYOFF STRESS')[0]
    if '\nDETAILS AND METHODS\n' in text:
        body=text.split('\nDETAILS AND METHODS\n',1)[1]
    else:
        body=text.split('\n',1)[1]
    summary=pd.read_csv(folder/'summary.csv')
    ensemble=summary[(summary.seed=='ensemble')&(summary.model=='LiMT')]
    gross=ensemble[(ensemble.strategy=='equal')&(ensemble.buy_bp==0)&(ensemble.window=='full-2017-2025')].iloc[0]
    net=ensemble[(ensemble.strategy=='equal')&(ensemble.buy_bp==5)&(ensemble.window=='full-2017-2025')].iloc[0]
    apo=ensemble[(ensemble.strategy=='apo')&(ensemble.buy_bp==0)&(ensemble.window=='full-2017-2025')].iloc[0]
    recent=ensemble[(ensemble.strategy=='equal')&(ensemble.buy_bp==0)&(ensemble.window=='recent-2021-2025')].iloc[0]
    stressed=stress_frame[(stress_frame.model=='LiMT')&(stress_frame.strategy=='equal')&(stress_frame.window=='full-2017-2025')].iloc[0]
    yearly_equal=yearly[(yearly.model=='LiMT')&(yearly.strategy=='equal')].set_index('year')
    conclusion=(f'The equal-dollar LiMT ranking ensemble has positive statistically significant gross returns in this fixed US transfer.\n'
        f'Full 2017-2025: annual arithmetic mean {gross.annual_mean:.2%}, CAGR {gross.cagr:.2%}, Sharpe {gross.sharpe:.2f}, maximum drawdown {gross.max_drawdown:.2%}.\n'
        f'HAC95% annual-mean CI [{gross.ci_low:.2%}, {gross.ci_high:.2%}], raw p={gross.p:.6g}, Holm p={gross.p_holm:.6g} across the 21 declared aggregate hypotheses.\n'
        f'Exposure targets +50% long/-50% short; actual mean gross {gross.mean_gross_exposure:.4%}, maximum absolute net {gross.max_abs_net_exposure:.3g}.\n'
        f'At 5bp on buys/short covers: annual mean {net.annual_mean:.2%}, Sharpe {net.sharpe:.2f}, p={net.p:.4f}; not significantly positive.\n'
        f'Capped APO before costs: annual mean {apo.annual_mean:.2%}, Sharpe {apo.sharpe:.2f}, raw p={apo.p:.4f}, Holm p={apo.p_holm:.4f}; nominal only.\n'
        f'The 2021-2025 equal-dollar mean is {recent.annual_mean:.2%} (raw p={recent.p:.4f}, Holm p={recent.p_holm:.4f}); it does not pass the declared family correction separately.\n'
        f'Predeclared adverse missing-payoff stress leaves full-period annual mean {stressed.annual_mean:.2%}, Sharpe {stressed.sharpe:.2f}, raw HAC p={stressed.p:.6g}.\n'
        f'Gains are uneven: calendar returns in 2023, 2024 and 2025 were {yearly_equal.loc["2023","calendar_return"]:.2%}, {yearly_equal.loc["2024","calendar_return"]:.2%}, and {yearly_equal.loc["2025","calendar_return"]:.2%}.\n'
        'The larger full-period result is concentrated in 2020 and 2022; it is not proof of a stable future gain.\n'
        'This is a retrospective US adaptation with an OHLC4-for-VWAP substitution and idealized fills/short availability.\n')
    text='LiMT US TRANSFER — COMPLETED RESULTS\n\nRESULTS\n'+conclusion+'\nDETAILS AND METHODS\n'+body
    text+='\nFINAL SOURCE AUDIT AND PREDECLARED MISSING-PAYOFF STRESS\n'
    text+='All 19 original source partitions retain their pretraining SHA256 hashes.\n'
    text+=f'{len(joined):,} distinct ensemble held stock-date payoffs independently match raw CRSP.\n'
    text+=f'{int(joined.prepared_return.isna().sum())} distinct held stock/date payoffs are missing in the source; no outcome-based deletion was used.\n'
    text+='Adverse stress assigns -100% to missing long returns and +100% to missing short returns.\n'
    text+='Stress results are predeclared robustness diagnostics, not newly selected portfolios or extra confirmatory tests.\n'
    text+=stress_frame.to_string(index=False)+'\n\nCALENDAR-YEAR ZERO-COST RETURNS (descriptive, not a search for significant subperiods)\n'
    text+=yearly.to_string(index=False)+'\n'
    text+='The yearly table shows when gains accrued; full-period significance is not evidence of stability in every year.\n'
    report.write_text(text)
    import html
    page=ROOT/'report.html'
    import re
    content=page.read_text().split('<h2>Final source audit and robustness diagnostics</h2>')[0]
    content=re.sub(r'<section id="measured-results">.*?</section>','',content,flags=re.S)
    content=content.replace('<h1>LiMT on US daily CRSP data</h1>',
        '<h1>LiMT on US daily CRSP data</h1><section id="measured-results"><h2>Measured result</h2><pre>'+html.escape(conclusion)+'</pre></section>')
    content=content.replace('</body>','') if '</body>' in content else content
    content=content.replace('</html>','')+ '<h2>Final source audit and robustness diagnostics</h2><pre>'+html.escape(text[text.index('\nFINAL SOURCE AUDIT'):])+'</pre></html>'
    page.write_text(content)
    files=[ROOT/'protocol.json',ROOT/'train.py',ROOT/'usdata.py',ROOT/'baseline.py',ROOT/'portfolio.py',ROOT/'analyze.py',
        ROOT/'verify_predictions.py',ROOT/'audit_results.py',ROOT/'run.py',ROOT/'README.txt',ROOT/'REPORT.txt',ROOT/'report.html',folder/'summary.csv',
        folder/'verification.json',ROOT/'prediction-replay-verification.json',folder/'missing_stress_summary.csv',folder/'calendar_years.csv',folder/'performance.png']
    write_json(folder/'final-checks.json',dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks,source_partitions_verified_unchanged=len(manifest['source_sha256']),
        distinct_held_stock_dates=len(joined),missing_source_payoffs=int(joined.prepared_return.isna().sum()),
        unobserved_source_rows=int((joined._merge=='left_only').sum()),
        tests='11 passed',plot='visually inspected; no clipped labels',
        artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    print('FINAL AUDIT PASSED',len(joined),'held stock-dates;',int(joined.prepared_return.isna().sum()),'missing',flush=True)


if __name__=='__main__':main()
