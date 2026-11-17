"""Report every recent test, with no seed or period selection."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.no_skip_significance import describe
from hermes.metrics import PORTFOLIO_PROTOCOL


def stats(gross,net):
    return {'gross_bp_day':float(gross.mean()*10000),'net_bp_day':float(net.mean()*10000),
            'gross_SR':float(gross.mean()/gross.std()*np.sqrt(252)),
            'net_SR':float(net.mean()/net.std()*np.sqrt(252)),
            'gross_return':float(np.prod(1+gross)-1),'net_return':float(np.prod(1+net)-1),
            'net_p_HAC5':describe(net)['hac']['5']['p_two_sided_normal']}


def main():
    root=ROOT/'results/recent'
    cases=[];combined={};series={}
    paths=sorted(root.glob('*/metrics.json'))
    if len(paths)!=10:raise ValueError(f'expected frozen run plus all nine rolling runs; found {len(paths)}')
    for path in paths:
        label=path.parent.name;d=np.load(path.parent/'test_predictions.npz')
        if label.startswith('frozen'):
            series[label]=(d['dates'],d['net_portfolio_return'])
        else:
            seed=int(label[-1]);combined.setdefault(seed,[]).append((d['dates'],d['portfolio_return'],d['net_portfolio_return']))
        for year in sorted(set(str(x)[:4] for x in d['dates'])):
            mask=np.char.startswith(d['dates'],year);gross,net=d['portfolio_return'][mask],d['net_portfolio_return'][mask]
            cases.append({'run':label,'year':int(year),'start':str(d['dates'][mask][0]),'end':str(d['dates'][mask][-1]),
                          'days':len(net),'mean_cost_bp':float(d['transaction_fees'][mask].mean()*10000),**stats(gross,net)})
    with (root/'yearly_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    aggregate={}
    for seed,parts in combined.items():
        parts=sorted(parts,key=lambda x:str(x[0][0]));dates,gross,net=[np.concatenate([part[k] for part in parts]) for k in range(3)]
        aggregate[str(seed)]={'start':str(dates[0]),'end':str(dates[-1]),'days':len(net),**stats(gross,net),'net_uncertainty':describe(net)}
        series[f'rolling seed {seed}']=(dates,net)
    means={}
    for year in [2023,2024,2025]:
        values=[row for row in cases if row['run'].startswith('rolling') and row['year']==year]
        means[str(year)]={key:{'mean':float(np.mean([r[key] for r in values])),
                               'min':float(min(r[key] for r in values)),'max':float(max(r[key] for r in values))}
                          for key in ['net_bp_day','net_SR','net_return']}
    summary={'portfolio_protocol':PORTFOLIO_PROTOCOL,'yearly':cases,'rolling_seed_means':means,'rolling_combined_per_seed':aggregate,
             'notes':['fixed model settings and seeds; validation-loss checkpoint selection only',
                      'CRSP total-return P&L including dividends; regression target is CRSP price return',
                      'frozen Yahoo-benchmark model transferred with pre-2022-05-26 vendor-unit calibration',
                      'fixed original 474-security universe; permanent IDs and prior-day tradability prevent trading inactive securities',
                      '5bp buy-only fees include long purchases and short covers, entry and final liquidation',
                      'rolling test years start flat and finish liquidated; annual refits and no cross-year order netting',
                      'idealized close-to-close execution; additional borrow/financing/slippage not included',
                      'asymptotic two-sided HAC(5) p-values are exploratory and unadjusted for multiple tests']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['RECENT PERIOD TESTS: FIXED NO-SKIP SETTINGS',
           '50% long / 50% short; gross exposure 100%, net exposure zero.',
           'Cost: 5bp on purchases/short covers only. Sales/new shorts free. Actual drift-adjusted turnover.',
           'Source: local CRSP daily through 2025-12-31; no 2026 observations are available locally.',
           '',f'{"Run":32s} {"Year":>4s} {"Gross bp/d":>11s} {"Net bp/d":>10s} {"Gross SR":>10s} {"Net SR":>9s} {"Net return":>11s} {"p(HAC5)":>9s}']
    for row in cases:
        lines.append(f'{row["run"]:32s} {row["year"]:4d} {row["gross_bp_day"]:11.3f} {row["net_bp_day"]:10.3f} {row["gross_SR"]:10.3f} {row["net_SR"]:9.3f} {row["net_return"]:11.2%} {row["net_p_HAC5"]:9.4f}')
    lines+=['','Rolling results summarized across all three seeds (means, not a selected best seed):']
    for year,values in means.items():
        lines.append(f'{year}: net mean {values["net_bp_day"]["mean"]:.3f} bp/day; net SR mean {values["net_SR"]["mean"]:.3f}, range [{values["net_SR"]["min"]:.3f},{values["net_SR"]["max"]:.3f}]')
    lines+=['','Full 2023-2025 rolling sequence per seed (includes each annual fold\'s entry/exit fees):']
    for seed,value in aggregate.items():
        lines.append(f'Seed {seed}: net {value["net_bp_day"]:.3f} bp/day, SR {value["net_SR"]:.3f}, compounded {value["net_return"]:.2%}, HAC(5) p={value["net_p_HAC5"]:.4f}')
    lines+=['','Protocol: analysis/recent/protocol.json. All results are historical tests, not secured daily returns.']+summary['notes']
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4))
    for label,(dates,net) in series.items():
        mask=dates>='2023-01-01'
        ax.plot(dates[mask].astype('datetime64[D]'),np.cumprod(1+net[mask]),label=label)
    ax.axhline(1,color='gray',linestyle='--',linewidth=1)
    ax.set(title='Recent no-skip tests: dollar-neutral, 5bp buys only',ylabel='Growth of $1 after trading fees')
    ax.legend(frameon=False);ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout();fig.savefig(root/'recent_performance.png');plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
