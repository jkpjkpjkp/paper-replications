"""Write a comparison report and cumulative-excess-return figure from completed runs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run=args.run
    protocol=json.loads((run/'protocol.json').read_text())
    paper=protocol['paper_targets']
    rows=[]
    groups={}
    for label,root in [('Demo-derived limits',run),('Opening-price limits',run/'open-limit-sensitivity')]:
        files=sorted(root.glob('seed-*/metrics.json'))
        if not files:
            continue
        records=[json.loads(f.read_text()) for f in files]
        curves=[]
        for f,r in zip(files,records):
            report=pd.read_csv(f.parent/'daily-report.csv',index_col=0,parse_dates=True)
            excess=report['return']-report['bench']-report['cost']
            curves.append(excess.cumsum().rename(f"seed {r['seed']}"))
            for annualizer in ['238','252']:
                m=r['metrics_by_annualization'][annualizer]
                rows.append(dict(protocol=label,seed=r['seed'],annualizer=int(annualizer),
                    AER=m['annualized_excess_return'],IR=m['information_ratio']))
        groups[label]=pd.concat(curves,axis=1)
    table=pd.DataFrame(rows)
    table.to_csv(run/'comparison.csv',index=False)
    lines=['Kronos-base pretrained CSI300 — completed numerical evidence',
           'Paper: https://arxiv.org/html/2508.02739v1 (Table 10)',
           'Paper targets: AER 19.1100%/year, IR 1.378200.',
           'Test period: 2024-07-01 to 2025-06-05; 225 backtest sessions.',
           '66,338 forecasts per seed; 317 constituents; no fine-tuning.',
           'Primary annualizer: 238 (Qlib). All SD values below are seed SD, not CI.', '']
    aggregate={}
    for label in groups:
        frame=table[(table.protocol==label)&(table.annualizer==238)]
        means=frame[['AER','IR']].mean()
        sds=frame[['AER','IR']].std(ddof=1) if len(frame)>1 else None
        aggregate[label]=dict(seeds=frame.seed.tolist(),metrics={})
        lines.append(f'{label}: {len(frame)} completed seeds')
        for row in frame.itertuples():
            lines.append(f'  seed {row.seed}: AER {row.AER*100:.6f}%/year; IR {row.IR:.9f}')
        for short,long in [('AER','annualized_excess_return'),('IR','information_ratio')]:
            avg=float(means[short]);sd=float(sds[short]) if sds is not None else None
            target=paper[long];err=avg-target
            aggregate[label]['metrics'][long]=dict(mean=avg,seed_sd=sd,paper=target,
                signed_error=err,absolute_error=abs(err),relative_error=err/target,
                rounding_tolerance=0.00005)
            sd_text='not estimable' if sd is None else f'{sd:.9f}'
            lines.append(f'  {short} mean {avg:.9f}; seed SD {sd_text}; paper {target:.6f}; signed error {err:.9f}; relative error {100*err/target:.4f}%.')
        lines.append('')
    lines.extend(['Interpretation limits:',
        '- Community snapshot and demo-derived terminal date do not establish exact author-data replay.',
        '- The demo-derived closing-change limit filter can look ahead when executing at open.',
        '- Opening-price limits remove that particular timing issue but retain a simplified 9.5% limit proxy.',
        '- Paper execution details and Table 10 seed uncertainty are unavailable.',
        '- No claim is made about the other Kronos tasks, CSI800, or pretraining.',
        '- See STATUS.txt and WRONG.txt in /home/jkp/data/Kronos for the assessed grade and scope.',
        '', 'Evidence: protocol.json, manifest.json, seed-N/daily-report.csv, seed-N/metrics.json,',
        'open-limit-sensitivity/seed-N/daily-report.csv, comparison.csv, comparison.json.',
        'All reported errors use dimensionless returns/IR; multiply AER errors by 100 for percentage points/year.'])
    (run/'COMPARISON.txt').write_text('\n'.join(lines)+'\n')
    (run/'comparison.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,ax=plt.subplots(figsize=(10.5,5.3))
    colors=['#2463a4','#c46a25']
    for (label,curves),color in zip(groups.items(),colors):
        mean=curves.mean(axis=1)
        ax.plot(mean.index,mean,color=color,lw=2,label=f'{label} (n={curves.shape[1]} seeds)')
        if curves.shape[1]>1:
            sd=curves.std(axis=1,ddof=1)
            ax.fill_between(mean.index,mean-sd,mean+sd,color=color,alpha=.15)
    ax.axhline(0,color='#777777',lw=.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Cumulative excess return after costs (arithmetic sum)')
    ax.set_title('Kronos-base pretrained CSI300 reconstruction\nCommunity data; shaded bands show ±1 seed SD, not confidence intervals')
    ax.grid(alpha=.2);ax.legend(loc='upper left',frameon=False)
    fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(run/'cumulative-excess-returns.png',dpi=160)
    print('\n'.join(lines[:30]))


if __name__=='__main__':
    main()
