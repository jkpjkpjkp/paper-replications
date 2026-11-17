"""Create tables and scientific plots from completed runs, never from invented values."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes.metrics import evaluate, PORTFOLIO_PROTOCOL


def main():
    results = ROOT / 'results'
    runs = []
    for path in sorted(results.glob('sp500-*/metrics.json')):
        r = json.loads(path.read_text())
        c = json.loads((path.parent / 'config.json').read_text())
        runs.append((path.parent, r, c))
    if not runs:
        raise RuntimeError('no completed runs')
    rows = []
    for path, r, c in runs:
        rows.append({'run': path.name, 'ablation': r['ablation'], 'seed': r['seed'],
                     'rank_reduction': c['rank_reduction'], 'best_epoch': r['best_epoch'],
                     'epochs_run': r['epochs_run'], 'parameters': r['parameters'],
                     **{k:r['test'][k] for k in ['IC','ICIR','SR_daily','SR_annualized','net_SR_annualized','mean_daily_return','mean_net_daily_return','mean_transaction_fee','mse','cumulative_return','net_cumulative_return','gross_exposure','net_exposure']}})
    with (results / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    for mode in ['sum', 'mean']:
        group = [r for _, r, c in runs if r['ablation'] == 'full' and c['rank_reduction'] == mode]
        if group:
            summaries[mode] = {'seeds': [r['seed'] for r in group], 'portfolio_protocol':PORTFOLIO_PROTOCOL}
            for metric in ['IC', 'ICIR', 'SR_daily', 'SR_annualized', 'net_SR_annualized', 'mean_net_daily_return','mse']:
                x = [r['test'][metric] for r in group]
                summaries[mode][metric] = {'mean': float(np.mean(x)), 'std_across_seeds': float(np.std(x, ddof=1)) if len(x) > 1 else None}
    (results / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    lines = [
        'HERMES REPRODUCTION RESULTS',
        'Paper: https://arxiv.org/html/2509.23668v3',
        'Independent equation-based reconstruction; these are measured results, not an exact author-code rerun.',
        'S&P500 test: 2021-01-04 through 2022-05-25; 352 days, 474 stocks, 11 sectors.',
        'Paper S&P500 Table 1: IC=0.050, RIC/ICIR=0.334, SR=2.247.',
        'Portfolio correction requested 2026-11-06: top 5 long, bottom 5 short; +50%/-50% NAV, gross 100%, net zero at each daily rebalance.',
        'Gross return = 0.5*(mean long underlying return - mean short underlying return).',
        'Net returns pay 5bp on buys only: long purchases and short covers. Sells/new shorts free. Turnover uses drifted positions; initial entry and final liquidation included.',
        'The paper portfolio convention is unresolved; its reported SR is not directly comparable to this user-specified dollar-neutral protocol.',
        'Historical long-only outputs are superseded and archived in results/archive/long_only_2026-11-05/.',
        '',
        f'{"Run":43s} {"IC":>9s} {"ICIR":>9s} {"Gross SR":>10s} {"Net SR":>10s} {"Net bp/day":>11s} {"Epoch":>7s}',
    ]
    if 'sum' in summaries:
        value = summaries['sum']['IC']
        lines.insert(3, f'Main summed-pair loss: mean test IC={value["mean"]:.6f} across {len(summaries["sum"]["seeds"])} seeds; paper IC=0.050. The published accuracy is not matched by this reconstruction.')
    for row in rows:
        lines.append(f'{row["run"]:43s} {row["IC"]:9.5f} {row["ICIR"]:9.5f} {row["SR_annualized"]:10.4f} {row["net_SR_annualized"]:10.4f} {row["mean_net_daily_return"]*10000:11.4f} {row["best_epoch"]:7d}')
    companions = []
    for path in sorted(results.glob('companion-*/metrics.json')):
        r = json.loads(path.read_text())
        companions.append({'run':path.parent.name, 'dataset':r['dataset'], 'seed':r['seed'],
                           'test_days':r['test']['days'], 'best_epoch':r['best_epoch'],
                           **{k:r['test'][k] for k in ['IC','ICIR','SR_daily','SR_annualized','net_SR_annualized','mean_net_daily_return','mse']}})
    if companions:
        lines += ['', 'NASDAQ/NYSE companion reconstructions (237 test days; processed moving-average features):',
                  'These differ from the paper\'s Table 4 and are NOT exact replications of its headline rows.']
        for row in companions:
            lines.append(f'{row["run"]:43s} {row["IC"]:9.5f} {row["ICIR"]:9.5f} {row["SR_annualized"]:10.4f} {row["net_SR_annualized"]:10.4f} {row["mean_net_daily_return"]*10000:11.4f} {row["best_epoch"]:7d}')
        with (results / 'companion_comparison.csv').open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(companions[0]))
            writer.writeheader()
            writer.writerows(companions)
    lines += ['', 'Across-seed means and sample standard deviations (seeds are not independent market samples):']
    for mode, summary in summaries.items():
        lines.append(f'Pair reduction={mode}; seeds={summary["seeds"]}')
        for metric in ['IC','ICIR','SR_daily','SR_annualized','net_SR_annualized']:
            value = summary[metric]
            lines.append(f'  {metric}: {value["mean"]:.6f} +/- {value["std_across_seeds"] if value["std_across_seeds"] is not None else "NA"}')
    baseline = runs[0][1]['previous_day_return_control']
    yearly=[]
    for path in sorted(results.glob('sp500-*/test_predictions.npz')) + sorted(results.glob('companion-*/test_predictions.npz')):
        d=np.load(path)
        for year in sorted(set(str(x)[:4] for x in d['dates'])):
            mask=np.char.startswith(d['dates'],year)
            gross,net=d['portfolio_return'][mask],d['net_portfolio_return'][mask]
            yearly.append({'run':path.parent.name,'year':year,'start':str(d['dates'][mask][0]),'end':str(d['dates'][mask][-1]),
                           'days':len(net),'gross_return':float(np.prod(1+gross)-1),'net_return':float(np.prod(1+net)-1),
                           'net_mean_bp':float(net.mean()*10000),'net_SR':float(net.mean()/net.std()*np.sqrt(252))})
    with (results/'yearly_returns.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(yearly[0]));writer.writeheader();writer.writerows(yearly)
    lines += ['', 'No-skip seed 0 by calendar year (2022 is partial; continuous positions/costs across the year boundary):']
    for row in yearly:
        if row['run']=='sp500-no_skip-seed0':
            lines.append(f'{row["start"]} to {row["end"]}: gross {row["gross_return"]:.2%}, net {row["net_return"]:.2%}, net mean {row["net_mean_bp"]:.3f} bp/day, net annualized SR {row["net_SR"]:.3f}')
    lines += ['', 'Previous-day-return control on identical dates:', json.dumps(baseline, indent=2), '',
              'Interpretation: a completed implementation/experiment does not establish replication of the published advantage.',
              'Because several experimental details and author code were unavailable, these measurements do not isolate why the published results differ.',
              'No hyperparameters were selected using test metrics. Ablations use seed 0 only; they cannot establish statistical significance.',
              'NASDAQ and NYSE headline tables are not reproduced: public processed releases have fewer dates than Table 4 and different feature definitions. Companion rows, when present, are separate reconstructions.',
              'See README.txt and sources/rsr-data-audit.json for protocol choices and evidence.', '',
              'Research workflow tooling reference:',
              'Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026).',
              'Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.',
              'https://doi.org/10.48550/arXiv.2609.00065 (metadata checked against arXiv).']
    (results / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for path, r, c in runs:
        if r['ablation'] not in ('full','no_skip') or r['seed'] != 0:
            continue
        d = np.load(path / 'test_predictions.npz')
        portfolio = d['net_portfolio_return']
        dates = d['dates'].astype('datetime64[D]')
        axes[0].plot(dates, np.cumprod(1 + portfolio), label='No skip' if r['ablation']=='no_skip' else f'Full, pair {c["rank_reduction"]}')
        if r['ablation']=='no_skip':continue
        history = [json.loads(line) for line in (path / 'history.jsonl').read_text().splitlines()]
        axes[1].plot([h['epoch'] for h in history], [h['val_loss'] for h in history], label=f'Pair {c["rank_reduction"]}')
    axes[0].axhline(1,color='gray',linestyle='--',label='Zero return')
    axes[0].set(title='Dollar-neutral, 5bp buys only (seed 0)', ylabel='Growth of $1 after trading fees')
    axes[1].set(title='Checkpoint selection uses validation only', xlabel='Epoch', ylabel='Validation objective', yscale='log')
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(results / 'reproduction.png')
    plt.close(fig)
    print('\n'.join(lines[:8]))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
