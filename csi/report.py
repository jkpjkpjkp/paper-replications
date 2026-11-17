#!/usr/bin/env python3
"""Render a local progress/results report for the CSI daily adaptation."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import html
import json
import os
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
UNIVERSES = ['csi300', 'csi500', 'csi1000']
METRICS = ['ic', 'icir', 'rank_ic', 'rank_icir', 'top_return_10sessions']


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def atomic_text(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def number(value, digits=5):
    if value is None:
        return 'pending'
    return f'{float(value):.{digits}f}'


def mean_sd(metric):
    if not isinstance(metric, dict) or metric.get('mean') is None:
        return 'pending'
    return f"{number(metric['mean'])} ± {number(metric.get('seed_std'))}"


def phase(status):
    return status.get('phase', status.get('state', status.get('stage', 'not started')))


def plot_horizons(root, summaries):
    completed = [(u, s) for u, s in summaries.items() if s.get('complete') and s.get('horizon_sweep')]
    if not completed:
        return None
    target = root / 'results/horizon_curves.png'
    sources = [root / 'results' / u / 'summary.json' for u, _ in completed]
    if target.exists() and target.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return 'results/horizon_curves.png'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, len(completed), figsize=(4.4 * len(completed), 3.8),
                             squeeze=False, constrained_layout=True)
    for ax, (universe, summary) in zip(axes[0], completed):
        sweep = sorted(summary['horizon_sweep'], key=lambda row: row['horizon'])
        horizon = np.array([row['horizon'] for row in sweep])
        ic = np.array([row['ic']['mean'] for row in sweep])
        sd = np.array([row['ic'].get('seed_std', 0) for row in sweep])
        ax.plot(horizon, ic, '-o', color='#146b91', markersize=4, label='Five-seed mean IC')
        ax.fill_between(horizon, ic - sd, ic + sd, color='#146b91', alpha=.16, label='Across-seed SD')
        ax.axvline(10, color='#c86c30', ls='--', alpha=.8, label='Inference horizon')
        ax.set(title=universe.upper(), xlabel='Supervision horizon (sessions)',
               ylabel='Held-out IC on 10-session target', xticks=[1, 3, 5, 7, 10])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Daily CSI adaptation: descriptive horizon sweep', fontsize=12)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return 'results/horizon_curves.png'


def inline(text):
    value = html.escape(text)
    value = re.sub(r'\[([^\]]+)\]\(([^\s)]+)\)', r'<a href="\2">\1</a>', value)
    value = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', value)
    value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
    return value


def markdown_html(markdown):
    """Render this generator's small Markdown subset without extra packages."""
    lines = markdown.splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if line.startswith('# '):
            out.append('<h1>' + inline(line[2:]) + '</h1>')
        elif line.startswith('## '):
            out.append('<h2>' + inline(line[3:]) + '</h2>')
        elif line.startswith('### '):
            out.append('<h3>' + inline(line[4:]) + '</h3>')
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
            if match:
                out.append(f'<img src="{html.escape(match[2], quote=True)}" alt="{html.escape(match[1], quote=True)}">')
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [s.strip() for s in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append('<div class="table"><table>')
            for j, row in enumerate(rows):
                tag = 'th' if j == 0 else 'td'
                out.append('<tr>' + ''.join(f'<{tag}>{inline(c)}</{tag}>' for c in row) + '</tr>')
            out.append('</table></div>')
            continue
        elif line.startswith('- '):
            out.append('<ul>')
            while i < len(lines) and lines[i].startswith('- '):
                out.append('<li>' + inline(lines[i][2:]) + '</li>')
                i += 1
            out.append('</ul>')
            continue
        else:
            out.append('<p>' + inline(line) + '</p>')
        i += 1
    return '\n'.join(out)


def write_report(root=HERE):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run = read(root / 'run_status.json')
    build = read(root / 'cache/build_status.json')
    inputs = {u: read(root / 'cache' / u / 'manifest.json') for u in UNIVERSES}
    summaries = {u: read(root / 'results' / u / 'summary.json') for u in UNIVERSES}
    statuses = {u: read(root / 'results' / u / 'status.json') for u in UNIVERSES}
    finished = [u for u, s in summaries.items() if s.get('complete')]
    headline = 'complete' if len(finished) == 3 else 'in progress'
    parts = [f'# CSI daily method adaptation: {headline}', '',
             '**Scope: CSI 300, CSI 500 and CSI 1000; Alpha158 inputs; a fixed 10-session VWAP target.**', '',
             'This is a daily adaptation of Song et al.’s supervision-horizon method. '
             'The paper’s original minute experiments and the authors’ separate daily demo use different inputs/settings. '
             'Their published numerical results are not treated as results from this run.', '',
             '[Paper](https://proceedings.mlr.press/v306/song26l.html) · '
             '[Authors’ demo](https://github.com/EFundAI/label-horizon-paradox) · '
             '[Data snapshot](https://huggingface.co/datasets/joshuaxql/qlib_data) · '
             '[Run instructions](README.txt)', '',
             f"Updated: {dt.datetime.now(dt.timezone.utc).isoformat()}", '',
             '## Observed progress', '',
             f"Runner: **{phase(run)}**. Data builder: **{phase(build)}**. "
             f'Completed universe comparisons: **{len(finished)} / 3**.', '',
             '| Universe | Verified tensor manifest | Decision dates by split | Training state | Financial summary |',
             '|---|---|---|---|---|']
    for u in UNIVERSES:
        m = inputs[u]
        split = m.get('summary', {}).get('by_split')
        if split is None and m.get('records'):
            split = dict(Counter(r['split'] for r in m['records']))
        dates = ', '.join(f'{k}: {v}' for k, v in (split or {}).items()) or 'pending'
        parts.append(f"| {u.upper()} | {'complete' if m.get('complete') else 'pending'} | "
                     f"{dates} | {phase(statuses[u])} | {'complete' if summaries[u].get('complete') else 'pending'} |")
    parts += ['', 'Only completed `summary.json` artifacts count as financial results. '
              'Input construction, an engineering check, or a running job does not establish financial replication.', '',
              '## Fixed protocol', '',
              'Signals use information through decision-date close. For h = 1,…,10, the supervision label is '
              '`adjusted_vwap[t+1+h] / adjusted_vwap[t+1] - 1`. All held-out models are evaluated on h = 10. '
              'Membership follows historical constituent intervals on the decision date. '
              'Adjusted price fields are constructed from each dated raw price and factor before Alpha158 expressions; '
              'raw vendor volume is retained. The [publisher’s companion Qlib fork](https://github.com/joshuaxql/qlib) '
              'documents post-adjustment before expression calculation; this builder applies it explicitly to '
              'the raw snapshot. Source adjustment and membership accuracy remain limitations.', '',
              'Decision-date splits: train January 2019–June 2023; validation July 2023–June 2024; '
              'test July 2024–July 2025. Train/validation decisions whose longest candidate label exits '
              'outside the split are purged. August 2025 observations only finish final test labels. '
              'Feature imputation/scaling uses training observations only; held-out labels are untrimmed.', '',
              'The model is AlphaMLP 158 → 256 → 128 → 1 with dropout 0.2. '
              'Every supervision horizon is trained with five seeds (0–4), AdamW at 1e-3, '
              '20-date batches, a 50-epoch cap and validation patience 5. '
              'BLO uses a three-epoch mean-of-standardized-levels warm-up, a differentiable inner step at 1e-6, '
              'outer learning rate 1e-3 and entropy weight 1e-3. The selected horizon is retrained. '
              'All fits and selections finish before test evaluation.', '',
              'Within each training batch, support/query decision dates are disjoint, but their ten-session '
              'return windows can overlap. This paper-style batching does not make the meta-query temporally '
              'independent. Purging at split boundaries protects the held-out validation and test periods.', '',
              '## Primary held-out comparison', '']
    if not finished:
        parts += ['**No completed CSI financial result is available yet.** The table will populate '
                  'after each full five-seed universe comparison finishes.', '']
    else:
        parts += ['Values are means ± standard deviations across five fitted seeds. '
                  'Correlations are in raw units; top-decile return is a decimal TEN-session return.', '',
                  '| Universe | Method | IC | ICIR | RankIC | RankICIR | Top-decile 10-session return |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for u in finished:
            for method in ['standard', 'selected']:
                metric = summaries[u]['primary'][method]
                parts.append('| ' + u.upper() + ' | ' + method + ' | ' +
                             ' | '.join(mean_sd(metric.get(k)) for k in METRICS) + ' |')
        parts += ['', 'The primary comparison uses the standard h=10 model and a model retrained '
                  'on the BLO-selected horizon. Test-sweep ranking does not select the primary model.', '',
                  '| Universe | BLO horizons by seed | Paired mean IC change | 95% date-block interval |',
                  '|---|---|---:|---|']
        for u in finished:
            s = summaries[u]
            bootstrap = s['paired_ic_bootstrap']
            interval = bootstrap['ci_95']
            horizons = ', '.join(f"{seed}: h{h}" for seed, h in s['selected_horizons'].items())
            parts.append(f"| {u.upper()} | {horizons} | {number(bootstrap['mean_difference'])} | "
                         f"[{number(interval[0])}, {number(interval[1])}] |")
    parts += ['', 'The bootstrap uses 20-date circular blocks and 10,000 draws, conditional on the fitted seeds. '
              'Ten-session targets overlap; dependence can last longer than the chosen block. '
              'ICIR and RankICIR describe the overlapping label series and are not annualized.', '',
              '**No annualized Sharpe is calculated from these overlapping ten-session label returns.** '
              'Top-decile returns are signal-quality statistics; a tradable portfolio result requires a separate '
              'holding/rebalancing simulation with execution assumptions and costs.', '',
              '## Descriptive horizon sweep', '']
    if finished:
        parts += ['The following best horizons are selected retrospectively from the held-out sweep '
                  'for description only; they never determine the BLO primary fits.', '',
                  '| Universe | Highest-IC test-sweep horizon | Mean rank correlation: learned weights vs test IC |',
                  '|---|---:|---:|']
        for u in finished:
            s = summaries[u]
            rc = s.get('lambda_test_ic_rank_correlation', {}).get('mean')
            parts.append(f"| {u.upper()} | {s.get('descriptive_best_test_horizon', 'pending')} | {number(rc)} |")
        plotted = plot_horizons(root, summaries)
        if plotted:
            parts += ['', f'![Five-seed daily CSI horizon curves]({plotted})', '',
                      'Shading is across-seed standard deviation, not a confidence interval.']
    else:
        parts += ['Pending all ten supervision horizons and their held-out evaluation.']
    parts += ['', '## Remaining original-paper scope', '',
              'The source contains daily bars. The original CSI experiments need one-minute OHLC, amount, '
              'volume and transaction count, historical constituents, and adjusted minute prices. '
              'The full-session close-to-close scenario uses 16 fifteen-minute patches; the intraday '
              '30/90-minute scenarios use eight morning patches. Those minute experiments remain pending '
              'a suitable data source. Daily values are not expanded into fabricated minutes.', '',
              'The authors’ public daily demo also differs in universe, date splits, preprocessing, '
              'warm-up label, optimization and ensemble settings. See [README.txt](README.txt) for all '
              'disclosed choices and artifact locations.', '']
    markdown = '\n'.join(parts)
    atomic_text(root / 'REPORT.md', markdown)
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
    document += 'content="width=device-width,initial-scale=1"><title>CSI daily method adaptation</title>'
    document += '<style>body{max-width:1120px;margin:36px auto;padding:0 24px;color:#1d3546;'
    document += 'background:#f7fafb;font:16px/1.65 system-ui}h1{line-height:1.2}h2{margin-top:2em}'
    document += 'a{color:#086b99}code{font-size:.9em;background:#e7eef2;padding:2px 4px;border-radius:3px}'
    document += 'table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px 12px;'
    document += 'text-align:left;border-bottom:1px solid #dbe5eb;vertical-align:top}th{background:#eaf1f5}'
    document += '.table{overflow-x:auto;margin:18px 0}img{width:100%;height:auto;background:white}'
    document += 'p{max-width:100ch}strong{color:#16344a}</style><body>'
    document += markdown_html(markdown) + '</body></html>\n'
    atomic_text(root / 'REPORT.html', document)
    return {'complete_universes': finished, 'financial_complete': len(finished) == 3,
            'runner_phase': phase(run), 'data_builder_phase': phase(build)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    print(write_report(parser.parse_args().root))
