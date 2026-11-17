#!/usr/bin/env python3
"""Render current evidence and actual protocol, without embedded stale settings."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from progress import collect_progress, describe

ROOT = Path(__file__).resolve().parent.parent

def read(path):
    return json.loads(path.read_text()) if path.exists() else {}

def write_report(root=ROOT):
    root=Path(root);here=root/'reproduction'
    config=read(here/'protocol.json');audit=read(root/'validation.json')
    tensors=read(here/'tensor_status.json');pipeline=read(here/'pipeline_status.json')
    references=read(here/'paper_reference.json')
    progress=collect_progress(root)
    lines=['# Song et al.: current independent minute implementation','',
        '**Exact published-result replication remains unestablished.** Original complete features and configurations are unavailable.', '',
        f"Updated: {datetime.now(timezone.utc).isoformat()}", '',
        f"Protocol: `{config.get('protocol_version')}`. Pipeline: `{pipeline.get('phase','not started')}`.",
        f"Raw audit: complete={audit.get('complete',False)}, verified sessions={audit.get('summary',{}).get('verified_sessions',0)}.",
        f"Tensor dates: {dict(Counter(r['split'] for r in tensors.get('cached',[])))}; complete={tensors.get('complete',False)}.", '',
        '## Method', '',
        'Twenty-six fifteen-minute patches from seven one-minute fields; 66 patch-local statistics/interactions. '
        'All390 candidate labels are cumulative returns from the decision close. Missing intermediate prices carry forward '
        'causally; the final minute must be observed. This price policy and the exact66 features are disclosed approximations.', '',
        'Training-only feature mean/std, without clipping. Double-precision model parameters, logits and losses. '
        'Cross-sectional standardized MSE summed over dates. Twenty-date batches; three warm-up epochs against '
        'mean standardized cumulative levels. Random10/10 support/query split; one differentiable SGD inner step1e-6; '
        'SGD on softmax logits1e-3; entropy weight1e-3. AdamW for supervised training and warm-up. '
        'Patience5 with no primary epoch cap; best checkpoint restored. Supervised validation uses its training horizon; '
        'BLO validation and all financial evaluation use the final target. Retrain on argmax from fresh same-seed initialization.', '',
        'The explicit gradient-descent equations take precedence over the paper’s general AdamW wording for BLO. '
        'Optimizer coordinates, initial supervised LR/weight decay, model widths, data vendor policies and exact July cutoffs '
        'remain assumptions. No settings were chosen to maximize the already-inspected historical test results.', '',
        '## Current five-seed results', '',
        '| Model | State | Standard IC | Selected IC | IC-change95% interval |',
        '|---|---|---:|---:|---|']
    completed=[]
    for name in ['LSTM','GRU','TCN']:
        folder=here/'results' if name=='LSTM' else here/'results/backbones'/name.lower()
        result=read(folder/'summary.json');state=read(folder/'status.json')
        if progress[name]['financial_complete']:
            completed.append(name);m=result['five_seed_metrics'];b=result['paired_ic_bootstrap']
            lines.append(f"| {name} | Complete | {m['standard']['ic']['mean']:.6f} | {m['selected']['ic']['mean']:.6f} | {b['ci_95']} |")
        else:
            lines.append(f"| {name} | {describe(progress[name])} | — | — | — |")
    lines += ['', 'Fit counts require matching checkpoint hashes and run signatures. '
              'Each backbone needs five standard fits, five horizon selections, and five retrained fits. '
              'Process checks distinguish live training from saved progress.']
    for name, entry in progress.items():
        state=entry['state']
        if 'maximum_weight_over_uniform' in state:
            lines += ['', f"{name} last saved BLO diagnostic (epoch {state['epoch']}): "
                      f"maximum weight / uniform weight = {state['maximum_weight_over_uniform']:.12f}, "
                      f"provisional argmax = {state['selected_horizon']} minutes. "
                      'Nearly uniform weights do not establish a stable horizon preference.']
    lines += ['', 'Only completed summaries above are new financial evidence. Previous-protocol results are in '
        '[archive/us-v1](../archive/us-v1/REPORT.md), not included as current results.', '',
        '## Published Table6 references', '',
        '| Model | Standard IC | Selected IC |', '|---|---:|---:|']
    for name,r in references.items():lines.append(f"| {name} | {r['standard']['ic']:.4f} | {r['selected']['ic']:.4f} |")
    lines += ['', 'IC and RankIC are raw correlations (published ×10 columns divided by10). '
        'ICIR is not annualized; Sharpe uses sqrt(252). Top-decile returns are gross price returns, '
        'not an executed portfolio with costs. Summaries average five seeds, not a score ensemble.', '',
        '## Limits and verification', '',
        'Original CSI minute experiments, seven remaining architectures, ablations, stress tests, costed execution and '
        'H20 efficiency measurements remain incomplete. Completed daily CSI adaptations do not reproduce minute experiments.', '',
        '[Protocol](protocol.json) · [Published-protocol checks](PROTOCOL_CHECK.md) · [Implementation notes](README.txt) · '
        '[Pre-change diagnosis](../methodology_audit_20261111/REPORT.md) · [Live status](../STATUS.md)', '',
        'Run `python ../resume.py --device cpu`; inspect `python ../status.py`.', '']
    text='\n'.join(lines);(here/'REPORT.md').write_text(text)
    (here/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>Song reproduction</title><pre style="white-space:pre-wrap">'+html.escape(text)+'</pre>')
    return {'protocol':config.get('protocol_version'),'completed_backbones':completed,'full_paper_complete':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    print(json.dumps(write_report(p.parse_args().root),indent=2))
