"""Retrain the predefined three-seed Figure-3 correction and audit its outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes.evaluation import predict_checkpoint
from hermes.metrics import evaluate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    protocol_path = ROOT/'configs/sp500_figure3.json'
    protocol = json.loads(protocol_path.read_text())
    dest = ROOT/'results/sp500-figure3-2026-11-11'
    dest.mkdir(parents=True, exist_ok=True)
    provenance = {'protocol': protocol, 'protocol_sha256': sha(protocol_path),
                  'data_sha256': sha(ROOT/'data/sp500.npz'),
                  'code_sha256': {str(f.relative_to(ROOT)): sha(f) for f in
                                  [ROOT/'scripts/train.py', ROOT/'hermes/model.py',
                                   ROOT/'hermes/metrics.py', ROOT/'hermes/evaluation.py',
                                   ROOT/'hermes/costs.py']}}
    record = dest/'protocol.json'
    if record.exists():
        assert json.loads(record.read_text()) == provenance, 'protocol/data/code changed'
    else:
        record.write_text(json.dumps(provenance, indent=2)+'\n')
    rows, checks = [], []
    data = np.load(ROOT/'data/sp500.npz')
    for seed in protocol['seeds']:
        run = dest/f'seed{seed}'
        cmd = [sys.executable, str(ROOT/'scripts/train.py'), '--output', str(run),
               '--device', args.device, '--seed', str(seed)]
        for key, value in protocol['training'].items():
            cmd.append('--'+key.replace('_','-'))
            cmd.extend(map(str, value if isinstance(value, list) else [value]))
        (dest/f'seed{seed}.command.json').write_text(json.dumps(cmd, indent=2)+'\n')
        if not (run/'metrics.json').exists():
            print(f'Training Figure-3 settings, seed {seed}', flush=True)
            with (dest/f'seed{seed}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        config = json.loads((run/'config.json').read_text())
        for key, value in protocol['training'].items():
            assert config[key] == value, (key, config[key], value)
        assert config['seed'] == seed
        metrics = json.loads((run/'metrics.json').read_text())
        arrays = np.load(run/'test_predictions.npz')
        np.testing.assert_array_equal(arrays['target'], data['returns'][1259:])
        np.testing.assert_array_equal(arrays['dates'], data['dates'][1259:])
        np.testing.assert_array_equal(arrays['tickers'], data['tickers'])
        np.testing.assert_array_equal(arrays['valid'], data['valid'][1259:])
        recalculated, _, _ = evaluate(arrays['prediction'], arrays['target'], arrays['valid'], 5)
        for key, value in recalculated.items():
            if value is None:
                assert metrics['test'][key] is None
            else:
                np.testing.assert_allclose(value, metrics['test'][key], rtol=1e-10, atol=1e-12)
        replay, checkpoint = predict_checkpoint(run/'best.pt', data, 1259, 1611, args.device)
        np.testing.assert_array_equal(replay, arrays['prediction'])
        history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
        assert min(history, key=lambda r:r['val_loss'])['epoch'] == metrics['best_epoch'] == checkpoint['epoch']
        rows.append({'seed':seed, 'best_epoch':metrics['best_epoch'],
                     'epochs_run':metrics['epochs_run'], 'validation_IC':metrics['validation']['IC'],
                     **{k:metrics['test'][k] for k in ['IC','ICIR','SR_annualized','net_SR_annualized']}})
        checks.append({'seed':seed, 'config_and_targets':'pass', 'metrics':'pass',
                       'checkpoint_replay':'bit_identical', 'validation_selection':'pass',
                       'checkpoint_sha256':sha(run/'best.pt'),
                       'predictions_sha256':sha(run/'test_predictions.npz')})
        print(f'Seed {seed}: test IC={metrics["test"]["IC"]:.6f}; replay verified', flush=True)
    with (dest/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'runs':rows,'verification':checks,'paper':{'IC':.050,'ICIR':.334,'SR':2.247},
               'retained_uncertainties':protocol['retained_uncertainties']}
    lines = ['S&P500 FIGURE-3 CORRECTION: THREE PREDEFINED SEEDS',
             'Changed only hidden dimension 8 -> 5 and lag windows [3,2,1] -> [3,3,3].',
             'Figure 3 supports these candidates, but does not establish the complete author headline configuration.',
             'Other settings, preprocessing and validation-loss selection retained. Sum-pair loss follows Eq.24.',
             'Historical outputs remain separate. No seed selected or settings adjusted using these test results.', '',
             'Metric                       Historical mean     Figure-3 mean     Across-seed SD     Paper']
    summary['means'], summary['historical_means'] = {}, {}
    for metric, paper in [('IC',.050),('ICIR',.334),('SR_annualized',2.247),('net_SR_annualized',None)]:
        vals = [r[metric] for r in rows]
        old = [json.loads((ROOT/f'results/sp500-full-seed{s}/metrics.json').read_text())['test'][metric] for s in protocol['seeds']]
        mean, sd = float(np.mean(vals)), float(np.std(vals,ddof=1))
        summary['means'][metric] = {'mean':mean,'std_across_seeds':sd}
        summary['historical_means'][metric] = float(np.mean(old))
        lines.append(f'{metric:27s} {np.mean(old):15.6f} {mean:17.6f} {sd:18.6f} {str(paper):>9s}')
    lines += ['', 'Sharpe rows use the local dollar-neutral protocol and are not directly comparable to paper SR.',
              'SD is across three initialization seeds, not uncertainty across independent market samples.',
              '', 'Per-seed test IC: '+', '.join(f'{r["seed"]}: {r["IC"]:.6f}' for r in rows),
              '', 'Verification: all configs, target/date/ticker alignment, recomputed test metrics,',
              'minimum-validation-loss checkpoint selection and bit-identical GPU replay passed.',
              '', 'Remaining reconstruction choices:'] + protocol['retained_uncertainties']
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (dest/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
