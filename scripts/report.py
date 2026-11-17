"""Refresh a factual report from completed artifacts; never turn partial into full."""
from datetime import datetime, timezone
import json
from pathlib import Path
from runtime import ROOT
from build_data import save_json

PAPER = {
    'csi300': dict(RankIC=.0646, RankICIR=.4224, AR=.3077, MDD=.1924, SR=1.5694),
    'sp500': dict(RankIC=.0141, RankICIR=.1208, AR=.1442, MDD=.1616, SR=.6701),
}


def maybe(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    report = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        exact_published_results_reproduced=False, paper_values_reference_only=PAPER,
        markets={}, verification=maybe(ROOT / 'results/verification.json'))
    lines = ['PRISM-VQ reproduction status', '',
        'Paper: https://ijcai.org/proceedings/2026/492',
        'Extended paper: https://arxiv.org/html/2605.13407',
        'Author code: https://github.com/finxlab/PRISM-VQ',
        'Pinned author commit: 7d02635d0cdeec2f4e7278a9e73acd0eb67a8fa6', '',
        'This is an adapted empirical reproduction. Exact published results have NOT been reproduced.',
        'CSI uses a community Qlib source beginning in 2010, with raw-price * cumulative-factor adjustment.',
        'US uses CRSP with historical membership; missing daily VWAP0 is normalized/fill-zero, not proxied.',
        'Train/validation labels crossing split boundaries are purged. Priors end at t-1.',
        'Stage 1 seed 42 is shared across Stage 2 seeds 0..4, following the released code.',
        'Maximum 50 epochs per stage follows the paper; the release defaults to 70.',
        'The released architecture/losses are used, including its auxiliary beta norm and soft cap.',
        'Frozen Stage 1 modules stay in eval mode; the VQ codebook cannot mutate during Stage 2.',
        'Fused Transformer evaluation is disabled so custom RMSNorm executes as implemented.',
        'Portfolio numbers use a documented reconstructed next-close broker, not the authors unprovided portfolio code.',
        'The independent ridge reference is not one of the published baseline experiments.', '']
    complete = True
    for universe in PAPER:
        market = dict(stage1=maybe(ROOT / f'results/{universe}/stage1/status.json'),
            data=maybe(ROOT / f'data/{universe}/build_status.json'),
            ridge=maybe(ROOT / f'results/{universe}/ridge/metrics.json'), seeds={})
        lines += [universe.upper()]
        for seed in range(5):
            folder = ROOT / f'results/{universe}/stage2/seed{seed}'
            m = maybe(folder / 'metrics.json')
            market['seeds'][str(seed)] = m or maybe(folder / 'status.json')
            if m:
                lines.append(f"  seed {seed}: RankIC={m['RankIC']:.6f}, RankICIR={m['RankICIR']:.6f}, selected epoch={m['best_epoch']}, trained={m['trained_epochs']}")
            else:
                complete = False
                lines.append(f'  seed {seed}: full test result not available')
        market['ensemble'] = maybe(ROOT / f'results/{universe}/ensemble/metrics.json')
        if market['ensemble']:
            m = market['ensemble']
            lines.append(f"  5-seed mean RankIC={m['seed_mean_RankIC']:.6f}; RankICIR={m['seed_mean_RankICIR']:.6f}")
            lines.append(f"  Reconstructed ensemble portfolio: AR={m['AR']:.4f}, MDD={m['MDD']:.4f}, SR={m['SR']:.4f}")
        else:
            complete = False
        if market['ridge']:
            m = market['ridge']
            lines.append(f"  Independent ridge: RankIC={m['RankIC']:.6f}, RankICIR={m['RankICIR']:.6f}, AR={m['AR']:.4f}, MDD={m['MDD']:.4f}, SR={m['SR']:.4f}")
        else:
            complete = False
        lines.append('  Published PRISM-VQ values (reference, not measured here): '+str(PAPER[universe]))
        lines.append('')
        report['markets'][universe] = market
    passed = bool(report['verification'] and report['verification']['passed'])
    report['status'] = 'adapted_empirical_run_complete' if complete and passed else 'in_progress'
    lines += ['Overall status: '+report['status'], 'Independent full-run verification passed: '+str(passed), '',
              'Updated UTC: '+report['updated_utc']]
    save_json(ROOT / 'results/report.json', report)
    (ROOT / 'RESULTS.txt').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
