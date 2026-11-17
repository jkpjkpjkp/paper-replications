"""Report every declared run; never pick seeds by held-out performance."""
import json
import os
from pathlib import Path
import pandas as pd
from evaluate import TARGET

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1, 1000, 2000, 3000, 4000, 5000]
DEVICE = os.environ.get('TRA_DEVICE', 'cuda')
RUNTIME_DIR = 'paper-runtime' if DEVICE == 'cuda' else 'paper-runtime-cpu'


def main():
    records = []
    for seed in SEEDS:
        path = ROOT/'results'/RUNTIME_DIR/str(seed)/'tra/complete.json'
        if path.exists():
            obj = json.loads(path.read_text())
            records.append(dict(seed=seed, **obj['evaluation']['metrics']))
    lines = ['TRA paper-runtime reproduction', '',
             'Completed main-stage seeds: {}/6'.format(len(records)),
             'Execution device: {}.'.format(DEVICE),
             'Qlib 0.7.1, Torch 1.7.0+cu110, NumPy 1.17.4, Pandas 1.1.2.',
             'Unmodified released model/dataset/YAML hyperparameters; original released data.',
             'Evaluation is checked against the original notebook with no fresh routing draws.', '']
    result = dict(device=DEVICE, completed_seeds=[r['seed'] for r in records], expected_seeds=SEEDS,
                  exact_reproduction_established=False)
    if records:
        frame = pd.DataFrame(records).set_index('seed')
        frame.to_csv(ROOT/'results/per-seed.csv')
        means, std = frame.mean(), frame.std(ddof=1)
        comparison = []
        for key, target in TARGET.items():
            digits = 3
            formatted = format(means[key], '.3f')
            paper = format(target, '.3f')
            match = formatted == paper
            comparison.append(dict(metric=key, paper=target, mean=float(means[key]),
                                   std=float(std[key]) if len(frame)>1 else None,
                                   difference=float(means[key]-target),
                                   matches_paper_rounding=match))
        pd.DataFrame(comparison).to_csv(ROOT/'results/comparison.csv', index=False)
        result['comparison'] = comparison
        result['all_rounded_means_match'] = all(r['matches_paper_rounding'] for r in comparison)
        lines += ['All completed seeds, equal-weight means (no selection):',
                  pd.DataFrame(comparison).to_string(index=False), '',
                  'Across-seed standard deviations use ddof=1, as in the notebook.',
                  'Published MSE/MAE standard deviations both round to 0.000; other SDs are not reported.', '']
        if len(frame) == 6:
            # Exhaustively expose the five-versus-six ambiguity; none is designated successful.
            leave_one_out = []
            for omitted in SEEDS:
                subset = frame.drop(index=omitted)
                leave_one_out.append(dict(omitted_seed=omitted, **subset.mean().to_dict()))
            pd.DataFrame(leave_one_out).to_csv(ROOT/'results/all-five-seed-subsets.csv', index=False)
            result['five_seed_subsets'] = 'All six subsets reported descriptively; none selected as the paper subset.'
    lines += ['Limits on an exact-reproduction claim:',
              '- Table 1 says five seeds; run.sh names six. The actual table subset is unavailable.',
              '- Python patch version, original GPU/driver and all transitive package versions are not supplied.',
              '- The released sequence builder crosses stock boundaries for some short histories. Preserved for source fidelity.',
              '- The YAML ALSTM width is 256; Appendix B search space lists 32, 64, 128. The notebook confirms width 256.',
              '- Reproduction uses the supplied return file and its notebook treatment, not a reconstructed execution-price panel.',
              '- Matching rounded cells alone would not establish bitwise identity or resolve these source inconsistencies.',
              '- No CZ adaptation or gate approval is performed by this job.']
    (ROOT/'results/report.txt').write_text('\n'.join(lines)+'\n')
    (ROOT/'results/aggregate.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('\n'.join(lines[:8]))


if __name__ == '__main__':
    main()
