"""Publish verified follow-up evidence while retaining the original experiment."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def main():
    summary = json.loads((ROOT / 'SUMMARY.json').read_text())
    assert summary['all_checks_passed'] and summary['training_runs'] == 2
    for name in ['STATUS.txt', 'STATUS.json', 'REPORT.txt', 'MANIFEST.json']:
        backup = PROJECT / 'evidence' / ('before_paper_settings_' + name)
        if not backup.exists():backup.write_bytes((PROJECT / name).read_bytes())
    original = json.loads((PROJECT / 'evidence/before_paper_settings_STATUS.json').read_text())
    s, sd = summary['mean_sharpe'], summary['sample_seed_sd']
    first = (f'1: STABLE US revised paper-settings reconstruction (two seeds), '
             f'mean annualized Sharpe: rerun {s:.3f} +/- {sd:.3f} seed SD; paper 1.85.')
    text = f'''{first}

Updated: {summary['updated_at']}
Scope: completed US follow-up after a paper-settings audit; original four-market,
three-seed reconstruction preserved separately. Fourteen total 200-epoch fits.

Latest evidence:
Seeds 20261120 and 20261121 completed 200 epochs with full fitting through
2024-09-30, March 2025 universe selection, revised features and literal displayed
noise/L2 objective. Mean ASR {s:.6f}, sample seed SD {sd:.6f}, range
[{summary['min_sharpe']:.6f}, {summary['max_sharpe']:.6f}]. Seed SD is not a confidence interval.
Equal-weight ASR {summary['equal_weight_sharpe']:.6f} versus paper 0.82. No exact metric cell certified.
{summary['ledgers_independently_verified']} follow-up ledgers independently reconstructed from prices/holdings.
Main equal-weight fallbacks: {summary['primary_fallbacks']}/{summary['primary_rebalances']}; gross caps: {summary['primary_gross_caps']}.

Historical evidence retained:
Original US mean ASR -0.004 from [-0.446, 0.681, -0.247], with equal weight 0.124.
Original Europe/Korea/China means remain 0.346/-0.246/-0.606. Do not pool the two
new US seeds with the original three: the protocols and selected universes differ.

Limits:
Author repository remains unavailable; exact IDs, historical GICS, UNet,
optimizer/sampling specifications and numeric seed are unresolved. Covariance
ridge, gross cap and fallback remain explicit repairs. End-of-test universe
selection is retrospective. This is closer to documented settings, not an exact
author-code replication or a clean rejection of the paper. Full competitors and
regime experiments remain outside scope. Distinct from stable-alpha.

Evidence:
- PAPER_SETTINGS_AUDIT.txt
- experiments/us_paper_settings_20261117/REPORT.txt
- experiments/us_paper_settings_20261117/SUMMARY.json
- experiments/us_paper_settings_20261117/results/best_effort/portfolio_metrics.csv
- evidence/openreview-complete-20261117.txt
- evidence/before_paper_settings_STATUS.json (original assessment)
'''
    (PROJECT / 'STATUS.txt').write_text(text)
    current = dict(original)
    current.update(updated_at=summary['updated_at'],status='paper_settings_audit_and_two_us_seeds_completed',
                   training_runs=original['training_runs']+2,
                   primary_market_backtests=original['primary_market_backtests']+2,
                   completed_runs_independently_checked=original['completed_runs_independently_checked']+2,
                   portfolio_ledgers_including_baselines_costs_sensitivities=original['portfolio_ledgers_including_baselines_costs_sensitivities']+summary['ledgers_independently_verified'],
                   fallback_decisions=original['fallback_decisions']+summary['primary_fallbacks'],
                   total_primary_rebalances=original['total_primary_rebalances']+summary['primary_rebalances'],
                   latest_experiment=str(ROOT.relative_to(PROJECT)),latest_summary=summary,
                   original_assessment='evidence/before_paper_settings_STATUS.json',
                   no_cross_protocol_seed_pooling=True,
                   seeds_by_setup={'original':original['seeds'],'us_paper_settings':summary['seeds']},
                   remaining_exact_reproduction_gaps=json.loads((ROOT/'protocol.json').read_text())['known_limits'])
    current.pop('seeds',None)
    (PROJECT / 'STATUS.json').write_text(json.dumps(current,indent=2)+'\n')
    previous = (PROJECT / 'evidence/before_paper_settings_REPORT.txt').read_text()
    (PROJECT / 'REPORT.txt').write_text((ROOT / 'REPORT.txt').read_text()+
        '\nDetailed settings audit: PAPER_SETTINGS_AUDIT.txt\n\n'+
        'Historical original four-market reconstruction (different setup)\n'+previous)
    index = PROJECT.parent / 'REPLICATION_STATUS.txt'
    content = index.read_text()
    updated, count = re.subn(r'^  1    stable/STATUS\.txt[^\n]*$',
                            '  1    stable/STATUS.txt — '+first.removeprefix('1: '),content,flags=re.M)
    assert count==1
    index.write_text(updated)
    manifest = {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in PROJECT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                and p != PROJECT/'MANIFEST.json' and 'logs' not in p.parts}
    (PROJECT/'MANIFEST.json').write_text(json.dumps({'created_at':summary['updated_at'],'sha256':manifest},indent=2)+'\n')
    print(first)


if __name__=='__main__':main()
