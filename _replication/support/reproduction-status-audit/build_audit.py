"""Summarize saved local evidence; never launch experiments or change source data."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reproduction-status-audit'

# Verdicts refer to the whole paper, including reported baselines and ablations.
# NO = documented discrepancy/substitution/partial coverage; UNVERIFIED = no
# completed comparable experiment; N/A = new research rather than a replication.
PRIMARY = [
('2505.11243', 'UNVERIFIED', '0/297 published finance targets reproduced; zero model fits.', 'Original CoreLogic mortgage panel and exact equity cohort/preprocessing are unavailable; Set-Sequence trainer and full baseline suite are unimplemented.', ['reports/REPRODUCTION_STATUS.txt', 'targets/published_values.csv', 'configs/run_plan_summary.json']),
('DRO', 'NO', 'Full portfolio experiment executed; all nine author endpoint comparisons differ (maximum relative difference 0.286607%).', 'Original numerical environment, posterior random draws and solver behavior are not fully pinned; the completed independent run differs from all nine saved author endpoints.', ['README.md', 'results/full/authors_endpoints_comparison.csv']),
('RADAR', 'NO', 'All seven S&P500 benchmark metrics match at 3 decimals; equal-weight Sharpe 0.579489 vs 0.795; no full native model suite.', 'Original 377-stock price/news universe and full experiment/seed protocols are unresolved; local 285-stock reconstruction and CZ202 adaptation do not reproduce the paper.', ['REPORT.txt', 'GITHUB/results/reproduction_audit/paper_comparison.txt']),
('alpha-research-reproduction', 'NO', 'Only reduced pretrained Kronos inference is complete; other staged paper pipelines are not full reproductions.', 'Kronos pretraining and paper-size evaluation were not repeated; CogAlpha/FaVOR/FinCAD/DSTPO and other staged pipelines lack completed comparable experiments.', ['STATUS.json', 'results/kronos/summary.json']),
('alphaprobe-reproduction', 'NO', '12/12 predictive point estimates below paper targets; three seed-0 substituted runs completed.', '0.6B embedding and DeepSeek Flash replace the paper\'s 4B embedding/V3.1 generator; exact data/protocol, full seeds, baselines and ablations remain unresolved.', ['STATUS.json', 'results/small-all-universes/comparison.csv', 'README.txt']),
('alphasage-reproduction', 'NO', 'CSI300 AlphaSAGE IC 0.04913 vs 0.079 and RankIC 0.06713 vs 0.094; seed-0 adaptation only.', 'Community Qlib data, purged split and evaluator/integration choices differ; complete original-market seeds, baselines and ablations have not been matched.', ['README.txt', 'results/comparison.csv']),
('arboreal-reproduction', 'NO', '55/55 configured public benchmark runs complete; published metrics are unmatched under local splits/search.', 'Exact author splits/search and private TabCredit are unavailable; paper Equation 2 and Algorithm 1 conflict, requiring an explicit implementation choice.', ['artifacts/benchmark/REPORT.md', 'README.md', 'PROTOCOL.md']),
('archetypetrader-reproduction', 'UNVERIFIED', '0/56 original paper configurations complete; equity transfer and module checks are not native reference results.', 'Continuous original four-pair 25-level historical books and venue/features/timing/seed protocol are unresolved; native full training/integration is unexecuted and third-party reference causality checks fail.', ['STATUS.json', 'native-reference/STATUS.json', 'from-scratch/STATUS.json', 'fidelity-audit/REPORT.txt']),
('corporate-graph-anomalies', 'N/A', 'New relationship-graph/return study; no identified paper-number replication and no anomaly-to-price result.', '', ['REPORT.txt']),
('earnings-timing', 'N/A', 'Original timing research and retrospective reconstruction; not an exact replication of an identified external paper.', '', ['README.txt', 'gross_reproduction/README.txt']),
('error-propagation-reproduction', 'NO', '100 main fits complete; only 1/10 main cell means lies inside the paper interval; no exact numerical match established.', 'Independent full-KRR/direct-Nystrom solvers substitute for FALKON; original experiment code/configuration and random draws are unavailable and measured pricing intervals differ.', ['README.txt', 'results/report_metrics.json']),
('expanding-chaos-reproduction', 'NO', 'Synthetic OU/Heston/Phi4 experiments executed; published benchmark accuracy not matched and several tasks omitted.', 'Official release inspected contained no implementation; independent SDENO/F-SPDENO choices and incomplete baseline/task coverage prevent exact published-number reproduction.', ['README.md', 'report/index.html', 'report/finance.html']),
('hermes-reproduction', 'NO', 'Main three-seed mean IC 0.000535 vs paper 0.050; ICIR 0.005667 vs 0.334.', 'Exact original data/calendar, architecture/training choices and portfolio convention are unresolved; independently reconstructed and transfer results differ from the paper.', ['results/REPORT.txt', 'results/exact-audit-2026-11-11/REPORT.txt']),
('hga-mt-reproduction', 'NO', 'Six reduced diagnostic fits complete; no full original result replicated; published MRR and average-Sharpe inconsistencies remain.', 'Original RESSET 39-feature data, Hundsun news/MSC checkpoint, Qichacha typed edges and walk-forward/seeds are missing; published metric/table inconsistencies require clarification.', ['REPORT.txt', 'STATUS.json']),
('learning-augmented-ops-reproduction', 'UNVERIFIED', 'No target algorithm or empirical experiment executed; paper numerical tables unavailable locally.', 'Exact target full manuscript/supplement remains unavailable in the saved access attempts, so algorithms, experiment settings and numerical targets cannot be audited.', ['reproduction_status.json']),
('limt-reproduction', 'NO', '0 exact original-data targets verified out of 270 means/scalars plus 140 SDs; alternative-data counterparts do not establish matching.', 'Six original Quark pickle downloads require a supported client/alternate link; selected configurations, confidential APO engine and Chinese FF5/accounting protocol are unresolved.', ['published-replication/STATUS.txt', 'published-replication/blockers.json', 'README.md']),
('limt-us-reproduction', 'N/A', 'Completed US transfer with OHLC4 substituted for VWAP; not the paper\'s Chinese benchmark.', '', ['REPORT.txt']),
('liu26y-reproduction', 'NO', '147/147 core stock values match published precision (3 significant figures), conditional on author forecast caches; PI/electricity/synthetic results do not all match.', 'Exact PI configuration and synthetic seeds/generation are unspecified and electricity results differ; base forecasting models were not retrained, so the full paper is not exactly reproduced.', ['RESULTS.md', 'README.md', 'results/paper_comparison.csv', 'STATUS.json']),
('monthly-coforecast', 'N/A', 'New CRSP model-family comparison; explicitly not an exact replication of the five cited papers.', '', ['README.txt']),
('multihorizon-forecast', 'N/A', 'New multi-horizon coforecasting experiment; no original-paper numerical target comparison established.', '', ['README.txt', 'results/REPORT.txt']),
('no-cost-neutral-audit', 'N/A', 'Cross-project portfolio re-evaluation under new exposure/cost rules; not a paper-native numerical replication.', '', ['REPORT.txt']),
('ohnemus26a-reproduction', 'UNVERIFIED', 'Author source/paper staged; local results directory empty.', 'No original experiment has been executed; the original synthetic-only study was excluded from the earlier observed-finance programme.', ['upstream/README.md', '../paper-reproduction-audit/reviews/064.json']),
('paper-124-reproduction', 'UNVERIFIED', 'Source/poster audit only; zero empirical experiments complete.', 'Two-page paper/poster do not supply an executable architecture/training recipe, exact data/splits, seeds or complete baseline protocol.', ['REPORT.txt']),
('paper-claim-validation', 'NO', 'Aggregate controls: Vision Table 2 52/52 cells, four SDF coefficient/t cells and Prophets Brier 39/39 points match at printed precision; no full paper certified.', 'Conditional artifact replays leave original training, full SDF model selection, unavailable forecast lanes and other paper-specific data/method gaps unresolved.', ['REPORT.txt']),
('portfoliopilot-reproduction', 'UNVERIFIED', 'No fixed published quantitative benchmark exists in the inspected demonstration paper; component audit is separate.', 'Paper does not identify immutable ticker/date/configuration/seed targets or a numerical results table; component runs cannot establish exact original-demo reproduction.', ['README.txt', 'audit/source_findings.json']),
('prism-vq-reproduction', 'NO', 'Original-feature artifacts and completed CZ202 transfer are explicitly non-exact; root exact_published_results_reproduced=false.', 'Author-exact input snapshot/VWAP, original portfolio/accounting semantics and full published configurations are unresolved; current CZ202 run changes the paper inputs.', ['STATUS.json', 'RESULTS.txt', 'results/report.json', 'README.txt']),
('prophets-profit-reproduction', 'NO', '39/39 available Brier Table 7 points and 467 curves match release rounding; only 13/18 model lanes (78706/94453 forecasts) available.', 'Five withheld forecast lanes, event/question identifiers and original cohorts are missing; Log/Spherical and baseline/live-ledger definitions do not recover all published values.', ['README.txt', 'results/status.json']),
('proximal-portfolio-reproduction', 'NO', '288 main runs complete; 0/24 proposed-method Table 5 wealth cells within 1% of paper values.', 'Original update/window/initialization and accounting ambiguities remain unresolved; original-data equation-based runs differ materially and full competing baselines are incomplete.', ['REPORT.txt', 'reproduction_status.json']),
('song26l-reproduction', 'NO', 'Daily CSI adaptations differ; saved November 11 audit says US LSTM/GRU do not match paper IC/BLO gain; no full original paper replication.', 'Original minute feature definitions, vendor cleaning/universe and complete backbone/ablation/portfolio protocols remain unresolved; TCN comparison failed; raw US data completion alone is insufficient.', ['PUBLISHED_RESULT_AUDIT.md', 'methodology_audit_20261111/REPORT.md', 'STATUS.md']),
('song26l-us', 'UNVERIFIED', 'US data acquisition and independent LSTM pipeline are not an exact original-feature reproduction.', 'Original proprietary minute feature formulas and exact cleaning/universe choices are undisclosed; explicit 72-feature reconstruction and limited backbone scope cannot establish exact paper numbers.', ['reproduction/README.txt', 'README.txt', '../song26l-reproduction/PUBLISHED_RESULT_AUDIT.md']),
('sparse-tangent-original-reproduction', 'NO', 'Release-faithful PFL replay and fidelity audits completed in part; no exact published-number reproduction established.', 'Original Yahoo/KRX sample vintages, full experiment settings and SD-relaxation cardinality mapping are unavailable; Sharpe convention and method/source differences remain unresolved.', ['README.txt', 'results/fidelity_audit/audit.txt']),
('stable-alpha-reproduction', 'UNVERIFIED', '0 original market experiments and 0 published metric cells; eight synthetic software checks are not empirical results.', 'Exact manuscript/targets and input feature/universe/preprocessing specification remain unavailable; source defaults and synthetic smoke tests cannot reconstruct the financial experiment.', ['REPORT.txt', 'STATUS.json']),
('transalpha-reproduction', 'NO', '27/27 daily adaptation fits complete; paper\'s private intraday benchmark not reproduced.', 'Exact anonymous 400-factor 15-minute HDF5 panel, aligned labels/IDs, selected hyperparameters and original execution rules are missing; the daily adaptation changes the experiment.', ['README.txt', 'results/REPORT.txt']),
('vision-to-value-evaluation', 'NO', 'Independent portfolio/CZ202 evaluation, not image-network retraining or full 200-seed SDF selection; conditional paper controls exist in paper-claim-validation.', 'Original full image training/inference and complete 200-seed-per-factor selection are not reproduced; changed portfolio protocol does not test equality to all paper-native values.', ['REPORT.txt', '../paper-claim-validation/REPORT.txt']),
]

# Payment can remove a named capacity/access obstacle, not guarantee matching.
# A quote is intentionally required: the local evidence supplies no current price.
BUDGET_PRIMARY = {
 '2505.11243': 'PARTIAL: quote required for lawful access to the exact CoreLogic mortgage history; paper-based equity plan estimates 525 A6000 GPU-hours (not benchmarked); code/cohort/protocol gaps remain.',
 'alphaprobe-reproduction': 'PARTIAL: rent GPU capacity for the paper\'s 4B embedding instead of 0.6B (quote required); exact V3.1 generator availability/data/seeds still need resolution and matching is not guaranteed.',
 'archetypetrader-reproduction': 'PARTIAL: obtain a vendor quote for continuous historical 25-level books for the four paper pairs after confirming venue/dates; feature/configuration and reference-code defects remain.',
 'hga-mt-reproduction': 'PARTIAL: quotes required for matching RESSET, Hundsun and Qichacha data rights; purchase cannot recover undisclosed preprocessing/seeds or resolve inconsistent published metrics.',
}
BUDGET_PAPER = {
 20: 'PARTIAL: quote required for the exact historical Nasdaq ITCH data after confirming dates/sampling; full manuscript/checkpoint/seed gaps remain.',
 42: 'PARTIAL: quote required for the exact Massive.com 512-security history after confirming universe/date/field manifest; original processed input and evaluation protocol still require verification.',
 323: 'PARTIAL: quote required for the exact EODHD historical OHLCV/adjustment panels including delisted assets; frozen original universe/snapshot and protocol equivalence remain unverified.',
 333: 'PARTIAL: rent a GPU meeting the saved 4096 MiB workload plus 768 MiB reserve (quote required); this clears capacity admission only, not original data/method gaps.',
 337: 'PARTIAL: quote required for matching 2022-2024 Binance/Tardis four-pair history; exact feature definitions, processing and released configuration gaps remain.',
 387: 'PARTIAL: quote required for matching Financial Modeling Prep historical OHLC/adjClose data and API access; original DJ30/SP500 snapshot and 152-feature reconstruction still require verification.',
 397: 'PARTIAL: rent GPU capacity meeting the saved 4864 MiB admission requirement (quote required); original-source/protocol constraints still apply and no numerical match is guaranteed.',
 435: 'PARTIAL: quote required for lawful June 2014-March 2015 SSE ten-level quotes and trades for the exact eight stocks; selected days, fill rules and training settings remain unresolved.',
}

DATA_EXCLUSIONS = [
 'Chen_Zimmermann_Factor_Zoo_2026_10', 'compustat', 'crsp-daily',
 'crsp-monthly', 'csi', 'factors-daily', 'global-market-data', 'ibes',
 'ngboost', 'outputs', 'ravenpack', 'sp500-earnings-transcripts',
 'wrds-crsp-download',
]

def line(value):
    return re.sub(r'\s+', ' ', str(value)).strip()

def flatten(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return line(value)
    if isinstance(value, list):
        return '; '.join(filter(None, (flatten(v) for v in value)))
    if isinstance(value, dict):
        keys = ('requirement', 'required', 'blocker', 'description', 'detail',
                'requirements', 'item', 'reason', 'summary')
        values = [flatten(value[k]) for k in keys if value.get(k)]
        return '; '.join(values) if values else '; '.join(
            f'{k}: {flatten(v)}' for k, v in value.items()
            if k not in ('evidence', 'evidence_paths', 'affects'))
    return line(value)

def relative_evidence(folder, paths):
    return [str((ROOT/folder/p).resolve().relative_to(ROOT)) for p in paths
            if (ROOT/folder/p).exists()]

def compact(value, limit=750):
    text = flatten(value)
    if len(text) <= limit:
        return text
    # Preserve a full first sentence if possible; avoid cutting IDs/paths midword.
    cut = text.rfind('; ', 0, limit)
    if cut < limit // 2:
        cut = text.rfind(' ', 0, limit)
    return text[:cut].rstrip(' ;,') + ' [full requirements in cited review]'

def make_records():
    records = []
    primary = {}
    for folder, verdict, finding, blocker, paths in PRIMARY:
        row = dict(path=folder, category='primary_project', verdict=verdict,
                   all_published_numbers_verified=False if verdict != 'N/A' else None,
                   finding=finding, blocker=blocker,
                   budget=BUDGET_PRIMARY.get(folder, ''),
                   evidence=relative_evidence(folder, paths),
                   underlying_record_status=None)
        assert (ROOT/folder).is_dir(), folder
        assert row['evidence'], folder
        records.append(row)
        primary[folder] = row

    # Older programme review records are superseded where a newer direct project
    # report exists; separate workspaces remain visible, not counted as new papers.
    overrides = {
        9:'alphaprobe-reproduction', 11:'hermes-reproduction',
        13:'learning-augmented-ops-reproduction', 19:'stable-alpha-reproduction',
        22:'prophets-profit-reproduction', 36:'proximal-portfolio-reproduction',
        38:'sparse-tangent-original-reproduction', 39:'error-propagation-reproduction',
        46:'song26l-reproduction', 51:'arboreal-reproduction',
        59:'expanding-chaos-reproduction', 64:'ohnemus26a-reproduction',
        67:'liu26y-reproduction', 99:'alphasage-reproduction',
        134:'alpha-research-reproduction', 136:'archetypetrader-reproduction',
        159:'transalpha-reproduction', 160:'prism-vq-reproduction',
    }
    parents = ['paper-reproduction-audit/workspaces',
               'paper-reproduction-audit/reproductions', 'paper-reproductions']
    for parent in parents:
        for folder in sorted((ROOT/parent).iterdir()):
            if not folder.is_dir():
                continue
            match = re.search(r'\d+', folder.name)
            assert match, folder
            pid = int(match[0])
            review = ROOT/'paper-reproduction-audit/reviews'/f'{pid:03}.json'
            j = json.loads(review.read_text())
            excluded = j.get('status') == 'excluded'
            category = 'programme_excluded' if excluded else 'nested_attempt'
            comparison = j.get('paper_comparison')
            finding = compact(comparison, 600) or 'No completed numerical comparison is recorded in the saved review.'
            verdict = 'UNVERIFIED'
            # These two folders are explicitly in-progress source reviews only.
            if pid in (373, 375):
                blocker = 'Source/eligibility review is unfinished; no complete experiment specification, executed benchmark or paper-number comparison is recorded.'
            else:
                blocker = compact(j.get('missing_requirements'), 750)
                if not blocker:
                    blocker = compact(j.get('remaining_blocked_work'), 750)
                if not blocker:
                    blocker = 'No complete, independently checked paper-number comparison is recorded; original experiments remain unfinished or unverified.'
            if j.get('status') in ('partial', 'complete'):
                verdict = 'NO'
            row = dict(path=str(folder.relative_to(ROOT)), category=category,
                       verdict=verdict, all_published_numbers_verified=False,
                       paper_id=pid, title=j.get('title'), finding=finding,
                       blocker=blocker, budget=BUDGET_PAPER.get(pid, '') if not excluded else '',
                       evidence=[str(review.relative_to(ROOT))],
                       underlying_record_status=j.get('status'),
                       review_updated_at=j.get('updated_at'),
                       original_missing_requirements=j.get('missing_requirements'),
                       original_paper_comparison=comparison)
            for name in ['REPORT.txt','REPORT.md','STATUS.txt','README.txt']:
                if (folder/name).is_file():
                    row['evidence'].append(str((folder/name).relative_to(ROOT)))
            if pid in overrides and not excluded:
                source = primary[overrides[pid]]
                for key in ['verdict','finding','blocker','budget']:
                    row[key] = source[key]
                row['finding'] = f"Shared paper with {source['path']}: " + row['finding']
                row['evidence'] += source['evidence']
                row['superseded_by_primary_project'] = source['path']
            if excluded:
                row['finding'] = 'Excluded from the earlier programme, not a completed reproduction: ' + compact(j.get('eligibility_reason'), 650)
                row['blocker'] = ''
            if pid == 362:
                row['verdict'] = 'NO'
                row['finding'] = 'Complete execution is not exact agreement: Tables 1-3 and 7-13 match printed rounding; Table 4 89/130, Table 5 0/20, Table 6 0/20.'
                row['blocker'] = 'Original Monte Carlo seeds are unreleased and native host timezone affects Table 4; full execution still differs in Tables 4-6, so exact paper-wide numerical agreement is unestablished.'
                row['evidence'].append(str((folder/'results/paper_comparison.json').relative_to(ROOT)))
            if pid == 63:
                row['blocker'] = 'Full original grids/ablations remain incomplete; source configuration/selection and deep-predictor forwarding defects plus unverifiable original FICO input hashes prevent exact matching.'
            if pid == 97:
                row['blocker'] = 'Original 1.7B/4B GRPO execution was excluded by the earlier heavy-LLM protocol; released-artifact review is unfinished and no complete comparable numerical reproduction is recorded.'
            followup = j.get('followup_20261112')
            if followup and not excluded:
                row['finding'] = followup['finding']
                row['evidence'] += [p for p in followup.get('evidence', []) if (ROOT/p).is_file()]
                if followup.get('missing'):
                    row['blocker'] = compact(followup['missing'], 750)
                row['latest_execution_followup'] = followup
            records.append(row)
    return records

def main():
    records = make_records()
    note_backup = {}
    backup_path = OUT/'previous-notes.json'
    if backup_path.exists():
        note_backup = json.loads(backup_path.read_text())
    existing_audit = OUT/'audit.json'
    previously_written = set(json.loads(existing_audit.read_text()).get('modified_note_files', [])) if existing_audit.exists() else set()
    changed = []
    for row in records:
        if row['category'] == 'programme_excluded' or row['verdict'] == 'N/A':
            continue
        folder = ROOT/row['path']
        for filename, text in [('BLOCKER.txt', row['blocker']), ('BUDGET.txt', row['budget'])]:
            if not text:
                continue
            path = folder/filename
            content = ('EXACT PAPER NUMBERS: ' + row['verdict'] + '; ' + text
                       if filename == 'BLOCKER.txt' else text)
            content = line(content) + '\n'
            key = str(path.relative_to(ROOT))
            if path.exists() and key not in note_backup and key not in previously_written:
                note_backup[key] = path.read_text()
            path.write_text(content)
            changed.append(key)
    backup_path.write_text(json.dumps(note_backup, indent=2, ensure_ascii=False)+'\n')
    timestamp = datetime.now(timezone.utc).isoformat()
    counts = dict(Counter(row['verdict'] for row in records if row['category'] != 'programme_excluded'))
    manifest = dict(audited_at_utc=timestamp, evidence_scope='Saved local artifacts only; no fresh experiments or web availability checks.',
                    exact_definition='Every published numerical result must match at its stated precision, using the original experimental definition; bitwise equality is a separate stronger condition and is not established.',
                    verdicts={'NO':'Documented mismatch, substituted experiment, or partial coverage; not a full numerical reproduction.', 'UNVERIFIED':'No complete comparable numerical experiment/target set is available.', 'N/A':'New research/transfer evaluation rather than an original-paper reproduction.'},
                    counts_by_workspace_not_unique_paper=counts,
                    full_exact_reproductions_verified=0,
                    programme_excluded_count=sum(r['category']=='programme_excluded' for r in records),
                    data_and_download_folders_excluded=DATA_EXCLUSIONS,
                    other_excluded={'ijcai-paper':'Manuscript authoring workspace; no external-paper reproduction target.', 'paper-reproduction-audit':'Coordinator/catalogue; individual materialized workspaces audited below.', 'paper-reproductions':'Container; its child project audited below.'},
                    budget_policy='Only identified paid data/capacity remedies; PARTIAL means a specific obstacle can be purchased away but exact reproduction is not guaranteed. No current prices available; obtain quotes. Queue contention and withheld author artifacts are not treated as purchasable fixes.',
                    modified_note_files=changed, records=records)
    (OUT/'audit.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    columns = ['path','category','verdict','finding','blocker','budget','evidence']
    with (OUT/'audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({k: '; '.join(row[k]) if k=='evidence' else row[k] for k in columns})
    lines = ['PUBLISHED-NUMBER REPRODUCTION AUDIT', f'Local evidence inspected: {timestamp}',
             'No full exact paper reproduction is verified in the inspected artifacts.',
             'NO = documented mismatch/substitution/partial coverage; UNVERIFIED = insufficient completed comparable evidence; N/A = original research or transfer evaluation.',
             'Exact means equality at the paper\'s printed precision across the whole reported experiment set, not bitwise identity. Partial printed-precision matches are stated explicitly.',
             'This audit reads saved outputs/reports; it does not rerun training or claim an implementation is correct. Saved running labels are not evidence of process liveness.',
             'Nested programme review snapshots can be older than direct project reports; direct reports supersede them where mapped. Workspaces for the same paper are not independent replications.',
             f'Nonexcluded workspace verdicts: {counts}. Earlier programme exclusions: {manifest["programme_excluded_count"]}.',
             'BUDGET.txt identifies a purchasable partial remedy, not a promise that money reproduces the numbers; prices need quotes.',
             'Data/download folders excluded: ' + ', '.join(DATA_EXCLUSIONS),
             'Unmaterialized catalogue entries and upstream/vendor source checkouts are not counted as reproduction projects.', '']
    for category, title in [('primary_project','PRIMARY PROJECTS'),('nested_attempt','NESTED REPRODUCTION ATTEMPTS'),('programme_excluded','EARLIER PROGRAMME EXCLUSIONS (NO NOTES WRITTEN)')]:
        lines.append(title)
        for row in records:
            if row['category'] != category:
                continue
            lines.append(f"{row['path']} | {row['verdict']} | {row['finding']}")
            if row['blocker']:
                lines.append('  BLOCKER: ' + row['blocker'])
            if row['budget']:
                lines.append('  BUDGET: ' + row['budget'])
            lines.append('  Evidence: ' + '; '.join(row['evidence']))
        lines.append('')
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    # Check note shape and source existence without touching the experiments.
    for key in changed:
        text = (ROOT/key).read_text()
        assert len(text.splitlines()) == 1 and text.endswith('\n'), key
    for row in records:
        for key in row['evidence']:
            assert (ROOT/key).is_file(), key
    assert not any(key.split('/')[0] in DATA_EXCLUSIONS for key in changed)
    (OUT/'verification.json').write_text(json.dumps(dict(
        note_files=len(changed), blocker_files=sum(k.endswith('/BLOCKER.txt') for k in changed),
        budget_files=sum(k.endswith('/BUDGET.txt') for k in changed),
        all_notes_one_line=True, all_evidence_paths_exist=True,
        data_folder_note_writes=0, record_count=len(records),
        sha256={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['REPORT.txt','audit.json','audit.csv']}
    ), indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k in ['counts_by_workspace_not_unique_paper','full_exact_reproductions_verified','programme_excluded_count']}, indent=2))
    print(f'{len(records)} records; {len(changed)} one-line notes written; all evidence paths checked.')

if __name__ == '__main__':
    main()
