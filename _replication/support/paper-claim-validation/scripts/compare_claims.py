#!/usr/bin/env python3
"""Refresh the claim comparisons, evidence registry and honest progress report."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'paper-claim-validation'
OUT=BASE/'results'


def read(path):
    return json.loads((ROOT/path).read_text())


def entry(name,scope,source,status,blockers,evidence):
    return dict(paper=name,scope=scope,paper_source=source,native_claim_status=status,
                blockers=blockers,new_evidence=evidence,
                full_paper_reproduced=False,
                false_negative_due_to_bugs_ruled_out=False)


def main():
    full_dir=OUT/'graph_full'
    if len(list(full_dir.glob('CS_GAT_*.parquet')))==24 and not (full_dir/'verification.json').exists():
        with (full_dir/'verification.log').open('a') as log:
            subprocess.run([sys.executable,str(BASE/'scripts/verify_graph_full.py')],check=True,stdout=log,stderr=log)
    comparisons=[]
    prism=read('prism-vq-reproduction/results/report.json')
    for universe,target in prism['paper_values_reference_only'].items():
        observations=[]
        for seed in range(5):
            p=ROOT/f'prism-vq-reproduction/results/{universe}/stage2/seed{seed}/daily_rankic.csv'
            x=pd.read_csv(p).RankIC.to_numpy()
            observations.append(dict(RankIC=np.nanmean(x),RankICIR=np.nanmean(x)/np.nanstd(x,ddof=1)))
        for metric in ['RankIC','RankICIR']:
            values=[r[metric] for r in observations]
            comparisons.append(dict(paper='PRISM-VQ',dataset=universe,metric=metric,
                paper_value=target[metric],recomputed_value=np.mean(values),seed_sd=np.std(values,ddof=1),
                scope='Five separate seed metrics averaged, original Alpha158 adaptation; different data',
                exact_agreement_claimed=False))
    limt=OUT/'limt_native_metrics.csv'
    if limt.exists():
        d=pd.read_csv(limt)
        for u,target in [('csi300',.0575),('csi500',.0496)]:
            x=d.loc[d.universe.eq(u),'IC']
            comparisons.append(dict(paper='LiMT',dataset=u,metric='IC',paper_value=target,
                recomputed_value=x.mean(),seed_sd=x.std(ddof=1),
                scope='Three seeds; original Alpha158 adaptation with 2010-start snapshot',
                exact_agreement_claimed=False))
    graph=pd.read_csv(ROOT/'financial-graph-reproduction/results/reconstruction/metrics.csv').set_index('model')
    targets=pd.read_csv(ROOT/'financial-graph-reproduction/sources/published_table2.csv')
    for row in targets.itertuples():
        if row.model not in graph.index:continue
        for metric in ['MAE_pct','MSE_x100','RMSE_pct','R2_zero_pct']:
            comparisons.append(dict(paper='Financial Graph',dataset='1999-2022 balanced 164 firms',
                metric=row.model+'/'+metric,paper_value=getattr(row,'paper_'+metric),
                recomputed_value=float(graph.loc[row.model,metric]),
                scope='192 fitted model-years replayed; reduced search; 165-firm original snapshot absent',
                exact_agreement_claimed=False))
    if (full_dir/'metrics.csv').exists():
        full=pd.read_csv(full_dir/'metrics.csv').set_index('model')
        if 'CS_GAT' in full.index:
            paper=targets.set_index('model').loc['CS_GAT']
            for metric in ['MAE_pct','MSE_x100','RMSE_pct','R2_zero_pct']:
                comparisons.append(dict(paper='Financial Graph full budget',dataset='1999-2022 balanced 164 firms',
                    metric='CS_GAT/'+metric,paper_value=float(paper['paper_'+metric]),
                    recomputed_value=float(full.loc['CS_GAT',metric]),
                    scope='Full 12-candidate/10-member posthoc budget diagnostic on reconstructed data',exact_agreement_claimed=False))
    arb=pd.read_csv(ROOT/'arboreal-reproduction/artifacts/benchmark/summary.csv')
    for row in arb.loc[arb.method.eq('arbnn_soft')].itertuples():
        comparisons.append(dict(paper='ArbNN',dataset=row.dataset,metric=row.metric,paper_value=row.paper,
            recomputed_value=row.mean,seed_sd=row.std,
            scope='Five seeds; native tabular tasks with unmatched splits/search; 55 checkpoints replayed',
            exact_agreement_claimed=False))
    probe=pd.read_csv(ROOT/'alphaprobe-reproduction/results/small-all-universes/comparison.csv')
    for r in probe.itertuples():
        comparisons.append(dict(paper='AlphaPROBE',dataset=r.universe,metric=r.metric,
            paper_value=r.paper_percent/100,recomputed_value=r.local_signed_percent/100,
            scope='One seed, 0.6B substituted embedding and generator; saved artifacts rechecked',
            exact_agreement_claimed=False))
    hermes=pd.read_csv(ROOT/'hermes-reproduction/results/comparison.csv')
    h=hermes[hermes.ablation.eq('full') & hermes.rank_reduction.eq('sum') & hermes.run.str.startswith('sp500-full')]
    for metric,target in [('IC',.05),('ICIR',.334)]:
        comparisons.append(dict(paper='HERMES',dataset='S&P500',metric=metric,paper_value=target,
            recomputed_value=h[metric].mean(),seed_sd=h[metric].std(ddof=1),
            scope='Three-seed equation reconstruction; no author implementation validation',
            exact_agreement_claimed=False))
    pd.DataFrame(comparisons).to_csv(OUT/'native_claim_comparison.csv',index=False)
    conditional=OUT/'vision_sdf/published_penalty_table3.json'
    if conditional.exists():
        for row in json.loads(conditional.read_text()):
            for metric,target in [('sdf_loading','paper_loading'),('sdf_loading_t','paper_loading_t')]:
                comparisons.append(dict(paper='From Vision to Value',dataset=row['model'],metric=metric,
                    paper_value=row[target],recomputed_value=row[metric],
                    scope='Table 3 conditional on supplied published penalties; CV selection not certified',
                    exact_agreement_claimed=abs(row[metric]-row[target])<.000500001))
        pd.DataFrame(comparisons).to_csv(OUT/'native_claim_comparison.csv',index=False)
    entries=[
        entry('LiMT','Current CZ202 and earlier neutral tests','limt-reproduction/sources/paper-browser.txt',
              'Original-market forecasts replayed; printed IC not matched',
              ['Paper trains from 2008; substitute data starts 2010. Author processed panel requires Quark login.',
               'Original APO execution engine is not released. Six local checkpoints are adaptations.'],
              ['results/limt_native_verification.json','results/limt_native_metrics.csv','results/limt_author_equivalence_tests.log']),
        entry('PRISM-VQ','Current CZ202','prism-vq-reproduction/sources/arxiv.txt',
              'Ten native-feature checkpoints replayed; positive ranking performance recovered; full paper unmatched',
              ['CSI training starts 2010 rather than 2009; US source is CRSP with VWAP0 unavailable.',
               'Original Top30/Drop5 portfolio implementation/data differ. CZ202 reuses the native training loop but changes inputs.',
               'Temporal normalization removes time-constant levels from the latent branch; monthly repeated CZ inputs are materially unlike daily Alpha158. Raw-feature loading branch still exists.'],
              ['results/prism_replay.json','results/prism_native_tests.log','results/prism_input_diagnostic.json']),
        entry('Financial Graph','Current CZ202','financial-graph-reproduction/sources/paper-html.txt',
              '192 fitted model-years replayed; full CS-GAT budget diagnostic running',
              ['164-firm current-vintage panel versus paper 165; original snapshot unavailable.',
               'Original released ED graph constructor is undefined.',
               'Completed reduced search had one ensemble member and a 30-epoch cap. Full search is a posthoc diagnostic, not exact-data replication.',
               'Released models ignore stored edge weights; complete-graph GCN cannot rank firms.'],
              ['results/graph_replay.json','results/graph_author_equivalence_tests.log','evidence/graph_full_config.json','results/graph_full/training.log']),
        entry('From Vision to Value','Current CZ202','vision-to-value-evaluation/sources/paper.txt',
              '52 Table 2 cells reproduced; four Table 3 loading/t cells reproduced at published penalties; full CV in progress',
              ['Correction to initial audit: all ten pretrained image checkpoints ARE present and load successfully. Original raw-price sorting matrices and image-training data are not supplied.',
               'After resolving released seed-column ordering, the observed CRSP checkpoint probe matches 350 of 360 probabilities within 1e-4. Two chart cases remain unmatched; ensemble averages are invariant to that seed reorder.',
               'Full 200-seed CV for each image factor is being reexecuted. The four conditional Table 3 matches do not validate penalty selection.'],
              ['results/vision/verification.json','results/vision/paper_comparison.csv','results/vision/fee_cancellation_witness.json',
               'results/vision/unit_consistent_diagnostics.csv','results/vision_checkpoint/verification.json','results/vision_checkpoint/seed_column_reconciliation.json',
               'results/vision_sdf/published_penalty_table3.json']),
        entry('ArchetypeTrader','Current CZ202','paper-reproduction-audit/workspaces/136-archetypetrader/sources/aaai40166.pdf',
              'No learned native policy result; equity negative remains unvalidated',
              ['Continuous 25-level books for BTC/ETH/DOT/BNB USDT, June 2021 through September 2024, are unavailable.',
               'Only genuine first-of-month book fixtures are available; candles cannot reconstruct historical books.',
               'Exact venue, feature/lookback settings and several optimizer/execution details require explicit reconstruction assumptions.'],
              ['results/core_checks.json','archetypetrader-reproduction/from-scratch/README.txt']),
        entry('HERMES','Earlier 5bp and no-cost neutral tests','hermes-reproduction/sources/paper-browser.txt',
              'Saved adapted metrics reverified; original IC advantage unmatched',
              ['No verified author implementation; equation/model and ranking-loss conventions unresolved.',
               'NASDAQ/NYSE released processed data have shorter/different panels. Source dataset includes stale delisting observations.'],
              ['results/hermes_replay.json','results/hermes_equation_tests.log']),
        entry('AlphaPROBE','Earlier 5bp and no-cost neutral tests','alphaprobe-reproduction/results/small-all-universes/REPORT.txt',
              'Three saved mined-factor runs reverified; original model configuration unmatched',
              ['0.6B embedding substituted for paper 4B; generation model differs.',
               'One seed per universe and reduced fixed mining budget; full baselines and ablations absent.'],
              ['results/alphaprobe_verification.json']),
        entry('ArbNN','Earlier finance adaptation','arboreal-reproduction/sources/paper-camera-ready.pdf',
              '55 native tabular runs re-scored and checkpoint-replayed; published metrics unmatched',
              ['No author source, exact split IDs, Optuna settings or TabCredit release.',
               'Paper equations and Appendix Algorithm 1 disagree on branch signs/depth weights; follows Appendix.',
               'Finance ranking is a separate CRSP task, absent from the paper.'],
              ['results/arboreal_native_verification.log']),
        entry('Prophets Profit','Earlier equity adaptation','prophets-profit-reproduction/sources/paper-html.txt',
              '39 of 39 available Brier Table 7 cells reproduced; native subset validated',
              ['Five of 18 model lanes omitted (15,747 forecasts). Original generation is not rerun.',
               'Event-contract proper-score guarantee does not transfer to stock-return portfolios.'],
              ['results/prophets/verification.json','results/prophets/brier_reproduction.csv','results/prophets/theory_checks.json']),
        entry('Song label-horizon method','Earlier daily/minute adaptations','song26l-reproduction/reproduction/REFERENCE_COMPARISON.md',
              'Official daily-demo engineering checks executed; original minute-paper result absent',
              ['Original full minute features and private cleaning unavailable; author daily demo is a different experiment.',
               'US minute build/download completion is not established by stale running status. No completed minute result is a negative.'],
              ['results/author_reference_checks.json']),
        entry('TransAlpha','Earlier daily CSI adaptation','transalpha-reproduction/README.txt',
              'Original claimed intraday result not reproduced; prior adaptation only',
              ['Private 400-factor, 15-minute panel and exact processing/training protocol absent.',
               'Daily Alpha158 cannot reproduce the original task.'],[]),
        entry('Sparse Tangent','Earlier no-cost neutral tests','paper-reproduction-audit/reviews/038.json',
              'Author source recovered at exact prior commit; original benchmark unreplicated',
              ['Old project directory is absent; source was restored to recovered-sparse-upstream at exact commit 8bc1e802b834bf82fe144e98cdd9e2227607c3e7.',
               'The release distributes no financial CSVs. Original snapshots, FTSE panel and seed/update details remain unresolved.',
               'Existing source audit identifies full-sample scaling leakage, covariance Cholesky orientation mismatch, and soft gate cardinality discrepancies.'],
              ['evidence/sparse_recovered_source.json','paper-reproduction-audit/workspaces/038-original-audit/experiment_coverage.json']),
        entry('AlphaSAGE','Earlier programme no-cost tests','paper-reproduction-audit/reviews/099.json',
              'Corrected-price adaptations exist; original numerical replication unresolved',
              ['Original author panel/snapshot and native executable portfolio are not established.',
               'Released return helper uses overlapping 20-day labels and future-label finite-pair selection; this is not a daily ledger.',
               'Historical raw-price training was corrected in later runs; old uncorrected results must remain superseded.'],
              ['paper-reproduction-audit/workspaces/099-alphasage/results/released-portfolio-proxies/summary.json'])
    ]
    # Preserve the broader catalogue without interpreting unexecuted/blocked
    # records as negative experiments. Supplement actual evaluated extensions.
    snapshot=[]
    for p in sorted((ROOT/'paper-reproduction-audit/reviews').glob('*.json')):
        r=json.loads(p.read_text())
        snapshot.append(dict(paper_id=r.get('paper_id'),title=r.get('title'),status=r.get('status'),
            extensions={k:{x:r.get(k,{}).get(x) for x in ['status','applicable','outcome']}
                        for k in ['market_neutral_nocost','market_neutral_5bp']}))
    (BASE/'evidence/programme_review_snapshot.json').write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n')
    supplementary=[]
    for r in snapshot:
        if r['paper_id'] in [132,169,301,397]:
            source=read(f"paper-reproduction-audit/reviews/{r['paper_id']:03d}.json")
            supplementary.append(dict(**r,source_review=f"paper-reproduction-audit/reviews/{r['paper_id']:03d}.json",
                interpretation='Prior extension evidence only; not newly reproduced or certified by this pass',
                missing_requirements=source.get('missing_requirements'),
                reproduction_level=source.get('reproduction_level')))
    (BASE/'evidence/additional_evaluated_extensions.json').write_text(json.dumps(supplementary,indent=2,ensure_ascii=False)+'\n')
    for e in entries:
        e['evidence_exists']={p:(BASE/p).exists() or (ROOT/p).exists() for p in e['new_evidence']}
    unit=subprocess.run(['systemctl','--user','show','paper-claim-graph-full.service',
                         '-p','ActiveState','-p','MainPID'],capture_output=True,text=True).stdout
    full_dir=OUT/'graph_full'
    completed=sorted(p.stem for p in full_dir.glob('CS_GAT_*.parquet'))
    full_verified=(full_dir/'verification.json').exists() and json.loads((full_dir/'verification.json').read_text()).get('passed')
    if full_verified:
        entries[2]['native_claim_status']='Full-budget reconstructed CS-GAT diagnostic completed and ensemble-replayed; exact original-data replication unresolved'
    graph_activity='completed and independently verified' if full_verified else ('running' if 'ActiveState=active\n' in unit else 'incomplete; inspect service/log before resuming')
    refreshed=datetime.now(timezone.utc).isoformat()
    ongoing={}
    for name,folder in [('graph_released_rule','graph_released_rule'),('vision_sdf','vision_sdf')]:
        path=OUT/folder/('FULL_STATUS.json' if name=='vision_sdf' and (OUT/folder/'FULL_STATUS.json').exists() else 'STATUS.json')
        ongoing[name]=json.loads(path.read_text()) if path.exists() else {'status':'running_or_queued_no_completed_stage_yet'}
    for name in ['CURRENT_FIT.json']:
        path=OUT/'vision_sdf'/name
        if path.exists():ongoing['vision_sdf_current_fit']=json.loads(path.read_text())
    for name in ['paper-claim-graph-released-rule.service','paper-claim-vision-sdf.service','paper-claim-vision-sdf-full.service']:
        ongoing[name]=subprocess.run(['systemctl','--user','show',name,'-p','ActiveState','-p','MainPID'],capture_output=True,text=True).stdout
    record=dict(updated_utc=refreshed,scope='Five current CZ202 cases plus eight earlier root-project cases; broader 437-review inventory retained',
                overall_status='incomplete_original_claim_validation',full_papers_certified=0,
                all_false_negatives_ruled_out=False,papers=entries,
                supplemental_prior_extensions=supplementary,
                graph_full_job=dict(unit='paper-claim-graph-full.service',observed_state=unit,
                                    completed_years=completed,required_years=24),
                continuation_jobs=ongoing,
                local_protocol_sha256=hashlib.sha256((ROOT/'protocol.txt').read_bytes()).hexdigest())
    (BASE/'STATUS.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'claim_registry.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['paper','scope','native_claim_status','full_paper_reproduced','false_negative_due_to_bugs_ruled_out','blockers'])
        writer.writeheader()
        for e in entries:writer.writerow({k:('; '.join(e[k]) if isinstance(e[k],list) else e[k]) for k in writer.fieldnames})
    lines=[
        'PAPER-CLAIM VALIDATION — '+refreshed,'',
        'Validation is INCOMPLETE. No full paper is certified, and this audit cannot rule out all code-caused false negatives.',
        'This pass reuses source code, checkpoints, recorded forecasts and released raw forecast records. It does not call LLM APIs or replace absent financial data with simulated observations.',
        'An inconclusive CZ202/neutral adaptation is not evidence that the original paper fails. A passing engineering test is not a reproduced financial claim. A nonsignificant alpha does not establish zero alpha.','',
        'NEW EXECUTIONS',
        '- Vision Table 2 control: executed the unmodified released portfolio_performance.py against released monthly portfolio artifacts. All 52 selected annual-return, volatility, Sharpe and drawdown cells match at printed precision. This control uses the released monthly portfolios.',
        '- Vision continuation: reused the released SDF numerical code at the published penalties. ViT loading 0.0516628 / t=2.948294 and CNN loading 0.0540829 / t=4.458405 match all four printed coefficient/t cells. This conditional reproduction does not validate selection; the full 200-seed-per-factor CV is running/queued with resumable files.',
        '- Correction: the first audit incorrectly described image-network checkpoints as absent. All ten released weights are present and loadable. Charts regenerated from 36 predeclared observed CRSP cases initially matched 72/360 values under the literal inference-script seed order. The released CSVs reorder seeds: checkpoints 1/42/123/999/12345 correspond to pred2/pred1/pred4/pred3/pred5. Inferred on the first date and checked on two later dates, this yields 350/360 matches within 1e-4. Only CNN PERMNO85374 on 2000-12-29 and ViT PERMNO10001 on 2012-12-31 remain unmatched. The seed reorder cannot change ensemble averages. Original image training and full-period image inference remain unreplicated.',
        '- Prophets Profit: recomputed 78,706 raw forecast records in 13 released model lanes. All 39 available Brier Table 7 point estimates and 467 daily curve points match release rounding. Five lanes remain missing.',
        '- PRISM-VQ: replayed all ten original-feature CSI300/S&P500 model artifacts, all 2,982,950 saved forecasts and 30 CUDA checkpoint date cross-sections. Metrics rejoin original observed labels. Nine existing native pipeline tests pass.',
        '- LiMT: all six CSI300/CSI500 checkpoints replayed at three test cross-sections each; every saved test label and full-period IC/RankIC recomputed. Eight author-equivalence/label/accounting tests pass.',
        '- Financial Graph: verified all 192 fitted model-years, 47,232 observations per model, and ten checkpoint-year forecasts. Eight tests include actual author-class value/input-gradient equivalence.',
        '- ArbNN: re-scored all 55 native tabular dataset/seed runs and replayed 128 checkpoint/symbolic-rule rows per run. Published metrics remain unmatched under the local splits/search.',
        '- HERMES and AlphaPROBE: re-executed saved-run verifiers; HERMES equation checks run separately. These validate reconstructed/substituted pipelines, not author-exact performance.',
        '- Song and ArchetypeTrader: reran official-demo hypergradient and independent native execution/DP checks, respectively. These are explicitly synthetic engineering checks, without financial performance inference.','',
        'FINDINGS',
        '1. LiMT original verifier fails after source evolution. The original model and trainer bytes were recovered and their SHA-256 values exactly match recorded provenance for five runs; the sixth completed after the changes. Current joint model code equals the historical source followed only by a new SingleTaskLiMT class. Trainer changes only move restored RNG states to CPU. Historical-source checkpoint replay succeeds. The initial verifier failure is retained in results/limt_replay.json.',
        '2. Vision native table arithmetic reproduces, but native common leg fees cancel when long-only returns are differenced. A scalar witness is saved. Its published Sharpe calculation also subtracts decimal RF from percentage returns; unit-consistent diagnostics are saved separately. The earlier CZ evaluation uses its own two-leg trade costs.',
        '3. Graph native code ignores stored edge weights. The local dense model matches the actual author classes, including input gradients. A complete-graph GCN cannot rank firms; the tied-leg zero portfolio is a structural consequence, not evidence of useful stock selection.',
        '4. PRISM-VQ original-feature mean RankIC is positive (CSI300 0.069198 vs paper 0.0646; S&P500 0.010321 vs paper 0.0141). CZ202 has a materially different feature time scale. This recovers a native forecasting effect on substitute data, but does not reproduce the paper portfolio claims.',
        '5. HERMES and ArbNN still miss their published native-task performance. Their finance negatives cannot be treated as author-method refutations until the implementation/data/split ambiguities are resolved.','',
        'FULL GRAPH BUDGET DIAGNOSTIC',
        'The existing CS-GAT training code is '+graph_activity+' with 12 configurations, 500-epoch cap, validation patience 5, ten ensemble members, and all 24 annual tests (1999–2022). It uses 2 CPU cores and at most 12 GiB RAM, with no swap/GPU. Completed annual artifacts resume on restart. This is a posthoc reconstruction diagnostic: 164 vs 165 firms and best-epoch vs release stop-epoch retraining remain differences.',
        f'Observed service state: {unit.strip()}; completed years {len(completed)}/24.',
        'Service Restart=no; it survives this chat turn but does not restart automatically after failure/reboot. Resume only after checking the actual service: bash scripts/graph_full.sh (prefer the same bounded systemd command in README.txt). It refreshes this report after successful completion.','',
        'ADDITIONAL NATIVE EXECUTIONS',
        'paper-claim-graph-released-rule.service reuses completed validation searches and retrains ten ensemble members through the released stopping epoch (best epoch + patience, capped at 500). This explicitly isolates the local best-epoch retraining departure. Original data still have 164 rather than 165 firms.',
        'paper-claim-vision-sdf.service executes literal full-grid seed-zero references for ViT and CNN. The optional Gram optimization failed the independent-fold grid-minimum agreement gate, so the full suite retains the literal solver. Two bounded seed workers execute seeds1..199 while seed-zero references finish, then reuse those references. Every seed remains required. Released broadcasting and string-index selection behavior is retained, not silently corrected. Numerical backend checks and declared tolerances are preserved in results/vision_sdf_backend*.',
        'LiMT original data were rechecked: six original train/validation/test pickle files are visible in the official Quark share. Anonymous share listing succeeds, but downloading requires login. A Quark sign-in question is pending; no credentials were invented or login bypass attempted.',
        'These units have no automatic boot/restart scheduler. Inspect actual state before resuming; queued or partial stages are not completed replications.','',
        'PER-PAPER STATUS / BLOCKERS']
    for e in entries:
        lines += ['',e['paper']+': '+e['native_claim_status'],*[f'  - {x}' for x in e['blockers']]]
    lines += ['',
        'Files: results/native_claim_comparison.csv gives numeric targets and observed values; results/claim_registry.csv and STATUS.json give coverage; evidence/session_inventory.json binds the 21 relevant non-dispatch sessions. The 437 registry records are not 437 negative experiments. Supplemental prior extension evidence for papers 132/169/301/397 is inventoried without a new reproduction claim.',
        '',
        'Workflow guidance: Kassis, Timothy; Agarwal, Vinayak; He, Yuhuan; Patel, Darshil; Brueckner, Aubrey M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065. Current record v2 verified in evidence/workflow_reference_check.txt. This is procedural guidance only and supplies no evidence for a finance result.'
    ]
    diagnostic=OUT/'prism_input_diagnostic.json'
    if diagnostic.exists():
        diagnostic=json.loads(diagnostic.read_text())
        lines.insert(lines.index('FULL GRAPH BUDGET DIAGNOSTIC'),
            'Additional PRISM input diagnostic: temporal-constant channels average '
            f"{diagnostic['mean_temporal_constant_channel_fraction']['CZ202']:.2%} with monthly CZ versus "
            f"{diagnostic['mean_temporal_constant_channel_fraction']['Alpha158']:.2%} with daily Alpha158. "
            'On 69 of 753 CZ test dates the entire input window comes from one source month. '
            'Released RevIN maps distinct constant stock levels identically before the latent encoder. '
            'The raw-feature loading branch survives; this is a plausible adaptation limitation, not a proven explanation of negative alpha.\n')
    (BASE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    manifest=[]
    for p in sorted(BASE.rglob('*')):
        if (not p.is_file() or p.name=='artifact_manifest.json' or '__pycache__' in p.parts
                or 'recovered-sparse-upstream' in p.parts or 'vision-deps' in p.parts or not p.resolve().is_relative_to(BASE)):
            continue
        if p.name.endswith('.tmp'):continue
        try:
            blob=p.read_bytes()
            manifest.append(dict(path=str(p.relative_to(BASE)),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
        except FileNotFoundError:
            continue  # A live training worker can atomically rename a staging file.
    (BASE/'artifact_manifest.json').write_text(json.dumps(dict(updated_utc=refreshed,
        scope='Audit artifacts at this report refresh; live training files may subsequently change',files=manifest),indent=2)+'\n')
    print(json.dumps(dict(status=record['overall_status'],cases=len(entries),native_metric_comparisons=len(comparisons),
                          full_papers_certified=0,graph_completed_years=len(completed)),indent=2))


if __name__=='__main__':main()
