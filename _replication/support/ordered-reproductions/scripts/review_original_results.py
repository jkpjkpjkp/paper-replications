"""Record evidence-bounded review of the two completed original experiments."""
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone.utc).isoformat()
for name in ['doubleadapt-original','tra-original']:
    out=ROOT/'results'/name
    s=json.loads((out/'summary.json').read_text());p=json.loads((out/'protocol.json').read_text())
    runner=ROOT/'scripts'/('run_doubleadapt_original.py' if name.startswith('double') else 'run_tra_original.py')
    assert sha(runner)==p['runner_sha256'],'Runner changed after original execution'
    repo=ROOT/'sources'/('DoubleAdapt-qlib' if name.startswith('double') else 'TRA-qlib')
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==p['source_commit']
    assert subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()==''
    metrics=s.get('metrics',s.get('source_report',{}).get('metric'))
    for key,value in s['independently_verified'].items():assert np.isclose(value,metrics[key],rtol=1e-7)
    common=dict(reviewed_utc=now,summary_sha256=sha(out/'summary.json'),protocol_sha256=sha(out/'protocol.json'),
        source_commit=p['source_commit'],runner_hash_verified=True,source_worktree_clean=True,independent_metrics_match=True)
    if name.startswith('double'):
        assert s['trading_days']==871 and s['dates']==['2017-01-03 00:00:00','2020-07-31 00:00:00']
        assert p['config']['step']==20 and p['config']['rank_label'] is False
        assert p['config']['num_head']==8 and p['config']['tau']==10
        gate=dict(**common,status='unresolved_configuration_and_performance',scope='Implementation fidelity checked; paper configuration/performance reconciliation still required',
            basis=['Pinned author primary code and author-linked dataset; original source unchanged',
                   'Paper train/validation/test split, Alpha360, raw-trend label, 20-day task interval and adapter settings',
                   'Complete offline early-stopping fit and online prediction/backtest; original method, no substitute baseline',
                   'Original prediction metrics independently agree with native metrics',
                   'Official README configuration used; no selection or tuning on test outcomes'],
            numerical_replication='NOT ESTABLISHED: this is one seed; paper reports 10-run averages. No tolerance or equivalence test was pre-registered.',
            limitations=['Official README/default validates from epoch 10 and overrides online data-adapter LR to 0.0001; run.sh validates from 30 and does not override online rates. This run is the README variant, not identical to run.sh.',
                         'Test IC 0.0650639 vs 0.0687; rank IC 0.0617836 vs 0.0621; net annual excess return 0.0802475 vs 0.1296. Do not call this exact table reproduction.',
                         'Paper says 10 runs, released run_all loops 5, requested experiment uses one.',
                         'Qlib git-diff code-capture warning comes from non-git output cwd; pinned source commit and runner hash are independently verified.',
                         'Portfolio gap is unresolved and must remain visible; prediction correctness is not evidence of identical trading performance.'],
            publication_claim='Implementation verified under released README configuration; published performance not fully reproduced.')
        (out/'gate.json').write_text(json.dumps(gate,indent=2))
    else:
        f=pd.read_pickle(ROOT/'data/TRA/kdd21/feature.pkl').sort_index(level=['instrument','datetime'])
        idx=f.index.to_frame(index=False)
        ordinal=idx.groupby('instrument').cumcount()
        instruments=idx.instrument.drop_duplicates().tolist()
        last=idx.groupby('instrument').datetime.max()
        previous={instruments[i]:last[instruments[i-1]] for i in range(1,len(instruments))}
        crosses=ordinal.lt(59)&idx.instrument.ne(instruments[0])
        future=crosses&idx.datetime.lt(idx.instrument.map(previous))
        counts={}
        for split,(start,end) in dict(train=('2007-10-30','2016-05-27'),valid=('2016-09-26','2018-05-29'),test=('2018-09-21','2020-06-30')).items():
            mask=idx.datetime.between(start,end)
            counts[split]=dict(rows=int(mask.sum()),cross_instrument_windows=int((mask&crosses).sum()),future_dated_windows=int((mask&future).sum()))
        review=dict(**common,status='held_for_sequence_boundary_correction',data_window_audit=counts,
            implementation='Native legacy author code and original released panel executed successfully; prediction metrics independently verified.',
            limitation='The native sequence slice start can cross the preceding instrument boundary, including later-dated rows. Counts concern raw feature-panel windows, not a direct estimate of performance bias.',
            next_action='Correct stock/calendar boundaries in the CZ adapter and verify chronological loss-memory access before CZ training. Preserve these native results with the limitation.',
            full_cz_authorized_by_gate=False)
        (out/'fidelity-review.json').write_text(json.dumps(review,indent=2))
        print(json.dumps(counts,indent=2))
qpath=ROOT/'queue.json';q=json.loads(qpath.read_text())
for row in q['papers']:
    if row['paper']=='DoubleAdapt':
        row.update(original_gate='held_configuration_and_performance_gap',stage='original_comparison_unresolved',fidelity_review='results/doubleadapt-original/gate.json')
    if row['paper']=='TRA':row.update(original_gate='held_sequence_boundary_issue',fidelity_review='results/tra-original/fidelity-review.json')
q['updated_utc']=now
tmp=qpath.with_suffix('.json.review.tmp');tmp.write_text(json.dumps(q,indent=2,ensure_ascii=False));tmp.replace(qpath)
print('DoubleAdapt implementation checks passed, but paper-reproduction gate remains unresolved. TRA boundary issue held.')
