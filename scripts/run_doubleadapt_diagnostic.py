"""Require replay agreement before attributing changes to online adapter learning rate."""
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/doubleadapt-online-lr-diagnostic';OUT.mkdir(exist_ok=True)
def status(stage,**extra):
    p=OUT/'pipeline-status.json';tmp=p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**extra),indent=2));tmp.replace(p)
python=str(ROOT/'.venv-doubleadapt/bin/python');script=str(ROOT/'scripts/run_doubleadapt_replay.py')
keys=['IC','ICIR','Rank IC','Rank ICIR','1day.excess_return_with_cost.annualized_return','1day.excess_return_with_cost.information_ratio']
try:
    status('control_replay_running')
    subprocess.run([python,'-u',script,'--control'],check=True)
    original=json.loads((ROOT/'results/doubleadapt-original/summary.json').read_text())['metrics']
    control=json.loads((ROOT/'results/doubleadapt-replay-control/summary.json').read_text())['metrics']
    delta={k:control[k]-original[k] for k in keys}
    if any(abs(v)>1e-7 for v in delta.values()):
        status('control_mismatch_no_causal_attribution',differences=delta)
        raise RuntimeError('Control replay does not match; investigate before treatment')
    status('online_lr_treatment_running',control_differences=delta)
    subprocess.run([python,'-u',script],check=True)
    treatment=json.loads((OUT/'summary.json').read_text())['metrics']
    paired={k:dict(original=original[k],control=control[k],online_lr_001=treatment[k],treatment_minus_control=treatment[k]-control[k]) for k in keys}
    (OUT/'paired-comparison.json').write_text(json.dumps(dict(comparison=paired,
        changed='Online data-adapter LR 0.0001 -> 0.01; fixed saved offline framework and optimizer states.',
        unchanged='Training result, data, CPU runtime, online task sequence and backtest.',
        interpretation='Diagnostic contrast, not a replacement selected by held-out performance. Does not test validation-start epoch or seed dispersion.'),indent=2))
    status('paired_diagnostic_complete')
except Exception as exc:
    (OUT/'pipeline-error.json').write_text(json.dumps(dict(error=repr(exc)),indent=2))
    raise
