"""Read-only progress report; does not start, stop or approve experiments."""
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
queue=json.loads((root/'queue.json').read_text())
for paper in queue['papers']:
    if paper['priority'] is None:continue
    print(f"{paper['priority']}. {paper['paper']}: {paper['stage']}; original gate={paper['original_gate']}")
    for field, label in [('active_data_recovery', 'Current data recovery'),
                         ('active_reproduction', 'Current reproduction'),
                         ('active_pipeline', 'Current continuation')]:
        if paper.get(field):
            current = root/paper[field]
            if current.is_file():print('   '+label+': '+json.dumps(json.loads(current.read_text())))
    path=root/paper.get('status_file','nonexistent')
    if path.is_file():print('   '+json.dumps(json.loads(path.read_text())))
    if paper.get('cz_status_file'):
        cz=root/paper['cz_status_file']
        print('   CZ: '+paper.get('full_cz_stage','unknown'))
        if cz.exists():print('   '+json.dumps(json.loads(cz.read_text())))
    if paper.get('corrected_status_file'):
        corrected=root/paper['corrected_status_file']
        if corrected.exists():print('   Corrected run: '+json.dumps(json.loads(corrected.read_text())))
    if paper.get('diagnostic_status_file'):
        diagnostic=root/paper['diagnostic_status_file']
        if diagnostic.exists():print('   Diagnostic: '+json.dumps(json.loads(diagnostic.read_text())))
    if paper.get('matched_status_file'):
        matched=root/paper['matched_status_file']
        if matched.exists():print('   Script-matched rerun: '+json.dumps(json.loads(matched.read_text())))
for service in ['ordered-factor-alphasage-data-20261115.service','ordered-factor-alphasage-recovered-20261115.service','tra-paper-runtime-cpu-20261115.service','ordered-factor-doubleadapt.service','ordered-factor-tra.service','ordered-factor-queue.service','ordered-factor-doubleadapt-cz.service','ordered-factor-tra-corrected.service','ordered-factor-doubleadapt-diagnostic.service','ordered-factor-doubleadapt-matched.service']:
    state=subprocess.run(['systemctl','--user','is-active',service],capture_output=True,text=True)
    print(service+': '+state.stdout.strip())
print('Historical CZ run exists (see per-paper fidelity status): '+str(queue.get('cz_plan',{}).get('full_training_started',False)))
print('Current continuation: original reproduction before any new CZ training')
print('Scheduled Codex checks cancelled: '+str((root/'monitor/cancelled.json').exists()))
