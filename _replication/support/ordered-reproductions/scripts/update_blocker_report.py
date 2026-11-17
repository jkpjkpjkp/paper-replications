"""Refresh requested root BLOCKER.txt from reviewed notes and live task metadata."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
q=json.loads((ROOT/'queue.json').read_text())
lines=['PAPER REPRODUCTION BLOCKERS',
       'Live status refreshed: '+datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
       'Reviewed blocker notes: 2026-11-13; subsequent substantive findings require updating ordered-reproductions/blocker-notes.txt.',
       'Scheduled Codex checks are cancelled. Refresh with scripts/update_blocker_report.py; edit the reviewed notes, not just this generated copy.',
       '', 'LIVE FACTOR-MODEL STATUS']
for row in q['papers']:
    if row['priority'] is None:continue
    lines.append(f"{row['priority']}. {row['paper']}: {row['stage']}; original gate={row['original_gate']}; CZ={row['full_cz_stage']}")
    for field,label in [('status_file','Original'),('corrected_status_file','Corrected'),('matched_status_file','Script-matched rerun'),('cz_status_file','CZ'),('diagnostic_status_file','Diagnostic')]:
        if row.get(field) and (ROOT/row[field]).exists():
            d=json.loads((ROOT/row[field]).read_text())
            lines.append('   '+label+': '+json.dumps(d,ensure_ascii=False))
gap=ROOT/'results/gap-resolution-20261113'
if gap.exists():
    lines += ['', 'DATA-CLEANING EXPERIMENTS (separate sensitivity runs; original gates unchanged)']
    for name,label in [('available-inputs','DoubleAdapt unavailable-input exclusion'),('tra-calendar-corrected','TRA calendar-aware histories'),('validation-rates-script-matched','DoubleAdapt paper-described validation rate selection')]:
        base=gap/name
        summary=base/'summary.json'
        status=base/'status.json'
        selection=base/'selection.json'
        progress=base/'validation-progress.json'
        if summary.exists():
            data=json.loads(summary.read_text())
            report={k:data[k] for k in ['status','independently_verified','metrics','observations','trading_days'] if k in data}
        elif status.exists():report=json.loads(status.read_text())
        elif selection.exists():report={'stage':'validation_selection_complete','winner':json.loads(selection.read_text())['winner']}
        elif progress.exists():report={'stage':'validation_selection_running','completed_candidates':len(json.loads(progress.read_text())['results'])}
        else:continue
        lines.append('   '+label+': '+json.dumps(report,ensure_ascii=False))
    lines.append('   Evidence: ordered-reproductions/results/gap-resolution-20261113/REPORT.txt')
paired=ROOT/'results/doubleadapt-online-lr-diagnostic/paired-comparison.json'
if paired.exists():lines+=['','NEW DIAGNOSTIC EVIDENCE (review against the notes below):',paired.read_text()]
lines+=['',(ROOT/'blocker-notes.txt').read_text(),'DEFERRED PAPERS: REGISTRY REQUIREMENTS, NOT REASSESSED IN FACTOR-FIRST PHASE']
for row in q['papers']:
    if row['priority'] is not None:continue
    lines += ['',row['paper']+' — deferred by user',
              '  Data: '+row['data'],'  Work/verification required: '+row['known_requirements'],
              '  Evidence: reference.xlsx '+row['reference_location']]
out=ROOT.parent/'BLOCKER.txt';tmp=out.with_suffix('.txt.tmp')
tmp.write_text('\n'.join(lines)+'\n');tmp.replace(out)
print(str(out))
