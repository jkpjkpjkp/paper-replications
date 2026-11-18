"""Keep the known baseline assessment current as the authorized fits finish."""
import json
from pathlib import Path
from datetime import datetime,timezone

PROJECT=Path('/home/jkp/data/Lookahead')
START='\n--- AUTOMATIC EMPIRICAL PROGRESS ---\n'
END='--- END AUTOMATIC EMPIRICAL PROGRESS ---\n'


def refresh(out):
    report=json.loads((out/'comparison.json').read_text())
    completed=list(out.glob('*/seed-*/complete.json'))
    count=len(completed)
    lines=[f'Updated UTC: {datetime.now(timezone.utc).isoformat()}',
           f'Completed neural fits: {count}/40; run_status.json is authoritative for active/failed/completed state.']
    for kind in ['mlp','uq_lstm','lstm','uq_mlp']:
        if kind in report['models']:
            r=report['models'][kind]
            lines.append(f"{kind}: {r['ensemble_members_completed']}/10 members; test EBIT MSE {r['standardized_ebit_mse']:.9f}; paper {r['paper_mse']:.2f}.")
    lines+=['Grade 1 remains scoped to the documented substituted-data baseline mismatch.',
            'This progress update does not upgrade fidelity or claim a paper match.']
    for path in [PROJECT/'STATUS.txt',PROJECT/'reproduction/STATUS.txt']:
        text=path.read_text()
        if START in text:text=text.split(START)[0]
        if count==40:
            text=text.replace('Neural work is running: four full-size model variants, ten seeds each, maximum',
                              'Neural work completed: four full-size model variants, ten seeds each, maximum')
        path.write_text(text.rstrip()+START+'\n'.join(lines)+'\n'+END)
    if count==40:
        path=PROJECT/'BLOCKER.txt';text=path.read_text()
        text=text.replace('The neural ensembles are running; see STATUS.txt for scope and completed evidence.',
                          'The neural ensembles have completed; see STATUS.txt for scope and completed evidence.')
        path.write_text(text)
        path=PROJECT.parent/'REPLICATION_STATUS.txt';text=path.read_text()
        text=text.replace('0.58/0.52. The neural ensembles are running. Gross monthly portfolio diagnostics',
                          '0.58/0.52. All 40 neural fits completed; see the project STATUS files for their metrics.\nGross monthly portfolio diagnostics')
        text=text.replace('QFM EBIT MSE 1.094017 vs paper 0.65; neural ensembles running.',
                          'QFM EBIT MSE 1.094017 vs paper 0.65; 40 neural fits completed.')
        path.write_text(text)
