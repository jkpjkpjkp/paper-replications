"""One-shot completion receipt for the matched run; no Codex invocation or gate approval."""
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'ordered-factor-doubleadapt-matched.service'
OUT = ROOT/'results/doubleadapt-script-matched'


def save(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def main():
    while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', SERVICE]).returncode == 0:
        time.sleep(20)
    success = (OUT/'summary.json').is_file()
    if success:
        original = json.loads((ROOT/'results/doubleadapt-original/summary.json').read_text())
        matched = json.loads((OUT/'summary.json').read_text())
        metrics = {}
        for key, target in matched['comparison'].items():
            metrics[key] = dict(baseline=original['metrics'][key],
                                matched=matched['metrics'][key], paper=target['paper'],
                                delta=matched['metrics'][key]-original['metrics'][key])
        save(OUT/'comparison-to-baseline.json', dict(
            reviewed_utc=datetime.now(timezone.utc).isoformat(),
            status='completed_pending_fidelity_review', metrics=metrics,
            interpretation='Two prespecified changes together: validation begins at epoch 30 and online lr_da remains 0.01. Same seed and data. This comparison alone does not isolate each change or validate the paper aggregate.',
            full_cz_authorized_by_gate=False))
    with (ROOT/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        qpath = ROOT/'queue.json'
        queue = json.loads(qpath.read_text())
        row = next(r for r in queue['papers'] if r['paper'] == 'DoubleAdapt')
        row['stage'] = 'script_matched_complete_pending_review' if success else 'script_matched_run_failed'
        if success:
            row['matched_summary_file'] = 'results/doubleadapt-script-matched/summary.json'
        queue['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save(qpath, queue)
    subprocess.run([sys.executable, str(ROOT/'scripts/update_blocker_report.py')], check=True)


if __name__ == '__main__':
    main()
