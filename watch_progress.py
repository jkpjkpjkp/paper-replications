#!/usr/bin/env python3
"""Refresh live evidence while the existing coordinator runs; never train."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import time

from status import inspect, save, write_markdown

ROOT = Path(__file__).resolve().parent


def main(interval):
    with (ROOT / '.progress.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT / 'reproduction/progress.pid').write_text(str(os.getpid()) + '\n')
        sys.path.insert(0, str(ROOT / 'reproduction'))
        spec = importlib.util.spec_from_file_location('live_song_report', ROOT / 'reproduction/report.py')
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        while True:
            state = inspect(ROOT)
            save(ROOT / 'STATUS.json', state)
            write_markdown(ROOT, state)
            report.write_report(ROOT)
            print(state['checked_utc'],
                  {name: (p['completed_fits'], p['state'].get('phase'), p['state'].get('epoch'))
                   for name, p in state['verified_fit_progress'].items()}, flush=True)
            if not state['processes']['us_pipeline']['live']:
                break
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if not 5 <= args.interval <= 60:
        parser.error('interval must be between 5 and 60 seconds')
    main(args.interval)
