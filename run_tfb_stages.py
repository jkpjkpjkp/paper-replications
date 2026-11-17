#!/usr/bin/env python3
"""Foreground-only sequential interpreters inside the shared guarded cgroup."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
commands=[
    [sys.executable,'-u',str(ROOT/'run_rise_tfb.py'),'--device','cuda'],
    [sys.executable,'-u',str(ROOT/'run_classical_tfb.py')],
]
for command in commands:
    code=subprocess.run(command,cwd=ROOT).returncode
    if code:sys.exit(code)
