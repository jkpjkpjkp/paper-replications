#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
commands=[['python3',str(root/'prepare_remaining_models.py')],
 [sys.executable,'-u',str(root/'run_remaining_tfb.py')],
 [sys.executable,'-u',str(root/'verify_and_compare_tfb.py')]]
for command in commands:
 result=subprocess.run(command,cwd=root).returncode
 if result:sys.exit(result)
