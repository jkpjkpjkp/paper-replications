#!/usr/bin/env python3
"""Re-score existing mined artifacts without making generation/API calls."""
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'alphaprobe-reproduction'
OUT=ROOT/'paper-claim-validation/results'
sys.path.insert(0,str(PROJECT/'scripts'))
import common  # Sets the pinned upstream import paths.
module=runpy.run_path(str(PROJECT/'scripts/verify_outputs.py'))
records={}
module['verify_run'].__globals__['write_json']=lambda path,value:None
for universe in ['csi300','csi500','csi1000']:
    records[universe]=module['verify_run'](PROJECT/f'results/small-{universe}-seed0')
record=dict(passed=True,checks=records,api_calls_made=0,paper_results_reproduced=False,
            reason='One-seed substituted generator/0.6B encoder; validates existing artifacts, not full paper protocol')
(OUT/'alphaprobe_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
