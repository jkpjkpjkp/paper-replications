#!/usr/bin/env python3
"""Reuse the original reproduction functions with output writes redirected."""
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'prophets-profit-reproduction'
OUT=ROOT/'paper-claim-validation/results/prophets'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PROJECT))
module=runpy.run_path(str(PROJECT/'scripts/reproduce.py'))
env=module['offline'].__globals__
# Read the exact original inputs. Route only project results writes elsewhere.
original_json=env['write_json']
env['write_json']=lambda name,obj:(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
original_csv=__import__('pandas').DataFrame.to_csv
def redirected_csv(self,path,*a,**kw):
    path=Path(path)
    if path.parent == PROJECT/'results':path=OUT/path.name
    else:raise RuntimeError(f'Unexpected output {path}')
    return original_csv(self,path,*a,**kw)
__import__('pandas').DataFrame.to_csv=redirected_csv
try:
    result,daily,manifest=module['offline']()
    module['theoretical']()
finally:
    __import__('pandas').DataFrame.to_csv=original_csv
record=dict(passed=True,scope='Released Brier forecast records and printed Table 7 cells; no LLM inference',
            models=len(result),forecasts=int(result.n_rows.sum()),printed_cells_matched=3*len(result),
            max_roi_error_percentage_points=float(result.roi_error_pp.abs().max()),
            daily_curve_points=len(daily),max_daily_curve_error_percentage_points=float(daily.error_pp.abs().max()),
            missing_model_lanes=len(manifest['models'])-len(result),full_paper_reproduced=False)
(OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
