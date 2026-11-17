"""Check resumed full-stride SDR evaluation against the upstream evaluator."""
import sys
from pathlib import Path
from native_runtime import ROOT,atomic
sys.path.insert(0,str(ROOT/'upstream/src'))
import numpy as np
import torch
from data import Split
from main import build_argparser,make_problem
from evaluate import EvalConfig,evaluate_split
from resumable_sdr import evaluate_sdr
args=build_argparser().parse_args(['--mode','historic','--dataset','euro50','--cuda','-1'])
problem=make_problem(args,ROOT,torch.device('cpu'))
s=problem.test
short=Split('test',s.X_flat[:3],s.y_norm[:3],s.y_raw[:3],s.covs[:3],s.Ls[:3],s.dates[:3])
mu=problem.historic_mu(short)
cfg=EvalConfig(compute_pfl_loss=False)
expected,trace,last=evaluate_split(problem,short,mu,5,'sdr',cfg)
observed,other,other_last=evaluate_sdr(problem,short,mu,5,'sdr',cfg,{'purpose':'SDR engineering equivalence, 3 real test dates'})
checks=[]
for key in expected:
    checks.append(bool(np.isclose(expected[key],observed[key],rtol=1e-12,atol=1e-12,equal_nan=True)))
for date in trace:
    for key in trace[date]:checks.append(bool(np.isclose(trace[date][key],other[date][key],rtol=1e-12,atol=1e-12,equal_nan=True)))
checks.append(bool(torch.allclose(last['w'],other_last['w'],rtol=1e-12,atol=1e-12)))
atomic(ROOT/'runtime/sdr_equivalence.json',{'passed':all(checks),'checks':len(checks),'market':'euro50','real_dates':short.dates,'interpolation':False})
assert all(checks)
print('Resumable SDR evaluation matches upstream',len(checks),'checks')
