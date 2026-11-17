"""Explicit precision sensitivity wrapper; original source remains unchanged."""
import runpy
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
torch.backends.cuda.matmul.allow_tf32 = True
print('DIAGNOSTIC ONLY: matmul.allow_tf32 =', torch.backends.cuda.matmul.allow_tf32, flush=True)
sys.path.insert(0, str(root))
sys.argv = [str(root/'run.py')] + sys.argv[1:]
runpy.run_path(str(root/'run.py'), run_name='__main__')
