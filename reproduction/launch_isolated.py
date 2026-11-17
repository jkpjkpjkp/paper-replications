"""Freeze CUDA_VISIBLE_DEVICES before author's device setup changes the env.
One process per physical GPU; author run.py, RNG seed and model are unchanged.
"""
import runpy
import sys
import torch

torch.cuda.init()
torch.set_num_threads(8)
sys.argv = ['run.py'] + sys.argv[1:]
runpy.run_path('run.py', run_name='__main__')
