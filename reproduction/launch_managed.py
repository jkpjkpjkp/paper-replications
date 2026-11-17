"""Run an unchanged Python script using CUDA managed allocations.

The wrapper changes storage allocation only, allowing driver-managed paging
to RAM. This is an execution-environment deviation and must be disclosed.
"""
import ctypes
import json
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
library = ROOT / 'reproduction/managed_allocator.so'
lib = ctypes.CDLL(str(library))
lib.sit_managed_peak_bytes.restype = ctypes.c_size_t
assert lib.sit_managed_supported() == 1, 'GPU does not support concurrent managed memory'
allocator = torch.cuda.memory.CUDAPluggableAllocator(str(library), 'sit_managed_malloc', 'sit_managed_free')
torch.cuda.memory.change_current_allocator(allocator)
# The non-caching allocator cudaFrees each tensor, so there is no cache to flush.
torch.cuda.empty_cache = lambda: None
print('Execution deviation: CUDA managed memory; unchanged FP32 model and batch size.', flush=True)
script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
sys.path.insert(0, str(ROOT))
try:
    runpy.run_path(str(script), run_name='__main__')
finally:
    print(json.dumps({'managed_allocator_peak_mib': lib.sit_managed_peak_bytes() / 2**20}), flush=True)
