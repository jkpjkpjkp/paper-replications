"""Verify pinned runtime, source/data identity and allocator arithmetic."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from controller import call_worker, write_json


def main():
    import torch
    from worker import environment, verify_source
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert torch.__version__==protocol['required_torch'],torch.__version__
    assert hashlib.sha256((ROOT/'asset_data/full_dataset.csv').read_bytes()).hexdigest()==protocol['dataset_sha256']
    verify_source()
    cache_manifest=json.loads((HERE/'cache_manifest.json').read_text())
    for item in cache_manifest['files']:
        path=ROOT/item['path']
        assert path.stat().st_size==item['bytes']
        hasher=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):
                hasher.update(block)
        assert hasher.hexdigest()==item['sha256'],item['path']
    for pool in protocol['asset_pools']:
        check=json.loads((ROOT/f'reproduction/cache_equivalence_{pool}.json').read_text())
        assert check['csv_sha256']==protocol['dataset_sha256']
        for split in ['train','val','test']:
            for stage in ['raw','scaled']:
                assert all(error==0 for error in check['split_checks'][split][stage].values())
    if not (HERE/'managed_allocator.so').exists():
        subprocess.run(['g++','-O2','-shared','-fPIC',str(HERE/'managed_allocator.cpp'),
            '-I/usr/local/cuda/include','-L/usr/local/cuda/lib64',
            '-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-o',str(HERE/'managed_allocator.so')],check=True)
    for allocator in ['native','managed']:
        output=HERE/'preflight'/allocator
        assert call_worker('probe',40,0,2023,output,allocator)==0,output/'worker.log'
    a=torch.load(HERE/'preflight/native/probe.pth',map_location='cpu',weights_only=True)
    b=torch.load(HERE/'preflight/managed/probe.pth',map_location='cpu',weights_only=True)
    assert a.keys()==b.keys()
    errors={key:float((a[key]-b[key]).abs().max()) for key in a}
    assert all(value==0 for value in errors.values()),errors
    write_json(HERE/'preflight.json',dict(passed=True,environment=environment(),
       allocator_check=dict(tensors=len(a),bitwise_equal=True,max_error=max(errors.values())),
       scope='Three small-batch optimizer steps; source/input hashes and previous cache equivalence. Not proof for every tensor allocation or original hardware equivalence.'))


if __name__=='__main__':
    main()
