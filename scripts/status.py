"""Print a compact view of the six long-running experiments."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
print('Universe  Seed  Phase     Epoch  Best  Stale  Validation IC')
for u in ['csi300','csi500']:
    for seed in range(3):
        path=ROOT/'results/main'/u/f'seed{seed}'/'status.json'
        if not path.exists():
            print(f'{u:9} {seed:<5} queued')
            continue
        j=json.loads(path.read_text())
        ic=j.get('best_validation_ic',j.get('validation',{}).get('return',{}).get('IC'))
        print(f"{u:9} {seed:<5} {j['phase']:9} {str(j.get('epoch',j.get('epochs','—'))):<6} {str(j.get('best_epoch','—')):<5} {str(j.get('stale','—')):<6} {ic if ic is not None else '—'}")
