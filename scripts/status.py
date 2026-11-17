"""Read local run status without importing the training environment."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
status = root/'results/status.json'
if status.exists():
    value = json.loads(status.read_text())
    print(json.dumps(value, indent=2))
    if value.get('state') == 'running':
        current = root/'results'/value.get('runtime_dir','paper-runtime')/str(value['seed'])/value['stage']/'status.json'
        if current.exists():
            print(current.read_text())
for runtime_dir in ['paper-runtime', 'paper-runtime-cpu']:
    print(runtime_dir, 'completed main stages:',
          sum((root/'results'/runtime_dir/str(s)/'tra/complete.json').exists()
              for s in [1,1000,2000,3000,4000,5000]), '/ 6')
