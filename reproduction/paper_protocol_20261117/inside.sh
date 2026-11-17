#!/usr/bin/env bash
set -euo pipefail
sit_run=/workspace/reproduction/paper_protocol_20261117
cd "$sit_run"
if [[ ! -f deps_ready.json ]]; then
  python -m pip install --no-cache-dir --target "$sit_run/deps" numpy==1.26.4 pandas==2.2.2 scipy==1.14.1 scikit-learn==1.5.2 matplotlib==3.10.0 einops==0.8.0 tqdm==4.67.1 joblib==1.4.2 sympy==1.13.1 > dependencies.log 2>&1
  python -c 'import json, numpy, pandas, scipy, sklearn, matplotlib, torch; from pathlib import Path; assert numpy.__version__ == "1.26.4"; assert torch.__version__ == "2.6.0a0+ecf3bae40a.nv25.1"; Path("deps_ready.json").write_text(json.dumps(dict(numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__, torch=torch.__version__), indent=2))'
fi
python -m pip freeze > container_environment_freeze.txt
python preflight.py > preflight.log 2>&1
exec python -u controller.py
