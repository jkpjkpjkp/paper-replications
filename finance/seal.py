from __future__ import annotations
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels
import torch
import xgboost
from .data import ROOT, digest

verification = json.loads((ROOT / "results/verification.json").read_text())
assert verification["status"] == "passed" and verification["trained_models"] == 15
environment = {"python": platform.python_version(), "platform": platform.platform(),
               "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
               "statsmodels": statsmodels.__version__, "xgboost": xgboost.__version__,
               "model_fits": 15, "test_months": 252, "stock_month_forecasts": 417706,
               "one_way_cost_bps": 10, "annual_short_borrow": .01,
               "long_exposure": 1, "short_exposure": -1,
               "tests_passed": 25}
(ROOT / "results/environment.json").write_text(json.dumps(environment, indent=2) + "\n")
project = ROOT.parent
files = [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
         and p.name != "SHA256SUMS.json" and not p.name.endswith(".part")]
files += [project / "arboreal/model.py", project / "tests/test_finance.py"]
manifest = {str(p.relative_to(project)): digest(p) for p in sorted(files)}
(ROOT / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")
for name, expected in manifest.items():
    assert digest(project / name) == expected, name
print(f"Sealed and verified {len(manifest)} finance/source artifacts")
