"""Torch-free experiment metadata, shared by the two existing environments."""
from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
from usdata import sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASK_INDEX = {"return": 0, "volatility": 2}
VARIANTS = {
    "joint": ("return", "volatility"),
    "return_only": ("return",),
    "volatility_only": ("volatility",),
}


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def input_hashes():
    paths = [ROOT / "protocol.json", ROOT / "data/manifest.json", ROOT / "usdata.py",
             ROOT / "limt/model.py", ROOT / "upstream/limt.py", ROOT / "tests/test_joint_prediction.py"]
    paths += [HERE / "protocol.json"] + sorted(HERE.glob("*.py"))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    a, b = x[valid].astype(float), y[valid].astype(float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        raise ValueError("Undefined daily IC; cannot silently change paired evaluation days")
    return float(np.corrcoef(a, b)[0, 1])
