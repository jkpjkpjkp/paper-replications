from pathlib import Path
import json
from datetime import datetime, timezone
from usdata import ROOT, sha

HERE = Path(__file__).resolve().parent
OLD = ROOT / "experiments/joint_price_volatility"
TASK_INDEX = {"return": 0, "volume": 1, "volatility": 2}


def now():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def input_hashes():
    paths = [HERE / "protocol.json"] + sorted(HERE.glob("*.py"))
    paths += [ROOT / name for name in ["protocol.json", "usdata.py", "portfolio.py", "limt/model.py",
              "limt/portfolio.py", "limt/neutral.py", "upstream/limt.py", "data/manifest.json"]]
    paths += [OLD / name for name in ["common.py", "model.py", "train.py", "protocol.json"]]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def reused_folders():
    for seed in [0, 1, 2]:
        yield "joint", seed, ROOT / "results" / f"seed{seed}"
        for task in ["return", "volatility"]:
            yield task, seed, OLD / "results" / f"{task}_only" / f"seed{seed}"


def source_hashes():
    paths = []
    for _, _, folder in reused_folders():
        paths += [folder / name for name in ["status.json", "provenance.json", "best.pt", "test_predictions.parquet"]]
    paths += [ROOT / "analysis" / f"LiMT_ensemble_apo_buy{cost}_ledger.csv" for cost in [0, 5]]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}
