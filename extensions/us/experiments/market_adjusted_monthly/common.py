from pathlib import Path
import json
from datetime import datetime, timezone
from usdata import ROOT, sha

HERE = Path(__file__).resolve().parent
FACTORS = ROOT.parent / "factors-daily"
TASKS = ("return", "volume", "volatility")
INDEX = {task: i for i, task in enumerate(TASKS)}
REGIMES = ("raw_monthly", "market_adjusted")


def now():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def input_hashes():
    paths = [HERE / "protocol.json"] + sorted(HERE.glob("*.py"))
    paths += [ROOT / name for name in ["protocol.json", "usdata.py", "limt/model.py", "upstream/limt.py",
        "limt/portfolio.py", "limt/neutral.py", "data/manifest.json",
        "experiments/separate_three_task_portfolio/portfolio_engine.py"]]
    paths += [FACTORS / name for name in ["ff5_daily.csv", "verification.json", "factor-description.txt"]]
    return {str(path): sha(path) for path in paths}


def jobs():
    for seed in [0, 1, 2]:
        yield "raw_monthly", "volume", seed
        for regime in ["market_adjusted", "raw_monthly"]:
            for model in ["joint", "return", "volatility"]:
                yield regime, model, seed


def folder(regime, model, seed):
    if model == "volume":
        regime = "raw_monthly"
    return HERE / "results" / regime / model / f"seed{seed}"
