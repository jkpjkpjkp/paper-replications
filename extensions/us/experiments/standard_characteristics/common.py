from pathlib import Path
from datetime import datetime, timezone
import json

from usdata import ROOT, sha, write_json

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CZ = ROOT.parent / "Chen_Zimmermann_Factor_Zoo_2026_10"
FACTORS = ROOT.parent / "factors-daily/ff5_momentum_daily.parquet"


def config():
    return json.loads((HERE / "protocol.json").read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def input_hashes():
    paths = sorted(HERE.glob("*.py")) + [HERE / "protocol.json", ROOT / "train.py",
        ROOT / "usdata.py", ROOT / "limt/model.py", ROOT / "limt/neutral.py", ROOT / "upstream/limt.py",
        ROOT / "data/manifest.json", FACTORS, FACTORS.parent / "verification.json",
        CZ / "SignalDoc.csv", CZ / "Firm Level Characteristics/download_receipt.json"]
    return {str(path): sha(path) for path in paths}
