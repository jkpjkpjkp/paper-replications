import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transalpha.data import ROOT, build_universe, sha, split_indices
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universes", default="csi300,csi500,csi1000")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    for u in args.universes.split(","):
        if not (ROOT / "data" / u / "manifest.json").exists() or args.force:
            build_universe(u, config, args.workers)
        else:
            folder = ROOT / "data" / u
            manifest = json.loads((folder / "manifest.json").read_text())
            if not manifest["complete"] or manifest["source_revision"] != config["source_revision"]:
                raise ValueError("Incomplete or mismatched existing panel: " + u)
            if split_indices(np.load(folder / "dates.npy"), config) != manifest["indices"]:
                raise ValueError("Changed split configuration requires --force: " + u)
            for name, expected in manifest["array_hashes"].items():
                if sha(folder / (name + ".npy")) != expected:
                    raise ValueError("Existing derived array hash mismatch: " + u + "/" + name)
            print(json.dumps({"universe": u, "verified_existing_panel": True}), flush=True)
