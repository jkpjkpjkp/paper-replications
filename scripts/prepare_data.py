from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from arboreal.data import DATASETS, prepare

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--root", default="data")
    p.add_argument("--split-seed", type=int, default=0)
    args = p.parse_args()
    # Bounded independent file downloads, not model/agent parallelism.
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lambda name: prepare(name, args.root, args.split_seed), args.datasets):
            print(result, flush=True)
