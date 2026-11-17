#!/usr/bin/env python3
"""Appendix F.5/Figure 13 sensitivity with identical paths across windows."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.reproduce import run_sde


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--windows",nargs="+",type=int,default=[50,60,70,75,80])
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    for window in args.windows:
        if not 1<=window<100:raise ValueError("window must be between 1 and 99")
        for seed in args.seeds:
            cfg=json.loads((ROOT/"configs/sde.json").read_text())
            cfg["train_steps"]=window
            if window!=75:cfg["label"]=f"heston_window{window}"
            run_sde("heston",cfg,seed,torch.device(args.device))
    for seed in args.seeds:
        cfg=json.loads((ROOT/"configs/sde.json").read_text())
        cfg["train_steps"]=cfg["steps"]
        cfg["label"]="heston_full_horizon"
        run_sde("heston",cfg,seed,torch.device(args.device))


if __name__=="__main__":main()
