from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from limt.model import LiMT
from limt.data import Panel
from scripts.build_data import write_json, sha

ROOT = Path(__file__).resolve().parents[1]


def corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x)<3 or x.std()==0 or y.std()==0:
        return np.nan
    return float(np.corrcoef(x,y)[0,1])


def masked_loss(pred, target, weights):
    valid = torch.isfinite(target)
    diff = (pred - torch.nan_to_num(target)).square()
    return ((diff * valid).sum(dim=0)/valid.sum(dim=0).clamp_min(1)*weights).sum()


@torch.no_grad()
def evaluate(model, panel, days, device, save=False):
    model.eval()
    scores, rows = [], []
    for t in days:
        ids, x, _, y = panel.day(t)
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
        scores.append([corr(pred[:,k],y[:,k]) for k in range(3)])
        if save:
            rows.append(pd.DataFrame(dict(date=panel.dates[t], realized_date=panel.dates[t+2],
                symbol=panel.symbols[ids], pred_return=pred[:,0], pred_volume=pred[:,1],
                pred_volatility=pred[:,2], label_return=y[:,0], label_volume=y[:,1], label_volatility=y[:,2])))
    scores = np.array(scores)
    metrics = {}
    for k, name in enumerate(["return","volume","volatility"]):
        values = scores[:,k]
        metrics[name] = dict(IC=float(np.nanmean(values)),
            ICIR=float(np.nanmean(values)/np.nanstd(values,ddof=1)), days=int(np.isfinite(values).sum()))
    return metrics, pd.concat(rows, ignore_index=True) if save else None


def run(args):
    out = ROOT/"results"/args.run_name/args.universe/f"seed{args.seed}"
    out.mkdir(parents=True,exist_ok=True)
    config = vars(args).copy()
    existing = out/"config.json"
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise ValueError("Refusing to overwrite a run with a different configuration")
    write_json(existing,config)
    panel = Panel(ROOT/"data"/args.universe)
    days = {s:panel.split(s) for s in ["train","valid","test"]}
    if args.smoke:
        days = {s:d[:args.smoke] for s,d in days.items()}
    summary = {s:dict(days=len(d),first=str(panel.dates[d[0]]),last=str(panel.dates[d[-1]]),
        last_realization=str(panel.dates[d[-1]+2]),
        observations=int(panel.eligible[:,d].sum())) for s,d in days.items()}
    write_json(out/"data_summary.json",summary)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    if device.type=="cuda":
        torch.cuda.manual_seed_all(args.seed)
    model=LiMT(dropout=args.dropout, experts=args.experts,variant=args.variant).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    weights=torch.tensor([1.,0.,0.] if args.variant=="no_aux" else args.weights,device=device)
    best, stale, start, history, best_epoch = -np.inf,0,0,[],0
    latest=out/"latest.pt"
    if latest.exists():
        ck=torch.load(latest,map_location=device,weights_only=False)
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        best,stale,start,history,best_epoch=[ck[k] for k in ["best","stale","epoch","history","best_epoch"]]
        torch.set_rng_state(ck["rng"].cpu())
        if device.type=="cuda":torch.cuda.set_rng_state(ck["cuda_rng"].cpu(),device)
    write_json(out/"status.json",dict(phase="training",start_epoch=start,config=config,
        updated_utc=datetime.now(timezone.utc).isoformat()))
    for epoch in range(start,args.epochs):
        if stale>=args.patience:break
        clock=time.perf_counter();model.train();losses=[]
        order=np.random.default_rng(args.seed+epoch).permutation(days["train"])
        for t in order:
            _,x,y,_=panel.day(t)
            optimizer.zero_grad(set_to_none=True)
            pred=model(torch.from_numpy(x).to(device))
            loss=masked_loss(pred,torch.from_numpy(y).to(device),weights)
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite training loss")
            loss.backward();torch.nn.utils.clip_grad_value_(model.parameters(),3.)
            optimizer.step();losses.append(loss.item())
        metrics,_=evaluate(model,panel,days["valid"],device)
        valid=metrics["return"]["IC"]
        if valid>best:
            best,stale,best_epoch=valid,0,epoch+1
            torch.save(model.state_dict(),out/"best.pt")
        else:stale+=1
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation=metrics,
            seconds=time.perf_counter()-clock,best_epoch=best_epoch,stale=stale)
        history.append(record)
        write_json(out/"history.json",history)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,
            stale=stale,epoch=epoch+1,history=history,best_epoch=best_epoch,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type=="cuda" else None),latest)
        write_json(out/"status.json",dict(phase="training",**record))
        print(json.dumps(dict(universe=args.universe,seed=args.seed,**record)),flush=True)
    model.load_state_dict(torch.load(out/"best.pt",map_location=device,weights_only=True))
    results={}
    for split in ["valid","test"]:
        metrics,pred=evaluate(model,panel,days[split],device,save=True)
        # RankIC is evaluated only after checkpoint selection.
        rankic=pred.groupby("date",sort=True).apply(lambda f:f.pred_return.corr(f.label_return,method="spearman"),include_groups=False)
        metrics["return"]["RankIC"]=float(rankic.mean())
        metrics["return"]["RankICIR"]=float(rankic.mean()/rankic.std(ddof=1))
        results[split]=metrics
        pred.to_parquet(out/f"{split}_predictions.parquet",index=False)
    write_json(out/"metrics.json",results)
    write_json(out/"provenance.json",dict(data_manifest_sha256=sha(panel.folder/"manifest.json"),
        model_code_sha256=sha(ROOT/"limt/model.py"),trainer_sha256=sha(__file__),
        author_commit="c0232623d50db26b02275b6f77b682f25effeec5",torch=torch.__version__,
        best_sha256=sha(out/"best.pt"),parameters=sum(p.numel() for p in model.parameters()),
        device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None))
    write_json(out/"status.json",dict(phase="complete",epochs=len(history),best_epoch=best_epoch,
        stop_reason="early_stopping" if stale>=args.patience else "epoch_limit",best_validation_ic=best,
        test=results["test"],updated_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(universe=args.universe,seed=args.seed,phase="complete",test=results["test"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--universe",choices=["csi300","csi500"],default="csi300")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--run-name",default="main")
    p.add_argument("--device",default="cuda")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--patience",type=int,default=21)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--weights",nargs=3,type=float,default=[.65,.30,.05])
    p.add_argument("--experts",type=int,default=8)
    p.add_argument("--dropout",type=float,default=.5)
    p.add_argument("--variant",choices=["full","no_aux","no_ldl","no_cross"],default="full")
    p.add_argument("--tf32",action="store_true")
    p.add_argument("--smoke",type=int,default=0,help="Days per split; 0 uses all dates")
    run(p.parse_args())
