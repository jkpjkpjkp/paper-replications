"""Train the explicitly repaired PDU reconstruction on a prepared real panel.

Checkpoint selection uses validation MSE only. Test data are evaluated once,
after selection. The default 300-epoch cap comes from the released main.py;
early stopping (20) and validation selection are declared reconstruction choices.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from model import PDUModel, objective
from metrics import evaluate

ROOT=Path(__file__).resolve().parent


def target_days(start, end, window=4):
    return np.arange(max(start, window), end)


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    panel=ROOT/'data'/f'{args.market}.npz'
    meta=json.loads(panel.with_suffix('.json').read_text())
    data=np.load(panel, allow_pickle=False)
    n,t,_=data['features'].shape
    if args.stocks:
        n=min(args.stocks,n)
    edge=data['edges'];edge=edge[:,(edge<n).all(axis=0)]
    tensors={name:torch.as_tensor(data[name][:n].astype(np.float32),device=args.device)
             for name in ['features','price','truth','mask']}
    graph=torch.as_tensor(edge,device=args.device)
    model=PDUModel(n,device=args.device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.001)
    out=ROOT/'results'/args.tag;out.mkdir(parents=True,exist_ok=False)
    config=vars(args)|{'data_sha256':hashlib.sha256(panel.read_bytes()).hexdigest(),
        'data_status':meta['status'],'stocks':n,'days':t,'train_targets':[4,755],
        'validation_targets':[756,1007],'test_targets':[1008,t-1],
        'selection':'minimum validation masked MSE','precision':'float32, TF32 disabled',
        'status':'reconstructed_protocol; not exact paper reproduction'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    def batch(day):
        return (tensors['features'][:,day-4:day],tensors['price'][:,day-1],
                tensors['truth'][:,day],tensors['mask'][:,day-4:day+1].amin(1))
    @torch.no_grad()
    def predict(days):
        model.eval();preds=[];ys=[];masks=[]
        for day in days:
            x,base,y,mask=batch(day)
            pred,_=model(x,graph,contrast=False)
            ratio=(pred[:,0]-base)/base.clamp_min(1e-8)
            preds.append(ratio.cpu().numpy());ys.append(y.cpu().numpy());masks.append(mask.cpu().numpy())
        return tuple(np.stack(v,axis=1) for v in [preds,ys,masks])
    started=time.time();best=float('inf');wait=0;history=[]
    for epoch in range(args.epochs):
        model.train();days=target_days(0,756);np.random.shuffle(days)
        if args.train_steps:
            days=days[:args.train_steps]
        loss_sum=0
        for step,day in enumerate(days):
            x,base,y,mask=batch(day)
            optimizer.zero_grad(set_to_none=True)
            pred,contrast=model(x,graph)
            loss,_=objective(pred,base,y,mask,contrast)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at {epoch=} {day=}')
            loss.backward();optimizer.step();model.update_moving_average();loss_sum+=loss.item()
            if step%50==0:
                print(json.dumps({'epoch':epoch,'step':step,'loss':loss.item(),'elapsed':time.time()-started}),flush=True)
        prediction,truth,mask=predict(target_days(756,1008))
        valid_mse=float(np.sum((prediction-truth)**2*mask)/mask.sum())
        row={'epoch':epoch,'train_loss':loss_sum/len(days),'validation_mse':valid_mse,'elapsed':time.time()-started}
        history.append(row);print(json.dumps(row),flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if valid_mse<best:
            best=valid_mse;wait=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation_mse':best,'config':config},out/'best.pt')
        else:
            wait+=1
        if wait>=args.patience:
            break
    checkpoint=torch.load(out/'best.pt',map_location=args.device,weights_only=False)
    model.load_state_dict(checkpoint['model'])
    prediction,truth,mask=predict(target_days(1008,t))
    np.savez_compressed(out/'predictions.npz',prediction=prediction,truth=truth,mask=mask,
                        dates=data['dates'][1008:],tickers=data['tickers'][:n])
    result=evaluate(prediction,truth,mask)
    daily=result.pop('daily_returns')
    np.savetxt(out/'daily_returns.csv',np.column_stack([data['dates'][1008:],daily]),
               delimiter=',',fmt='%s',header='date,top5_close_to_close_return',comments='')
    result|={'selected_epoch':checkpoint['epoch'],'validation_mse':checkpoint['validation_mse'],
             'elapsed_seconds':time.time()-started,'max_gpu_memory_bytes':torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
             'status':'diagnostic_only' if args.epochs==1 or args.train_steps or args.stocks else 'reconstructed_protocol',
             'matches_paper':False}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--market',choices=['NASDAQ','NYSE'],required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--stocks',type=int,default=0,help='Diagnostic subset only; first N source tickers')
    p.add_argument('--train-steps',type=int,default=0,help='Diagnostic cap; 0 uses all 752 days')
    main(p.parse_args())
