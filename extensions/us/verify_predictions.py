"""Replay first/middle/last held-out dates against each selected checkpoint."""
import json
import numpy as np
import pandas as pd
import torch
from limt.model import LiMT
from usdata import ROOT,Panel,config,sha,write_json


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    panel=Panel();p=config();days=panel.split('test')
    check_days=days[[0,len(days)//2,len(days)-1]]
    records=[]
    for seed in [0,1,2]:
        folder=ROOT/'results'/f'seed{seed}'
        lock=json.loads((folder/'lock.json').read_text())
        for field,path in [('protocol_sha256',ROOT/'protocol.json'),('model_sha256',ROOT/'limt/model.py'),
                           ('trainer_sha256',ROOT/'train.py'),('usdata_sha256',ROOT/'usdata.py'),
                           ('author_code_sha256',ROOT/'upstream/limt.py')]:
            assert lock[field]==sha(path),(seed,field)
        m=p['model'];model=LiMT(d_model=m['d_model'],cross_heads=m['cross_heads'],
            temporal_heads=m['temporal_heads'],experts=m['experts'],dropout=m['dropout']).cuda().eval()
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True))
        pred=pd.read_parquet(folder/'test_predictions.parquet')
        for t in check_days:
            ids,x,_,_=panel.day(t)
            with torch.no_grad():actual=model(torch.from_numpy(x).cuda()).cpu().numpy()
            saved=pred[pred.date==panel.dates[t]].set_index('symbol').reindex(panel.symbols[ids])
            expected=saved[['pred_return','pred_volume','pred_volatility']].to_numpy()
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
            records.append(dict(seed=seed,date=str(panel.dates[t]),stocks=len(ids),
                                max_absolute_difference=float(abs(actual-expected).max())))
        del model;torch.cuda.empty_cache()
    write_json(ROOT/'prediction-replay-verification.json',dict(status='passed',records=records,
        code_sha256=sha(__file__),selection='first, middle and last test dates; no return-based sampling'))
    print('Prediction replay passed for all three seeds and nine full cross-sections',flush=True)


if __name__=='__main__':main()
