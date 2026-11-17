"""Replay saved checkpoints and independently rebuild labels from source closes."""
from pathlib import Path
import json
import numpy as np
import torch
from model import Model
from prepare import read_field, DATA
from metrics import ranking
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2)
    z=np.load(ROOT/'data/price_only.npz');cfg=json.loads((ROOT/'protocol.json').read_text())
    calendar=np.array((DATA/'calendars/day.txt').read_text().splitlines())
    closes=np.stack([read_field(s,'close',calendar)[0] for s in z['symbols']],axis=1)
    lookup={d:i for i,d in enumerate(calendar)}
    origin_ids=np.array([lookup[d] for d in z['dates'][z['test']]])
    expected_return=closes[origin_ids+1]/closes[origin_ids]-1
    expected_risk=np.stack([np.std(closes[t-4:t+6],axis=0,ddof=0) for t in origin_ids])
    assert np.allclose(expected_return,z['yret'][z['test']],atol=2e-8,rtol=1e-6,equal_nan=True)
    assert np.allclose(expected_risk,z['yrisk'][z['test']],atol=2e-7,rtol=1e-6,equal_nan=True)
    assert np.allclose(ranking(np.array([0,3,2,1]),np.array([3,2,1,0]),2)[:2],[.5,5/12])
    x=torch.from_numpy(np.stack([z['x'][t-6:t+1].transpose(1,0,2) for t in z['test']]))
    replay=[]
    for variant in ['return_only','multitask']:
        for seed in cfg['seeds']:
            d=ROOT/f'results/{variant}_seed{seed}'
            model=Model(x.shape[-1],hidden=cfg['hidden_per_direction'],head=cfg['heads'],dropout=cfg['dropout'])
            model.load_state_dict(torch.load(d/'best.pt',map_location='cpu',weights_only=True));model.eval()
            pred=[]
            with torch.no_grad():
                for ids in np.array_split(np.arange(len(x)),8):pred.append(model(x[ids])[0].numpy())
            pred=np.concatenate(pred);saved=np.load(d/'predictions.npz')
            assert np.array_equal(pred,saved['pred'])
            hist=json.loads((d/'history.json').read_text());fit=json.loads((d/'fit.json').read_text())
            best=min(hist,key=lambda r:r['valid_return_mae'])
            assert fit['best_epoch']==best['epoch']
            assert np.array_equal(saved['y'][...,0],z['yret'][z['test']],equal_nan=True)
            replay.append({'variant':variant,'seed':seed,'predictions_identical':True,'best_checkpoint_validated':True})
    out={'passed':True,'independent_source_label_reconstruction':True,'hand_worked_ranking_example':True,'checkpoint_replays':replay}
    (ROOT/'results/verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
