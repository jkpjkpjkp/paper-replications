"""Reconstruct the paper's attention/routing case study from a frozen model.

Uses the released attention scaling and gating; attention is descriptive and
does not identify a causal effect. No parameters or forecasts are changed.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from limt.data import Panel
from limt.model import LiMT
from scripts.build_data import write_json,sha
ROOT=Path(__file__).resolve().parents[1]


def attention(module,x,cross):
    z=module.norm1(x)
    if cross:z=z.transpose(0,1)
    batch,length,width=z.shape
    def split(layer):
        return layer(z).reshape(batch,length,module.nhead,width//module.nhead).transpose(1,2)
    q,k=[split(layer) for layer in [module.qtrans,module.ktrans]]
    logits=q@k.transpose(-1,-2)
    if cross:logits=logits/math.sqrt(width//module.nhead)
    return logits.softmax(dim=-1).mean(dim=1).cpu().numpy()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--universe',default='csi300',choices=['csi300','csi500'])
    p.add_argument('--seed',type=int,default=0);p.add_argument('--run-name',default='main')
    p.add_argument('--date',default='2020-03-20');p.add_argument('--target',default='600000.SH')
    p.add_argument('--source',default='600004.SH');args=p.parse_args()
    folder=ROOT/'results'/args.run_name/args.universe/f'seed{args.seed}'
    assert json.loads((folder/'status.json').read_text())['phase']=='complete'
    config=json.loads((folder/'config.json').read_text())
    if config['variant']!='full':raise ValueError('Case-study extraction requires the full released model')
    panel=Panel(ROOT/'data'/args.universe)
    t=int(np.flatnonzero(panel.dates==args.date)[0])
    ids,x,_,_=panel.day(t);symbols=panel.symbols[ids].tolist()
    if args.target not in symbols:raise ValueError(f'Target is not an eligible dated member: {args.target}')
    target=symbols.index(args.target)
    source=symbols.index(args.source) if args.source in symbols else next(i for i in range(len(ids)) if i!=target)
    torch.set_num_threads(1)
    model=LiMT(experts=config['experts'],dropout=config['dropout'],variant=config['variant']).eval()
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True))
    tensor=torch.from_numpy(x)
    projected=model.net.feature_layer(tensor)
    cross=attention(model.net.s_attn,projected,True)
    spatial=model.net.s_attn(projected)
    temporal=attention(model.net.t_attn,spatial,False)
    shared=model.net.t_attn(spatial)[:,-1,:]
    tasks=['return','volume','volatility']
    routing=np.stack([model.net.mmoe_fusion.gates[name](shared).softmax(dim=-1).numpy() for name in tasks],axis=1)
    for values in [cross,temporal,routing]:
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values.sum(axis=-1),1.,atol=1e-6)
    predicted=model(tensor).numpy()
    saved=pd.read_parquet(folder/'test_predictions.parquet')
    saved=saved[saved.date==args.date].set_index('symbol').loc[symbols]
    np.testing.assert_allclose(predicted,saved[['pred_return','pred_volume','pred_volatility']].to_numpy(),atol=1e-4,rtol=5e-4)
    max_prediction_error=float(np.max(np.abs(predicted-saved[['pred_return','pred_volume','pred_volatility']].to_numpy())))
    peers=np.random.default_rng(31).choice([i for i in range(len(ids)) if i!=target],size=min(100,len(ids)-1),replace=False)
    dates=panel.dates[t-panel.lookback+1:t+1]
    # Explicit indexing avoids NumPy advanced-index axis reordering.
    cross_target=cross[:,target,:][:,peers]
    propagation=temporal[target,-1,:,None]*cross_target
    out=folder/'interpretability';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'attention_and_routing.npz',cross_attention=cross,
        temporal_attention=temporal,routing=routing,symbols=np.array(symbols),dates=dates,
        peer_indices=peers,propagation=propagation)
    fig,axes=plt.subplots(2,1,figsize=(11,6),constrained_layout=True)
    for ax,values,title in [(axes[0],cross_target,'Cross-stock attention, averaged over heads'),
                            (axes[1],propagation,'Propagation to the final representation (paper Eq. 27)')]:
        im=ax.imshow(values.T,aspect='auto',origin='lower',cmap='viridis')
        ax.set_title(title);ax.set_ylabel('100 fixed sampled peers')
        ticks=np.arange(0,len(dates),5);ax.set_xticks(ticks,dates[ticks],rotation=20,ha='right')
        fig.colorbar(im,ax=ax,pad=.01,label='Weight')
    fig.suptitle(f'{args.target}, signal {args.date} | {args.universe.upper()}, seed {args.seed}')
    fig.savefig(out/'cross_stock_and_propagation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,a,b in [(axes[0],target,source),(axes[1],source,target)]:
        values=temporal[a]*cross[:,a,b][None,:]
        im=ax.imshow(values,origin='lower',aspect='auto',cmap='viridis')
        ax.set(title=f'{symbols[b]} → {symbols[a]}',xlabel='Source history step',ylabel='Output history step')
        ax.set_xticks([0,5,10,15,20]);ax.set_yticks([0,5,10,15,20])
        fig.colorbar(im,ax=ax,pad=.02)
    fig.suptitle('Attention propagation weights; descriptive, not causal effects')
    fig.savefig(out/'directional_propagation.png',dpi=160);plt.close(fig)
    chosen=[]
    for symbol in [args.target,symbols[source],'000001.SZ']+symbols:
        if symbol in symbols and symbol not in chosen:chosen.append(symbol)
        if len(chosen)==3:break
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    vmax=max(float(routing[symbols.index(s)].max()) for s in chosen)
    for ax,symbol in zip(axes,chosen):
        im=ax.imshow(routing[symbols.index(symbol)],aspect='auto',vmin=0,vmax=vmax,cmap='Blues')
        ax.set(title=symbol,xlabel='Shared expert')
        ax.set_xticks(np.arange(config['experts']),np.arange(1,config['experts']+1))
        ax.set_yticks(np.arange(3),['Return','Volume shock','Range/VWAP'])
        fig.colorbar(im,ax=ax,pad=.02)
    fig.suptitle(f'Task-specific expert gates | {args.date} | seed {args.seed}')
    fig.savefig(out/'expert_routing.png',dpi=160);plt.close(fig)
    write_json(out/'metadata.json',dict(signal_date=args.date,target=args.target,source=symbols[source],
        source_fallback=args.source not in symbols,routing_stocks=chosen,peer_sampling_seed=31,
        checkpoint_sha256=sha(folder/'best.pt'),data_manifest_sha256=sha(panel.folder/'manifest.json'),
        script_sha256=sha(__file__),model_code_sha256=sha(ROOT/'limt/model.py'),
        maximum_forecast_reconstruction_error=max_prediction_error,
        implementation='Released core: scaled cross-stock attention, unscaled temporal attention; head means; Eq27 products.',
        prediction_check='CPU reconstructed forecasts agree with saved test forecasts within 1e-4 absolute / 5e-4 relative.',
        interpretation='Descriptive learned weights, not causal effects or full feature attributions; residual/FFN paths are omitted. Data and peer sampling differ from the author figures.',
        output_hashes={f.name:sha(f) for f in sorted(out.iterdir()) if f.suffix in ['.png','.npz']}))
    print(json.dumps(dict(folder=str(out),target=args.target,source=symbols[source],stocks=len(ids))),flush=True)


if __name__=='__main__':main()
