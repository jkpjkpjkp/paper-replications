"""Numerical tests of ranking bounds, label timing and matched gradient controls."""
from pathlib import Path
import json
import numpy as np
import torch
from model import Model
from metrics import ranking,portfolio,inference
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2)
    y=np.arange(86,dtype=float)
    for k in [1,3,5,10]:
        p,m,r=ranking(y,y,k)
        assert p==1 and np.isclose(m,np.sum(1/np.arange(1,k+1))/k)
        for seed in range(10):
            _,mr,_=ranking(np.random.default_rng(seed).normal(size=86),y,k)
            assert mr<=m+1e-14
    assert np.isclose(portfolio([.1,-.2])['mdd'],.2)
    assert np.isclose(portfolio([.1,-.2])['irr_sum_eq21'],-.1)
    assert inference(np.zeros(166))['hac_p_two_sided']==1
    torch.manual_seed(5);model=Model(24,hidden=5,dropout=0)
    xx=torch.randn(2,4,7,24);p,h=model(xx)
    p[...,0].abs().mean().backward()
    assert all(z.grad is None or torch.count_nonzero(z.grad)==0 for z in model.risk_head.parameters())
    base=[q.grad.clone() for q in model.encoder.parameters()]
    model.zero_grad();p,h=model(xx);(.7*p[...,0].abs().mean()+.3*(p[...,1]-1).abs().mean()).backward()
    assert any(not torch.equal(a,b.grad) for a,b in zip(base,model.encoder.parameters()))
    assert any(q.grad is not None and q.grad.abs().sum()>0 for q in model.risk_head.parameters())
    graph=Model(24,hidden=5,graph_layers=2,dropout=0)
    adj=torch.stack([torch.eye(4) for _ in range(3)])
    p,_=graph(xx,adj);assert p.shape==(2,4,2)
    try:graph(xx)
    except ValueError:pass
    else:raise AssertionError('Must refuse missing graphs')
    z=np.load(ROOT/'data/price_only.npz')
    dates=z['dates']
    for split,end in [('train','2020-05-25'),('valid','2020-09-21')]:
        assert (dates[z[split]+5]<=end).all()
    assert len(z['symbols'])==86 and len(z['test'])==166
    result={'passed':True,'checks':['Equation20 perfect prediction upper bound and random predictions',
      'portfolio initial wealth and drawdown','constant paired difference inference',
      'return-only has no auxiliary-head gradients; multitask changes encoder gradients',
      'two graph blocks shape and missing-graph refusal','risk-label split embargo',
      '86 paper symbols and 166 test origins']}
    (ROOT/'results/checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
