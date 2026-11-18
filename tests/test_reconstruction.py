import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from setseq.features import rank_features,training_scale,transform_medians,rolling_sum
from setseq.model import PortfolioModel,LongConv,sharpe_loss


def config():
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/reconstruction.json').read_text())
    c.update(hidden_dim=8,ffn_hidden_dim=8,layers=2,kernel_length=5,dropout=0)
    return c


def test_causality_and_permutation():
    torch.manual_seed(1)
    for kind in ['set_sequence','longconv']:
        m=PortfolioModel(config(),kind).eval()
        x=torch.randn(21,7,79)
        z=m(x)
        future=x.clone();future[12:]=torch.randn_like(future[12:])*50
        torch.testing.assert_close(m(future)[:12],z[:12],atol=1e-6,rtol=1e-5)
        permutation=torch.randperm(7)
        torch.testing.assert_close(m(x[:,permutation]),z[:,permutation],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(m.weights(x).abs().sum(-1),torch.ones(21))


def test_fft_equals_direct_causal_convolution():
    torch.manual_seed(9)
    m=LongConv(4,5,2,threshold=.003)
    x=torch.randn(3,17,4)
    u=x.transpose(1,2)
    k=m.kernel.sign()*F.relu(m.kernel.abs()-.003)
    expected=[]
    for c in range(2):
        z=F.conv1d(F.pad(u,(4,0)),k[c].flip(-1)[:,None],groups=4)+u*m.skip[c][None,:,None]
        expected.append(z)
    expected=torch.cat(expected,dim=1).transpose(1,2)
    torch.testing.assert_close(m.convolve(x),expected,atol=1e-6,rtol=1e-5)


def test_overlap_preserves_inference():
    torch.manual_seed(2)
    for kind in ['set_sequence','longconv']:
        m=PortfolioModel(config(),kind).eval()
        x=torch.randn(60,7,79)
        z=m(x)
        start=35;begin=start-m.receptive_history
        torch.testing.assert_close(m(x[begin:])[start-begin:],z[start:],atol=1e-6,rtol=1e-5)


def test_train_only_scaling_and_log_missingness():
    train=np.array([True,True,True,False])
    z,center,scale=training_scale(np.array([1.,2.,3.,1e10]),train)
    z2,c2,s2=training_scale(np.array([1.,2.,3.,-1e10]),train)
    np.testing.assert_array_equal(z[:3],z2[:3]);assert center==c2 and scale==s2
    np.testing.assert_allclose(transform_medians([1.,np.e,np.e**3],3)[1:],[1.,2.])
    assert np.isnan(transform_medians([1.,-1.,2.],3)).all()
    np.testing.assert_allclose(rank_features([[1,1,3,np.nan]]),[[-.25,-.25,.5,0]])


def test_gradient_cross_section_and_loss():
    torch.manual_seed(3)
    m=PortfolioModel(config(),'set_sequence').eval()
    x=torch.randn(20,5,79,requires_grad=True)
    m(x)[-1,0].backward()
    assert x.grad[-3:,1:].abs().sum()>0
    w=torch.tensor([[.2,-.8],[.5,-.5],[.6,-.4]],requires_grad=True)
    y=torch.tensor([[.01,-.02],[-.03,.01],[.02,-.01]])
    r=(w*y).sum(-1)
    torch.testing.assert_close(sharpe_loss(w,y),-r.mean()/r.std(unbiased=False))
