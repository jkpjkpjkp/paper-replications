import numpy as np
import torch
from limt.data import labels_from_market, fill_window, Panel
from limt.model import LiMT
from limt.portfolio import linear_norm, clip_trades, observed_moments, apo_weights


def test_optimized_matches_released_values_and_gradients():
    torch.set_num_threads(1);torch.manual_seed(7)
    a=LiMT(d_model=32,dropout=0.,optimized=False).double()
    b=LiMT(d_model=32,dropout=0.,optimized=True).double()
    b.load_state_dict(a.state_dict())
    x=torch.randn(7,5,158,dtype=torch.float64,requires_grad=True)
    z=x.detach().clone().requires_grad_()
    ya,yb=a(x),b(z)
    torch.testing.assert_close(ya,yb,atol=1e-10,rtol=1e-9)
    ya.square().sum().backward();yb.square().sum().backward()
    torch.testing.assert_close(x.grad,z.grad,atol=1e-10,rtol=1e-8)
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,atol=1e-10,rtol=1e-8)


def test_exact_target_offsets():
    c=np.array([[10.,12.,15.,18.,20.,25.,30.,36.,40.,50.]])
    v=np.exp(np.arange(10,dtype=float)[None,:])
    y=labels_from_market(c,v,c*1.1,c*.9,c)
    np.testing.assert_allclose(y[0,4],[30/25-1,3.,.2],rtol=1e-6)
    assert np.isnan(y[0,-2:]).all()


def test_window_fill_uses_only_observed_window():
    x=np.array([[[np.nan,1.],[3.,np.nan],[np.nan,4.]]])
    np.testing.assert_equal(fill_window(x),np.array([[[3.,1.],[3.,1.],[3.,4.]]]))
    assert np.isnan(x[0,0,0])


def test_degenerate_weight_and_trade_cap():
    np.testing.assert_allclose(linear_norm([4,4,4]),[1/3]*3)
    current,request,fill=clip_trades(np.array([0.,-7.]),np.array([100.,0.]),np.array([10.,0.]),.02)
    np.testing.assert_allclose(current,[.2,-7.])
    np.testing.assert_allclose(fill,[.2,0.])


def test_apo_cannot_read_future():
    rng=np.random.default_rng(6)
    m=rng.uniform(10,20,(5,15,7));member=np.ones((5,15),bool)
    pred=rng.normal(size=(5,3));ids=np.arange(5)
    before=apo_weights(pred,ids,m,member,8)
    m[:,9:]=99999
    np.testing.assert_array_equal(before,apo_weights(pred,ids,m,member,8))


def test_split_purges_label_boundary_and_eligibility_is_causal(tmp_path):
    import json
    dates=np.array(['2014-12-26','2014-12-29','2014-12-30','2014-12-31','2015-01-05','2015-01-06'])
    (tmp_path/'manifest.json').write_text('{}')
    arrays=dict(dates=dates,symbols=np.array([str(x) for x in range(60)]),
        features=np.ones((60,6,158),np.float32),market=np.ones((60,6,7),np.float32),
        member=np.ones((60,6),bool),labels=np.ones((60,6,3),np.float32))
    arrays['labels'][:,2:]=np.nan
    for k,a in arrays.items():np.save(tmp_path/f'{k}.npy',a)
    p=Panel(tmp_path,lookback=1)
    assert p.split('train').tolist()==[0,1]
    assert p.eligible[:,3].all()  # Missing future labels cannot change inference universe.
