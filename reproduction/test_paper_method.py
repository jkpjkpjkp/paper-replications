import random
import numpy as np
import pandas as pd
import pytest
import torch

from paper_method import Panel, PrintedDecision, batches, covariance, vector_loss, train_epoch
from selector import soft_topk_1d


def test_vector_loss_uses_same_target_and_sum_over_assets():
    p=torch.tensor([2.,3.],requires_grad=True);y=torch.tensor([1.,1.]);w=torch.tensor([.2,.8])
    loss=vector_loss(p,y,w)
    assert loss.item()==2.0  # .5*(-1)+.5*(1+4)
    loss.backward()
    torch.testing.assert_close(p.grad,torch.tensor([1.,2.]))


def test_epoch_covers_every_sample_once_including_remainder():
    selected=batches(list(range(137)),64,random.Random(16))
    assert list(map(len,selected))==[64,64,9]
    assert sorted(i for b in selected for i in b)==list(range(137))


def test_covariance_shrinkage_has_no_unconditional_raw_unit_jitter():
    x=torch.tensor([[.01,.02],[-.02,.01],[.03,-.01],[.01,.01]],dtype=torch.float64)
    c,L=covariance(x);raw=torch.cov(x.T)
    expected=.9*raw+.1*raw.trace()/2*torch.eye(2,dtype=torch.float64)
    torch.testing.assert_close(c,expected)
    torch.testing.assert_close(L@L.T,c)


def test_soft_gate_mass_and_gradient_are_not_hard_cardinality():
    x=torch.tensor([.1,.2,.3,.4],dtype=torch.float64,requires_grad=True)
    g=soft_topk_1d(x,2,alpha=5.,iters=32)
    assert abs(float(g.detach().sum())-2)<1e-8
    assert int((g.detach()>0).sum())==4
    assert torch.autograd.gradcheck(lambda z:soft_topk_1d(z,2,alpha=5.,iters=64),(x,),eps=1e-5,atol=1e-5)


def test_printed_constraint_orientation_against_closed_form():
    decision=PrintedDecision(2,1,solver_eps=1e-8)
    L=torch.tensor([[2.,0.],[1.,1.]],dtype=torch.float64)
    mean=torch.tensor([3.,2.],dtype=torch.float64,requires_grad=True)
    w,info=decision.solve(mean,L)
    A=L.T@L
    direction=torch.linalg.solve(A,mean.detach())
    expected=direction/direction.sum()
    torch.testing.assert_close(w.detach(),expected,atol=1e-5,rtol=1e-5)
    assert info['t']>0
    (w@torch.tensor([.02,-.01],dtype=torch.float64)).backward()
    assert torch.isfinite(mean.grad).all()


def test_preconditioned_layer_matches_original_and_gradients():
    L=torch.tensor([[2.,0.],[1.,1.]],dtype=torch.float64)
    outputs=[];grads=[]
    for precondition in [False,True]:
        layer=PrintedDecision(2,1,solver_eps=1e-9,precondition=precondition)
        mean=torch.tensor([3.,2.],dtype=torch.float64,requires_grad=True)
        w,_=layer.solve(mean,L)
        (w@torch.tensor([.02,-.01],dtype=torch.float64)).backward()
        outputs.append(w.detach());grads.append(mean.grad)
    torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(grads[0],grads[1],rtol=1e-5,atol=1e-7)


def test_printed_constraint_allows_singular_zero_scale():
    decision=PrintedDecision(2,1,solver_eps=1e-8)
    # The exact optimizer is proportional to (1,-1), with sum(y)=t=0.
    _,info=decision.solve(torch.tensor([1.,-2.],dtype=torch.float64),torch.eye(2,dtype=torch.float64))
    assert abs(info['t'])<1e-6
    assert info['gross_exposure']>1e6


def test_test_windows_use_only_preceding_observations(tmp_path):
    values=np.arange(600).reshape(300,2)/100000
    path=tmp_path/'panel.csv'
    pd.DataFrame(values,index=pd.date_range('2020-01-01',periods=300)).to_csv(path)
    panel=Panel(path)
    assert panel.train==list(range(100,240))
    assert panel.test[0]==240
    x,target,_,_=panel.sample(240)
    torch.testing.assert_close(x,torch.tensor(values[140:240].flatten(),dtype=torch.float32))
    torch.testing.assert_close(target,torch.tensor(values[240],dtype=torch.float64))


def test_hard_baseline_restricts_weight_support():
    decision=PrintedDecision(3,1,solver_eps=1e-8)
    cov=torch.eye(3,dtype=torch.float64)
    w,info=decision.hard(torch.tensor([.01,.03,.02],dtype=torch.float64),cov,cov)
    assert int((w.abs()>1e-8).sum())==1
    assert info['budget_error']<1e-6


def test_streamed_batch_is_same_sgd_update_as_full_batch_graph(tmp_path):
    import copy
    class SmallPanel:
        train=[0,1,2]
        def sample(self,i):
            return (torch.tensor([.01*(i+1),.02],dtype=torch.float64),
                    torch.tensor([.01,.02,-.01],dtype=torch.float64),
                    torch.eye(3,dtype=torch.float64),torch.eye(3,dtype=torch.float64))
    panel=SmallPanel()
    model=torch.nn.Linear(2,3,dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(.1)
        model.bias.copy_(torch.tensor([.2,.3,.4],dtype=torch.float64))
    reference=copy.deepcopy(model)
    reference_decision=PrintedDecision(3,1,solver_eps=1e-8)
    order=batches(panel.train,64,random.Random(16))[0]
    losses=[]
    for i in order:
        x,target,_,L=panel.sample(i)
        prediction=reference(x)
        weights,_=reference_decision.soft(prediction,L)
        losses.append(vector_loss(prediction,target,weights))
    opt=torch.optim.SGD(reference.parameters(),lr=.001)
    torch.stack(losses).mean().backward();opt.step()
    history=[]
    diagnostic={"training_decisions":0,"training_near_zero_t":0,"training_budget_error_max":0.,"training_gate_mass_error_max":0.}
    count=train_epoch(model,panel,PrintedDecision(3,1,solver_eps=1e-8),
                      torch.optim.SGD(model.parameters(),lr=.001),"dfl",random.Random(16),tmp_path,1,0,history,diagnostic)
    assert count==1 and history[0]['batch_size']==3
    for actual,expected in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-9)


def test_full_epoch_checkpoint_resume_preserves_final_parameters(tmp_path,monkeypatch):
    import paper_method
    panel=tmp_path/'panel.csv'
    # Small generated fixture tests checkpoint mechanics, not financial results.
    values=np.random.default_rng(7).normal(.001,.01,size=(150,10))
    pd.DataFrame(values,index=pd.date_range('2020-01-01',periods=150)).to_csv(panel)
    direct=tmp_path/'direct';resumed=tmp_path/'resumed'
    paper_method.run(panel,direct,'pfl',16,10,2)
    original=paper_method.train_epoch
    def interrupt_after_one(*args,**kwargs):
        if args[7]==2:
            raise RuntimeError('simulated interruption after full epoch checkpoint')
        return original(*args,**kwargs)
    monkeypatch.setattr(paper_method,'train_epoch',interrupt_after_one)
    interrupted=paper_method.run(panel,resumed,'pfl',16,10,2)
    assert interrupted['status']=='failed'
    monkeypatch.setattr(paper_method,'train_epoch',original)
    result=paper_method.run(panel,resumed,'pfl',16,10,2,resume=True)
    assert result['completed_updates']==2
    a=torch.load(direct/'checkpoint.pt',weights_only=False)['model']
    b=torch.load(resumed/'checkpoint.pt',weights_only=False)['model']
    assert all(torch.equal(a[k],b[k]) for k in a)
