"""Reference comparison against two unchanged upstream Adam updates."""
import copy,json,random,sys
from pathlib import Path
from native_runtime import ROOT,streamed_step,release,atomic
sys.path.insert(0,str(ROOT/'upstream/src'))
import numpy as np
import torch
from main import build_argparser,make_problem,new_model
from train import train_model
from losses import get_loss_fn
from utils import set_seed
args=build_argparser().parse_args(['--mode','dfl','--dataset','euro50','--hidden_dims','512','256','--batchsize','64','--cuda','-1'])
problem=make_problem(args,ROOT,torch.device('cpu'))
set_seed(0);reference=new_model(problem,args,torch.device('cpu'))
initial=copy.deepcopy(reference.state_dict());rng_state=torch.get_rng_state().clone();np_state=np.random.get_state()
loss_fn=get_loss_fn('dfl',problem,k=5,alpha=.5)
reference,h=train_model(reference,problem,loss_fn,2,64,.001,.0001,0,'UNMODIFIED REFERENCE')
expected={n:p.detach().clone()for n,p in reference.named_parameters()}
expected_grad={n:p.grad.detach().clone()for n,p in reference.named_parameters()}
expected_rng=torch.get_rng_state().clone();release(problem);del reference
model=new_model(problem,args,torch.device('cpu'));model.load_state_dict(initial)
torch.set_rng_state(rng_state);np.random.set_state(np_state)
optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
sample_rng=random.Random(0);observed=[]
for _ in range(2):observed.append(streamed_step(model,problem,loss_fn,optimizer,sample_rng.sample(range(len(problem.train)),64))[0])
checks={};passed=True
for kind,wanted in [('parameter',expected),('gradient',expected_grad)]:
    errors=[]
    for name,param in model.named_parameters():
        actual=param if kind=='parameter'else param.grad
        errors.append(float((actual.detach()-wanted[name]).abs().max()))
        passed &= bool(torch.allclose(actual.detach(),wanted[name],rtol=2e-4,atol=2e-6))
    checks[kind+'_max_absolute_error']=max(errors)
checks['same_torch_rng']=torch.equal(torch.get_rng_state(),expected_rng)
checks['losses_close']=np.allclose(observed,[r['loss']for r in h],rtol=2e-6,atol=2e-6)
passed &= checks['same_torch_rng']and checks['losses_close']
atomic(ROOT/'runtime/equivalence.json',{'passed':bool(passed),'market':'euro50','assets':problem.n_assets,'batchsize':64,'adam_updates':2,'checks':checks,'reference_losses':[r['loss']for r in h],'streamed_losses':observed,'scope':'Finite-loss paths only; same upstream loss/solvers, no approximation of objective'})
assert passed,checks
print('Runtime equivalence passed',checks,flush=True)
