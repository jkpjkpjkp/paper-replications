"""Resource-only CPU/CUDA timing; no experimental model or result is saved."""
import gc,json,random,sys,time
from pathlib import Path
from native_runtime import ROOT,streamed_step,release
sys.path.insert(0,str(ROOT/'upstream/src'))
import torch
from main import build_argparser,make_problem,new_model
from losses import get_loss_fn
from utils import set_seed
rows=[]
for device_name in ['cpu','cuda:0']:
    args=build_argparser().parse_args(['--mode','dfl','--dataset','euro50','--hidden_dims','512','256','--batchsize','64','--cuda','-1'if device_name=='cpu'else'0'])
    device=torch.device(device_name)
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    problem=make_problem(args,ROOT,device);set_seed(999)
    model=new_model(problem,args,device);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    loss_fn=get_loss_fn('dfl',problem,k=5,alpha=.9);rng=random.Random(999);times=[]
    for step in range(4):
        if device.type=='cuda':torch.cuda.synchronize()
        start=time.monotonic()
        streamed_step(model,problem,loss_fn,optimizer,rng.sample(range(len(problem.train)),64))
        if device.type=='cuda':torch.cuda.synchronize()
        times.append(time.monotonic()-start)
    rows.append({'device':device_name,'effective_batch':64,'updates':4,'update_seconds':times,'steady_mean_seconds':sum(times[1:])/3,
                 'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda'else 0})
    release(problem);del model,optimizer,loss_fn,problem;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
(ROOT/'runtime/gpu_benchmark.json').write_text(json.dumps({'purpose':'Resource timing only; no strategy/seed selection','results':rows},indent=2))
print(json.dumps(rows,indent=2))
