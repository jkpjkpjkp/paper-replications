"""Validation-only sensitivity to sampler respacing; never chooses the test checkpoint."""
import argparse
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
from core import ROOT
from run import load_data, make_model, evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='local');args=ap.parse_args()
    base=ROOT/'results'/args.output
    cfg=json.loads((base/'config.json').read_text())
    cfg.pop('torch_version');cfg.pop('panel_sha256')
    config=SimpleNamespace(**cfg)
    torch.set_num_threads(2)
    results=[]
    # This diagnostic is predefined and uses validation only, on the first seed.
    seed=config.seeds[0]
    x,y,splits,meta=load_data(seed,config.n,config.device)
    for method in ['diffusion_ts','diffusion_reparam','diffusion_score']:
        checkpoint=base/f'seed{seed}'/method/'model.pt'
        model,d=make_model('diffusion',x.shape[1],config.n,x.device)
        model.load_state_dict(torch.load(checkpoint,map_location=x.device,weights_only=True))
        for steps in [50,1000]:
            config.ddim_steps=steps
            started=time.time()
            metric,_,_=evaluate('diffusion',model,d,x,y,splits[1],config,seed+10000)
            results.append(dict(method=method,seed=seed,split='validation',ddim_steps=steps,
                                seconds=time.time()-started,**metric))
            print(json.dumps(results[-1]),flush=True)
        del model,d
        if x.is_cuda:torch.cuda.empty_cache()
    (base/'sampler_validation_check.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
