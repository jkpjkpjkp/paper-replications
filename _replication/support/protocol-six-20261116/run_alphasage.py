import argparse, json, os, time, types
import numpy as np
import torch
from symbolic_common import HERE,ROOT,NAMES,CZData,prepare,target_class,protocol,write_json,evaluate_pool

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2)
    out=HERE/'results'/('AlphaSAGE-smoke' if args.smoke else 'AlphaSAGE');out.mkdir(parents=True,exist_ok=True)
    runtime=prepare('alphasage-reproduction');protocol('alphasage-reproduction',out)
    if (out/'evaluation/report.json').exists():return
    if not (out/'mining-summary.json').exists():
        source=(runtime/'train_gfn.py').read_text().replace('from src.','from ')
        source=source.replace("start_time='2010-01-01', end_time='2016-12-31'","start_time='2009-01-01', end_time='2019-12-31'")
        source=source.replace("start_time='2018-01-01', end_time='2020-12-31'","start_time='2020-01-01', end_time='2021-12-31'")
        source=source.replace("torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')","torch.device('cpu')")
        source=source.replace('close = Feature(FeatureType.CLOSE)','close = None').replace('target = Ref(close, -20) / close - 1','target = MonthlyTarget()')
        source=source.replace('        weight_scheduler.step()','        weight_scheduler.step()\n        logger.on_episode(episode, loss)')
        (out/'adapted_train.py').write_text(source)
        module=types.ModuleType('cz_gfn');exec(compile(source,str(out/'adapted_train.py'),'exec'),module.__dict__)
        module.StockData=CZData;module.MonthlyTarget=target_class()
        episodes=64 if args.smoke else 10000;start=time.time()
        class Logger:
            def __init__(self,model,pool,log_dir,test_data,target):
                self.model=model;self.pool=pool;self.episode=-1;self.bad=0
                self.writer=types.SimpleNamespace(add_scalar=lambda *a,**k:None)
                assert len(module.FEATURES)==202
            def on_episode(self,episode,loss):
                self.episode=episode;self.bad+=int(loss is not None and not bool(torch.isfinite(loss)))
                if (episode+1)%64==0:
                    write_json(out/'status.json',dict(stage='mining',episodes=episode+1,budget=episodes,pool_size=self.pool.size,nonfinite_losses=self.bad,elapsed=time.time()-start,pid=os.getpid()))
            def log_metrics(self,episode):pass
            def show_pool_state(self):pass
            def save_checkpoint(self,episode):
                torch.save(self.model.state_dict(),out/'model.pt');write_json(out/'pool.json',self.pool.to_dict())
            def close(self):
                self.save_checkpoint(self.episode)
                assert self.pool.size>0
                write_json(out/'mining-summary.json',dict(episodes=self.episode+1,pool_size=self.pool.size,nonfinite_losses=self.bad,elapsed=time.time()-start))
        module.GFNLogger=Logger
        os.chdir(out)
        module.train(argparse.Namespace(seed=0,instrument='cz202',pool_capacity=50,encoder_type='gnn',entropy_coef=.01,
            entropy_temperature=1.,mask_dropout_prob=1.,ssl_weight=1.,nov_weight=.3,n_episodes=episodes,
            update_freq=64,log_freq=500,weight_decay_type='linear',final_weight_ratio=0.))
    if not args.smoke:
        evaluate_pool('alphasage-reproduction',out,20)
        write_json(out/'status.json',dict(stage='complete'))

if __name__=='__main__':main()
