"""Launch upstream main with only resource-equivalent training substitution."""
import json,sys
from pathlib import Path
from native_runtime import ROOT,bounded_train,release
sys.path.insert(0,str(ROOT/'upstream/src'))
import main as author
original_args=author.build_argparser().parse_args()
original_evaluate=author.evaluate_split
data_digest=__import__('hashlib').sha256((ROOT/'data/daily'/f'{original_args.dataset}_160101_251231.csv').read_bytes()).hexdigest()


def train(*args,**kwargs):
    identity={'arguments':vars(original_args),'log_prefix':kwargs.get('log_prefix',''),
              'data_sha256':data_digest,
              'protocol_sha256':__import__('hashlib').sha256((ROOT/'reproduction/protocol.json').read_bytes()).hexdigest()}
    return bounded_train(*args,**kwargs,identity=identity)


author.train_model=train
def evaluate(problem,split,mu,k,selector,cfg):
    if selector=='sdr':
        from resumable_sdr import evaluate_sdr
        result=evaluate_sdr(problem,split,mu,k,selector,cfg,{'arguments':vars(original_args),'data_sha256':data_digest})
    else:result=original_evaluate(problem,split,mu,k,selector,cfg)
    release(problem)
    return result
author.evaluate_split=evaluate
author.main()
