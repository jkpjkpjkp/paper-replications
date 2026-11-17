import sys,torch
from symbolic_common import HERE,prepare,evaluate_pool
name=sys.argv[1];project=name.lower()+'-reproduction'
torch.set_num_threads(2 if name=='AlphaSAGE' else 4);prepare(project)
evaluate_pool(project,HERE/'results'/name,20 if name=='AlphaSAGE' else 15,verify_only=True)
print(name,'saved-expression scores replay and direct OLS checks passed')
