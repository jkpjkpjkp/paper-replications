import importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path('/home/jkp/data');OUT=ROOT/'protocol-six-20261116/results/TRA'
path=ROOT/'ordered-reproductions/sources/TRA-qlib/examples/benchmarks/TRA/src/model.py'
spec=importlib.util.spec_from_file_location('original_tra',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2)
info=json.loads((OUT/'tra/info.json').read_text());cfg=info['config']
model=m.LSTM(**cfg['model_config']);tra=m.TRA(model.output_size,**cfg['tra_config'])
state=torch.load(OUT/'tra/model.bin',map_location='cpu',weights_only=True);model.load_state_dict(state['model']);tra.load_state_dict(state['tra']);model.eval();tra.eval()
device=torch.device('cuda');model.to(device);tra.to(device)
features=pd.read_pickle(OUT/'features.pkl');pred=pd.read_pickle(OUT/'tra/pred.pkl')
cols=[f'score_{i}' for i in range(10)];probs=[f'prob_{i}' for i in range(10)]
selected=pred[probs].to_numpy().argmax(1)
np.testing.assert_allclose(pred.score,pred[cols].to_numpy()[np.arange(len(pred)),selected],rtol=0,atol=0)
checks=[]
for date in ['2022-01-31','2023-07-31','2025-12-31']:
    d=pd.Timestamp(date);saved=pred.xs(d,level='datetime');ids=saved.index
    months=pd.date_range(end=d,periods=12,freq='M')
    index=pd.MultiIndex.from_product([months,ids],names=['datetime','instrument'])
    x=features.reindex(index).fillna(0).to_numpy(dtype=np.float32).reshape(12,len(ids),202).transpose(1,0,2).copy()
    with torch.no_grad():actual=tra.predictors(model(torch.from_numpy(x).to(device))).cpu().numpy()
    expected=saved[cols].to_numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)
    checks.append(dict(date=date,rows=len(ids),max_expert_error=float(abs(actual-expected).max())))
# Recompute headline predictive metric from actual held-out saved labels.
ic=pred.groupby(level='datetime').apply(lambda x:x.score.corr(x.label,method='spearman'))
np.testing.assert_allclose(ic.mean(),info['metric']['IC'],atol=1e-10)
(OUT/'checkpoint-verification.json').write_text(json.dumps(dict(passed=True,checks=checks,
    selected_router_score_matches_saved_probabilities=True,independent_test_spearman_ic=float(ic.mean()),
    limitation='Expert forecasts replayed. Stochastic Gumbel routing checked against saved probabilities, not a newly selected routing draw.'),indent=2))
print('TRA checkpoint replay passed',checks)
