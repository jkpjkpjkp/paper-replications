import argparse,json
import numpy as np
import pandas as pd
import torch
from symbolic_common import BASE,HERE,NAMES,CZData,prepare,target_class,write_json
ap=argparse.ArgumentParser();ap.add_argument('project');args=ap.parse_args()
prepare(args.project)
from alphagen.data.expression import Feature,FeatureType,Ref,TsMean
from alphagen.utils.correlation import batch_pearsonr
torch.set_num_threads(2)
d=CZData(start_time='2020-01-01',end_time='2020-12-31')
f=Feature(FeatureType.BM);current=f.evaluate(d).clone()
p=pd.read_parquet(BASE/'data/cz-monthly/panel.parquet').set_index(['return_month','permno'])
expected=p.xs(202001).reindex(d._stock_ids).BM.to_numpy()
np.testing.assert_allclose(current[0].numpy(),expected,equal_nan=True)
lag=Ref(f,1).evaluate(d)[0].numpy()
np.testing.assert_allclose(lag,p.xs(201912).reindex(d._stock_ids).BM.to_numpy(),equal_nan=True)
rolling=TsMean(f,3).evaluate(d).clone()
d.data[106:]=1234
torch.testing.assert_close(TsMean(f,3).evaluate(d)[:6],rolling[:6],equal_nan=True)
d.y[:]=999
torch.testing.assert_close(f.evaluate(d)[:6],current[:6],equal_nan=True)
eligible=d.eligible
x=torch.nan_to_num(current);y=x*.2+torch.sin(x*3)
a=batch_pearsonr(x,y)
x[~eligible]=999;y[~eligible]=-999
b=batch_pearsonr(x,y)
torch.testing.assert_close(a,b)
manual=np.array([np.corrcoef(xx[mm],yy[mm])[0,1] for xx,yy,mm in zip(x.numpy(),y.numpy(),eligible.numpy())])
np.testing.assert_allclose(a.numpy(),manual,atol=2e-6,rtol=1e-5)
assert len(list(FeatureType))==202
write_json(HERE/(args.project+'-adapter-verification.json'),dict(passed=True,checks=['202 distinct real CZ terminals',
 'Feature row equals saved lagged CZ panel','One-month Ref uses previous calendar month','Future feature changes leave prior expressions invariant',
 'Outcome changes leave expression inputs invariant','Inactive stocks cannot change correlations','Correlations match independently masked NumPy']))
print(args.project,'adapter checks passed')
