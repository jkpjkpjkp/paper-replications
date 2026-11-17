from pathlib import Path
import sys,json
from datetime import datetime,timezone
import numpy as np,pandas as pd
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
from screen import residuals,trial,ALPHAS
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ocp import run,controller,metrics

def main():
 frame=pd.read_csv(HERE/'discovery_sweep.csv')
 source=pd.read_csv(ROOT/'data/raw/djia.csv')
 locked=[]
 for (ticker,predictor),part in frame.groupby(['ticker','predictor']):
  for method,group in part.groupby('method'):
   eligible=group[(group.coverage>=.95)&np.isfinite(group.width)]
   if len(eligible):
    r=eligible.loc[eligible.width.idxmin()]
    locked.append(dict(ticker=ticker,predictor=predictor,method=method,alpha=r.alpha,lr=r.lr,scale=r.scale))
 pd.DataFrame(locked).to_csv(HERE/'locked_configs.csv',index=False)
 (HERE/'confirmation_protocol.json').write_text(json.dumps(dict(locked_at=datetime.now(timezone.utc).isoformat(),discovery_end='2011-12-31',confirmation_start='2012-01-01',scope='all 27 unused stock tickers and both declared point forecasts',selection='per-method minimum discovery mean width with discovery coverage >=.95',updates='continue same online run, fixed normalization and alpha/LR chosen before holdout',uncertainty='descriptive chronological transfer; cross-stock series are dependent'),indent=2))
 results=[]
 for r in locked:
  d=source[source.Name==r['ticker']].sort_values('Date').dropna(subset=['Open','Close'])
  s,dates=residuals(d,r['predictor']);s=s/r['scale']
  q=controller(s,r['alpha'],r['lr'],100) if r['method']=='P' else run(s,r['method'],r['alpha'],r['lr'])
  mask=dates>='2012-01-01';m=metrics(s[mask],q[mask])
  results.append(dict(**r,**m))
 pd.DataFrame(results).to_csv(HERE/'confirmation.csv',index=False)
 print('Wrote',len(results),'fixed-configuration chronological confirmation runs.')
if __name__=='__main__':main()
