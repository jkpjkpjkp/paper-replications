from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ocp import run,controller
OUT=Path(__file__).parent
ALPHAS=np.r_[.0025,.005,.01,np.arange(.02,.301,.01)]
RATES=np.geomspace(.01,100,9)
PRATES=[0,.01,.025,.05,.1,.25,.5,1,2]

def residuals(d, predictor):
 y=d.Open.to_numpy(float)
 if predictor=='previous_close':
  pred=d.Close.shift(1).to_numpy(float)
  return abs(y[1:]-pred[1:]),d.Date.to_numpy()[1:]
 window=100; logy=np.log(y); x=np.arange(window); xc=x-x.mean(); den=xc@xc
 scores=[]
 for t in range(window,len(y)):
  z=logy[t-window:t]; slope=xc@z/den
  pred=np.exp(z.mean()+slope*(window-x.mean()))
  scores.append(abs(y[t]-pred))
 return np.array(scores),d.Date.to_numpy()[window:]

def trial(s,method,alpha,lr=1):
 q=controller(s,alpha,lr,window=100) if method=='P' else run(s,method,alpha,lr)
 q=q[100:]; v=s[100:]
 return float(np.mean(v<=q)),float(np.mean(2*np.maximum(q,0)))

def main():
 allrows=[];frontiers=[]; started=time.time()
 d=pd.read_csv(ROOT/'data/raw/djia.csv')
 for name,part in d.groupby('Name'):
  if name in ['AXP','AAPL','AMZN','GOOGL']:continue
  part=part.sort_values('Date').dropna(subset=['Open','Close'])
  for predictor in ['previous_close','loglinear100']:
   scores,dates=residuals(part,predictor)
   discovery=scores[dates<'2012-01-01']
   scale=float(np.quantile(discovery[:100],.95));s=discovery/scale
   for method in ['UP','KT','DtACI','SFOGD','P']:
    rates=RATES if method=='SFOGD' else PRATES if method=='P' else [1]
    for lr in rates:
     for alpha in ALPHAS:
      cov,width=trial(s,method,float(alpha),float(lr))
      allrows.append(dict(ticker=name,predictor=predictor,method=method,alpha=alpha,lr=lr,coverage=cov,width=width,scale=scale))
   frame=pd.DataFrame(allrows); f=frame[(frame.ticker==name)&(frame.predictor==predictor)]
   for target in [.9,.95]:
    eligible=f[(f.coverage>=target)&np.isfinite(f.width)]
    if not any(eligible.method=='UP'):continue
    up=eligible[eligible.method=='UP'].sort_values('width').iloc[0]
    best=eligible[eligible.method!='UP'].sort_values('width').iloc[0]
    frontiers.append(dict(ticker=name,predictor=predictor,target=target,up_width=up.width,best_width=best.width,best_method=best.method,gain=1-up.width/best.width,up_alpha=up.alpha,best_alpha=best.alpha,best_lr=best.lr))
   print(name,predictor,round(time.time()-started,1),flush=True)
  pd.DataFrame(allrows).to_csv(OUT/'discovery_sweep.csv',index=False)
  pd.DataFrame(frontiers).sort_values('gain',ascending=False).to_csv(OUT/'discovery_frontiers.csv',index=False)
 print(pd.DataFrame(frontiers).sort_values('gain',ascending=False).head(16).to_string(index=False))
if __name__=='__main__':main()
