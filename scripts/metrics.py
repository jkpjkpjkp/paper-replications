"""Literal ranking equations, explicit portfolio conventions and paired inference.

HAC/circular-bootstrap implementation adapted from local
monthly-coforecast/scripts/evaluate.py::inference, with daily lags/block length,
5000 resamples, centered null p-values and degenerate-case handling added.
"""
import numpy as np
from scipy.stats import norm

def ranking(pred,y,k):
    order=np.argsort(-pred,kind='stable')[:k]
    truth=np.argsort(-y,kind='stable')
    rank=np.empty(len(y),dtype=int);rank[truth]=np.arange(1,len(y)+1)
    precision=np.isin(order,truth[:k]).mean()
    mrr=(1/rank[order]).mean()  # Eq20, not an inferred replacement for the paper's table.
    return precision,mrr,float(y[order].mean())

def portfolio(r):
    r=np.asarray(r,dtype=float)
    wealth=np.r_[1,np.cumprod(1+r)]
    mdd=float(np.max(1-wealth/np.maximum.accumulate(wealth)))
    return {'irr_sum_eq21':float(r.sum()),'compounded_return':float(wealth[-1]-1),
            'sharpe_annualized':float(np.sqrt(252)*(r.mean()-((1.025)**(1/252)-1))/r.std(ddof=0)) if r.std() else None,
            'mdd':mdd}

def inference(d,lag=20,block=10,reps=5000):
    d=np.asarray(d,dtype=float);n=len(d);x=d-d.mean();lag=min(lag,n-1)
    lr=x@x/n
    for k in range(1,lag+1):lr+=2*(1-k/(lag+1))*(x[k:]@x[:-k]/n)
    se=np.sqrt(max(lr,0)/n)
    stat=d.mean()/se if se>0 else (0 if d.mean()==0 else np.sign(d.mean())*np.inf)
    rng=np.random.default_rng(20261110)
    starts=rng.integers(n,size=(reps,int(np.ceil(n/block))))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    boot=d[ix].mean(-1)
    p=(1+np.sum(np.abs(boot-d.mean())>=abs(d.mean())))/(reps+1)
    return {'delta':float(d.mean()),'hac_t':float(stat),'hac_p_two_sided':float(2*norm.sf(abs(stat))),
            'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),
            'bootstrap_p_two_sided':float(p),'days':n,'block':block}

def holm(p):
    p=np.asarray(p);ix=np.argsort(p);out=np.empty(len(p))
    out[ix]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[ix]))
    return out
