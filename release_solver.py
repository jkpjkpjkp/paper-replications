"""Batched diagonal-only equivalent of the pinned author token/entropy routine.

No covariance. Preserve the original input order, caps, one-swap step and final
item-ID tie order. A guarded parity test against unchanged author code is required
before this implementation can be used for comprehensive artifact execution.
"""
import numpy as np


def solve(p,r,var,pop,rho,B,valid,asset_codes,lam):
    delta=1./30;eps=1e-10
    n,m=p.shape;row=np.arange(n)
    w=np.zeros_like(p);support=np.zeros(n,dtype=int);tokens=np.zeros(n,dtype=int)
    activation=np.full((n,m),32767,dtype=np.int16)
    lengths=valid.sum(axis=1)
    def marginal(weights):
        util=p*(B*r*delta-.5*rho*(B**2)*var*((weights+delta)**2-weights**2))
        div=-lam*((weights+delta)*np.log(np.maximum(weights+delta,eps))-weights*np.log(np.maximum(weights,eps)))
        return util+div-.1*pop*delta
    gains=np.where(valid,marginal(w),-np.inf)
    running=np.ones(n,bool)
    for iteration in range(100):
        gains[(support>=10)[:,None]&(w==0)]=-np.inf
        best=np.argmax(gains,axis=1);bestgain=gains[row,best]
        running&=(tokens<30)&(bestgain>1e-9)
        if not running.any():break
        budget=w.sum(axis=1)
        # Match the scalar reference reduction length at the final budget step.
        for j in np.flatnonzero(running&(tokens>=29)):
            budget[j]=np.sum(w[j,:lengths[j]])
        step=np.minimum(delta,np.minimum(.4-w[row,best],1-budget))
        commit=running&(step>1e-18)
        invalid=running&~commit
        gains[row[invalid],best[invalid]]=-np.inf
        old=w[row,best].copy();new=old+np.where(commit,step,0)
        fresh=commit&(old==0)
        activation[row[fresh],best[fresh]]=iteration
        support[fresh]+=1
        w[row[commit],best[commit]]=new[commit]
        tokens[commit]+=np.rint(step[commit]/delta).astype(int)
        # Only the selected marginal changes for the diagonal objective.
        rr=r[row,best];vv=var[row,best];pp=p[row,best];ss=pop[row,best]
        bb=B[:,0];rh=rho[:,0]
        g=pp*(bb*rr*delta-.5*rh*(bb**2)*vv*((new+delta)**2-new**2))
        g+=-lam*((new+delta)*np.log(np.maximum(new+delta,eps))-new*np.log(np.maximum(new,eps)))-.1*ss*delta
        gains[row[commit],best[commit]]=g[commit]
        running&=w.sum(axis=1)<1-1e-15
    else:raise RuntimeError('Batched reference token loop failed to terminate')
    # The original one-round swap has a separable objective with no covariance.
    smallest=np.min(np.where(w>0,w,np.inf),axis=1)
    eligible=(support==10)&np.isfinite(smallest)
    choice=np.where((w>0)&(w==smallest[:,None]),activation,32767)
    jstar=np.argmin(choice,axis=1)
    amount=w[row,jstar]
    f=p*(B*r*amount[:,None]-.5*rho*(B**2)*var*(amount[:,None]**2))-.1*pop*amount[:,None]
    outsiders=valid&(w==0)
    outside_values=np.where(outsiders,f,-np.inf)
    bestoutside=np.argmax(outside_values,axis=1)
    improvement=outside_values[row,bestoutside]-f[row,jstar]
    swap=eligible&(improvement>1e-9)
    # Record near-tie cases so the caller can use the exact scalar reference.
    second=np.partition(outside_values,-2,axis=1)[:,-2]
    ambiguous=eligible&((np.abs(outside_values[row,bestoutside]-second)<1e-10)|(np.abs(improvement-1e-9)<1e-10))
    w[row[swap],jstar[swap]]=0
    w[row[swap],bestoutside[swap]]=np.minimum(.4,amount[swap])
    total=w.sum(axis=1)
    over=total>1+1e-9
    w[over]/=total[over,None]
    order=np.lexsort((asset_codes,-w),axis=1)[:,:10]
    selected=np.take_along_axis(w,order,axis=1)
    ids=np.take_along_axis(asset_codes,order,axis=1)
    ids=np.where(selected>0,ids,-1)
    return ids,selected,ambiguous
