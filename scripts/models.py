"""Estimators for the coforecasting experiment. All fit methods receive past data only."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
import torch
from torch import nn
from numba import njit

torch.set_num_threads(2)
SCALE=.1

class Transform:
    def fit(self,x):
        self.imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        a=self.imp.transform(x)
        self.lo=np.nanquantile(a,.005,axis=0);self.hi=np.nanquantile(a,.995,axis=0)
        self.sc=StandardScaler().fit(np.clip(a,self.lo,self.hi))
        return self
    def __call__(self,x):
        return self.sc.transform(np.clip(self.imp.transform(x),self.lo,self.hi)).astype('float32')

def sampled(n,limit=60000,seed=42):
    return np.sort(np.random.default_rng(seed).choice(n,min(n,limit),replace=False))

class Independent:
    def __init__(self,kind='ridge',complexity=100): self.kind=kind;self.complexity=complexity
    def fit(self,x,r,v):
        if self.kind=='ridge':
            self.a=Ridge(alpha=self.complexity).fit(x,r/SCALE)
            self.b=Ridge(alpha=self.complexity).fit(x,np.log(v/SCALE**2))
            self.smear=np.mean((v/SCALE**2)/np.exp(self.b.predict(x)))
        else:
            self.a=HistGradientBoostingRegressor(max_iter=self.complexity,max_leaf_nodes=7,
                    min_samples_leaf=100,learning_rate=.05,l2_regularization=10,early_stopping=False).fit(x,r/SCALE)
            self.b=HistGradientBoostingRegressor(loss='poisson',max_iter=self.complexity,max_leaf_nodes=7,
                    min_samples_leaf=100,learning_rate=.05,l2_regularization=10,early_stopping=False).fit(x,v/SCALE**2)
        return self
    def predict(self,x):
        m=self.a.predict(x)*SCALE
        v=(np.exp(self.b.predict(x))*self.smear if self.kind=='ridge' else self.b.predict(x))*SCALE**2
        return m,np.clip(v,1e-8,10)

class Trees:
    def __init__(self,joint): self.joint=joint
    def fit(self,x,r,v):
        y=np.column_stack([r/SCALE,np.log(v/SCALE**2)])
        self.scale=y.std(axis=0)
        kw=dict(n_estimators=100,max_depth=8,min_samples_leaf=150,max_features=.8,n_jobs=2,random_state=51)
        if self.joint:
            self.mod=ExtraTreesRegressor(**kw).fit(x,y/self.scale)
            lv=self.mod.predict(x)[:,1]*self.scale[1]
        else:
            self.mod=[ExtraTreesRegressor(**kw).fit(x,y[:,j]/self.scale[j]) for j in range(2)]
            lv=self.mod[1].predict(x)*self.scale[1]
        self.smear=np.mean((v/SCALE**2)/np.exp(lv))
        return self
    def predict(self,x):
        y=(self.mod.predict(x) if self.joint else np.column_stack([m.predict(x) for m in self.mod]))*self.scale
        return y[:,0]*SCALE,np.clip(np.exp(y[:,1])*self.smear*SCALE**2,1e-8,10)

class NGB:
    """Gaussian NGBoost; defaults retain the monthly study's configuration."""
    def __init__(self,n=150,*,max_depth=2,min_samples_leaf=100,learning_rate=.03,
                 target_scale=SCALE,train_limit=40000,variance_bounds=(1e-8,10),
                 random_state=42):
        self.n=n;self.max_depth=max_depth;self.min_samples_leaf=min_samples_leaf
        self.learning_rate=learning_rate;self.target_scale=target_scale
        self.train_limit=train_limit;self.variance_bounds=variance_bounds
        self.random_state=random_state
    @classmethod
    def paper_benchmark(cls,n=2000,*,random_state=42):
        """Duan et al. (2020), Section 4: raw UCI target units, no risk clipping."""
        return cls(n,max_depth=3,min_samples_leaf=1,learning_rate=.01,
                   target_scale=1.,train_limit=None,variance_bounds=None,
                   random_state=random_state)
    def fit(self,x,r,v=None):
        ix=sampled(len(x),self.train_limit) if self.train_limit is not None else np.arange(len(x))
        self.mod=NGBRegressor(Dist=Normal,Base=DecisionTreeRegressor(
            max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf),
            n_estimators=self.n,learning_rate=self.learning_rate,minibatch_frac=1.,
            natural_gradient=True,verbose=False,random_state=self.random_state
        ).fit(x[ix],r[ix]/self.target_scale)
        return self
    def predict(self,x,*,max_iter=None):
        d=self.mod.pred_dist(x,max_iter=max_iter)
        variance=d.scale**2*self.target_scale**2
        if self.variance_bounds is not None:variance=np.clip(variance,*self.variance_bounds)
        return d.loc*self.target_scale,variance

class Net:
    """Matched 32->16 architectures. Independent networks each have their own encoder."""
    def __init__(self,kind='joint',weight=1.,epochs=35,seeds=(11,29,47)):
        self.kind=kind;self.weight=weight;self.epochs=epochs;self.seeds=seeds
    def fit(self,x,r,v):
        ix=sampled(len(x));xx=torch.tensor(x[ix]);rr=torch.tensor(r[ix]/SCALE,dtype=torch.float32)
        vv=torch.tensor(v[ix]/SCALE**2,dtype=torch.float32)
        self.mods=[]
        for seed in self.seeds:
            torch.manual_seed(seed)
            def make(n):return nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,n))
            mod=make(2) if self.kind!='independent' else nn.ModuleList([make(1),make(1)])
            opt=torch.optim.AdamW(mod.parameters(),lr=.002,weight_decay=.01)
            epochs=self.epochs if isinstance(self.epochs,tuple) else (self.epochs,self.epochs)
            for epoch in range(max(epochs)):
                order=torch.randperm(len(xx))
                for idx in order.split(4096):
                    a=xx[idx];y=rr[idx];z=vv[idx]
                    if self.kind=='independent':
                        mu=mod[0](a).flatten();lv=mod[1](a).flatten().clamp(-8,5)
                    else:
                        out=mod(a);mu=out[:,0];lv=out[:,1].clamp(-8,5)
                    if self.kind=='gaussian': loss=.5*(lv+(y-mu)**2*torch.exp(-lv))
                    elif self.kind=='student':
                        # t_5 parameterized by variance h; scale^2 = 3h/5.
                        loss=.5*lv+3*torch.log1p((y-mu)**2/(3*torch.exp(lv)))
                    elif self.kind=='independent':
                        loss=(y-mu)**2*(epoch<epochs[0])+(lv+z*torch.exp(-lv))*(epoch<epochs[1])
                    else: loss=(y-mu)**2+self.weight*(lv+z*torch.exp(-lv))
                    opt.zero_grad();loss.mean().backward()
                    if self.kind=='independent':
                        for j in range(2):
                            if epoch>=epochs[j]:
                                for par in mod[j].parameters():par.grad=None
                            else:torch.nn.utils.clip_grad_norm_(mod[j].parameters(),5)
                    else:torch.nn.utils.clip_grad_norm_(mod.parameters(),5)
                    opt.step()
            self.mods.append(mod.eval())
        return self
    def predict(self,x,individual=False):
        results=[]
        with torch.no_grad():
            for mod in self.mods:
                means=[];vars=[]
                for a in torch.tensor(x).split(16384):
                    if self.kind=='independent':mu=mod[0](a).flatten();lv=mod[1](a).flatten()
                    else:out=mod(a);mu=out[:,0];lv=out[:,1]
                    means.extend(mu.numpy()*SCALE);vars.extend(torch.exp(lv.clamp(-8,5)).numpy()*SCALE**2)
                results.append((np.asarray(means),np.asarray(vars)))
        if individual:return results
        # Average variance forecasts, rather than adding epistemic spread across seeds.
        return tuple(np.mean([p[j] for p in results],axis=0) for j in range(2))

@njit(cache=True)
def recurs(y,ids,months,alpha,beta,psi,initial,sharv):
    pred=np.empty(len(y));loss=np.empty(len(y));state=initial
    for i in range(len(y)):
        if i==0 or ids[i]!=ids[i-1] or months[i]!=months[i-1]+1:state=initial
        if sharv:
            b=beta*state;a=alpha+psi*state
            pred[i]=b+3*a
            state=(b+np.sqrt(b*b+4*a*y[i]*y[i]))/2
            state=max(state,1e-12)
            loss[i]=.5*y[i]*y[i]/state-.5*np.log(state)+np.log(max(2*state-b,1e-12))+.5*np.log(2*np.pi)
        else:
            pred[i]=state
            loss[i]=.5*(np.log(2*np.pi*state)+y[i]*y[i]/state)
            state=alpha+beta*state+psi*y[i]*y[i]
    return pred,loss

class Recursion:
    def __init__(self,sharv):self.sharv=sharv
    def fit_predict(self,history,test):
        import pandas as pd
        self.mean=history.yret.mean()
        scale=history.assign(sq=(history.yret-self.mean)**2).groupby('permno').sq.mean().pow(.5).clip(.02,.5)
        all_=pd.concat([history,test]).sort_values(['permno','target_month'])
        s=all_.permno.map(scale).fillna(scale.median()).to_numpy()
        y=(all_.yret.to_numpy()-self.mean)/s
        ids=all_.permno.to_numpy();months=all_.target_month.to_numpy().astype(np.int64)
        train=(all_.target_month<test.target_month.min()).to_numpy()
        def unpack(p):
            b,q=softmax(np.r_[p[1:3],0])[:2]*.995
            level=np.exp(p[0]);a=level*(1-b-q)/(3-2*b) if self.sharv else level*(1-b-q)
            initial=a/(1-b-q) if self.sharv else level
            return a,b,q,initial
        def obj(p):
            a,b,q,init=unpack(p)
            return recurs(y,ids,months,a,b,q,init,self.sharv)[1][train].mean()
        fits=[minimize(obj,p,method='L-BFGS-B',bounds=[(-3,3),(-7,7),(-7,7)],options={'maxiter':100}) for p in [np.array([0.,3.,0.]),np.array([0.,1.,1.])]]
        fit=min(fits,key=lambda x:x.fun);a,b,q,init=unpack(fit.x)
        pred,loss=recurs(y,ids,months,a,b,q,init,self.sharv)
        out=pd.DataFrame({'mu':self.mean,'variance':pred*s*s,'exact_nll':loss+np.log(s)},index=all_.index).loc[test.index]
        self.meta={'alpha':a,'beta':b,'psi':q,'objective':float(fit.fun),'success':bool(fit.success),'message':str(fit.message)}
        return out

class CMK:
    """2026 closed-form estimator, equations (30),(32),(36),(37), on reduced rank features.

    Independent-factor control uses the identical mean and uncorrected covariance.
    One-step option solves the joint b,V problem at pooled idiosyncratic variance.
    """
    def __init__(self,eta=1.,penalty=0.,control=False,onestep=False):
        self.eta=eta;self.penalty=penalty;self.control=control;self.onestep=onestep
    def fit(self,phi,r,months):
        fs=[];gs=[];rss=0;dfs=0
        for m in np.unique(months):
            ix=months==m;x=phi[ix];y=r[ix]/SCALE;p=x.shape[1]
            gram=x.T@x; inv=np.linalg.inv(gram+.001*np.trace(gram)/p*np.eye(p))
            f=inv@x.T@y;gp=inv@gram@inv
            fs.append(f);gs.append(gp);rss+=y@y-y@x@f;dfs+=len(y)-p
        fs=np.array(fs);self.b=fs.mean(axis=0);self.u=max(rss/dfs,1e-6)
        self.F=self.u*np.mean(gs,axis=0);second=fs.T@fs/len(fs);cov=second-np.outer(self.b,self.b)
        if self.control:self.C=cov
        else:
            self.lam=self.penalty*np.trace(cov)/len(self.b)
            a=cov-(1+self.eta)*self.F-self.lam/2*np.eye(len(self.b))
            w,q=np.linalg.eigh((a+a.T)/2)
            self.C=self.eta*self.F+(q*np.maximum(w,0))@q.T
        if self.onestep:
            import cvxpy as cp
            p=len(self.b);b=cp.Variable(p);v=cp.Variable((p,p),symmetric=True)
            aug=cp.bmat([[np.ones((1,1)),cp.reshape(b,(1,p),order='C')],[cp.reshape(b,(p,1),order='C'),v-self.eta*self.F]])
            prob=cp.Problem(cp.Minimize(2*cp.sum_squares(b-self.b)+cp.sum_squares(v-(second-self.F))+self.lam*cp.trace(v)),[aug>>0])
            prob.solve(solver='CLARABEL')
            if prob.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(prob.status)
            self.b=b.value;self.C=v.value-np.outer(b.value,b.value)
        assert np.linalg.eigvalsh(self.C).min()>-1e-7
        return self
    def predict(self,x):
        return x@self.b*SCALE,(np.einsum('ij,jk,ik->i',x,self.C,x)+self.u)*SCALE**2
    def density_score(self,x,r,mu=None):
        # Woodbury and determinant lemma, without constructing an N by N inverse.
        w,q=np.linalg.eigh(self.C);a=x@(q*np.sqrt(np.maximum(w,0)))
        mid=np.eye(a.shape[1])+a.T@a/self.u
        e=r/SCALE-(x@self.b if mu is None else mu/SCALE)
        z=a.T@e
        quad=e@e/self.u-z@np.linalg.solve(mid,z)/self.u**2
        logdet=len(x)*np.log(self.u)+np.linalg.slogdet(mid)[1]+2*len(x)*np.log(SCALE)
        return .5*(logdet+quad+len(x)*np.log(2*np.pi))/len(x)
