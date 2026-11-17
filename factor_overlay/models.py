"""Small deterministic regression library; penalties fixed before evaluation.

No external ML dependency. All preprocessing and conditional variance fits are
estimated using the supplied historical training rows only.
"""
import numpy as np

METHODS = ['ols', 'ridge', 'lasso', 'elastic_net', 'huber', 'quadratic_ridge']

def solve(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-10)[0]

class Regression:
    def __init__(self, method):
        self.method=method

    def transform(self,x):
        z=np.clip((np.where(np.isfinite(x),x,self.median)-self.center)/self.scale,-6,6)
        if self.method=='quadratic_ridge':
            # Main effects and all quadratic interactions, deterministic basis.
            z=np.column_stack([z,z[:,self.ii]*z[:,self.jj]])
        return np.column_stack([np.ones(len(z)),z])

    def fit(self,x,y,weight,variance_horizon=1):
        self.median=np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0)
        self.median=np.nan_to_num(self.median)
        q=np.where(np.isfinite(x),x,self.median)
        w=np.asarray(weight,float);w=w/w.sum()
        self.center=w@q
        self.scale=np.sqrt(w@((q-self.center)**2));self.scale[self.scale<1e-6]=1
        if self.method=='quadratic_ridge':
            # Restrict to squares and each feature x leg/factor/momentum/risk.
            pairs=sorted(set([(i,i) for i in range(q.shape[1])]+[(min(i,j),max(i,j)) for i in range(q.shape[1]) for j in [0,1,2,6]]))
            self.ii=np.array([i for i,j in pairs]);self.jj=np.array([j for i,j in pairs])
        a=self.transform(q)
        # Training labels capped at +/-20% per day; evaluation P&L never capped.
        y_raw=np.asarray(y,float)
        y=np.clip(y_raw,-.2,.2)
        g=a.T@(w[:,None]*a);h=a.T@(w*y)
        penalty=np.eye(a.shape[1]);penalty[0,0]=0
        self.iterations=1;self.converged=True
        if self.method in ['ols','ridge','quadratic_ridge']:
            alpha=0 if self.method=='ols' else .1
            self.beta=solve(g+alpha*penalty,h)
        elif self.method in ['lasso','elastic_net']:
            l1=.0001 if self.method=='lasso' else .00005
            l2=0 if self.method=='lasso' else .02
            b=np.zeros(a.shape[1]);self.converged=False
            for it in range(2000):
                old=b.copy()
                for j in range(len(b)):
                    u=h[j]-g[j]@b+g[j,j]*b[j]
                    b[j]=u/max(g[j,j],1e-12) if j==0 else np.sign(u)*max(abs(u)-l1,0)/max(g[j,j]+l2,1e-12)
                if np.max(abs(b-old))<1e-8:
                    self.converged=True;break
            self.beta=b;self.iterations=it+1
        elif self.method=='huber':
            b=solve(g+.05*penalty,h)
            scale=max(np.median(abs(y-a@b))*1.4826,.001)
            self.converged=False
            for it in range(30):
                resid=y-a@b;wh=w*np.minimum(1,1.345*scale/np.maximum(abs(resid),1e-12))
                bn=solve(a.T@(wh[:,None]*a)+.05*penalty,a.T@(wh*y))
                if np.max(abs(bn-b))<1e-8:self.converged=True;b=bn;break
                b=bn
            self.beta=b;self.iterations=it+1
        else:raise ValueError(self.method)
        # Conditional risk: log residual-square ridge, calibrated to arithmetic
        # residual variance, then shrunk 50% to observed lagged stock variance.
        residual=(y_raw-np.clip(a@self.beta,-.01,.01))*np.sqrt(variance_horizon)
        av=a[:,:x.shape[1]+1]
        pv=np.eye(av.shape[1]);pv[0,0]=0
        self.var_beta=solve(av.T@(w[:,None]*av)+.1*pv,av.T@(w*np.log(np.maximum(residual**2,1e-7))))
        vv=np.exp(np.clip(av@self.var_beta,-14,-2))
        self.var_calibration=float(np.clip((w@(residual**2))/(w@vv),.1,20))
        self.n_train=len(y)
        return self

    def predict(self,x):
        a=self.transform(x)
        mu=np.clip(a@self.beta,-.01,.01)
        variance=np.exp(np.clip(a[:,:len(self.var_beta)]@self.var_beta,-14,-2))*self.var_calibration
        lag_sigma=np.exp(np.where(np.isfinite(x[:,2]),x[:,2],self.median[2]))
        variance=np.clip(.5*variance+.5*lag_sigma**2,1e-5,.04)
        return mu,np.sqrt(variance)

    def audit(self):
        return dict(method=self.method,n_train=self.n_train,n_parameters=len(self.beta),iterations=self.iterations,converged=self.converged,coefficient_max=float(abs(self.beta).max()),variance_calibration=self.var_calibration)
