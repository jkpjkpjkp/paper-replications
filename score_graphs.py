#!/usr/bin/env python3
"""Chronological monthly EB graph-anomaly application to observed disclosed customers.
No stock return, price, future label, or test-dependent tuning enters these scores.
"""
import json,os,time,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.stats import kstest
from sklearn.neighbors import LocalOutlierFactor
from threadpoolctl import threadpool_limits
threadpool_limits(2)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'scores';OUT.mkdir(exist_ok=True)
KAPPAS=[0,.001,.003,.01,.03,.1,.3,.5,1,2,3,5,7,10,15,20]
GAMMAS=np.array([.1,.2,.5,.7,1.,2.,5.])
edges=pd.read_parquet(ROOT/'data/edges.parquet');records=pd.read_parquet(ROOT/'data/matched.parquet',columns=['supplier_permno','period_end','filing_date'])
records=records.dropna().drop_duplicates();records['supplier_permno']=records.supplier_permno.astype('int64')
for df in [edges,records]:
 for c in ['period_end','filing_date']:df[c]=pd.to_datetime(df[c])
# All parameters are frozen in analysis_config.json before reading outcomes.
def snap(date):
 valid=records[(records.filing_date<date)&(records.filing_date>=date-pd.Timedelta(days=540))&(records.period_end>=date-pd.Timedelta(days=730))]
 latest=valid.sort_values(['period_end','filing_date']).drop_duplicates('supplier_permno',keep='last')
 return edges.merge(latest,on=['supplier_permno','period_end','filing_date'],how='inner').drop_duplicates(['supplier_permno','customer_permno'])
def fit(S,lam,d):
 best={'ml':-np.inf}
 for k in KAPPAS:
  def ml(r):
   q=r*(k*k+lam)+(1-r)
   if np.any(q<=0):return -np.inf
   return float(-.5*np.dot(q,S)+d/2*np.log(q).sum())
  r=minimize_scalar(lambda r:-ml(r),bounds=(0,1),method='bounded',options={'xatol':1e-7})
  for rho in [0.,1.,float(r.x)]:
   v=ml(rho)
   if v>best['ml']:best={'ml':v,'rho':rho,'kappa':k}
 return best
summary=[]
for date in pd.date_range('2004-01-31','2024-12-31',freq='ME'):
 target=OUT/(date.strftime('%Y-%m-%d')+'.parquet');meta=target.with_suffix('.json')
 if target.exists() and meta.exists():summary.append(json.loads(meta.read_text()));continue
 start=time.time();e=snap(date);prev=snap(date-pd.DateOffset(years=1))
 if len(e)<50:raise RuntimeError('Insufficient relationship graph '+str(date))
 nodes=np.union1d(e.supplier_permno.unique(),e.customer_permno.unique()).astype('int64');n=len(nodes);loc={int(x):i for i,x in enumerate(nodes)}
 src=e.supplier_permno.map(loc).to_numpy();dst=e.customer_permno.map(loc).to_numpy()
 A=np.zeros((n,n),np.float64);A[src,dst]=1;A[dst,src]=1;np.fill_diagonal(A,0);deg=A.sum(1)
 inv=np.zeros(n);inv[deg>0]=1/np.sqrt(deg[deg>0]);L=np.eye(n)-(inv[:,None]*A)*inv[None,:]
 oldpairs=set(zip(prev.supplier_permno.astype(int),prev.customer_permno.astype(int)))
 out=np.bincount(src,minlength=n);inc=np.bincount(dst,minlength=n)
 priorout=prev.groupby('supplier_permno').size().to_dict();priorin=prev.groupby('customer_permno').size().to_dict()
 novel=np.bincount(src,weights=[(int(s),int(c)) not in oldpairs for s,c in zip(e.supplier_permno,e.customer_permno)],minlength=n)/np.maximum(out,1)
 outchange=np.array([(out[i]-priorout.get(int(p),0))/(1+priorout.get(int(p),0)) for i,p in enumerate(nodes)])
 inchange=np.array([(inc[i]-priorin.get(int(p),0))/(1+priorin.get(int(p),0)) for i,p in enumerate(nodes)])
 sal=e.max_reported_customer_sales.clip(lower=0).fillna(0);sals=pd.DataFrame({'src':src,'sales':sal.to_numpy()}).groupby('src').sales.agg(['sum','max'])
 concentration=np.zeros(n);idx=sals.index.to_numpy();concentration[idx]=sals['max']/np.maximum(sals['sum'],1e-8)
 X=np.column_stack([np.log1p(out),np.log1p(inc),novel,outchange,inchange,concentration])
 mu=X.mean(0);sd=np.maximum(X.std(0,ddof=1),1e-8);Z=(X-mu)/sd
 nz=np.linalg.norm(Z,axis=1);ei,ej=np.where(np.triu(A)>0);h=float(np.mean(np.sum(Z[ei]*Z[ej],axis=1)/np.maximum(nz[ei]*nz[ej],1e-12)))
 de=float(len(ei)/n);center=float(np.clip((.5+h)/np.sqrt(max(de,10)/10),.05,10));cand=GAMMAS[GAMMAS<=2*center]
 if len(cand)==0:cand=GAMMAS[:1]
 grid=np.sort(cand[np.argsort(np.abs(np.log(cand/center)))[:3]]);ga=float(grid[np.argmin(np.abs(np.log(grid/center)))])
 # Full graph and full spectrum: no subsampling. Symmetric double precision.
 lam,V=eigh(L,check_finite=False,driver='evr');rawmin=float(lam.min());lam=np.maximum(lam,0)
 delta=Z-V@((1/(ga*ga+lam))[:,None]*(V.T@Z));S=((V.T@delta)**2).sum(1);f=fit(S,lam,Z.shape[1]);q=f['rho']*(f['kappa']**2+lam)+1-f['rho']
 J=np.sum((V@(np.sqrt(np.maximum(q,1e-12))[:,None]*(V.T@delta)))**2,axis=1);R=J/np.maximum(np.sum(delta**2,axis=1),1e-12)
 def nullks(s):
  s=s[s>0];m=s.mean();v=s.var()
  if len(s)<10 or m<1e-12 or v<1e-12:return 0.
  return float(kstest(s/(v/(2*m)),'chi2',args=(max(.01,2*m*m/v),)).statistic)
 kj,kr=nullks(J),nullks(R);selected='J' if kj>=kr else 'R';score=J if selected=='J' else R
 clf=LocalOutlierFactor(n_neighbors=20,n_jobs=1).fit(Z);lof=-clf.negative_outlier_factor_
 # Independent equation/accounting checks, spectral residual and no input label check.
 er=float(np.max(np.linalg.norm(L@V-V*lam,axis=0)));orth=float(np.linalg.norm(V.T@V-np.eye(n)))
 assert er<1e-8 and orth<1e-7 and np.isfinite(score).all()
 assert np.max(np.abs(J.sum()-np.dot(q,S)))/max(J.sum(),1)<1e-9
 frame=pd.DataFrame({'signal_calendar_date':date,'permno':nodes,'is_supplier':out>0,'eb_selected':score,'eb_J':J,'eb_R':R,'feature_LOF':lof,'degree_change':np.abs(outchange),'out_degree':out,'in_degree':inc,'novel_fraction':novel,'out_degree_change':outchange,'in_degree_change':inchange,'disclosed_customer_concentration':concentration})
 tmp=target.with_suffix('.parquet.part');frame.to_parquet(tmp,index=False);tmp.replace(target)
 d={'date':str(date.date()),'n_nodes':n,'n_edges_undirected':len(ei),'n_suppliers':int((out>0).sum()),'full_spectrum':True,'eigen_residual_max':er,'orthogonality_frobenius':orth,'raw_min_eigenvalue':rawmin,'anchor_gamma':ga,'gamma0':center,'grid':grid.tolist(),'fit':f,'selected':selected,'NullKS_J':kj,'NullKS_R':kr,'seconds':time.time()-start,'max_input_filing_date':str(e.filing_date.max().date()),'no_price_inputs':True,'source_data_sha256':hashlib.sha256((ROOT/'data/edges.parquet').read_bytes()).hexdigest()}
 assert pd.Timestamp(d['max_input_filing_date'])<date
 meta.write_text(json.dumps(d,indent=2)+'\n');summary.append(d)
 (ROOT/'results/score_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(date.date(),n,int((out>0).sum()),selected,'seconds',round(d['seconds'],2),flush=True)
print('SCORE_COMPLETE',len(summary),flush=True)
