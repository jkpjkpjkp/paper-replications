"""Frozen-validation model selection and annual rolling 120-month walk-forward."""
from pathlib import Path
import os,json,time,gc,argparse
import numpy as np
import pandas as pd
from models import Transform,Independent,Trees,NGB,Net,Recursion,CMK

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
META=json.loads((ROOT/'data/metadata.json').read_text())
FEATURES=META['features']
CF=[f for f in FEATURES if f.startswith('rank_')]

def phi(df):return np.column_stack([np.ones(len(df)),df[CF].fillna(0).to_numpy()*np.sqrt(12)])
def scores(df,m,v):
    mse=(df.yret.to_numpy()-m)**2;rv=df.yrv.to_numpy()
    return dict(mse=mse.mean(),qlike=(np.log(v)+rv/v).mean(),nll=(.5*(np.log(2*np.pi*v)+mse/v)).mean())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['tune','test'],required=True)
    ap.add_argument('--start-year',type=int,default=2010);ap.add_argument('--end-year',type=int,default=2025);args=ap.parse_args()
    full=pd.read_parquet(ROOT/'data/panel.parquet')
    df=full[full.variance_observed].copy()
    if args.stage=='tune':
        tr=df[df.year.between(1995,2004)];va=df[df.year.between(2005,2009)]
        tf=Transform().fit(tr[FEATURES]);x=tf(tr[FEATURES]);z=tf(va[FEATURES])
        r=tr.yret.to_numpy();v=tr.yrv.to_numpy();records=[];selection={}
        for kind,grid in [('ridge',[10,1000,100000]),('hgb',[100,250])]:
            opts=[]
            for par in grid:
                mod=Independent(kind,par).fit(x,r,v);m,h=mod.predict(z);s=scores(va,m,h)
                s.update(model=kind,param=par);records.append(s);opts.append(s)
                print('tune',s,flush=True)
            selection[kind]=[min(opts,key=lambda s:s['mse'])['param'],min(opts,key=lambda s:s['qlike'])['param']]
        opts=[]
        for ep in [5,15,35]:
            mod=Net('independent',epochs=ep).fit(x,r,v);m,h=mod.predict(z);s=scores(va,m,h)
            s.update(model='nn_ind',param=ep);records.append(s);opts.append(s);print('tune',s,flush=True)
        selection['nn_ind_epochs']=[min(opts,key=lambda s:s['mse'])['param'],min(opts,key=lambda s:s['qlike'])['param']]
        opts=[]
        for ep in [10,35]:
            for w in [.1,1.,10.]:
                mod=Net('joint',w,epochs=ep).fit(x,r,v);m,h=mod.predict(z);s=scores(va,m,h)
                s.update(model='nn_joint',param=w,epochs=ep,objective=s['mse']/np.mean(r*r)+s['qlike'])
                records.append(s);opts.append(s);print('tune',s,flush=True)
        chosen=min(opts,key=lambda s:s['objective'])
        selection['nn_weight']=chosen['param'];selection['nn_joint_epochs']=chosen['epochs']
        opts=[]
        for n in [100,250]:
            mod=NGB(n).fit(x,r,v);m,h=mod.predict(z);s=scores(va,m,h)
            s.update(model='ngboost',param=n);opts.append(s);records.append(s);print('tune',s,flush=True)
        selection['ngboost']=min(opts,key=lambda s:s['nll'])['param']
        opts=[];pt=phi(tr);pv=phi(va)
        for eta in [0.,1.,10.]:
            for penalty in [0.,.1,1.]:
                mod=CMK(eta,penalty).fit(pt,r,tr.target_month.to_numpy())
                vals=[]
                for month in va.target_month.unique():
                    mask=(va.target_month==month).to_numpy();vals.append(mod.density_score(pv[mask],va.yret.to_numpy()[mask]))
                s={'model':'cmk','eta':eta,'penalty':penalty,'multivariate_nll':np.mean(vals)}
                opts.append(s);records.append(s);print('tune',s,flush=True)
        selection['cmk']=min(opts,key=lambda s:s['multivariate_nll'])
        pd.DataFrame(records).to_csv(OUT/'validation_scores.csv',index=False)
        selection.update(protocol={'train_tuning':'1995-2004','validation':'2005-2009','test':'2010-2025',
            'rolling_training_years':10,'refit':'annual','features':FEATURES,'neural_seeds':[11,29,47],
            'neural_epochs':35,'neural_train_cap':60000,'ngboost_train_cap':40000,
            'labels':'returns and realized variance untrimmed','test_selection':'none',
            'universe':'formation-month largest 1000 eligible stocks','cmk_features':['intercept']+CF})
        (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
        return
    selection=json.loads((OUT/'selection.json').read_text());logs=[]
    for year in range(args.start_year,args.end_year+1):
        target=OUT/f'predictions-{year}.parquet'
        if target.exists(): print('skip complete',year,flush=True);continue
        start=time.time();tr=df[df.year.between(year-10,year-1)];te=full[(full.year==year)&full.return_observed].copy()
        tf=Transform().fit(tr[FEATURES]);x=tf(tr[FEATURES]);z=tf(te[FEATURES]);r=tr.yret.to_numpy();v=tr.yrv.to_numpy()
        assert tr.target_month.max()<te.target_month.min()
        preds=[];fitted={};covscores=[]
        def save(name,m,h,extra=None):
            assert len(m)==len(te) and np.isfinite(m).all() and np.isfinite(h).all() and (h>0).all()
            p=te[['permno','target_month','date','year','yret','yrv','variance_observed','size_rank','cap','ydelist']].copy()
            p['model']=name;p['mu']=m;p['variance']=h
            if extra is not None:p['exact_nll']=extra
            preds.append(p);fitted[name]=(m,h)
            mask=te.variance_observed.to_numpy();print(year,name,scores(te[mask],m[mask],h[mask]),'seconds',round(time.time()-start),flush=True)
        save('zero_persistence',np.zeros(len(te)),np.exp(te.log_rv.to_numpy()))
        save('mean_rv3',np.full(len(te),r.mean()),np.exp(te.log_rv3.fillna(te.log_rv).to_numpy()))
        for kind in ['ridge','hgb']:
            aa,bb=selection[kind];mod=Independent(kind,aa).fit(x,r,v);m,h=mod.predict(z)
            if aa!=bb:h=Independent(kind,bb).fit(x,r,v).predict(z)[1]
            save('ind_'+kind,m,h)
        for joint in [False,True]:
            mod=Trees(joint).fit(x,r,v);save('trees_joint' if joint else 'trees_ind',*mod.predict(z));del mod;gc.collect()
        for name,kind,w in [('nn_ind','independent',1.),('nn_joint','joint',selection['nn_weight']),('nn_gaussian','gaussian',1.),('nn_student','student',1.)]:
            ep=tuple(selection['nn_ind_epochs']) if name=='nn_ind' else selection['nn_joint_epochs'] if name=='nn_joint' else 35
            mod=Net(kind,w,epochs=ep).fit(x,r,v);individual=mod.predict(z,individual=True)
            save(name,*tuple(np.mean([p[j] for p in individual],axis=0) for j in range(2)))
            if name in ['nn_ind','nn_joint']:
                for seed,p in zip([11,29,47],individual):save(f'{name}_seed{seed}',*p)
            del mod;gc.collect()
        mod=NGB(selection['ngboost']).fit(x,r,v);save('ngboost',*mod.predict(z));del mod;gc.collect()
        # Validation-free shrinkage/ensembles fixed before viewing test results.
        save('half_joint_half_hgb',.5*(fitted['nn_joint'][0]+fitted['ind_hgb'][0]),.5*(fitted['nn_joint'][1]+fitted['ind_hgb'][1]))
        save('half_ind_half_hgb',.5*(fitted['nn_ind'][0]+fitted['ind_hgb'][0]),.5*(fitted['nn_ind'][1]+fitted['ind_hgb'][1]))
        save('hgb_mean_joint_var',fitted['ind_hgb'][0],fitted['nn_joint'][1])
        for sharv in [False,True]:
            mod=Recursion(sharv);out=mod.fit_predict(tr,te)
            save('sharv' if sharv else 'garch',out.mu.to_numpy(),out.variance.to_numpy(),out.exact_nll.to_numpy())
            logs.append({'year':year,'model':'sharv' if sharv else 'garch',**mod.meta})
        pt=phi(tr);pz=phi(te);hp=selection['cmk']
        for name,control,onestep,eta in [('factor_ind',True,False,0),('cmk',False,False,hp['eta']),('cmk_onestep',False,True,hp['eta']),('cmk_no_floor',False,False,0)]:
            mod=CMK(eta,hp['penalty'],control,onestep).fit(pt,r,tr.target_month.to_numpy());save(name,*mod.predict(pz))
            for month in te.target_month.unique():
                ix=(te.target_month==month).to_numpy();y=te.yret.to_numpy()[ix];p=pz[ix]
                covscores.append({'year':year,'month':month,'model':name,'nll':mod.density_score(p,y),
                    'nll_hgb_mean':mod.density_score(p,y,fitted['ind_hgb'][0][ix]),'n':sum(ix)})
        pd.concat(preds,ignore_index=True).to_parquet(target,index=False)
        pd.DataFrame(covscores).to_csv(OUT/f'covariance-{year}.csv',index=False)
        (OUT/f'recursion-{year}.json').write_text(json.dumps([l for l in logs if l['year']==year],indent=2))
        print('COMPLETE YEAR',year,'seconds',round(time.time()-start),flush=True)
        gc.collect()
if __name__=='__main__':main()
