#!/usr/bin/env python3
"""Independent numerical checks against SciPy and statsmodels optimizers."""
from pathlib import Path
import hashlib
import json
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from models import Regression, METHODS

HERE=Path(__file__).resolve().parent


def run():
    rng=np.random.default_rng(153910)
    x=rng.normal(size=(1200,7))
    x[:,2]=np.log(.025)+.2*x[:,2]
    x[:,6]=rng.choice([-1.,1.],len(x))
    y=.0008+.0011*x[:,0]-.0008*x[:,1]+.0005*x[:,6]+rng.normal(0,.014,len(x))
    weight=rng.uniform(.5,3,len(x))
    w=weight/weight.sum()
    checks={}
    models={}
    for method in METHODS:
        models[method]=Regression(method).fit(x,y,weight)
    a=models['ols'].transform(x)
    ols=sm.WLS(y,a,weights=weight).fit()
    err=float(np.max(np.abs(ols.params-models['ols'].beta)))
    assert err<1e-11,err
    checks['ols_vs_statsmodels_wls']={'max_abs_coefficient_error':err,'passed':True}

    for method in ['lasso','elastic_net']:
        m=models[method];a=m.transform(x)
        l1=.0001 if method=='lasso' else .00005
        l2=0 if method=='lasso' else .02
        p=a.shape[1]-1
        def unpack(v):
            return np.r_[v[0],v[1:1+p]-v[1+p:]]
        def obj(v):
            b=unpack(v);r=a@b-y
            value=.5*(w@(r*r))+l1*np.sum(v[1:])+.5*l2*(b[1:]@b[1:])
            g=a.T@(w*r)+np.r_[0,l2*b[1:]]
            return value,np.r_[g[0],g[1:]+l1,-g[1:]+l1]
        result=minimize(obj,np.zeros(1+2*p),jac=True,method='L-BFGS-B',
                        bounds=[(None,None)]+[(0,None)]*(2*p),
                        options={'ftol':1e-15,'gtol':1e-12,'maxiter':10000,'maxls':50})
        b=unpack(result.x)
        diff=float(np.max(np.abs(b-m.beta)))
        residual=a@m.beta-y
        grad=a.T@(w*residual)+np.r_[0,l2*m.beta[1:]]
        active=np.abs(m.beta[1:])>1e-8
        kkt=max(abs(grad[0]),
                np.max(np.abs(grad[1:][active]+l1*np.sign(m.beta[1:][active]))) if active.any() else 0,
                np.max(np.maximum(np.abs(grad[1:][~active])-l1,0)) if (~active).any() else 0)
        assert result.success,result.message
        assert diff<1e-6,diff
        assert kkt<1e-7,kkt
        checks[f'{method}_vs_scipy']={'max_abs_coefficient_error':diff,'max_kkt_violation':float(kkt),
                                    'objective':'0.5*weighted_MSE + l1*sum(abs(slopes)) + 0.5*l2*sum(slopes^2)',
                                    'l1':l1,'l2':l2,'passed':True}

    # Huber threshold is frozen from the initial ridge fit, matching IRLS.
    m=models['huber'];a=m.transform(x)
    penalty=np.eye(a.shape[1]);penalty[0,0]=0
    start=np.linalg.solve(a.T@(w[:,None]*a)+.05*penalty,a.T@(w*y))
    scale=max(np.median(abs(y-a@start))*1.4826,.001)
    delta=1.345*scale
    def huber_obj(b):
        r=a@b-y;ar=abs(r)
        loss=np.where(ar<=delta,.5*r*r,delta*(ar-.5*delta))
        value=w@loss+.5*.05*(b[1:]@b[1:])
        gradient=a.T@(w*np.clip(r,-delta,delta))+.05*penalty@b
        return value,gradient
    opt=minimize(huber_obj,start,jac=True,method='BFGS',options={'gtol':1e-12,'maxiter':2000})
    err=float(np.max(abs(opt.x-m.beta)))
    assert err<1e-6,err
    checks['huber_vs_scipy']={'max_abs_coefficient_error':err,'fixed_delta':float(delta),'passed':True}

    # Every baseline feature and its deterministic polynomial term is retained
    # when adding cycle features. Neither scaling nor imputation changes it.
    cycle=rng.normal(size=(len(x),24));aug=np.column_stack([x,cycle])
    base=Regression('quadratic_ridge').fit(x,y,weight)
    enhanced=Regression('quadratic_ridge').fit(aug,y,weight)
    assert np.array_equal(base.median,enhanced.median[:7])
    assert np.allclose(base.center,enhanced.center[:7],atol=1e-14)
    assert np.allclose(base.scale,enhanced.scale[:7],atol=1e-14)
    ab=base.transform(x);ae=enhanced.transform(aug)
    pairs_b=list(zip(base.ii,base.jj));pairs_e=list(zip(enhanced.ii,enhanced.jj))
    mapping=[0]+list(range(1,8))+[1+31+pairs_e.index(p) for p in pairs_b]
    err=float(np.max(abs(ab-ae[:,mapping])))
    assert err<1e-12,err
    checks['quadratic_baseline_nested_in_augmented']={'base_parameters':len(base.beta),
        'augmented_parameters':len(enhanced.beta),'shared_basis_error':err,'passed':True}

    # Prediction is pure: future data never modify fitted preprocessing.
    xm=x.copy();xm[::17,1]=np.nan;xm[::37,3]=np.inf
    m=Regression('ridge').fit(xm,y,weight)
    snapshot={k:v.copy() for k,v in m.__dict__.items() if isinstance(v,np.ndarray)}
    extreme=np.full((10,7),1e8);extreme[:,2]=np.log(.03);extreme[0]=np.nan
    m.predict(extreme)
    assert all(np.array_equal(m.__dict__[k],v,equal_nan=True) for k,v in snapshot.items())
    expected=np.nanmedian(np.where(np.isfinite(xm),xm,np.nan),axis=0)
    assert np.array_equal(m.median,expected)
    checks['preprocessing_training_only']={'imputer_matches_training_medians':True,
       'future_predict_leaves_all_fitted_arrays_unchanged':True,
       'caller_run_py_review':'fit_all uses year-10 <= training_year < prediction_year and asserts maximum training_year < prediction_year; fixed annual models loaded for evaluate',
       'passed':True}

    m=models['ridge'];a=m.transform(x);residual=y-np.clip(a@m.beta,-.01,.01)
    av=a[:,:8];raw=np.exp(np.clip(av@m.var_beta,-14,-2))
    calibration=float(np.clip((w@(residual**2))/(w@raw),.1,20))
    assert abs(calibration-m.var_calibration)<1e-12
    expected=np.clip(.5*raw*calibration+.5*np.exp(x[:,2])**2,1e-5,.04)
    mu,sigma=m.predict(x)
    err=float(np.max(abs(sigma**2-expected)))
    assert err<1e-14,err
    checks['variance_formula_and_calibration']={'calibration':calibration,
        'mean_before_shrink':float(w@(raw*calibration)),'training_residual_mean_square':float(w@(residual**2)),
        'prediction_variance_error':err,'passed':True}

    # Guard against using capped mean-regression labels to estimate tail risk.
    tail_y=y.copy();tail_y[0]=.30;tail_y[1]=-.30
    tm=Regression('ridge').fit(x,tail_y,weight)
    cm=Regression('ridge').fit(x,np.clip(tail_y,-.2,.2),weight)
    assert np.max(abs(tm.beta-cm.beta))<1e-12
    ta=tm.transform(x);tr=tail_y-np.clip(ta@tm.beta,-.01,.01)
    va=ta[:,:8];penalty=np.eye(8);penalty[0,0]=0
    expected_beta=np.linalg.solve(va.T@(w[:,None]*va)+.1*penalty,
                                  va.T@(w*np.log(np.maximum(tr**2,1e-7))))
    err=float(np.max(abs(expected_beta-tm.var_beta)))
    assert err<1e-10,err
    raw=np.exp(np.clip(va@tm.var_beta,-14,-2))
    expected_cal=float(np.clip((w@(tr**2))/(w@raw),.1,20))
    assert abs(expected_cal-tm.var_calibration)<1e-10
    assert np.mean(tm.predict(x)[1]**2)>np.mean(cm.predict(x)[1]**2)
    checks['raw_tail_residual_guard']={'mean_coefficients_unchanged_by_equivalent_clip':True,
        'variance_beta_matches_raw_residuals_error':err,
        'variance_calibration_matches_raw_residuals':True,
        'raw_tail_mean_predicted_variance':float(np.mean(tm.predict(x)[1]**2)),
        'capped_tail_mean_predicted_variance':float(np.mean(cm.predict(x)[1]**2)),
        'passed':True}

    report={'created_by':'independent data/model audit agent','all_numerical_checks_passed':True,
            'model_source_sha256':hashlib.sha256((HERE/'models.py').read_bytes()).hexdigest(),
            'checks':checks,'synthetic_model_audits':[m.audit() for m in models.values()],
            'limitations':['Models are predictive estimators; these checks do not establish investment alpha.',
               'Diagonal conditional variance does not model cross-stock covariance.',
               'Variance calibration is deliberately bounded0.1..20; shrinkage and final bounds mean final forecasts need not exactly match training residual variance.',
               'OLS uses normalized weighted normal equations with SVD tolerance1e-10, so nearly singular columns may be dropped.',
               'The optimizer scales one-day conditional mean/variance by five for weekly positions; persistence and zero serial covariance are approximations, not separately learned five-day forecasts.'],
            'variance_tail_note':'Confirmed final models.py retains raw uncapped training returns for variance residuals; only mean-regression fitting labels are capped.'}
    actual=[]
    for path in sorted((HERE/'fits').glob('*.json')):
        actual.extend(json.loads(path.read_text()))
    if actual:
        assert all(r['training_last_year']==r['prediction_year']-1 and
                   r['training_first_year']==r['prediction_year']-10 for r in actual)
        assert all(r['converged'] for r in actual)
        report['actual_fit_audit']={'records':len(actual),'expected_for_complete_study':21*6*6*2,
            'all_converged':True,'all_training_windows_strictly_precede_prediction_year':True,
            'min_training_rows':min(r['n_train'] for r in actual),
            'max_training_rows':max(r['n_train'] for r in actual),
            'variance_calibration_min':min(r['variance_calibration'] for r in actual),
            'variance_calibration_max':max(r['variance_calibration'] for r in actual)}
    (HERE/'model_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    run()
