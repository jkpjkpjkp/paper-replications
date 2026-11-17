"""Isolated CZ vocabulary/data adapter; reuse original expression and mining code."""
from pathlib import Path
import hashlib, json, re, shutil, sys
import numpy as np
import pandas as pd
import torch

ROOT=Path('/home/jkp/data')
HERE=Path(__file__).resolve().parent
BASE=ROOT/'ordered-reproductions'
NAMES=json.loads((BASE/'data/cz-monthly/manifest.json').read_text())['features']
MASKS={}

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)

def prepare(project):
    origin=ROOT/project/'upstream'
    runtime=HERE/'runtime'/project
    if not runtime.exists():
        shutil.copytree(origin/'src',runtime/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for p in origin.glob('*.py'):shutil.copy2(p,runtime/p.name)
        # Feature identifiers are genuine CZ names, never aliases for price or volume.
        p=runtime/'src/alphagen_qlib/stock_data.py';s=p.read_text()
        s=re.sub(r'class FeatureType\(IntEnum\):.*?(?=\ndef )',
                 'class FeatureType(IntEnum):\n'+''.join(f'    {n.upper()} = {i}\n' for i,n in enumerate(NAMES))+'\n',s,flags=re.S)
        p.write_text(s)
        for rel in ['src/alphagen/config.py','src/alpha_gfn/config.py']:
            p=runtime/rel
            if p.exists():
                s=p.read_text();s=re.sub(r'FEATURES\s*=\s*\[.*?\]', 'FEATURES = list(FeatureType)',s,flags=re.S)
                # Monthly lag vocabulary fixed before training.
                s=re.sub(r'DELTA_TIMES\s*=\s*\[.*?\]','DELTA_TIMES = [1, 3, 6, 12, 24]',s,flags=re.S);p.write_text(s)
        p=runtime/'src/alphagen/data/tree.py'
        if p.exists():
            s=p.read_text();s=re.sub(r'feature_map\s*=\s*\{.*?\}',"feature_map = {'$'+f.name.lower(): f for f in FeatureType}",s,flags=re.S);p.write_text(s)
        write_json(runtime/'source_manifest.json',{str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.rglob('*.py')})
    sys.path[:0]=[str(runtime/'src'),str(runtime)]
    # Preserve average-tie ranks without the released O(stocks^2) allocation.
    import alphagen.utils.correlation as correlation
    from scipy.stats import rankdata
    original=correlation.batch_pearsonr
    def corr(x,y):
        mask=MASKS.get(tuple(x.shape))
        if mask is not None:
            mask=mask.to(x.device);x=x.masked_fill(~mask,float('nan'));y=y.masked_fill(~mask,float('nan'))
        return original(x,y)
    correlation.batch_pearsonr=corr
    def rank(x,mask):
        a=x.detach().cpu().numpy().copy();m=mask.cpu().numpy();a[m]=np.nan
        r=rankdata(a,axis=1,method='average',nan_policy='omit');r[m]=0
        return torch.as_tensor(r,dtype=x.dtype,device=x.device)
    correlation._rank_data=rank
    return runtime

class CZData:
    def __init__(self,instrument='cz',start_time='2009-01-01',end_time='2019-12-31',**kwargs):
        self.max_backtrack_days=100;self.max_future_days=0;self.device=torch.device('cpu')
        self.raw=False
        start=pd.Period(start_time,freq='M');end=pd.Period(end_time,freq='M')
        calendar=pd.period_range(start-100,end,freq='M')
        self._dates=calendar.to_timestamp('M');self.dates=self._dates[100:]
        self.n_days=len(self.dates)
        panel=pd.read_parquet(BASE/'data/cz-monthly/panel.parquet')
        # Asset IDs may include future entrants; masked cells never enter scoring or losses.
        self._stock_ids=pd.Index(sorted(panel.permno.unique()))
        self.n_stocks=len(self._stock_ids)
        codes=np.array([p.year*100+p.month for p in calendar])
        index=pd.MultiIndex.from_product([codes,self._stock_ids],names=['return_month','permno'])
        frame=panel.set_index(['return_month','permno']).reindex(index)
        values=frame[NAMES].to_numpy(dtype=np.float32).reshape(len(codes),self.n_stocks,202).transpose(0,2,1).copy()
        self.data=torch.from_numpy(values)
        self.y=torch.from_numpy(frame.target_return.to_numpy(dtype=np.float32).reshape(len(codes),self.n_stocks)[100:].copy())
        self.eligible=torch.from_numpy(frame.lagged_market_cap.notna().to_numpy().reshape(len(codes),self.n_stocks)[100:].copy())
        self.codes=codes[100:]
        MASKS[(self.n_days,self.n_stocks)]=self.eligible
        if start.year>=2009:assert torch.isfinite(self.y[self.eligible]).all()

def target_class():
    from alphagen.data.expression import Expression
    class ForwardReturn(Expression):
        def evaluate(self,data,period=slice(0,1)):
            assert period==slice(0,1)
            return data.y
        @property
        def is_featured(self):return True
        def __str__(self):return 'NextMonthReturn'
    return ForwardReturn

def protocol(project,out):
    write_json(out/'protocol.json',dict(project=project,features=NAMES,train=[200901,201912],validation=[202001,202112],test=[202201,202512],
        scope='User-authorized monthly CZ202 adaptation',universe='All eligible US common stocks; lagged NYSE20',
        input='CZ ranks as of previous month; neutral imputation inherited from verified panel',
        label='Realized next-month total return; not an expression terminal',
        adaptation='202 CZ terminals replace 6 OHLCV terminals. Rolling lag units become months. Historical eligibility is masked in all training correlations.',
        selection='Fixed seed0; no test-based choice; frozen mining then causal expanding factor OLS',
        cost_bps=10,source_panel=str(BASE/'data/cz-monthly/panel.parquet'),
        source_manifest=str(HERE/'runtime'/project/'source_manifest.json')))

def evaluate_pool(project,out,n_factors,verify_only=False):
    from alphagen.data.expression import Feature,FeatureType
    import alphagen.data.expression as exprs
    from alphagen.models.alpha_pool import AlphaPool
    from scipy.stats import rankdata
    data=CZData(start_time='2009-01-01',end_time='2025-12-31')
    namespace={k:getattr(exprs,k) for k in dir(exprs) if not k.startswith('_')}
    namespace.update({f.name.lower():Feature(f) for f in FeatureType})
    expressions=json.loads((out/'pool.json').read_text())['exprs'];assert expressions
    arrays=[];daily=[]
    for text in expressions:
        expr=eval(text.replace('$',''),{'__builtins__':{}},namespace)
        raw=expr.evaluate(data);raw[~data.eligible]=float('nan')
        values=AlphaPool._normalize_by_day(raw).numpy()
        values=np.nan_to_num(values,nan=0,posinf=0,neginf=0)
        arrays.append(values)
        ic=[]
        for a,b,mask in zip(values,data.y.numpy(),data.eligible.numpy()):
            ar=rankdata(a[mask]);br=rankdata(b[mask]);sd=ar.std()*br.std()
            ic.append(float(np.corrcoef(ar,br)[0,1]) if sd>0 else 0.)
        daily.append(ic)
    x=np.stack(arrays,axis=-1);y=data.y.numpy();eligible=data.eligible.numpy();daily=np.array(daily).T
    # Expanding QR retains OLS information; only outcome months strictly before query are incorporated.
    k=x.shape[-1];R=np.empty((0,k+2));rows=[];selections=[];ols_checks=[]
    for t,code in enumerate(data.codes):
        if t>0:
            a=np.column_stack([x[t-1,eligible[t-1]],np.ones(eligible[t-1].sum()),y[t-1,eligible[t-1]]])
            R=np.linalg.qr(np.vstack([R,a]),mode='r')
        if code<202201:continue
        hist=daily[:t];mean=hist.mean(0);std=hist.std(0,ddof=1)
        ir=np.divide(mean,std,out=np.zeros_like(mean),where=std>0)
        selected=np.flatnonzero((abs(mean)>.015)&(abs(ir)>.15))[:n_factors]
        if not len(selected):selected=np.array([np.argmax(abs(ir))])
        cols=np.r_[selected,k];coef=np.linalg.lstsq(R[:,cols],R[:,-1],rcond=None)[0]
        pred=np.column_stack([x[t][:,selected],np.ones(data.n_stocks)])@coef
        if code in [202201,202512]:
            direct_x=x[:t][eligible[:t]][:,selected]
            direct_x=np.column_stack([direct_x,np.ones(len(direct_x))])
            direct_y=y[:t][eligible[:t]]
            direct_coef=np.linalg.lstsq(direct_x,direct_y,rcond=None)[0]
            direct_pred=np.column_stack([x[t][:,selected],np.ones(data.n_stocks)])@direct_coef
            err=float(np.max(abs(pred[eligible[t]]-direct_pred[eligible[t]])))
            np.testing.assert_allclose(pred[eligible[t]],direct_pred[eligible[t]],rtol=1e-7,atol=1e-8)
            ols_checks.append(dict(month=int(code),max_prediction_error=err,fit_through=int(data.codes[t-1])))
        assert np.isfinite(pred[eligible[t]]).all()
        rows.extend(dict(return_month=int(code),permno=int(p),score=float(s)) for p,s in zip(data._stock_ids[eligible[t]],pred[eligible[t]]))
        selections.append(dict(return_month=int(code),fit_through=int(data.codes[t-1]),selected=selected.tolist()))
    if verify_only:
        actual=pd.DataFrame(rows).sort_values(['return_month','permno']).reset_index(drop=True)
        saved=pd.read_csv(out/'scores.csv').sort_values(['return_month','permno']).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[['return_month','permno']],saved[['return_month','permno']])
        np.testing.assert_allclose(actual.score,saved.score,rtol=1e-7,atol=1e-8)
    else:
        pd.DataFrame(rows).to_csv(out/'scores.csv',index=False)
        write_json(out/'selections.json',selections)
    write_json(out/'ols-verification.json',dict(passed=True,checks=ols_checks,method='Expanding QR predictions versus direct historical-only OLS'))
    if verify_only:return
    import subprocess
    subprocess.run([str(BASE/'.venv-tra/bin/python'),str(BASE/'scripts/evaluate_cz.py'),'--scores',str(out/'scores.csv'),'--output',str(out/'evaluation')],check=True)
