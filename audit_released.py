#!/usr/bin/env python3
"""Guard-only independent audit of author-released metrics; never trains.
No generated trial scores, prices, predictions, or substituted datasets.
"""
import hashlib, json, math, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parent
UP=ROOT/'official/results'
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
SEEDS=[3407+10*t for t in range(10)]
DATASETS=['yelp','amazon','weibo','tfinance','reddit','tolokers','questions']
FIELDS={'AUROC':('auc_per_trial','auc_list'),'AUPRC':('pre_per_trial','pre_list'),'RecK':('rec_per_trial','rec_list')}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_text())
def atomic(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def vec(d,m):
    for k in FIELDS[m]:
        if k in d:return np.array(d[k],dtype=float)
    return None

def summaries():
    rows=[]
    for p in sorted(UP.rglob('*.json')):
        d=load(p)
        if not isinstance(d,dict):continue
        for metric in FIELDS:
            a=vec(d,metric)
            if a is None:continue
            valid=a[np.isfinite(a)]
            calculated_mean=float(valid.mean()) if len(valid) else None
            calculated_std=float(valid.std(ddof=0)) if len(valid) else None
            errors=[]
            if len(valid)!=len(a):errors.append('nonfinite_trials')
            if any((valid<0)|(valid>1)):errors.append('metric_out_of_unit_interval')
            if d.get('trials') is not None and len(a)!=d['trials']:errors.append('trial_count_mismatch')
            for suffix,computed in [('mean',calculated_mean),('std',calculated_std)]:
                reported=d.get(metric+'_'+suffix)
                if reported is not None and computed is not None and not math.isclose(reported,computed,rel_tol=1e-10,abs_tol=1e-12):errors.append('summary_'+suffix+'_mismatch')
            rows.append({'path':str(p.relative_to(ROOT)), 'sha256':sha(p), 'metric':metric,'n':len(a),'finite_n':len(valid),'mean':calculated_mean,'std_population':calculated_std,'declared_seeds':d.get('seeds'),'seed_alignment_verified':d.get('seeds')==SEEDS,'issues':errors})
    return rows

def compare(a,b,seed):
    assert len(a)==len(b)==10 and np.isfinite(a).all() and np.isfinite(b).all()
    diff=a-b; n=len(diff); variance=float(diff.var(ddof=1)); mu=float(diff.mean())
    corrected_se=math.sqrt((1/n+1)*variance)
    t=mu/corrected_se if corrected_se else 0.
    p_nb=float(2*stats.t.sf(abs(t),n-1)) if corrected_se else 1.
    w=stats.wilcoxon(diff,alternative='two-sided',zero_method='wilcox')
    rng=np.random.default_rng(seed)
    boots=np.mean(rng.choice(diff,size=(10000,n),replace=True),axis=1)
    lo,hi=np.quantile(boots,[.025,.975])
    ci_nb=[mu-stats.t.ppf(.975,n-1)*corrected_se,mu+stats.t.ppf(.975,n-1)*corrected_se]
    return {'n_paired':n,'mean_a':float(a.mean()),'mean_b':float(b.mean()),'mean_difference':mu,'wins':int((diff>0).sum()),'ties':int((diff==0).sum()),'wilcoxon_two_sided_p':float(w.pvalue),'nadeau_bengio_t':t,'nadeau_bengio_two_sided_p':p_nb,'nadeau_bengio_ci95':list(map(float,ci_nb)),'split_bootstrap_ci95':[float(lo),float(hi)],'split_bootstrap_resamples':10000,'split_bootstrap_seed':seed,'bootstrap_limit':'Resampling matched random-split differences is descriptive; overlaps violate independent-split inference. Not a time-series or return bootstrap.','differences':diff.tolist()}

def holm(rows,key,outkey):
    ordered=sorted(rows,key=lambda x:x[key]);prev=0.
    for i,r in enumerate(ordered):
        prev=max(prev,min(1.,(len(ordered)-i)*r[key]));r[outkey]=prev

checks=summaries(); pairs=[]
for ds in DATASETS:
    a_path=UP/f'fraudgnam_refined_{ds}.json'
    b_path=UP/'secgfd_corrected'/f'secgfd_{ds}_paired.json'
    a,b=load(a_path),load(b_path)
    assert a['seeds']==b['seeds']==SEEDS
    for metric in FIELDS:
        result=compare(vec(a,metric),vec(b,metric),20261107+len(pairs))
        result.update(dataset=ds,metric=metric,a_path=str(a_path),b_path=str(b_path),a_sha256=sha(a_path),b_sha256=sha(b_path))
        # Recompute deterministic numerical summaries twice; boot seed is fixed.
        assert result==compare(vec(a,metric),vec(b,metric),20261107+len(pairs))|{k:result[k] for k in ['dataset','metric','a_path','b_path','a_sha256','b_sha256']}
        pairs.append(result)
holm(pairs,'wilcoxon_two_sided_p','wilcoxon_holm_21_p')
holm(pairs,'nadeau_bengio_two_sided_p','nadeau_bengio_holm_21_p')
gnan=[]
a=load(UP/'fraudgnam_refined_tfinance.json');b=load(UP/'gnan/gnan_tfinance.json')
assert a['seeds']==b['seeds']==SEEDS
for j,metric in enumerate(FIELDS):
    x=compare(vec(a,metric),vec(b,metric),42900+j);x.update(dataset='tfinance',metric=metric,a_path=str(UP/'fraudgnam_refined_tfinance.json'),b_path=str(UP/'gnan/gnan_tfinance.json'));gnan.append(x)
holm(gnan,'wilcoxon_two_sided_p','wilcoxon_holm_3_p')
holm(gnan,'nadeau_bengio_two_sided_p','nadeau_bengio_holm_3_p')
original=load(UP/'real_paired_tests.json');old_agreement=[]
for r in pairs:
    ds=r['dataset'];key='refined:vs_secgfd:'+r['metric'];old=original.get(ds,{}).get(key,{})
    old_agreement.append({'dataset':ds,'metric':r['metric'],'source':'real_paired_tests.json','recomputed_using':'secgfd_corrected','same_inputs_as_saved':sha(UP/f'secgfd_{ds}_paired.json')==r['b_sha256'],'wilcoxon_saved':old.get('p_wilcoxon'),'wilcoxon_recomputed':r['wilcoxon_two_sided_p'],'mean_saved':old.get('mean_diff'),'mean_recomputed':r['mean_difference']})
tf=load(UP/'tfinance_full.json')
inactive_checks={}
for variant in ['no_highpass','no_nce','order2']:
    d=load(UP/f'tfinance_{variant}.json')
    inactive_checks[variant]={m:bool(np.array_equal(vec(tf,m),vec(d,m))) for m in FIELDS}
report={'paper_id':429,'created_at':datetime.now(timezone.utc).isoformat(),'scope':'independent arithmetic/statistical audit of author-released metric vectors; no locally trained models or raw-data verification','source_commit':'6f13f4f7363a5a39448fd759a14963ed7e7b8cb9','n_summary_vectors':len(checks),'summary_checks':checks,'paired_secgfd':pairs,'paired_gnan_tfinance':gnan,'saved_real_test_agreement':old_agreement,'inactive_tfinance_variants_identical':inactive_checks,'simulated_legacy_statistics':{'excluded_file':str(UP/'statistical_tests.json'),'generator':str(ROOT/'official/exp_stats.py'),'reason':'sample_paired_trials uses rng.multivariate_normal to simulate scores from table means/stds and assumed rho=.7. Never empirical evidence. Superseded by release metric vectors.'},'selection_multiplicity':{'secgfd_family_tests':21,'gnan_tfinance_family_tests':3,'adjustment':'Holm separately for Wilcoxon and corrected t statistics; report all comparisons','limits':['Refined rules were developed after examining benchmark results; this is retrospective released-artifact analysis.','Random-split observations overlap; Wilcoxon nominal p values do require assumptions about independent pairs.','The 21 metric-dataset outcomes are dependent; their binomial win p-value is not treated as an independent confirmatory test.','No untouched external test, checkpoint predictions or raw split masks verified.']},'market_neutral_nocost':{'status':'not_applicable','applicable':False,'reason':'Nontrading fraud classifier; accounts are not investable ranked securities and no asset-return series or portfolio signal is produced.'}}
atomic(OUT/'released-artifact-audit.json',report)
tf_pairs=[x for x in pairs if x['dataset']=='tfinance']
print(json.dumps({'summary_vectors':len(checks),'summary_issues':sum(bool(x['issues']) for x in checks),'tfinance_vs_secgfd':[{k:x[k] for k in ['metric','mean_a','mean_b','mean_difference','wilcoxon_two_sided_p','nadeau_bengio_two_sided_p','wilcoxon_holm_21_p']} for x in tf_pairs],'output':str(OUT/'released-artifact-audit.json')},indent=2))
