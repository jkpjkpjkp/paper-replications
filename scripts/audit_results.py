"""Read-only audit of saved experiments; writes only inside ijcai-paper.

Uses a separate NumPy implementation of Bartlett HAC; never imports the
experiment's analysis code. Bootstrap intervals are newly added, post hoc
robustness diagnostics, not preregistered evidence.
"""
from pathlib import Path
import csv, hashlib, json, shutil
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
EXP=BASE/'limt-us-reproduction/experiments/standard_characteristics'
OUT=ROOT/'evidence'
NAMES=['mkt_rf','smb','hml','rmw','cma','mom']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def hac(y,x,lag):
    n,k=x.shape
    inv=np.linalg.inv(x.T@x)
    b=np.linalg.lstsq(x,y,rcond=None)[0]
    z=x*(y-x@b)[:,None]
    meat=z.T@z
    for h in range(1,lag+1):
        g=z[h:].T@z[:-h]
        meat+=(1-h/(lag+1))*(g+g.T)
    cov=(n/(n-k))*inv@meat@inv
    se=np.sqrt(cov[0,0]);stat=b[0]/se
    q=t.ppf(.975,n-k)
    return dict(annual_alpha=b[0]*252,annual_alpha_se=se*252,
                t=stat,p=2*t.sf(abs(stat),n-k),ci_low=(b[0]-q*se)*252,
                ci_high=(b[0]+q*se)*252)

def main():
    OUT.mkdir(exist_ok=True)
    paths=[BASE/'protocol.txt',BASE/'paper-reproduction-audit/programme-status.json',
           BASE/'paper-reproduction-audit/papers.json',BASE/'paper-reproduction-audit/SOURCE_PROVENANCE.json',
           EXP/'REPORT.txt',EXP/'protocol.json',EXP/'status.json',EXP/'locked_before_training.json',
           EXP/'analysis/verification.json',EXP/'data/manifest.json',EXP/'data/universe_audit.csv',
           BASE/'no-cost-neutral-audit/REPORT.txt',BASE/'limt-us-reproduction/REPORT.txt',
           BASE/'arboreal-reproduction/finance/REPORT.md',
           BASE/'financial-graph-reproduction/protocol202/protocol.json']
    paths+=list((EXP/'analysis').glob('*.csv'))
    records=[]
    for p in paths:
        assert p.is_file(),p
        records.append(dict(path=str(p.relative_to(BASE)),sha256=sha(p),bytes=p.stat().st_size))
    save('local_source_hashes.json',records)
    for p in [EXP/'protocol.json',EXP/'analysis/alpha.csv',EXP/'analysis/performance.csv',
              EXP/'analysis/missing_stress.csv',EXP/'analysis/hac_sensitivity.csv',
              EXP/'analysis/verification.json',EXP/'data/universe_audit.csv']:
        shutil.copy2(p,OUT/p.name)
    shutil.copy2(BASE/'protocol.txt',OUT/'user_protocol.txt')
    status=json.loads((EXP/'status.json').read_text());assert status['stage']=='complete'
    old=json.loads((EXP/'analysis/verification.json').read_text());assert old['status']=='passed'
    for name,h in old['output_sha256'].items():assert sha(EXP/'analysis'/name)==h,name
    frozen=json.loads((EXP/'locked_before_training.json').read_text())
    input_changes=[]
    for path,h in frozen['inputs_sha256'].items():
        if not Path(path).is_file() or sha(Path(path))!=h:input_changes.append(path)
    assert not input_changes, input_changes
    assert len(json.loads((EXP/'data/manifest.json').read_text())['feature_names'])==202
    factors=pd.read_parquet(BASE/'factors-daily/ff5_momentum_daily.parquet')
    factors['date']=factors.date.astype(str)
    original=pd.read_csv(EXP/'analysis/alpha.csv');assert len(original)==36
    recomputed=[];ledgers={}
    for weight in ['value','volume']:
        for cost in [0,10]:
            name=f'{weight}_cost{cost}'
            df=pd.read_csv(EXP/'analysis'/f'{name}_ledger.csv')
            assert len(df)==2260 and not df.date.duplicated().any()
            assert (pd.to_datetime(df.signal_date)<pd.to_datetime(df.execution_date)).all()
            assert (pd.to_datetime(df.execution_date)<pd.to_datetime(df.date)).all()
            np.testing.assert_allclose(df.return_net,(df.pnl-df.fees)/df.start_nav,atol=1e-13)
            np.testing.assert_allclose(df.end_nav,df.start_nav+df.pnl-df.fees,rtol=1e-12)
            np.testing.assert_allclose(df.start_nav.to_numpy()[1:],df.end_nav.to_numpy()[:-1],rtol=1e-12)
            np.testing.assert_allclose(df.fees,(df.traded_dollars+df.final_liquidation_dollars)*cost/10000,atol=1e-6)
            assert df.net_exposure.abs().max()<1e-8
            d=df.merge(factors,on='date',how='left',validate='one_to_one');assert d[NAMES].notna().all().all()
            ledgers[name]=d
            for _,row in original[(original.weighting==weight)&(original.cost_bp==cost)].iterrows():
                if row.window=='full-2017-2025':s=d
                elif row.window=='paper-2017-2020':s=d[d.date.between('2017-01-01','2020-08-01')]
                else:s=d[d.date.between('2021-01-01','2025-12-31')]
                cols=[] if row.factor_model=='mean' else (NAMES[:-1] if row.factor_model=='FF5' else NAMES)
                x=np.column_stack([np.ones(len(s)),s[cols].to_numpy()])
                stats=hac(s.return_net.to_numpy(),x,20)
                for k,v in stats.items():np.testing.assert_allclose(v,row[k],atol=1e-10,rtol=1e-8)
                recomputed.append(dict(weighting=weight,cost_bp=cost,window=row.window,
                                       factor_model=row.factor_model,**stats))
    recomputed=pd.DataFrame(recomputed)
    order=np.argsort(recomputed.p.to_numpy());adjusted=np.empty(36)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(recomputed.p.to_numpy()[order]*np.arange(36,0,-1)))
    recomputed['p_holm36']=adjusted
    merged=recomputed.merge(original,on=['weighting','cost_bp','window','factor_model'],suffixes=['_new','_old'])
    np.testing.assert_allclose(merged.p_holm36_new,merged.p_holm36_old,atol=1e-12)
    recomputed.to_csv(OUT/'independent_hac.csv',index=False)
    primary=recomputed[(recomputed.window=='full-2017-2025')&(recomputed.factor_model=='FF5+UMD')].copy()
    primary.to_csv(OUT/'primary_results.csv',index=False)
    # Circular paired (y, X) block bootstrap, percentile intervals for alpha.
    rng=np.random.default_rng(20261110);boots=[]
    for name in ['value_cost0','value_cost10']:
        d=ledgers[name];y=d.return_net.to_numpy();x=np.column_stack([np.ones(len(d)),d[NAMES].to_numpy()]);n=len(y)
        for block in [20,60]:
            estimates=[]
            for b in range(2000):
                starts=rng.integers(n,size=int(np.ceil(n/block)))
                idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
                estimates.append(np.linalg.lstsq(x[idx],y[idx],rcond=None)[0][0]*252)
            lo,hi=np.quantile(estimates,[.025,.975])
            boots.append(dict(case=name,block_sessions=block,resamples=2000,seed=20261110,
                              ci_low=lo,ci_high=hi,interpretation='post_hoc_paired_circular_percentile_alpha_CI'))
    pd.DataFrame(boots).to_csv(OUT/'posthoc_bootstrap.csv',index=False)
    # Preserve the programme snapshot's own categories; no outcome adjudication here.
    programme=json.loads((BASE/'paper-reproduction-audit/programme-status.json').read_text())
    papers={p['id']:p for p in json.loads((BASE/'paper-reproduction-audit/papers.json').read_text())}
    rows=[]
    for p in programme['market_neutral_nocost']['papers']:
        item=papers[p['paper_id']]
        rows.append(dict(paper_id=p['paper_id'],title=item['title'],conference=item['conference'],
                         source_url=item['url'],legacy_return_status=p['status'],
                         legacy_return_outcome=p['reported_outcome'],
                         protocol_match='not_established_by_legacy_status'))
    assert len(rows)==437 and len({r['paper_id'] for r in rows})==437
    pd.DataFrame(rows).to_csv(OUT/'catalogue_437.csv',index=False)
    snapshot=dict(updated_at=programme['updated_at'],total_papers=437,
        workflow_counts=programme['counts'],legacy_return_counts=programme['market_neutral_nocost']['reported_outcome_counts'],
        note='Legacy zero-cost +50/-50 protocol differs from protocol.txt; counts are metadata, not failure rates.')
    assert sum(snapshot['workflow_counts'].values())==437
    assert sum(snapshot['legacy_return_counts'].values())==437
    save('coverage_snapshot.json',snapshot)
    fig,ax=plt.subplots(figsize=(3.35,1.8))
    pp=primary[primary.weighting=='value'].sort_values('cost_bp')
    for i,(_,r) in enumerate(pp.iterrows()):
        ax.errorbar(r.annual_alpha*100,1-i,
                    xerr=[[100*(r.annual_alpha-r.ci_low)],[100*(r.ci_high-r.annual_alpha)]],
                    fmt='o' if i==0 else 's',color=['#19567A','#9F472A'][i],capsize=4,ms=6)
    ax.axvline(0,color='black',lw=.9,ls='--')
    ax.set_yticks([1,0],['Gross','10 bp'],fontsize=9)
    ax.set_xlabel('Annual alpha (%)',fontsize=9)
    ax.tick_params(axis='x',labelsize=8)
    ax.set_ylim(-.7,1.7);ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right','left']].set_visible(False)
    fig.tight_layout();fig.savefig(ROOT/'figures/alpha_intervals.pdf');fig.savefig(ROOT/'figures/alpha_intervals.png',dpi=200);plt.close(fig)
    save('audit_validation.json',dict(status='passed',rows_recomputed=36,ledger_cases=4,
        saved_output_hashes_checked=len(old['output_sha256']),frozen_inputs_changed=input_changes,
        new_model_training=False,bootstrap_added_posthoc=True,
        human_source_and_claim_approval='pending',submission_ready=False))
    print('PASS: 36 HAC rows, 4 ledgers, Holm36, output hashes, frozen inputs and 437 catalogue records.')
    print(pd.DataFrame(boots).to_string(index=False))

if __name__=='__main__':main()
