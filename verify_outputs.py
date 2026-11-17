#!/usr/bin/env python3
"""Guarded independent verification of saved native outputs; no retraining."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'release/data';OUT=ROOT/'results/released-algorithm'
sys.path.insert(0,str(ROOT/'upstream'))
from src.rura_core_fast import run_rura_for_user,ruraConfig
from src.metrics_core_fast import roi_from_prices,ndcg_at_k_binary
from src.groundtruth_utils import build_newbuy_labels,load_prices_at_dates
pa.set_cpu_count(1);pa.set_io_thread_count(1)


def main():
    status=json.loads((OUT/'status.json').read_text())
    assert status['status']=='complete_released_algorithm_artifact_scope'
    slices=[json.loads(x) for x in (DATA/'splits/slices.jsonl').read_text().splitlines()]
    cases=[];categories={x['user_id']:x['category'] for x in csv.DictReader((ROOT/'upstream/configs/user_categories.csv').open())}
    for meta in [slices[0],slices[-1]]:
        sid=meta['slice_id'];saved=np.load(OUT/sid/'rankings.npz');users=saved['user_ids'];assets=saved['asset_ids'];ids=saved['ranked_asset_indices'];weights=saved['learned_weights'];asset_index={a:i for i,a in enumerate(assets)}
        prof=pq.read_table(DATA/'profiles_riskmap_cat'/sid/'user_params.parquet').to_pandas().set_index('user_id')
        if any(categories.get(uid)!=str(prof.loc[uid,'category']) for uid in prof.index):raise RuntimeError('Profile and README utility category maps differ')
        stats=pq.read_table(DATA/'features'/sid/'market_stats.parquet').to_pandas().set_index('asset_id');pop=pq.read_table(DATA/'features'/sid/'popularity.parquet').to_pandas().set_index('asset_id');default=max(1e-8,float(np.nanmedian(stats.sigma2)))
        frame=pq.read_table(OUT/sid/'user-metrics.parquet').to_pandas().set_index(['user_id','lambda'])
        start=pd.Timestamp(meta['test_start']).date();end=pd.Timestamp(meta['test_end']).date()
        label=build_newbuy_labels(DATA/'clean/transactions.parquet',start,end,chunk_rows=200000)
        first,last=load_prices_at_dates(DATA/'clean/prices.parquet',list(assets),start,end,chunk_rows=200000)
        source=pq.ParquetFile(DATA/'candidates'/(sid+'_all.parquet'))
        if source.metadata.num_row_groups!=len(users):raise RuntimeError('Cannot assume one user per row group for verification')
        for j in np.unique(np.linspace(0,len(users)-1,64,dtype=int)):
            uid=str(users[j]);g=source.read_row_group(int(j)).to_pandas()
            if set(g.user_id.astype(str))!={uid}:raise RuntimeError('Source row group does not match saved user')
            g=g[g.eligible&~g.in_holdings_flag]
            rawitems=g.item_id.astype(str).to_numpy();pi=g.p_i.to_numpy();r=g.item_id.map(stats.r).fillna(0.).to_numpy();var=g.item_id.map(stats.sigma2).fillna(default).to_numpy();s=g.item_id.map(pop.s).fillna(0.).to_numpy()
            profile=prof.loc[uid]
            for a in range(9):
                lam=(a+1)/10
                ref=run_rura_for_user(sid,uid,rawitems,pi,r,var,s,float(profile.gamma_or_eta),float(profile.B),10,ruraConfig(method='rura_la',lambda_div=lam,beta_pop=.1,L=30,w_max=.4,device='cpu'),None,'unchanged_author',2025)
                refids=[asset_index[x.item_id] for x in ref];refw=np.array([x.w for x in ref]);keep=ids[j,a]>=0;actualids=ids[j,a,keep];actualw=weights[j,a,keep]
                same_ids=np.array_equal(actualids,refids);weight_gap=float(np.max(np.abs(actualw-refw))) if len(refw) else 0.
                items=assets[actualids]
                native_roi=roi_from_prices(items,actualw,first,last,10.,10.,10.)
                native_ndcg=ndcg_at_k_binary(items,label.get(uid,set()),len(items))
                result=frame.loc[(uid,lam)]
                roi_gap=abs(float(result.native_period_roi)-native_roi);ndcg_gap=abs(float(result.ndcg)-native_ndcg)
                cases.append({'slice':sid,'user':uid,'lambda':lam,'rank_ids_equal':same_ids,'max_abs_weight_difference':weight_gap,'roi_difference':roi_gap,'ndcg_difference':ndcg_gap})
                if not same_ids or weight_gap>1e-12 or roi_gap>1e-12 or ndcg_gap>1e-12:raise RuntimeError('Saved native output fails independent unchanged-author/accounting check: '+str(cases[-1]))
    summaries=[json.loads((OUT/x['slice_id']/'summary.json').read_text()) for x in slices]
    overall=[]
    for lam in [i/10 for i in range(1,10)]:
        cells=[g for s in summaries for g in s['groups'] if g['category']=='Overall' and g['lambda']==lam]
        assert len(cells)==16
        overall.append({'lambda':lam,'user_slices':sum(x['users'] for x in cells),'missing_exact_endpoint_slot_fraction':float(np.average([x['missing_exact_endpoint_slot_fraction'] for x in cells],weights=[x['users'] for x in cells]))})
    result={'status':'passed','cases':len(cases),'panel_positions':'64deterministic evenly-spaced users from first/last slices x9variants,including users not in initial prefix parity probe','native_rank_weights_match_unchanged_author':True,'native_roi_ndcg_match_unchanged_author_helpers':True,'all_category_maps_match_readme_csv':True,'max_abs_weight_difference':max(x['max_abs_weight_difference'] for x in cases),'max_abs_roi_difference':max(x['roi_difference'] for x in cases),'max_abs_ndcg_difference':max(x['ndcg_difference'] for x in cases),'all_variants':overall,'cases_detail':cases,'not_daily_ledger':'Native ROI is overlapping period metric with exact-date missing-price zero contribution,not a cashflow-complete daily backtest.','verification_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    aggregate=pd.read_csv(OUT/'aggregate.csv')
    inventory=json.loads((ROOT/'experiment-inventory.json').read_text())
    comparisons=[]
    for exp in inventory['original_paper_experiments']:
        if not exp['id'].startswith('rura_lambda') or 'reported' not in exp:continue
        lam=float(exp['id'].replace('rura_lambda',''));reported=exp['reported']
        overallrow=aggregate[(aggregate['lambda']==lam)&(aggregate.category=='Overall')].iloc[0]
        for metric in ['ndcg_percent','roi_percent']:
            got=float(overallrow[metric]);expected=reported[metric]
            comparisons.append({'lambda':lam,'category':'Overall','metric':metric,'paper':expected,'reproduced':got,'difference':got-expected,'within_printed_rounding':abs(got-expected)<=.00005,'primary_native_table_comparison':True})
        for cat,expected in zip(['Aggressive','Income','Balanced','Conservative','Unknown','Overall'],reported['utility_x1e4']):
            row=aggregate[(aggregate['lambda']==lam)&(aggregate.category==cat)].iloc[0]
            for col in ['utility_single_scale_x1e4','utility_readme_double_scale_x1e4']:
                got=float(row[col])
                comparisons.append({'lambda':lam,'category':cat,'metric':col,'paper':expected,'reproduced':got,'difference':got-expected,'within_printed_rounding':abs(got-expected)<=.0005,'primary_native_table_comparison':col=='utility_readme_double_scale_x1e4'})
    comp=pd.DataFrame(comparisons);comp.to_csv(OUT/'paper-comparison.csv',index=False)
    primary=comp[comp.primary_native_table_comparison]
    result['paper_comparison']={'native_primary_cells':len(primary),'within_printed_rounding':int(primary.within_printed_rounding.sum()),'native_metric_cells':int(primary.metric.isin(['ndcg_percent','roi_percent']).sum()),'native_metric_cells_within_rounding':int(primary[primary.metric.isin(['ndcg_percent','roi_percent'])].within_printed_rounding.sum()),'readme_double_utility_cells':18,'double_utility_cells_within_rounding':int(primary[primary.metric=='utility_readme_double_scale_x1e4'].within_printed_rounding.sum()),'single_scale_diagnostic_cells':18,'single_scale_diagnostic_matches':int(comp[comp.metric=='utility_single_scale_x1e4'].within_printed_rounding.sum()),'selection':'Both utility conventions fixed from paper/README before evaluation,all variants retained; no best-test configuration chosen.'}
    target=ROOT/'results/native-output-verification.json';tmp=target.with_name(target.name+'.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(target)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    native_curve=aggregate[aggregate.category=='Overall'].sort_values('lambda')
    eq=pd.read_csv(ROOT/'results/paper-equations/figure1-numerical-data.csv')
    fig,axes=plt.subplots(3,1,figsize=(8,9),sharex=True,layout='constrained')
    for ax,col,eqcol,label,target_values in zip(axes,['roi_percent','ndcg_percent','utility_readme_double_scale_x1e4'],['roi_percent','ndcg_percent','utility_x1e4'],['Period ROI (%)','nDCG@10 (%)','Proxy utility (x 1e-4)'],[[.2924,.3878,.5166],[.0955,.1053,.1155],[3.602,3.593,3.583]]):
        ax.plot(native_curve['lambda'],native_curve[col],marker='o',label='Released routine / README conventions')
        ax.plot(eq['lambda'],eq[eqcol],marker='.',label='Published equations / stated slate metric')
        ax.scatter([.1,.5,.9],target_values,color='#b54b32',marker='x',s=60,label='Reported table cells')
        ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);axes[-1].set_xlabel('Diversification lambda');axes[-1].set_xticks([i/10 for i in range(1,10)])
    fig.suptitle('RURA: separate algorithm and evaluation conventions\nAll fixed variants retained; these are not daily trading returns',fontsize=11)
    fig.savefig(OUT/'implementation-comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases_detail','all_variants']}),flush=True)


if __name__=='__main__':main()
