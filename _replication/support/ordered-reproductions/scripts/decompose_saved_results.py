"""Granular descriptive diagnostics of saved outputs; no held-out model selection."""
from pathlib import Path
import json,ast
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gap-resolution-20261113';OUT.mkdir(exist_ok=True)
result={}
# Run the inspected author notebook's daily formula exactly; aggregate independently.
nb=json.loads((ROOT/'sources/TRA-qlib/examples/benchmarks/TRA/Reports.ipynb').read_text())
source=''.join(nb['cells'][1]['source']).replace('%matplotlib inline','')
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='func')
namespace={'pd':pd,'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'author_notebook_daily','exec'),namespace)
ret=pd.read_pickle(ROOT/'data/TRA/kdd21/ret.pkl').clip(-.1,.1)
tra={};details={}
for run in ['tra-original','tra-boundary-corrected']:
 for phase in ['initialization','tra']:
  key=run+'-'+phase;p=ROOT/'results'/run/phase/'pred.pkl';f=pd.read_pickle(p);f=f.loc['2018-09-21':'2020-06-30'].copy();f['ret']=ret.reindex(f.index)
  assert f.ret.notna().all();d=f.groupby(level='datetime',sort=True).apply(namespace['func']);details[key]=d
  r=d.R.reindex(pd.date_range(d.index.min(),d.index.max())).fillna(0)
  metrics={k:float(d[k].mean()) for k in ['MSE','MAE','IC']};metrics.update(ICIR=float(d.IC.mean()/d.IC.std()),AR=float(r.mean()*365),AV=float(r.std()*np.sqrt(365)),SR=float(r.mean()/r.std()*np.sqrt(365)),MDD=float((r.cumsum().cummax()-r.cumsum()).max()))
  published=json.loads(p.with_name('info.json').read_text())['metric'];assert all(np.isclose(metrics[k],published[k],rtol=1e-7) for k in published)
  d.to_parquet(OUT/(key+'-daily.parquet'))
  byyear=d.groupby(d.index.year).agg(days=('IC','size'),IC=('IC','mean'),MSE=('MSE','mean'),mean_long_short_return=('R','mean')).to_dict('index')
  sc=[c for c in f if c.startswith('score_')];pr=[c for c in f if c.startswith('prob_')]
  rank=f[['label','score']+sc].groupby(level=0).rank(pct=True)
  head_ic={c:float(rank.groupby(level=0).apply(lambda g:g[c].corr(g.label)).mean()) for c in sc}
  choice=f[pr].to_numpy().argmax(1);heads=f[sc].to_numpy();assert np.allclose(f.score.to_numpy(),heads[np.arange(len(f)),choice])
  avg=pd.Series(heads.mean(1),index=f.index).groupby(level=0).rank(pct=True)
  mean_ic=float(pd.DataFrame({'average':avg,'label':rank.label}).groupby(level=0).apply(lambda g:g.average.corr(g.label)).mean())
  shares=np.bincount(choice,minlength=len(sc))/len(f)
  tra[key]={'metrics':metrics,'by_year':byyear,'expert_IC':head_ic,'selected_head_shares':shares.tolist(),'equal_weight_head_average_IC_diagnostic_only':mean_ic,'rows':len(f)}
  print(key,metrics,flush=True)
tra['phase_differences']={run:{k:tra[run+'-tra']['metrics'][k]-tra[run+'-initialization']['metrics'][k] for k in ['IC','MSE','AR','SR']} for run in ['tra-original','tra-boundary-corrected']}
tra['paper_main_table']={'MSE':.157,'MAE':.318,'IC':.059,'ICIR':.460,'AR':.124,'AV':.140,'SR':.885,'MDD':.204}
tra['author_notebook_K10_more_precision']={'IC':.059403,'MSE':.156766}
tra['cautions']=['Our initialization uses three experts, not the paper plain-ALSTM baseline.','Per-expert/average-head metrics are descriptive, not candidates selected on test.','Author per-stock predictions/checkpoints referenced by notebook are not bundled in this checkout.','Cross-boundary corrected run is a sensitivity variant, not untouched source reproduction.']
result['TRA']=tra
# Decompose actual DoubleAdapt gross/net ledger; infer and verify native annualizer.
da={}
for run in ['doubleadapt-original','doubleadapt-online-lr-diagnostic','doubleadapt-script-matched']:
 base=ROOT/'results'/run;paths=list((base/'mlruns').rglob('report_normal_1day.pkl'));assert len(paths)==1
 ledger=pd.read_pickle(paths[0]);summary=json.loads((base/'summary.json').read_text())['metrics'];gross=ledger['return']-ledger.bench;net=gross-ledger.cost
 ann=summary['1day.excess_return_with_cost.annualized_return']/net.mean();assert abs(ann-round(ann))<1e-6
 assert np.isclose(summary['1day.excess_return_with_cost.information_ratio'],net.mean()/net.std()*np.sqrt(ann))
 d=pd.DataFrame({'gross_excess':gross,'net_excess':net,'cost':ledger.cost,'turnover':ledger.turnover,'portfolio_return':ledger['return'],'benchmark_return':ledger.bench})
 d.to_parquet(OUT/(run+'-ledger.parquet'))
 pred=pd.read_pickle(base/'predictions-raw-labels.pkl');daily=pred.groupby(level=0).apply(lambda g:pd.Series({'IC':g.prediction.corr(g.raw_label),'RankIC':g.prediction.corr(g.raw_label,method='spearman')}))
 d=d.join(daily)
 byyear={str(y):{'days':len(g),'IC':float(g.IC.mean()),'RankIC':float(g.RankIC.mean()),'annual_gross_excess':float(g.gross_excess.mean()*ann),'annual_net_excess':float(g.net_excess.mean()*ann),'annual_cost_drag':float(g.cost.mean()*ann),'mean_daily_turnover':float(g.turnover.mean())} for y,g in d.groupby(d.index.year)}
 da[run]={'annualizer':float(ann),'days':len(d),'annual_gross_excess':float(gross.mean()*ann),'annual_net_excess':float(net.mean()*ann),'annual_cost_drag':float(ledger.cost.mean()*ann),'mean_daily_turnover':float(ledger.turnover.mean()),'by_year':byyear}
 print(run,da[run],flush=True)
a=da['doubleadapt-original'];b=da['doubleadapt-script-matched'];da['matched_minus_baseline']={k:b[k]-a[k] for k in ['annual_gross_excess','annual_net_excess','annual_cost_drag','mean_daily_turnover']}
da['limitations']='Without author daily predictions/holdings this decomposes local changes, not the residual paper-minus-local portfolio gap. Annualized returns are the native additive ledger convention.'
result['DoubleAdapt']=da
(OUT/'saved-result-decomposition.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('complete',flush=True)
