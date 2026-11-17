"""Inventory every published numerical claim; compare cached counterparts.

No alternative-data result is promoted to an exact-paper replication.
"""
from pathlib import Path
import hashlib,json,re,sys,importlib.util
from datetime import datetime,timezone
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PAPER=(HERE/'paper-v1.txt').read_text()
sys.path.insert(0,str(ROOT))
from limt.data import Panel
from limt.portfolio import backtest


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def table(name,end):
    return [l.strip() for l in PAPER.split(name,1)[1].split(end,1)[0].splitlines() if l.startswith('|')]


def numeric(s):
    m=re.search(r'[-+]?\d+(?:\.\d+)?',s)
    return float(m.group())


def targets():
    rows=[]
    def add(table,row,column,value,std=None):
        rows.append(dict(table=table,row=row,column=column,published=value,published_sd=std))
    for line in table('TABLE I:','TABLE II:'):
        cells=[s.strip() for s in line.split('|')[1:-1]]
        if len(cells)==9 and cells[0]=='Relative Improvement (%)':
            for j,c in enumerate(cells[1:]):
                add('Derived-I',cells[0],('csi300' if j<4 else 'csi500')+'/'+['IC','ICIR','AR_excess','IR_excess'][j%4],numeric(c)/100)
            continue
        if len(cells)!=9 or '±' not in cells[1]:continue
        for j,c in enumerate(cells[1:]):
            sd=numeric(c.split('\\pm')[-1])
            add('I',cells[0],('csi300' if j<4 else 'csi500')+'/'+['IC','ICIR','AR_excess','IR_excess'][j%4],numeric(c),sd)
    for line in table('TABLE II:','In this section'):
        cells=[s.strip() for s in line.split('|')[1:-1]]
        if len(cells)!=5 or '±' not in cells[1]:continue
        for col,c in zip(['IC','ICIR','AR_excess','IR_excess'],cells[1:]):
            # HTML extraction repeats some formula values: first mean, final SD.
            add('II',cells[0],col,float(re.match(r'[-+]?\d+\.\d{4}',c).group()),numeric(c.split('\\pm')[-1]))
    for line in table('TABLE III:','#### IV-C3'):
        cells=[s.strip() for s in line.split('|')[1:-1]]
        if len(cells)!=5 or not re.search(r'\d',cells[1]) or cells[0]=='Metric':continue
        for col,c in zip(['equal','risk_parity','rank','apo'],cells[1:]):
            add('III',cells[0],col,numeric(c)/(100 if '%' in c else 1))
    for line in table('TABLE IV:','The best return'):
        cells=[s.strip() for s in line.split('|')[1:-1]]
        if len(cells)!=10 or not re.match(r'^0\.\d$',cells[0]):continue
        for col,c in zip(['OSR','AR','ShR','MD','P/L','SoR','CR','DownsideVol','Turnover'],cells[1:]):
            add('IV',cells[0],col,numeric(c))
    for label,value in [('volume_IC',.3435),('volume_ICIR',2.95),('volatility_IC',.5157),('volatility_ICIR',4.36),('FF5_daily_alpha',.0035)]:
        add('Text',label,'value',value)
    assert sum(r['table']=='I' for r in rows)==120
    assert sum(r['table']=='II' for r in rows)==20
    assert sum(r['table']=='III' for r in rows)==36
    assert sum(r['table']=='IV' for r in rows)==81
    return rows


def main(recompute=False):
    records=targets(); measured={};sources={}
    def save(key,values,paths):
        measured[key]=(float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else None)
        sources[key]=[str(p) for p in paths]
    for u in ['csi300','csi500']:
        for model in ['LiMT','LightGBM']:
            folders=[ROOT/'results'/('main' if model=='LiMT' else 'baseline')/u/f'seed{s}' for s in range(3)]
            metrics=[]
            for folder in folders:
                d=json.loads((folder/'metrics.json').read_text())['test']
                metrics.append(d['return'] if model=='LiMT' else d['return_metrics'])
            for metric in ['IC','ICIR']:save(('I',model,u+'/'+metric),[d[metric] for d in metrics],[p/'metrics.json' for p in folders])
            if model=='LiMT' and u=='csi300':
                for task in ['volume','volatility']:
                    for metric in ['IC','ICIR']:
                        vals=[json.loads((p/'metrics.json').read_text())['test'][task][metric] for p in folders]
                        save(('Text',task+'_'+metric,'value'),vals,[p/'metrics.json' for p in folders])
        if u!='csi300':continue
        dest=HERE/'cached-counterparts'/u;dest.mkdir(parents=True,exist_ok=True)
        panel=Panel(ROOT/'data'/u)
        portfolio=[];sens=[]
        for seed in range(3):
            predpath=ROOT/'results/main'/u/f'seed{seed}/test_predictions.parquet'
            pred=pd.read_parquet(predpath)
            for strategy in ['equal','risk_parity','rank','apo']:
                cache=dest/f'seed{seed}_{strategy}.json'
                if recompute or not cache.exists():
                    daily,metric=backtest(panel,pred,strategy=strategy,r=.1,fee=.001,capital=1e9,participation=.02,volume_multiplier=100.)
                    daily.to_csv(dest/f'seed{seed}_{strategy}.csv',index=False)
                    cache.write_text(json.dumps(metric,indent=2)+'\n')
                portfolio.append(dict(seed=seed,strategy=strategy,**json.loads(cache.read_text())))
            for tenth in range(1,10):
                r=tenth/10;cache=dest/f'seed{seed}_apo_r{r:.1f}.json'
                if recompute or not cache.exists():
                    _,metric=backtest(panel,pred,strategy='apo',r=r,fee=.001,capital=1e9,participation=.02,volume_multiplier=100.)
                    cache.write_text(json.dumps(metric,indent=2)+'\n')
                sens.append(dict(seed=seed,r=r,**json.loads(cache.read_text())))
        pm=pd.DataFrame(portfolio);sm=pd.DataFrame(sens)
        mapping={'OSR':'order_success_rate','AR':'annualized_return','DownsideVol':'downside_vol','MD':'max_drawdown','Turnover':'mean_turnover','ShR':'sharpe','P/L':'profit_loss_ratio','SoR':'sortino','CR':'calmar'}
        for rec in records:
            key=(rec['table'],rec['row'],rec['column'])
            if rec['table']=='III':
                label=re.sub(r'[^A-Za-z/]','',rec['row'].replace('\\uparrow','').replace('\\downarrow',''))
                vals=pm[pm.strategy==rec['column']][mapping[label]].tolist()
                save(key,vals,list(dest.glob('seed*_'+rec['column']+'.json')))
            elif rec['table']=='IV':
                vals=sm[np.isclose(sm.r,float(rec['row']))][mapping[rec['column']]].tolist()
                save(key,vals,list(dest.glob('seed*_apo_r'+rec['row']+'.json')))
    for rec in records:
        key=(rec['table'],rec['row'],rec['column'])
        rec['exact_status']='pending_original_data_and_protocol'
        if key in measured:
            mean,sd=measured[key]
            rec.update(counterpart_mean=mean,counterpart_sd=sd,difference=mean-rec['published'],
                counterpart_status='measured_on_alternative_2010_start_data',counterpart_sources=sources[key])
        else:rec['counterpart_status']='not_measured'
    figs=[dict(figure=i,status='not_exactly_replicated',description=d,existing_code=c,missing=m) for i,d,c,m in [
        (1,'Architecture schematic','limt/model.py','Paper equations and released core disagree; no numerical curve to fit.'),
        (2,'March20 cross-stock attention','scripts/interpret.py','Original data, selected checkpoint, seed, 100-stock sample/order.'),
        (3,'March20 temporal propagation','scripts/interpret.py','Original data, selected checkpoint, seed, 100-stock sample/order.'),
        (4,'March20 MoE routing','scripts/interpret.py','Original checkpoint and selected stocks/experts.'),
        (5,'APO versus three baseline NAV paths','limt/portfolio.py','Original price/execution inputs and unreleased APO exchange/accounting settings.')]]
    tasks=[]
    names=['LSTM','GRU','Transformer','ALSTM','SFM','TCN','TabNet','TFT','Localformer','MASTER','XGBoost','CatBoost','LightGBM','DoubleEnsemble','LiMT']
    for u in ['csi300','csi500']:
        for model in names:
            for seed in range(3):tasks.append(dict(table='I',model=model,universe=u,seed=seed,status='waiting_original_data_and_selected_config',reuse='Qlib baseline implementation' if model!='LiMT' else 'Pinned author core, existing tested adapter'))
    for name in sorted({r['row'] for r in records if r['table']=='II'}):
        for seed in range(3):tasks.append(dict(table='II',model=name,universe='csi300',seed=seed,status='waiting_original_data_and_ablation_mapping'))
    (HERE/'targets.json').write_text(json.dumps(records,indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if k!='counterpart_sources'} for r in records]).to_csv(HERE/'comparison.csv',index=False)
    (HERE/'figures.json').write_text(json.dumps(figs,indent=2)+'\n')
    (HERE/'jobs.json').write_text(json.dumps(tasks,indent=2)+'\n')
    counts={t:sum(r['table']==t for r in records) for t in ['I','II','III','IV','Text','Derived-I']}
    matched=sum('counterpart_mean' in r for r in records)
    access=json.loads((HERE/'author-files.json').read_text())
    access_note=('Login is verified, but the authenticated server rejects the validation-file download with HTTP400/code23018 (download file size limit). A supported client download or alternate author link is needed.' if access.get('login_verified') else 'The browser currently requires account login.')
    report=f'''LiMT ALL-PUBLISHED-RESULTS REPLICATION TRACKER
Updated: {datetime.now(timezone.utc).isoformat()}

Scope: all four numerical tables, all five figures (Fig1 is conceptual),
the four auxiliary-task scores, and the FF5 daily-alpha claim. The text's
p<0.01 alpha significance is a separate inequality to validate, not an
invented exact p-value. Relative-improvement percentages in TableI are derived
from its model cells; they must be recomputed from reproduced means.

Published numeric targets: {len(records)} means/scalars, plus140 reported SDs.
Counts: {counts}. Training plan: {len(tasks)} three-seed model/universe fits,
with cached prediction reuse for portfolio and sensitivity experiments.
Existing alternative-data counterparts measured: {matched} targets.
Exact original-data targets verified: 0. The cached counterparts cannot
establish exact replication, even when a rounded number happens to coincide.

Blocking inputs observed today:
1. Six original train/valid/test pickle files, total3,121,620,953 bytes, visible
   on the authors' Quark share. {access_note}
2. Published-run selected hyperparameters, seeds/checkpoints and ablation
   mappings are not released as a full experiment configuration.
3. The authors' original APO/exchange engine remains unpublished. The public
   repository explicitly describes the business code as confidential; issue1
   also requests the missing APO module and currently has no displayed reply.
4. FF5 claim needs the factor series, alignment, excess-return convention and
   exact spread/NAV denominator. Do not substitute US factors for China.
5. Interpretability figures omit the exact checkpoint and 100-stock sample/order.

Reuse: upstream/*, limt/model.py, scripts/train.py, scripts/baseline.py,
limt/portfolio.py, scripts/interpret.py and all completed joint checkpoints.
The separate-task experiment is not a replacement for TableII's five ablations.
The US experiment is not evidence that any original-paper cell was replicated.

Artifacts:
  targets.json / comparison.csv — every published scalar and measured counterpart
  figures.json — figure coverage and exactness requirements
  jobs.json — all90 TableI fits and15 TableII fits, no substitute-data launch
  author-files.json — six original filenames, sizes and download IDs
  author_run.py — original-data entry point; requires explicit resolved configuration

Next dependency: obtain/hash/inspect the six original files via an authenticated supported download
files, resolve original/open-source preprocessing semantics, and confirm split
coverage before launching. Reproduction attempts report measured discrepancies;
published test returns must never be used to tune until matching numbers appear.
'''
    (HERE/'STATUS.txt').write_text(report)
    (HERE/'manifest.json').write_text(json.dumps(dict(status='waiting_supported_client_download_or_alternate_author_link' if access.get('login_verified') else 'waiting_original_data',
        numeric_targets=len(records),reported_sd_targets=140,cached_counterparts=matched,exact_targets_verified=0,
        paper_sha256=sha(HERE/'paper-v1.txt'),audit_sha256=sha(__file__),
        upstream_sha256={p.name:sha(p) for p in (ROOT/'upstream').glob('*.py')}),indent=2)+'\n')
    print(report)


if __name__=='__main__':main('--recompute' in sys.argv)
