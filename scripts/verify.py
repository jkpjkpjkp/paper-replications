"""Check artifacts independently of training, including real-data label offsets."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_data import sha,write_json
from limt.data import Panel
from limt.portfolio import observed_moments,portfolio_metrics
ROOT=Path(__file__).resolve().parents[1]


def verify(run_name='main',require_complete=True):
    records=[];datasets=[]
    assert subprocess.check_output(['git','-C',str(ROOT/'upstream'),'rev-parse','HEAD'],text=True).strip()=='c0232623d50db26b02275b6f77b682f25effeec5'
    assert not subprocess.check_output(['git','-C',str(ROOT/'upstream'),'status','--porcelain'],text=True).strip()
    benchmark_receipts=json.loads((ROOT/'sources/benchmarks/receipts.json').read_text())
    for universe in ['csi300','csi500']:
        benchmark_path=ROOT/'sources/benchmarks'/f'{universe}.csv'
        assert sha(benchmark_path)==benchmark_receipts[universe]['csv_sha256']
        assert sha(benchmark_path.with_suffix('.json'))==benchmark_receipts[universe]['response_sha256']
        panel=Panel(ROOT/'data'/universe)
        for name,digest in panel.manifest['output_hashes'].items():
            assert sha(panel.folder/name)==digest,(universe,name,'hash')
        source_hashes=json.loads((panel.folder/'source_hashes.json').read_text())
        for path,digest in source_hashes.items():
            assert sha(path)==digest,(universe,path,'source hash')
        assert panel.features.shape[-1]==158
        # Check actual source values, independently from the builder's label code.
        rng=np.random.default_rng(31)
        checked=0;feature_checks=0;source_cache={}
        norm=np.load(panel.folder/'normalization.npz')
        def source_field(symbol,field):
            key=(symbol,field)
            if key not in source_cache:
                data=np.fromfile(Path(panel.manifest['provider'])/'features'/symbol.lower()/f'{field}.day.bin',dtype='<f4')
                values=np.full(len(panel.dates),np.nan,dtype=float)
                start=int(data[0]);end=min(len(values),start+len(data)-1)
                values[start:end]=data[1:1+end-start]
                source_cache[key]=values
            return source_cache[key]
        for t in rng.choice(panel.split('test'),size=100,replace=False):
            i=int(rng.choice(np.flatnonzero(panel.eligible[:,t])))
            m=panel.market[i]
            expected=np.array([m[t+2,0]/m[t+1,0]-1,
                np.log(m[t+2,1])-np.log(m[t-3:t+2,1]).mean(),
                (m[t+2,4]-m[t+2,5])/m[t+2,6]])
            expected[~np.isfinite(expected)]=np.nan
            np.testing.assert_allclose(panel.labels[i,t],expected,rtol=5e-5,atol=1e-6,equal_nan=True)
            symbol=panel.symbols[i]
            c=source_field(symbol,'close')*source_field(symbol,'factor')
            v=source_field(symbol,'volume')
            o=source_field(symbol,'open')[t]
            raw_features={
                'ROC5':c[t-5]/c[t],
                'MA5':np.nanmean(c[t-4:t+1])/c[t],
                'VMA5':np.nanmean(v[t-4:t+1])/(v[t]+1e-12),
                'KLEN':(source_field(symbol,'high')[t]-source_field(symbol,'low')[t])/o,
            }
            for name,value in raw_features.items():
                j=panel.manifest['feature_names'].index(name)
                normalized=np.clip((value-norm['median'][j])/norm['scale'][j],-3,3)
                np.testing.assert_allclose(panel.features[i,t,j],normalized,rtol=5e-4,atol=1e-4,equal_nan=True,err_msg=f'{universe} {symbol} {panel.dates[t]} {name}')
                feature_checks+=1
            checked+=1
        bounds=dict(train=('2010-01-01','2014-12-31'),valid=('2015-01-01','2016-12-31'),test=('2017-01-01','2020-08-01'))
        for split,(start,end) in bounds.items():
            d=panel.split(split)
            assert np.all(panel.dates[d]>=start) and np.all(panel.dates[d+2]<=end)
        datasets.append(dict(universe=universe,real_data_label_checks=checked,
            real_data_feature_checks=feature_checks,normalization_sha256=sha(panel.folder/'normalization.npz'),
            source_files_verified=len(source_hashes),output_files_verified=len(panel.manifest['output_hashes'])))
        for seed in range(3):
            folder=ROOT/'results'/run_name/universe/f'seed{seed}'
            status=folder/'status.json'
            if not status.exists() or json.loads(status.read_text())['phase']!='complete':
                if require_complete:raise AssertionError(f'Run incomplete: {folder}')
                continue
            metrics=json.loads((folder/'metrics.json').read_text())
            pred=pd.read_parquet(folder/'test_predictions.parquet')
            assert not pred.duplicated(['date','symbol']).any()
            assert sorted(pred.date.unique())==panel.dates[panel.split('test')].tolist()
            assert np.isfinite(pred[['pred_return','pred_volume','pred_volatility']].to_numpy()).all()
            score=pred.groupby('date').apply(lambda f:f.pred_return.corr(f.label_return),include_groups=False)
            np.testing.assert_allclose(score.mean(),metrics['test']['return']['IC'],atol=1e-9)
            baseline_folder=ROOT/'results/baseline'/universe/f'seed{seed}'
            baseline_pred=pd.read_parquet(baseline_folder/'test_predictions.parquet')
            assert pred[['date','symbol']].equals(baseline_pred[['date','symbol']])
            np.testing.assert_allclose(pred.label_return,baseline_pred.label_return,equal_nan=True)
            baseline_ic=baseline_pred.groupby('date').apply(lambda f:f.pred_return.corr(f.label_return),include_groups=False)
            baseline_reported=json.loads((baseline_folder/'metrics.json').read_text())['test']['return_metrics']
            np.testing.assert_allclose(baseline_ic.mean(),baseline_reported['IC'],atol=1e-9)
            history=json.loads((folder/'history.json').read_text())
            best=max(history,key=lambda r:r['validation']['return']['IC'])
            assert best['epoch']==json.loads(status.read_text())['best_epoch']
            provenance=json.loads((folder/'provenance.json').read_text())
            assert provenance['data_manifest_sha256']==sha(panel.folder/'manifest.json')
            assert provenance['best_sha256']==sha(folder/'best.pt')
            assert provenance['model_code_sha256']==sha(ROOT/'limt/model.py')
            assert provenance['trainer_sha256']==sha(ROOT/'scripts/train.py')
            for strategy in ['equal','risk_parity','rank','apo','top50']:
                f=folder/f'backtest_{strategy}.csv'
                if not f.exists():
                    if require_complete:raise AssertionError(f'Missing backtest: {f}')
                    continue
                daily=pd.read_csv(f)
                assert np.isfinite(daily.return_net).all()
                assert np.all(daily.signal_date<daily.execution_date)
                assert np.all(daily.execution_date<daily.date)
                assert np.all(daily.filled_value<=daily.requested_value+1e-4)
                backtest_meta=json.loads((folder/'backtest_metrics.json').read_text())
                if require_complete or 'benchmark_sha256' in backtest_meta:
                    assert backtest_meta['benchmark_sha256']==sha(benchmark_path)
                assert backtest_meta['engine_sha256']==sha(ROOT/'limt/portfolio.py')
                assert backtest_meta['prediction_sha256']==sha(folder/'test_predictions.parquet')
                reported=backtest_meta['metrics'][strategy]
                recomputed=portfolio_metrics(daily.return_net,daily.turnover,daily.requested_value,daily.filled_value)
                for key,value in recomputed.items():
                    if value is not None:np.testing.assert_allclose(value,reported[key],atol=1e-8,rtol=1e-7)
                benchmark_series=pd.read_csv(benchmark_path).set_index('date')['return']
                np.testing.assert_allclose(daily.benchmark_return,benchmark_series.reindex(daily.date),atol=1e-12)
                excess=daily.return_net-daily.benchmark_return
                np.testing.assert_allclose(reported['excess_annualized_return'],excess.mean()*252,atol=1e-9)
                np.testing.assert_allclose(reported['excess_information_ratio'],excess.mean()/excess.std(ddof=1)*np.sqrt(252),atol=1e-8)
            baseline_book=baseline_folder/'backtest_top50.csv'
            if baseline_book.exists():
                daily=pd.read_csv(baseline_book)
                meta=json.loads((baseline_folder/'backtest_metrics.json').read_text())
                assert meta['engine_sha256']==sha(ROOT/'limt/portfolio.py')
                assert meta['prediction_sha256']==sha(baseline_folder/'test_predictions.parquet')
                assert meta['benchmark_sha256']==sha(benchmark_path)
                stats=portfolio_metrics(daily.return_net,daily.turnover,daily.requested_value,daily.filled_value)
                for key,value in stats.items():
                    if value is not None:np.testing.assert_allclose(value,meta['top50'][key],atol=1e-8,rtol=1e-7)
                excess=daily.return_net-daily.benchmark_return
                np.testing.assert_allclose(meta['top50']['excess_annualized_return'],excess.mean()*252,atol=1e-9)
                np.testing.assert_allclose(meta['top50']['excess_information_ratio'],excess.mean()/excess.std(ddof=1)*np.sqrt(252),atol=1e-8)
            elif require_complete:
                raise AssertionError(f'Missing baseline backtest: {baseline_book}')
            records.append(dict(universe=universe,seed=seed,predictions=len(pred),real_data_label_checks=checked))
    result=dict(passed=True,require_complete=require_complete,datasets=datasets,runs=records)
    case=ROOT/'results'/run_name/'csi300/seed0/interpretability'
    if require_complete:
        meta=json.loads((case/'metadata.json').read_text())
        assert meta['checkpoint_sha256']==sha(case.parent/'best.pt')
        assert meta['model_code_sha256']==sha(ROOT/'limt/model.py')
        assert meta['script_sha256']==sha(ROOT/'scripts/interpret.py')
        for name,digest in meta['output_hashes'].items():assert sha(case/name)==digest
        weights=np.load(case/'attention_and_routing.npz')
        for name in ['cross_attention','temporal_attention','routing']:
            values=weights[name]
            assert np.isfinite(values).all() and (values>=0).all()
            np.testing.assert_allclose(values.sum(axis=-1),1.,atol=1e-6)
        result['interpretability_verified']=True
    neutral_folder=ROOT/'results'/run_name/'market_neutral'
    neutral_cases=sorted(neutral_folder.glob('*_seed*_buy5_sell*.json'))
    if require_complete:
        expected={f'{u}_seed{s}_buy5_sell{fee}.json' for u in ['csi300','csi500'] for s in range(3) for fee in [5,10,15]}
        assert expected.issubset({p.name for p in neutral_cases})
    max_net=0.
    for meta_path in neutral_cases:
        meta=json.loads(meta_path.read_text());daily=pd.read_csv(meta_path.with_suffix('.csv'))
        assert meta['engine_sha256']==sha(ROOT/'limt/neutral.py')
        assert meta['prediction_sha256']==sha(ROOT/'results'/run_name/meta['universe']/f"seed{meta['seed']}"/'test_predictions.parquet')
        stats=meta['metrics']
        assert (daily.long_notional>=0).all() and (daily.short_notional>=0).all()
        assert daily.net_exposure.abs().max()<1e-8
        max_net=max(max_net,float(daily.net_exposure.abs().max()))
        equity_before=2e9*np.r_[1.,np.cumprod(1+daily.return_net.to_numpy())[:-1]]
        np.testing.assert_allclose((daily.long_notional-daily.short_notional)/equity_before,daily.net_exposure,atol=1e-14)
        np.testing.assert_allclose(daily.fees,daily.buy_value*stats['buy_fee']+daily.sell_value*stats['sell_fee'],atol=1e-5,rtol=1e-10)
        for key,value in portfolio_metrics(daily.return_net,daily.turnover).items():
            if value is not None:np.testing.assert_allclose(value,stats[key],atol=1e-8,rtol=1e-7)
    if neutral_cases:result['market_neutral']=dict(verified_cases=len(neutral_cases),maximum_absolute_net_exposure=max_net)
    write_json(ROOT/'results'/run_name/'verification.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='main');p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args();verify(a.run_name,not a.allow_incomplete)
