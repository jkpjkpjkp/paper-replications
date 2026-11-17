"""Resumable Kronos-base CSI300 Table 10 reconstruction, without fine-tuning.

The author data snapshot and exact test end are not released. This runner freezes
those substitutions in protocol.json; its results must not be called an exact replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/kronos'))
MODEL_REV = '2b554741eca47781b64468546e77fef3e85130e6'
TOKENIZER_REV = '0e0117387f39004a9016484a186a908917e22426'
MODEL_HASH = 'abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83'
TOKENIZER_HASH = '59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_bin(path, n):
    a = np.fromfile(path, dtype='<f4')
    result = np.full(n, np.nan, dtype=np.float32)
    start = int(a[0])
    result[start:start+len(a)-1] = a[1:]
    return result


def prepare(args):
    out, data = args.output, args.data
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        manifest = json.loads((out/'manifest.json').read_text())
        assert sha(out/'panel.npz') == manifest['panel_sha256']
        assert sha(out/'samples.csv') == manifest['samples_sha256']
        assert sha(out/'protocol.json') == manifest['protocol_sha256']
        print('Reusing verified prepared inputs', manifest['n_samples'], flush=True)
        return
    calendar = pd.to_datetime((data/'calendars/day.txt').read_text().splitlines())
    membership = pd.read_csv(data/'instruments/csi300.txt', sep='\t', header=None,
                             names=['symbol','start','end'], parse_dates=['start','end'])
    first = calendar.searchsorted(pd.Timestamp(args.start))
    last = calendar.searchsorted(pd.Timestamp(args.end), side='right')-1
    # Qlib forms the day's orders from the previous trading day's signal.
    origin_indices = np.arange(first-1, last)
    begin, stop = first-90, last+11
    dates = calendar[begin:stop]
    assert len(dates) == stop-begin and begin >= 0
    relevant = membership[(membership.start <= calendar[last-1]) &
                          (membership.end >= calendar[first-1])]
    symbols = sorted(relevant.symbol.unique())
    provider = out/'qlib_adjusted'
    (provider/'calendars').mkdir(parents=True, exist_ok=True)
    (provider/'instruments').mkdir(exist_ok=True)
    (provider/'calendars/day.txt').write_text('\n'.join(d.strftime('%Y-%m-%d') for d in dates)+'\n')
    # Keep quotes after membership exit for valuing/liquidating existing holdings.
    (provider/'instruments/all.txt').write_text(''.join(
        f'{s}\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n' for s in symbols))
    shutil.copy2(data/'instruments/csi300.txt', provider/'instruments/csi300.txt')
    receipts = {str(data/'calendars/day.txt'):sha(data/'calendars/day.txt'),
                str(data/'instruments/csi300.txt'):sha(data/'instruments/csi300.txt')}
    panels, rows, exclusions = [], [], []
    for symbol in symbols:
        source = data/'features'/symbol.lower()
        raw = {}
        for field in ['open','high','low','close','volume','factor']:
            file = source/f'{field}.day.bin'
            receipts[str(file)] = sha(file)
            raw[field] = read_bin(file, len(calendar))[begin:stop].astype(np.float64)
        valid = np.flatnonzero(np.isfinite(raw['close']*raw['factor']) &
                               (raw['close']>0) & (raw['factor']>0))
        assert len(valid), symbol
        # Convert raw Tushare prices/lots to the price/volume convention Qlib expects.
        anchor = raw['close'][valid[0]] * raw['factor'][valid[0]]
        factor = raw['factor']/anchor
        prices = np.stack([raw[f]*factor for f in ['open','high','low','close']], axis=1)
        volume = raw['volume']*100/factor
        volume = np.where(np.isfinite(volume), volume, 0.0)
        # Released qlib_data_preprocess.py constructs this proxy for turnover.
        amount = prices.mean(axis=1)*volume
        panel = np.column_stack([prices, volume, amount]).astype(np.float32)
        panels.append(panel)
        adjusted = dict(zip(['open','high','low','close','volume','amount'], panel.T))
        adjusted['factor'] = factor
        adjusted['change'] = np.r_[np.nan, prices[1:,3]/prices[:-1,3]-1]
        folder = provider/'features'/symbol.lower()
        folder.mkdir(parents=True, exist_ok=True)
        for field, values in adjusted.items():
            np.r_[0, values].astype('<f4').tofile(folder/f'{field}.day.bin')
        intervals = relevant[relevant.symbol == symbol]
        si = len(panels)-1
        for origin in origin_indices:
            day = calendar[origin]
            if not ((intervals.start <= day) & (intervals.end >= day)).any():
                continue
            t = origin-begin
            x = panel[t-89:t+1]
            # Missing price bars are hard boundaries; never compress trading time.
            if x.shape != (90,6) or not np.isfinite(x).all() or (x[:,:4]<=0).any():
                exclusions.append(dict(symbol=symbol, date=str(day.date()), reason='invalid_90_session_context'))
                continue
            rows.append(dict(symbol=symbol, symbol_index=si, date=str(day.date()),
                             time_index=t, last_close=float(panel[t,3])))
    frame = pd.DataFrame(rows).sort_values(['date','symbol']).reset_index(drop=True)
    frame.to_csv(out/'samples.csv', index=False)
    np.savez_compressed(out/'panel.npz', panel=np.stack(panels), dates=dates.values,
                        symbols=np.asarray(symbols))
    pd.DataFrame(exclusions).to_csv(out/'excluded.csv',index=False)
    protocol = dict(
        task='Kronos-base pretrained CSI300 investment simulation reconstruction',
        paper='https://arxiv.org/html/2508.02739v1#A4', table=10,
        paper_targets=dict(annualized_excess_return=0.1911, information_ratio=1.3782),
        rounding_tolerance=0.00005, paper_seed_uncertainty='not reported in Table 10',
        model_revision=MODEL_REV, tokenizer_revision=TOKENIZER_REV,
        lookback=90, horizon=10, temperature=0.6, top_p=0.9, top_k=0, sample_count=10,
        normalization='context-only per-feature population mean/std + 1e-5; clip [-5,5]',
        signal='mean of 10 forecast closes / current close - 1; average 10 Monte Carlo paths',
        execution='next trading session open, Qlib TopkDropoutStrategy',
        topk=50, n_drop=5, hold_thresh=5, risk_degree=0.95,
        open_cost=0.0015, close_cost=0.0015, min_cost=5, account=100_000_000,
        limit_threshold=0.095, annualization_primary=238, annualization_sensitivity=252,
        test_start=args.start, test_end=args.end, seeds=args.seeds, batch_size=args.batch_size,
        rng='seed*100000000 + global batch start; torch manual_seed before each batch',
        data_source='joshuaxql/qlib_data@ee0fa6b2299cb5e927d2df491888c84f6c9c027a',
        data_path=str(data), price_adjustment='raw OHLC * dated factor / historical anchor',
        volume_adjustment='raw Tushare lots * 100 / normalized factor',
        amount='mean(adjusted OHLC) * adjusted volume (released demo convention)',
        membership='historical membership at signal formation; no future-membership filter',
        limitations=[
            'Community snapshot differs from the unreleased author snapshot.',
            '2025-06-05 end comes from released finetune/config.py; paper gives only July 2024 start.',
            'Paper specifies Qlib, 0.15% cost, top-k/drop-n, hold 5; execution open, 9.5% limit, minimum fee and 95% allocation follow released demo/Qlib defaults.',
            'Adjusted community inputs reconstruct Qlib conventions; source vendor conventions not independently audited.',
            'No fine-tuning or pretraining; other datasets and tasks remain untested.',
            'Cached float32 implementation is numerically checked against released inference; PyTorch runtime differs from paper.'
        ])
    write_json(out/'protocol.json', protocol)
    write_json(out/'manifest.json', dict(created_utc=utc(), n_samples=len(frame),
        n_symbols=len(symbols), n_signal_days=frame.date.nunique(), excluded=len(exclusions),
        panel_sha256=sha(out/'panel.npz'), samples_sha256=sha(out/'samples.csv'),
        protocol_sha256=sha(out/'protocol.json'), source_sha256=receipts))
    print('Prepared',len(frame),'samples;',len(symbols),'symbols;',frame.date.nunique(),'days',flush=True)


def load_models(args):
    import torch
    from model import Kronos, KronosTokenizer
    torch.set_num_threads(args.threads)
    cache = Path.home()/'.cache/huggingface/hub'
    paths = [cache/'models--NeoQuasar--Kronos-Tokenizer-base'/'snapshots'/TOKENIZER_REV,
             cache/'models--NeoQuasar--Kronos-base'/'snapshots'/MODEL_REV]
    for p, expected in zip(paths,[TOKENIZER_HASH,MODEL_HASH]):
        assert sha(p/'model.safetensors') == expected, f'Checkpoint hash mismatch: {p}'
    physical_device = 'cuda:0' if args.device == 'cuda-offload' else args.device
    tok = KronosTokenizer.from_pretrained(str(paths[0])).eval().to(physical_device)
    model = Kronos.from_pretrained(str(paths[1])).eval()
    if args.device == 'cuda-offload':
        from kronos_cached import offload_transformers
        offload_transformers(model,physical_device,args.resident_layers)
    else:
        model.to(physical_device)
    write_json(args.output/'runtime.json', dict(python=sys.version, executable=sys.executable,
        torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, device=args.device,
        threads=args.threads,resident_layers=args.resident_layers,checkpoint_paths=list(map(str,paths)),
        model_sha256=MODEL_HASH, tokenizer_sha256=TOKENIZER_HASH,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                     [ROOT/'scripts/reproduce_pretrained.py', ROOT/'scripts/kronos_cached.py',
                      ROOT/'upstream/kronos/model/kronos.py',ROOT/'upstream/kronos/model/module.py']}))
    return tok, model


def inputs(panel, stamps, rows):
    x = np.stack([panel[int(r.symbol_index),int(r.time_index)-89:int(r.time_index)+1]
                  for r in rows.itertuples()])
    x_stamp = np.stack([stamps[int(r.time_index)-89:int(r.time_index)+1] for r in rows.itertuples()])
    y_stamp = np.stack([stamps[int(r.time_index)+1:int(r.time_index)+11] for r in rows.itertuples()])
    mean, std = x.mean(axis=1,keepdims=True), x.std(axis=1,keepdims=True)
    return np.clip((x-mean)/(std+1e-5),-5,5), x_stamp, y_stamp, mean, std


def infer(args):
    import torch
    from kronos_cached import cached_inference, verify_cache
    from model.kronos import auto_regressive_inference
    out = args.output
    protocol = json.loads((out/'protocol.json').read_text())
    assert args.batch_size == protocol['batch_size'], 'Do not change batch size on resume'
    assert args.seeds == protocol['seeds'], 'Requested seeds differ from frozen protocol'
    manifest = json.loads((out/'manifest.json').read_text())
    for name, key in [('panel.npz','panel_sha256'),('samples.csv','samples_sha256'),
                      ('protocol.json','protocol_sha256')]:
        assert sha(out/name) == manifest[key], f'Input integrity error: {name}'
    frame = pd.read_csv(out/'samples.csv')
    panel_file = np.load(out/'panel.npz')
    panel = panel_file['panel']
    dates = pd.DatetimeIndex(panel_file['dates'])
    stamps = np.stack([dates.minute,dates.hour,dates.weekday,dates.day,dates.month],axis=1).astype(np.float32)
    tok, model = load_models(args)
    physical_device = 'cuda:0' if args.device == 'cuda-offload' else args.device
    tensor = lambda a: torch.as_tensor(a, device=physical_device)
    x, xs, ys, mean, std = inputs(panel, stamps, frame.iloc[:2])
    started = time.monotonic()
    validation = verify_cache(model,tensor(np.concatenate([xs,ys],axis=1)))
    validation['device'] = args.device
    validation['resident_layers'] = args.resident_layers
    # End-to-end sampled forecast check on a real 90-day context, all 10 paths.
    torch.manual_seed(19)
    with torch.inference_mode():
        reference = auto_regressive_inference(tok,model,tensor(x),tensor(xs),tensor(ys),
            max_context=512,pred_len=10,T=0.6,top_p=0.9,sample_count=10)[:,-10:]
        torch.manual_seed(19)
        accelerated = cached_inference(tok,model,tensor(x),tensor(xs),tensor(ys))
    validation['sampled_forecast_max_absolute_error'] = float(np.max(np.abs(reference-accelerated)))
    np.testing.assert_allclose(reference,accelerated,atol=1e-4,rtol=1e-4)
    validation['wall_seconds'] = time.monotonic()-started
    validation['code_sha256'] = sha(ROOT/'scripts/kronos_cached.py')
    validation['sampled_forecast_batch'] = len(x)
    write_json(out/'cache_validation.json',validation)
    print('Cache validation:',validation,flush=True)
    if args.stage == 'verify':
        t0 = time.monotonic()
        rows = frame.iloc[:args.batch_size]
        x,xs,ys,mean,std = inputs(panel,stamps,rows)
        pred = cached_inference(tok,model,tensor(x),tensor(xs),tensor(ys))
        seconds = time.monotonic()-t0
        result=dict(batch_size=len(rows),batch_seconds=seconds,seconds_per_sample=seconds/len(rows),
                    estimated_hours_per_seed=seconds/len(rows)*len(frame)/3600,
                    shape=list(pred.shape),finite=bool(np.isfinite(pred).all()))
        write_json(out/'throughput.json',result)
        print('Throughput:',result,flush=True)
        return
    run_start = time.monotonic()
    completed_this_run = 0
    for seed in args.seeds:
        folder = out/f'seed-{seed}'
        folder.mkdir(exist_ok=True)
        # Shards are multiples of batch size, so crash/restart preserves RNG grouping.
        shard_size = args.batch_size*16
        for first in range(0,len(frame),shard_size):
            end = min(first+shard_size,len(frame))
            dest = folder/f'predictions-{first:07d}-{end:07d}.npz'
            if dest.exists():
                with np.load(dest) as saved:
                    assert np.array_equal(saved['indices'], np.arange(first,end))
                    assert saved['prediction'].shape == (end-first,10,6)
                    assert np.isfinite(saved['prediction']).all()
                continue
            predictions=[]
            for start in range(first,end,args.batch_size):
                rows=frame.iloc[start:min(start+args.batch_size,end)]
                x,xs,ys,mean,std=inputs(panel,stamps,rows)
                torch.manual_seed(seed*100000000+start)
                predicted=cached_inference(tok,model,tensor(x),tensor(xs),tensor(ys))
                predicted=predicted*(std+1e-5)+mean
                assert np.isfinite(predicted).all()
                predictions.append(predicted)
                completed_this_run+=len(rows)
                elapsed=time.monotonic()-run_start
                write_json(out/'progress.json',dict(state='running',updated_utc=utc(),pid=os.getpid(),
                    seed=seed,completed=start+len(rows),total=len(frame),device=args.device,
                    completed_this_process=completed_this_run,elapsed_seconds=elapsed,
                    seconds_per_sample=elapsed/completed_this_run,
                    estimated_hours_remaining_current_seed=(len(frame)-start-len(rows))*elapsed/completed_this_run/3600))
            with open(dest.with_suffix('.tmp'),'wb') as f:
                np.savez_compressed(f,indices=np.arange(first,end),prediction=np.concatenate(predictions),
                                    device=np.asarray(args.device),seed=np.asarray(seed),
                                    protocol_sha256=np.asarray(sha(out/'protocol.json')))
            dest.with_suffix('.tmp').replace(dest)
            print(f'seed={seed} {end}/{len(frame)} samples; {elapsed/completed_this_run:.3f} s/sample',flush=True)
        backtest(args, seed)
    write_json(out/'progress.json',dict(state='complete',updated_utc=utc(),seeds=args.seeds))
    summarize(args)


def backtest(args, seed, diagnostic=False):
    import qlib
    from qlib.config import REG_CN
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.strategy import TopkDropoutStrategy
    out=args.output
    frame=pd.read_csv(out/'samples.csv',parse_dates=['date'])
    protocol=json.loads((out/'protocol.json').read_text())
    folder=out/f'seed-{seed}'
    folder.mkdir(exist_ok=True)
    if diagnostic:
        scores=np.zeros(len(frame))
    else:
        predictions=np.full((len(frame),10,6),np.nan,dtype=np.float32)
        seen=np.zeros(len(frame),dtype=bool)
        for file in sorted(folder.glob('predictions-*.npz')):
            with np.load(file) as saved:
                idx=saved['indices']
                assert not seen[idx].any()
                seen[idx]=True
                predictions[idx]=saved['prediction']
        assert seen.all() and np.isfinite(predictions).all(), 'Inference is incomplete'
        scores=predictions[:,:,3].mean(axis=1)/frame.last_close.to_numpy()-1
    signal=pd.Series(scores,index=pd.MultiIndex.from_arrays([frame.symbol,frame.date],
                      names=['instrument','datetime'])).sort_index()
    if not diagnostic:
        signal.rename('score').to_csv(folder/'signals.csv')
    benchmark_raw=json.loads((out/'evidence/csi300-benchmark.json').read_text())
    assert benchmark_raw['data']['code']=='000300'
    benchmark_rows=[row.split(',') for row in benchmark_raw['data']['klines']]
    benchmark=pd.Series([float(r[2]) for r in benchmark_rows],
                         index=pd.to_datetime([r[0] for r in benchmark_rows])).pct_change()
    qlib.init(provider_uri=str(out/'qlib_adjusted'),region=REG_CN,kernels=1,
              expression_cache=None,dataset_cache=None,logging_level=30)
    start=protocol['test_start']
    end=str(frame.date.unique()[5])[:10] if diagnostic else protocol['test_end']
    strategy=TopkDropoutStrategy(topk=50,n_drop=5,hold_thresh=5,signal=signal)
    portfolio,_=qlib_backtest(start_time=start,end_time=end,strategy=strategy,
        executor={'class':'SimulatorExecutor','module_path':'qlib.backtest.executor',
                  'kwargs':{'time_per_step':'day','generate_portfolio_metrics':True}},
        benchmark=benchmark,account=100_000_000,
        exchange_kwargs=dict(freq='day',codes=sorted(frame.symbol.unique()),
            limit_threshold=0.095,deal_price='open',open_cost=0.0015,close_cost=0.0015,min_cost=5))
    report, positions=portfolio['1day']
    assert report[['return','bench','cost']].notna().all().all()
    report.to_csv(folder/('diagnostic-backtest.csv' if diagnostic else 'daily-report.csv'))
    if diagnostic:
        assert (report.cost>0).any(), 'Diagnostic backtest executed no trades'
        write_json(out/'backtest_validation.json',dict(status='passed',days=len(report),
            first=str(report.index.min().date()),last=str(report.index.max().date()),
            cost_sum=float(report.cost.sum()),scope='zero-signal plumbing check, not a model result',
            benchmark_sha256=sha(out/'evidence/csi300-benchmark.json')))
        return
    excess=report['return']-report['bench']-report['cost']
    metrics={}
    for annualizer in [238,252]:
        aer=float(excess.mean()*annualizer)
        ir=float(excess.mean()/excess.std(ddof=1)*np.sqrt(annualizer))
        metrics[str(annualizer)]=dict(annualized_excess_return=aer,information_ratio=ir,
            annualized_excess_return_error=aer-0.1911,information_ratio_error=ir-1.3782,
            annualized_excess_return_relative_error=(aer-0.1911)/0.1911,
            information_ratio_relative_error=(ir-1.3782)/1.3782)
    write_json(folder/'metrics.json',dict(status='complete',seed=seed,days=len(report),
        n_predictions=len(frame),first=str(report.index.min().date()),last=str(report.index.max().date()),
        metrics_by_annualization=metrics,source_report_sha256=sha(folder/'daily-report.csv'),
        protocol_sha256=sha(out/'protocol.json'),benchmark_sha256=sha(out/'evidence/csi300-benchmark.json')))
    print('Completed seed metrics',seed,metrics,flush=True)
    summarize(args)


def summarize(args):
    runs=[json.loads(p.read_text()) for p in sorted(args.output.glob('seed-*/metrics.json'))]
    if not runs:
        return
    metrics={}
    for key in ['annualized_excess_return','information_ratio']:
        values=np.array([r['metrics_by_annualization']['238'][key] for r in runs])
        target={'annualized_excess_return':0.1911,'information_ratio':1.3782}[key]
        metrics[key]=dict(values=values.tolist(),mean=float(values.mean()),
            seed_sd=float(values.std(ddof=1)) if len(values)>1 else None,
            paper=target,absolute_error=float(abs(values.mean()-target)),
            relative_error=float(abs(values.mean()-target)/abs(target)),rounding_tolerance=0.00005)
    write_json(args.output/'summary.json',dict(status='complete' if len(runs)==len(args.seeds) else 'partial',
        updated_utc=utc(),scope='pretrained CSI300 reconstruction; community data and assumed terminal date',
        completed_seeds=[r['seed'] for r in runs],requested_seeds=args.seeds,metrics=metrics,
        limitations=json.loads((args.output/'protocol.json').read_text())['limitations']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/home/jkp/data/csi/qlib/cn_data'))
    p.add_argument('--stage',choices=['prepare','verify','infer','backtest','diagnostic','all'],default='all')
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--resident-layers',type=int,default=4)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--start',default='2024-07-01')
    p.add_argument('--end',default='2025-06-05')
    args=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1'
    if args.stage in ['all','prepare']:
        prepare(args)
    if args.stage=='diagnostic':
        backtest(args,0,diagnostic=True)
    if args.stage in ['all','infer','verify']:
        try:
            infer(args)
        except Exception as exc:
            if args.stage != 'verify':
                write_json(args.output/'failure.json',dict(state='failed',updated_utc=utc(),
                    pid=os.getpid(),error_type=type(exc).__name__,message=str(exc)))
            raise
    if args.stage=='backtest':
        for seed in args.seeds:
            backtest(args,seed)


if __name__=='__main__':
    main()
