"""Pinned released Kronos inference on a predeclared 2026 daily-bar panel."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
import sys, json, hashlib, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'csi/qlib/cn_data'
OUT = ROOT / 'results/kronos'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'upstream/kronos'))
from model import Kronos, KronosTokenizer, KronosPredictor

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    t0 = time.monotonic()
    models = {}
    for name in ['Kronos-Tokenizer-base', 'Kronos-small', 'Kronos-base']:
        repo = 'NeoQuasar/' + name
        revision = HfApi().model_info(repo).sha
        directory = snapshot_download(repo, revision=revision,
                                      allow_patterns=['config.json', '*.safetensors'])
        models[name] = dict(repo=repo, revision=revision, directory=directory,
                            files={p.name: sha(p) for p in Path(directory).glob('*') if p.is_file()})
        (OUT / 'models.json').write_text(json.dumps(models, indent=2))
        print('downloaded', name, revision, flush=True)

    calendar = pd.to_datetime((DATA/'calendars/day.txt').read_text().splitlines())
    membership = pd.read_csv(DATA/'instruments/csi300.txt', sep='\t', header=None,
                              names=['symbol','start','end'])
    # Selection uses only membership at the start, never held-out performance.
    symbols = sorted(membership.loc[(membership.start <= '2026-01-05') &
                                     (membership.end >= '2026-01-05'), 'symbol'].unique())[:16]
    dates = [pd.Timestamp(x) for x in ['2026-01-05','2026-03-02','2026-05-06','2026-07-01']]
    fields = ['open','high','low','close','volume','amount','factor']
    inputs, xts, yts, identities, receipts = [], [], [], [], {}
    for symbol in symbols:
        frame = pd.DataFrame(index=calendar)
        for field in fields:
            p = DATA / 'features' / symbol.lower() / (field+'.day.bin')
            if not p.exists() and field == 'amount':
                p = DATA / 'features' / symbol.lower() / 'money.day.bin'
            if not p.exists():
                continue
            values = np.fromfile(p, dtype='<f4')
            start = int(values[0])
            frame.loc[calendar[start:start+len(values)-1], field] = values[1:]
            receipts[str(p)] = sha(p)
        # The publisher stores raw prices; apply its dated adjustment factors.
        for field in ['open','high','low','close']:
            frame[field] = frame[field] * frame['factor']
        for date in dates:
            end = int(calendar.searchsorted(date))
            hist, future = frame.iloc[end-128:end].copy(), frame.iloc[end:end+5].copy()
            use = [x for x in ['open','high','low','close','volume','amount'] if x in hist]
            if len(hist) != 128 or len(future) != 5 or hist[use].isna().any().any() or future.close.isna().any():
                identities.append(dict(symbol=symbol, date=str(date.date()), excluded='missing observations'))
                continue
            inputs.append(hist[use].reset_index(drop=True))
            xts.append(pd.Series(hist.index))
            yts.append(pd.Series(future.index))
            identities.append(dict(symbol=symbol,date=str(date.date()),batch_index=len(inputs)-1,
                                   last_close=float(hist.close.iloc[-1]),
                                   actual_close=future.close.tolist()))
    (OUT/'inputs.json').write_text(json.dumps(dict(symbols=symbols, windows=identities,
           source_files_sha256=receipts, calendar_sha256=sha(DATA/'calendars/day.txt'),
           membership_sha256=sha(DATA/'instruments/csi300.txt')), indent=2))
    rows = []
    for name in ['Kronos-small','Kronos-base']:
        tok = KronosTokenizer.from_pretrained(models['Kronos-Tokenizer-base']['directory'])
        model = Kronos.from_pretrained(models[name]['directory'])
        model.eval(); tok.eval()
        predictor = KronosPredictor(model, tok, device='cuda:0', max_context=512)
        for seed in [0,1,2]:
            torch.manual_seed(seed); np.random.seed(seed)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            predictions = []
            with torch.inference_mode():
                for first in range(0,len(inputs),8):
                    predictions.extend(predictor.predict_batch(inputs[first:first+8],
                        xts[first:first+8],yts[first:first+8],pred_len=5,
                        T=1.0,top_p=0.9,sample_count=5,verbose=False))
            for info in identities:
                if 'batch_index' not in info: continue
                pred = predictions[info['batch_index']]
                for h,(actual,estimate) in enumerate(zip(info['actual_close'],pred.close),1):
                    rows.append(dict(model=name,seed=seed,symbol=info['symbol'],date=info['date'],
                        horizon=h,actual_close=actual,predicted_close=float(estimate),
                        last_close=info['last_close'],actual_return=actual/info['last_close']-1,
                        predicted_return=float(estimate)/info['last_close']-1))
            pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False)
            print(name,seed,'seconds',time.monotonic()-start,'peak MiB',
                  torch.cuda.max_memory_allocated()/2**20,flush=True)
        del predictor, model, tok
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    metrics = []
    for (name,seed,h), group in frame.groupby(['model','seed','horizon']):
        error = group.predicted_return-group.actual_return
        metrics.append(dict(model=name,seed=int(seed),horizon=int(h),n=len(group),
            return_mae=float(abs(error).mean()),
            persistence_mae=float(abs(group.actual_return).mean()),
            direction_accuracy=float((np.sign(group.predicted_return)==np.sign(group.actual_return)).mean()),
            mean_cross_sectional_ic=float(group.groupby('date').apply(
                lambda x:x.predicted_return.corr(x.actual_return)).mean())))
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete',
        scope='released pretrained inference; reduced daily-data evaluation, not pretraining replication',
        n_windows=len(inputs),metrics=metrics,wall_seconds=time.monotonic()-t0,
        limitations=['community data and adjustment rules are not author snapshot',
                    'universe fixed at 2026-01-05; later membership exits are not simulated',
                    '64 windows and three sampling seeds do not establish portfolio profitability',
                    '2026 dates follow public model release; original pretraining corpus is not independently audited']),indent=2,allow_nan=False))

if __name__ == '__main__':
    main()
